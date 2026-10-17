"""Warstwa PRZYSZŁEGO popytu: pozwolenia na budowę budynków wielorodzinnych w Warszawie.

Źródło: GUNB — Rejestr Wniosków, Decyzji i Zgłoszeń w sprawach budowlanych (RWDZ),
dane.gov.pl/pl/dataset/3467, plik „Wnioski i decyzje … – województwo mazowieckie”
(https://wyszukiwarka.gunb.gov.pl/pliki_pobranie/wynik_mazowieckie.zip, CSV ';', UTF-8,
aktualizowany co dzień; jeden wiersz = jedna działka objęta decyzją).

Wynik: data/gunb_permits.parquet — GeoDataFrame EPSG:4326, jeden wiersz na pozwolenie
(punkt = środek ciężkości działek inwestycji), kolumna n_flats = szacowana liczba mieszkań.

Metoda:
  1. Filtr: terc 1465xxx (m.st. Warszawa), kategoria XIII (budynki mieszkalne
     wielorodzinne), rodzaj zamierzenia „budowa nowego/nowych obiektów”, data decyzji
     od YEAR_FROM (2016 — potrzebne do kalibracji prognozy ludności na latach 2011–2021).
     Odrzucamy decyzje zmieniające / projekty zamienne (duplikowałyby mieszkania
     z pierwotnego pozwolenia) i obiekty pomocnicze (instalacje gazowe, altany, oranżerie…).
     Rejestr nie ma pola „rodzaj rozstrzygnięcia” — wg opisu GUNB zawiera rozstrzygnięcia
     I instancji, w praktyce pozwolenia; decyzji odmownych nie da się odfiltrować.
  2. Lokalizacja: id działki <jednostka>.<obręb>.<nr> -> GUGiK ULDK (GetParcelById, EPSG:2180),
     cache na dysku w data/cache/uldk/. Punkt pozwolenia = centroid działek ważony
     powierzchnią (małe działki drogowe/dojazdowe prawie nie przesuwają punktu).
     Literówki w kodzie jednostki (np. 186518_8) poprawiamy na 1465xx_8. Gdy żadna działka
     się nie znajdzie (działki często dzielono po wydaniu decyzji) -> geokoder adresowy
     GUGiK UUG (ulica + nr domu), a w ostateczności centroid obrębu ewidencyjnego (ULDK
     GetRegionById). Kolumna geo_source: 'uldk' | 'uug_address' | 'uldk_obreb'.
  3. Liczba mieszkań: RWDZ NIE ma pola z liczbą mieszkań ani powierzchnią użytkową —
     jest tylko kubatura [m3]. n_flats = kubatura / M3_PER_FLAT, gdzie M3_PER_FLAT
     kalibrujemy tak, by suma szacunków za lata CALIB_YEARS zgadzała się z GUS BDL
     (P3816: mieszkania, na które wydano pozwolenia, powiat m.st. Warszawa, formy
     nieindywidualne: sprzedaż/wynajem + spółdzielcze + komunalne + społeczne czynszowe).
     Kubatura nieprawdopodobna (<300 m3 lub >600 tys. m3, w danych są wpisy „1” i literówki
     rzędu 10^7) -> mediana kubatury pozwoleń. n_flats_source = 'estimate' dla wszystkich
     ('field' zarezerwowane, gdyby GUNB dodał pole z liczbą mieszkań).

Uruchomienie:  .venv/bin/python mapping/gunb.py
"""

import json
import os
import re
import time
import urllib.parse
import zipfile
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely import wkt
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "gunb"
ULDK_CACHE = DATA_DIR / "cache" / "uldk"
UUG_CACHE = DATA_DIR / "cache" / "uug"
DISTRICTS_PATH = Path(os.environ.get("DANE_WARSZAWA", Path.home() / "Downloads" / "dane_warszawa")) / "warszawa_dzielnice.geojson"
OUT_PATH = DATA_DIR / "gunb_permits.parquet"
CRS = "EPSG:2180"

GUNB_URL = "https://wyszukiwarka.gunb.gov.pl/pliki_pobranie/wynik_mazowieckie.zip"
ULDK_URL = "https://uldk.gugik.gov.pl/"
UUG_URL = "https://services.gugik.gov.pl/uug/"
BDL_URL = "https://bdl.stat.gov.pl/api/v1/data/by-unit/071412865000"
# GUS BDL P3816 „styczeń-grudzień”: spółdzielcze, sprzedaż/wynajem, komunalne, społeczne czynszowe.
BDL_VARS = [747635, 747636, 747638, 747639]

YEAR_FROM = 2016
CALIB_YEARS = range(2022, 2026)   # lata, w których RWDZ jest już kompletny (patrz raport kalibracji)
KUB_MIN, KUB_MAX = 300, 600_000    # m3, poza tym kubatura uznana za błędną
MAX_WORKERS = 4
USER_AGENT = "optymalna-warszawa-hackathon/1.0"

AMENDMENT = re.compile(r"zamienn|zmian|przeniesieni|uchyl", re.I)
NON_BUILDING = re.compile(
    r"^\W*(?:budowa\s+|wykonanie\s+|montaż\s+)?(?:wewn\S*\s+|zewn\S*\s+)?"
    r"(?:instalac|przyłącz|sieci|oranżer|wiat|altan|zbiornik|stacji|zjazd|śmietnik)", re.I)
DISTRICT_FROM_ORGAN = {"Praga Płd.": "Praga-Południe", "Praga Pn.": "Praga-Północ"}


def download() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zpath = RAW_DIR / "wynik_mazowieckie.zip"
    if not zpath.exists():
        print(f"Pobieram {GUNB_URL} ...")
        with requests.get(GUNB_URL, stream=True, timeout=300, headers={"User-Agent": USER_AGENT}) as r:
            r.raise_for_status()
            with open(zpath, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    csv = RAW_DIR / "wynik_mazowieckie.csv"
    if not csv.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(RAW_DIR)
    return csv


def load_permits(csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zwraca (pozwolenia — wiersz na decyzję, działki — wiersz na działkę), wszystkie lata."""
    df = pd.read_csv(csv, sep=";", dtype=str, encoding="utf-8-sig")
    df = df[
        df["terc"].str.startswith("1465", na=False)
        & (df["kategoria"] == "XIII")
        & df["nazwa_zamierzenia_bud"].str.startswith("budowa nowego", na=False)
    ].copy()
    desc = df["nazwa_zam_budowlanego"].fillna("").str.strip()
    df = df[~desc.str.contains(AMENDMENT) & ~desc.str.contains(NON_BUILDING)]
    df = df.rename(columns={"numer_gunb": "permit_id"})

    first = df.groupby("permit_id", sort=False).first().reset_index()
    first = first.drop_duplicates(["nazwa_organu", "numer_decyzji_urzedu", "data_wydania_decyzji"])
    first["data_decyzji"] = pd.to_datetime(first["data_wydania_decyzji"]).dt.date
    first["rok"] = pd.to_datetime(first["data_wydania_decyzji"]).dt.year
    street = (first["cecha"].fillna("").str.strip() + " " + first["ulica_dalej"].fillna("").str.strip()
              + " " + first["ulica"].fillna("").str.strip()).str.split().str.join(" ")
    first["adres"] = (street + " " + first["nr_domu"].fillna("")).str.strip()
    first["opis"] = first["nazwa_zam_budowlanego"].fillna("").str.strip()
    first["inwestor"] = first["nazwa_inwestor"].fillna("")
    first["organ"] = first["nazwa_organu"]
    first["kubatura"] = pd.to_numeric(first["kubatura"], errors="coerce")
    permits = first[["permit_id", "data_decyzji", "rok", "adres", "opis", "inwestor", "organ",
                     "numer_decyzji_urzedu", "kubatura", "ulica", "nr_domu"]]

    parcels = df[df["permit_id"].isin(permits["permit_id"])][
        ["permit_id", "jednosta_numer_ew", "obreb_numer", "numer_dzialki"]].dropna().drop_duplicates()
    parcels["parcel_id"] = (parcels["jednosta_numer_ew"].str.strip() + "." + parcels["obreb_numer"].str.strip()
                            + "." + parcels["numer_dzialki"].str.strip())
    return permits.reset_index(drop=True), parcels


def gus_permitted_flats() -> pd.Series:
    """Mieszkania (formy nieindywidualne), na które wydano pozwolenia — Warszawa, rocznie."""
    q = "&".join(f"var-id={v}" for v in BDL_VARS)
    r = requests.get(f"{BDL_URL}?{q}&format=json", timeout=60)
    r.raise_for_status()
    res = r.json()["results"]
    s = pd.concat([pd.Series({int(v["year"]): v["val"] for v in var["values"]}) for var in res], axis=1)
    return s.sum(axis=1)


def _get(url: str, cache: Path, retries: int = 4) -> str:
    if cache.exists():
        return cache.read_text()
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            text = r.content.decode("utf-8")
            cache.write_text(text)
            return text
        except Exception:
            time.sleep(2 ** attempt)
    return ""


def uldk_parcel(parcel_id: str):
    """Geometria działki z ULDK (EPSG:2180) albo None."""
    url = f"{ULDK_URL}?request=GetParcelById&id={urllib.parse.quote(parcel_id)}&result=geom_wkt&srid=2180"
    text = _get(url, ULDK_CACHE / f"{sha1(parcel_id.encode()).hexdigest()}.txt").strip()
    lines = text.splitlines()
    if not lines or lines[0].strip() != "0" or len(lines) < 2:
        return None
    return wkt.loads(lines[1].split(";", 1)[-1])


def uldk_region(region_id: str):
    """Geometria obrębu ewidencyjnego z ULDK (EPSG:2180) albo None."""
    url = f"{ULDK_URL}?request=GetRegionById&id={urllib.parse.quote(region_id)}&result=geom_wkt&srid=2180"
    text = _get(url, ULDK_CACHE / f"region_{sha1(region_id.encode()).hexdigest()}.txt").strip()
    lines = text.splitlines()
    if not lines or lines[0].strip() != "0" or len(lines) < 2:
        return None
    return wkt.loads(lines[1].split(";", 1)[-1])


def fixed_unit(parcel_id: str) -> str | None:
    """Literówka w kodzie jednostki: 186518_8 / 145617_8 -> 1465xx_8 (Warszawa ma 146501..146519)."""
    unit, rest = parcel_id.split(".", 1)
    m = re.fullmatch(r"\d{4}(\d{2})_\d", unit)
    if m and not unit.startswith("1465") and 1 <= int(m.group(1)) <= 19:
        return f"1465{m.group(1)}_8.{rest}"
    return None


def locate_parcels(parcels: pd.DataFrame) -> pd.DataFrame:
    ULDK_CACHE.mkdir(parents=True, exist_ok=True)
    ids = parcels["parcel_id"].unique().tolist()
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        geoms = dict(zip(ids, ex.map(uldk_parcel, ids)))
    for pid in [p for p, g in geoms.items() if g is None]:
        alt = fixed_unit(pid)
        if alt:
            geoms[pid] = uldk_parcel(alt)
    out = parcels.copy()
    out["geom"] = out["parcel_id"].map(geoms)
    return out


def uug_address(street: str, number: str):
    if not street or not number:
        return None
    addr = f"Warszawa, {street} {number}"
    url = f"{UUG_URL}?request=GetAddress&address={urllib.parse.quote(addr)}&srid=2180"
    text = _get(url, UUG_CACHE / f"{sha1(addr.encode()).hexdigest()}.json")
    try:
        res = json.loads(text)["results"]
        best = res[sorted(res, key=int)[0]] if isinstance(res, dict) else None
    except Exception:
        return None
    if not best or "Warszawa" not in str(best.get("jednostka", "")):
        return None
    return Point(float(best["x"]), float(best["y"]))


def permit_points(permits: pd.DataFrame, parcels: pd.DataFrame) -> pd.DataFrame:
    """Centroid działek pozwolenia ważony powierzchnią; fallback: adres (UUG)."""
    ok = parcels[parcels["geom"].notna()].copy()
    ok["area"] = [g.area for g in ok["geom"]]
    ok["cx"] = [g.centroid.x for g in ok["geom"]]
    ok["cy"] = [g.centroid.y for g in ok["geom"]]
    agg = ok.groupby("permit_id").apply(
        lambda d: pd.Series({"x": np.average(d.cx, weights=d.area + 1e-6),
                             "y": np.average(d.cy, weights=d.area + 1e-6),
                             "n_parcels_found": len(d), "area_m2": d.area.sum()}),
        include_groups=False)
    p = permits.merge(agg, on="permit_id", how="left")
    p["n_parcels"] = p["permit_id"].map(parcels.groupby("permit_id").size()).fillna(0).astype(int)
    p["geo_source"] = np.where(p["x"].notna(), "uldk", None)

    UUG_CACHE.mkdir(parents=True, exist_ok=True)
    miss = p["x"].isna()
    for i in p.index[miss]:
        pt = uug_address(str(p.at[i, "ulica"] or "").strip(), str(p.at[i, "nr_domu"] or "").strip())
        if pt is not None:
            p.loc[i, ["x", "y", "geo_source"]] = [pt.x, pt.y, "uug_address"]

    # Ostatnia deska: centroid obrębu (działki często podzielono po wydaniu decyzji,
    # więc stary numer nie istnieje już w EGiB). Obręby w Warszawie mają ~0,2–1 km2.
    region = parcels["parcel_id"].str.rsplit(".", n=1).str[0]
    region = region.groupby(parcels["permit_id"]).agg(lambda r: r.mode().iloc[0])
    for i in p.index[p["x"].isna()]:
        rid = region.get(p.at[i, "permit_id"])
        if rid is None:
            continue
        geom = uldk_region(rid)
        if geom is None and fixed_unit(rid + ".x"):
            geom = uldk_region(fixed_unit(rid + ".x").rsplit(".", 1)[0])
        if geom is not None:
            c = geom.centroid
            p.loc[i, ["x", "y", "geo_source"]] = [c.x, c.y, "uldk_obreb"]
    return p


def estimate_flats(p: pd.DataFrame, gus: pd.Series) -> tuple[pd.DataFrame, float]:
    """n_flats = kubatura / M3_PER_FLAT, M3_PER_FLAT skalibrowane do GUS za CALIB_YEARS."""
    p = p.copy()
    valid = p["kubatura"].between(KUB_MIN, KUB_MAX)
    p["kubatura_ok"] = valid
    p["kubatura_est"] = p["kubatura"].where(valid, p.loc[valid, "kubatura"].median())
    calib = p[p["rok"].isin(CALIB_YEARS)]
    years = [y for y in CALIB_YEARS if y in gus.index]
    m3_per_flat = calib.loc[calib["rok"].isin(years), "kubatura_est"].sum() / gus.loc[years].sum()
    p["n_flats"] = np.maximum(1, np.round(p["kubatura_est"] / m3_per_flat)).astype(int)
    p["n_flats_source"] = "estimate"
    return p, m3_per_flat


def main() -> None:
    csv = download()
    all_permits, all_parcels = load_permits(csv)
    gus = gus_permitted_flats()
    all_permits, m3_per_flat = estimate_flats(all_permits, gus)

    rep = all_permits.groupby("rok").agg(pozwolenia=("permit_id", "size"), szac=("n_flats", "sum"))
    rep["gus"] = gus.reindex(rep.index)
    rep["szac/gus"] = (rep["szac"] / rep["gus"]).round(2)
    print(f"M3_PER_FLAT (kalibracja {CALIB_YEARS.start}-{CALIB_YEARS.stop - 1}) = {m3_per_flat:.0f} m3/mieszkanie")
    print(rep.to_string())

    permits = all_permits[all_permits["rok"] >= YEAR_FROM].reset_index(drop=True)
    parcels = locate_parcels(all_parcels[all_parcels["permit_id"].isin(permits["permit_id"])])
    print(f"Działki: {len(parcels)}, znalezione w ULDK: {parcels['geom'].notna().sum()}")
    p = permit_points(permits, parcels)

    gdf = gpd.GeoDataFrame(p[p["x"].notna()].copy(), crs=CRS,
                           geometry=gpd.points_from_xy(p.loc[p["x"].notna(), "x"], p.loc[p["x"].notna(), "y"]))
    districts = gpd.read_file(DISTRICTS_PATH).to_crs(CRS)[["dzielnica", "geometry"]]
    gdf = gpd.sjoin(gdf, districts, how="left", predicate="within").drop(columns="index_right")
    # Punkt poza granicą miasta = błędna lokalizacja (literówka w działce/adresie) -> odrzucamy.
    outside = gdf["dzielnica"].isna()
    organ_d = gdf["organ"].str.extract(r"Warszawa - (.+)$")[0].replace(DISTRICT_FROM_ORGAN)
    gdf["dzielnica_organ"] = organ_d

    cols = ["permit_id", "data_decyzji", "rok", "adres", "opis", "inwestor", "organ",
            "numer_decyzji_urzedu", "dzielnica", "dzielnica_organ", "kubatura", "kubatura_ok",
            "n_flats", "n_flats_source", "n_parcels", "n_parcels_found", "area_m2", "geo_source", "geometry"]
    out = gdf.loc[~outside, cols].to_crs("EPSG:4326").reset_index(drop=True)
    out["n_parcels_found"] = out["n_parcels_found"].fillna(0).astype(int)
    out.to_parquet(OUT_PATH)

    n = len(permits)
    print(f"\nPozwolenia {YEAR_FROM}+: {n}, zlokalizowane: {len(out)} ({len(out) / n:.1%}); "
          f"ULDK {(out.geo_source == 'uldk').sum()}, adres {(out.geo_source == 'uug_address').sum()}, "
          f"obręb {(out.geo_source == 'uldk_obreb').sum()}, poza miastem {outside.sum()}, brak {n - len(gdf)}")
    print(f"Mieszkania (szac.) w pliku: {out['n_flats'].sum():,} z {permits['n_flats'].sum():,}")
    print(out.groupby("rok")["n_flats"].agg(["size", "sum"]).to_string())
    print(out.groupby("dzielnica")["n_flats"].sum().sort_values(ascending=False).head(10).to_string())
    print(f"Zapisano {OUT_PATH}")


if __name__ == "__main__":
    main()
