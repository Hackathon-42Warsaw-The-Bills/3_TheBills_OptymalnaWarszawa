"""Cechy klastrów H3 do modelu „ile komunikacji potrzebuje klaster”.

Wynik: dane/klastry/cechy_klastrow.csv — jeden wiersz na klaster (h3), kolumny w grupach:

  CEL (y) — dzisiejsza obsługa ZTM z rozkładu GTFS (dzień roboczy, środa):
    kursy_dzien_bus_tram    różne kursy autobusów i tramwajów zatrzymujące się w klastrze, doba
    kursy_szczyt_bus_tram   to samo w szczycie 7:00–9:00 (na godzinę)
    kursy_dzien_autobus / _tramwaj / _metro / _kolej   osobno wg rodzaju (metro z frequencies.txt)
    przystanki_ztm, linie_ztm   liczba słupków i różnych linii w klastrze

  POPYT — kto mieszka i co przyciąga podróże:
    population (2025), pop_2031, pop_2036, mieszkania_w_budowie   (prognoza_ludnosci.csv)
    gestosc_os_km2, pow_km2
    udzial_0_14, udzial_65_   struktura wieku (siatka NSP 2021, ważona powierzchnią przecięcia)
    praca_biura_uslugi_m2   powierzchnia biur/handlu/usług w budynkach (OSM: obrys × kondygnacje)
    praca_przemysl_m2       powierzchnia budynków przemysłowych i magazynowych (OSM)
    punkty_uslug            liczba sklepów i punktów usługowych (OSM shop=*, amenity=*)
    infra_<grupa>           szpitale, szkoły, handel, kultura, urzędy, targowiska, parki (warszawa_infrastruktura)

  POŁOŻENIE I SIEĆ:
    odl_centrum_km          od Pałacu Kultury
    ma_metro, ma_kolej, dist_metro_m, dist_kolej_m, metro_plan_rok   (dane/kolej/klastry_kolej.csv)
    v_autobus_szczyt_kmh, korek   średnia prędkość autobusów na ulicach klastra (rozkład ZTM)

Uruchomienie (po build_graph.py, gtfs_predkosci.py, kolej.py, klastry_kolej.py, prognoza_ludnosci.py):
    .venv/bin/python mapping/cechy_klastrow.py
"""

import os
import re
import sys
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import osmium
import pandas as pd
from shapely import wkb

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DANE = ROOT / "dane"
GTFS = DATA_DIR / "gtfs" / "warsaw"
PBF = DATA_DIR / "osm" / "mazowieckie-latest.osm.pbf"
EXTERNAL_DIR = Path(os.environ.get("DANE_WARSZAWA", Path.home() / "Downloads" / "dane_warszawa"))
OUT = DANE / "klastry" / "cechy_klastrow.csv"
CRS = "EPSG:2180"
RES = 8
CENTRUM = (21.0067, 52.2318)                 # Pałac Kultury i Nauki
PEAK = (7 * 3600, 9 * 3600)
SERVICE_SUFFIX = ":PcS"                      # dzień roboczy (pon–czw); bierzemy środę jak w gtfs_predkosci.py
MODE = {"3": "autobus", "0": "tramwaj", "1": "metro", "2": "kolej"}

# OSM: budynki miejsc pracy i domyślna liczba kondygnacji, gdy brak building:levels
WORK_BUILDINGS = {"office": 4, "commercial": 3, "retail": 1, "supermarket": 1, "hotel": 5, "hospital": 4,
                  "university": 4, "college": 4, "school": 3, "kindergarten": 1, "government": 4, "civic": 3,
                  "public": 3, "bank": 3, "train_station": 2}
INDUSTRY_BUILDINGS = {"industrial": 1, "warehouse": 1, "manufacture": 1, "factory": 1, "depot": 1}
INFRA_GROUPS = {"szpital": "szpital", "szkola": "szkola", "centrum_handlowe": "handel", "park": "park",
                "kino": "kultura", "teatr": "kultura", "muzeum": "kultura", "urzad_dzielnicy": "urzad",
                "targowisko": "targowisko", "miejsce_kultu": "kult"}


def to_h3(lon, lat) -> np.ndarray:
    return np.array([h3.latlng_to_cell(y, x, RES) for x, y in zip(lon, lat)])


def to_sec(t: pd.Series) -> pd.Series:
    hms = t.str.split(":", expand=True).astype(int)
    return hms[0] * 3600 + hms[1] * 60 + hms[2]


# ---------- cel: obsługa ZTM z GTFS ----------

def gtfs_service(hex_ids: set) -> pd.DataFrame:
    routes = pd.read_csv(GTFS / "routes.txt", dtype=str)[["route_id", "route_short_name", "route_type"]]
    trips = pd.read_csv(GTFS / "trips.txt", dtype=str, usecols=["trip_id", "route_id", "service_id"])
    services = sorted(s for s in trips["service_id"].unique() if s.endswith(SERVICE_SUFFIX))
    day = services[min(2, len(services) - 1)]
    # Metro ma osobny kalendarz bez daty (PcM = dzień roboczy), reszta — konkretny dzień (…:PcS).
    keep = {day} | ({"PcM"} & set(trips["service_id"]))
    trips = trips[trips["service_id"].isin(keep)].merge(routes, on="route_id")
    trips["tryb"] = trips["route_type"].map(MODE).fillna("inne")
    print(f"   dzień rozkładu {day}: {len(trips)} kursów ({trips['tryb'].value_counts().to_dict()})")

    # Metro jest w GTFS jako kursy wzorcowe z frequencies.txt — liczba faktycznych kursów = okno / odstęp.
    mult = pd.Series(1.0, index=trips["trip_id"])
    freq_path = GTFS / "frequencies.txt"
    if freq_path.exists():
        fr = pd.read_csv(freq_path, dtype={"trip_id": str})
        fr = fr[fr["trip_id"].isin(set(trips["trip_id"]))].copy()
    if freq_path.exists() and len(fr):
        fr["runs"] = (to_sec(fr["end_time"]) - to_sec(fr["start_time"])) / fr["headway_secs"]
        mult.update(fr.groupby("trip_id")["runs"].sum())

    stops = pd.read_csv(GTFS / "stops.txt", dtype={"stop_id": str}, usecols=["stop_id", "stop_lat", "stop_lon"])
    stops["h3"] = to_h3(stops["stop_lon"], stops["stop_lat"])
    st = pd.read_csv(GTFS / "stop_times.txt", dtype={"trip_id": str, "stop_id": str},
                     usecols=["trip_id", "stop_id", "departure_time"])
    st = st[st["trip_id"].isin(set(trips["trip_id"]))].merge(stops[["stop_id", "h3"]], on="stop_id")
    st = st[st["h3"].isin(hex_ids)]
    st["t"] = to_sec(st["departure_time"])
    # Jeden kurs liczy się w klastrze raz, niezależnie od liczby jego przystanków w klastrze.
    first = st.sort_values("t").drop_duplicates(["trip_id", "h3"])
    first = first.merge(trips[["trip_id", "tryb", "route_short_name"]], on="trip_id")
    first["n"] = first["trip_id"].map(mult).fillna(1.0)
    first["szczyt"] = first["t"].between(*PEAK, inclusive="left")

    out = first.pivot_table(index="h3", columns="tryb", values="n", aggfunc="sum", fill_value=0)
    out.columns = [f"kursy_dzien_{c}" for c in out.columns]
    peak = first[first["szczyt"] & first["tryb"].isin(["autobus", "tramwaj"])].groupby("h3")["n"].sum() / 2
    out["kursy_szczyt_bus_tram"] = peak
    out["kursy_dzien_bus_tram"] = out.get("kursy_dzien_autobus", 0) + out.get("kursy_dzien_tramwaj", 0)
    out["linie_ztm"] = first[first["tryb"].isin(["autobus", "tramwaj"])].groupby("h3")["route_short_name"].nunique()
    out["przystanki_ztm"] = stops[stops["h3"].isin(hex_ids)].groupby("h3").size()
    return out.fillna(0).round(1)


# ---------- popyt: miejsca pracy z OSM ----------

def levels(tags, default: int) -> float:
    v = tags.get("building:levels")
    m = re.match(r"\d+(\.\d+)?", v) if v else None
    return float(m.group()) if m else default


def osm_jobs(hex_ids: set, bbox) -> pd.DataFrame:
    """Powierzchnia użytkowa biur/usług i przemysłu (obrys × kondygnacje) oraz punkty usługowe na klaster."""
    minx, miny, maxx, maxy = bbox
    fab = osmium.geom.WKBFactory()
    rows, pois = [], []
    fp = (osmium.FileProcessor(str(PBF)).with_areas()
          .with_filter(osmium.filter.KeyFilter("building", "shop", "amenity", "office")))
    for o in fp:
        tags = o.tags
        if o.is_node():
            if ("shop" in tags or "amenity" in tags or "office" in tags) and minx <= o.location.lon <= maxx and miny <= o.location.lat <= maxy:
                pois.append((o.location.lon, o.location.lat))
            continue
        if not o.is_area():
            continue
        b = tags.get("building")
        is_office = "office" in tags
        if not b and not is_office and "shop" not in tags and "amenity" not in tags:
            continue
        try:
            g = wkb.loads(fab.create_multipolygon(o), hex=True)
        except Exception:  # noqa: BLE001 — uszkodzona geometria w OSM
            continue
        c = g.representative_point()
        if not (minx <= c.x <= maxx and miny <= c.y <= maxy):
            continue
        if "shop" in tags or "amenity" in tags:
            pois.append((c.x, c.y))
        kind = None
        if b in WORK_BUILDINGS or (b and is_office):
            kind, lv = "biura_uslugi", levels(tags, WORK_BUILDINGS.get(b, 3))
        elif b in INDUSTRY_BUILDINGS:
            kind, lv = "przemysl", levels(tags, INDUSTRY_BUILDINGS[b])
        if kind:
            rows.append((c.x, c.y, kind, g, lv))
    print(f"   OSM: {len(rows)} budynków pracy, {len(pois)} punktów usług")
    b = gpd.GeoDataFrame(rows, columns=["lon", "lat", "kind", "geometry", "lv"], crs="EPSG:4326")
    b["m2"] = b.to_crs(CRS).area * b["lv"]
    b["h3"] = to_h3(b["lon"], b["lat"])
    b = b[b["h3"].isin(hex_ids)]
    out = b.pivot_table(index="h3", columns="kind", values="m2", aggfunc="sum", fill_value=0)
    out.columns = [f"praca_{c}_m2" for c in out.columns]
    p = pd.DataFrame(pois, columns=["lon", "lat"])
    p["h3"] = to_h3(p["lon"], p["lat"])
    out["punkty_uslug"] = p[p["h3"].isin(hex_ids)].groupby("h3").size()
    return out.fillna(0).round(0)


# ---------- popyt: struktura wieku z siatki NSP 2021 ----------

def age_shares(hexes: gpd.GeoDataFrame) -> pd.DataFrame:
    grid = gpd.read_file(EXTERNAL_DIR / "warszawa_siatka_1km_NSP2021.geojson").to_crs(CRS)
    grid = grid[grid["RES"] > 0][["RES", "RES_0_14", "RES_65_", "geometry"]]
    grid["A"] = grid.area
    pieces = gpd.overlay(grid, hexes[["h3", "geometry"]].to_crs(CRS), how="intersection")
    w = pieces.area / pieces["A"]
    agg = pd.DataFrame({"h3": pieces["h3"], "res": pieces["RES"] * w, "k": pieces["RES_0_14"] * w,
                        "s": pieces["RES_65_"] * w}).groupby("h3").sum()
    return pd.DataFrame({"udzial_0_14": (agg["k"] / agg["res"]).round(3), "udzial_65_": (agg["s"] / agg["res"]).round(3)})


# ---------- położenie, sieć, infrastruktura ----------

def bus_speed(hex_ids: set) -> pd.DataFrame:
    path = DATA_DIR / "predkosci_ulic.parquet"
    if not path.exists():
        return pd.DataFrame()
    sp = pd.read_parquet(path)
    e = gpd.read_parquet(DATA_DIR / "roads_edges.parquet")[["edge_id", "geometry", "length"]]
    e = e.merge(sp, on="edge_id")
    mid = e.geometry.interpolate(0.5, normalized=True)
    e["h3"] = to_h3(mid.x, mid.y)
    e = e[e["h3"].isin(hex_ids)]
    g = e.assign(w=e["length"]).groupby("h3")
    return pd.DataFrame({"v_autobus_szczyt_kmh": g.apply(lambda d: np.average(d["v_szczyt_kmh"], weights=d["w"]), include_groups=False).round(1),
                         "korek": g.apply(lambda d: np.average(d["korek"], weights=d["w"]), include_groups=False).round(3)})


def infra_counts(hex_ids: set) -> pd.DataFrame:
    inf = pd.read_csv(DANE / "infrastruktura" / "warszawa_infrastruktura.csv")
    inf["grupa"] = inf["kategoria"].map(INFRA_GROUPS).fillna("inne")
    t = pd.crosstab(inf["h3"], inf["grupa"])
    t.columns = [f"infra_{c}" for c in t.columns]
    return t[t.index.isin(hex_ids)]


def main() -> None:
    hexes = gpd.read_file(DANE / "klastry" / "kontury_400m" / "warszawa_h3_400m_pelna.geojson")
    ids = set(hexes["h3"])
    base = pd.read_csv(DANE / "klastry" / "kontury_400m" / "warszawa_h3_400m_pelna.csv")[
        ["h3", "dzielnica", "population", "gestosc_os_km2", "pow_km2", "lon", "lat"]].set_index("h3")

    print("== obsługa ZTM (GTFS)")
    service = gtfs_service(ids)
    print("== miejsca pracy i usługi (OSM)")
    jobs = osm_jobs(ids, hexes.total_bounds)
    print("== struktura wieku (NSP 2021)")
    ages = age_shares(hexes)
    print("== prognoza, kolej, prędkości, infrastruktura")
    prog = pd.read_csv(DANE / "ludnosc" / "prognoza_ludnosci.csv").set_index("h3")[["pop_2031", "pop_2036", "mieszkania_w_budowie"]]
    kolej = pd.read_csv(DANE / "kolej" / "klastry_kolej.csv").set_index("h3")[
        ["ma_metro", "ma_kolej", "dist_metro_m", "dist_kolej_m", "metro_plan_rok"]]

    out = base.join([service, jobs, ages, prog, kolej, bus_speed(ids), infra_counts(ids)])
    lon0, lat0 = CENTRUM
    out["odl_centrum_km"] = (np.hypot((out["lon"] - lon0) * 67.9, (out["lat"] - lat0) * 111.2)).round(2)
    zero_cols = [c for c in out.columns if c.startswith(("kursy_", "praca_", "infra_")) or c in ("punkty_uslug", "linie_ztm", "przystanki_ztm")]
    out[zero_cols] = out[zero_cols].fillna(0)
    out = out.drop(columns=["lon", "lat"])
    out.reset_index().to_csv(OUT, index=False, encoding="utf-8")

    print(f"== {len(out)} klastrów, {out.shape[1]} cech -> {OUT.relative_to(ROOT)}")
    print("   braki:", {c: int(n) for c, n in out.isna().sum().items() if n})
    num = out.select_dtypes("number")
    corr = num.corr()["kursy_dzien_bus_tram"].drop(["kursy_dzien_bus_tram"]).sort_values(ascending=False)
    print("   korelacja z kursy_dzien_bus_tram:\n" + corr.round(2).to_string())


if __name__ == "__main__":
    main()
