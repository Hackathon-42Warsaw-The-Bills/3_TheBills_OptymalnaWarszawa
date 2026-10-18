"""Pełna siatka klastrów H3 (Kontur, rozdz. 8) z ludnością policzoną z danych GUS.

Krok 1 — brakujące heksagony.

Kontur pomija heksagony z zerową ludnością, więc siatka ma dziury: lasy, lotnisko,
tereny przemysłowe i — co dla nas ważne — tereny pod przyszłą zabudowę. Klaster,
którego nie ma w siatce, nie dostanie żadnego wyniku w scoringu.

Dodajemy wszystkie heksagony rozdz. 8 przecinające granicę Warszawy, przycięte do niej
tak samo jak oryginał (zrodlo = "dodany"). `dzielnica` wg środka komórki H3, `lon`/`lat`
w CSV = środek komórki H3.

Krok 2 — ludność. Kontur ma błędy (Wola ~26 tys. zamiast ~150 tys., cała Warszawa 1,72 mln,
stan modelu ok. 2020), więc `population` liczymy od nowa:
  1. Ludność kwadratów GUS 1 km (NSP 2021, część w granicach miasta) dzielimy na kawałki
     kwadrat ∩ heksagon ∩ dzielnica według gęstości Kontur — Kontur tylko rozmieszcza,
     suma kwadratu zostaje zgodna ze spisem. Kwadrat, w którym Kontur nie ma nikogo,
     dzielimy po powierzchni.
  2. Kalibracja do ludności dzielnic GUS 2025 (BDL): kawałki × (GUS 2025 / suma kawałków).
Kolumny: population (= 2025, GUS), population_2021 (NSP), population_kontur (oryginał),
gestosc_os_km2 (2025, na powierzchnię heksagonu w granicach miasta), pow_km2.

Wejście:  dane/klastry/kontury_400m/warszawa_h3_400m_kontur2023.geojson
          dane_warszawa/warszawa_dzielnice.geojson (granica miasta = suma dzielnic, GUS 2021/2025)
          dane_warszawa/warszawa_siatka_1km_NSP2021.geojson
Wynik:    dane/klastry/kontury_400m/warszawa_h3_400m_pelna.geojson / .csv

Uruchomienie:  .venv/bin/python mapping/h3_siatka.py
"""

import os
from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent.parent
KONTUR_DIR = ROOT / "dane" / "klastry" / "kontury_400m"
SRC = KONTUR_DIR / "warszawa_h3_400m_kontur2023.geojson"
OUT = KONTUR_DIR / "warszawa_h3_400m_pelna"
EXTERNAL_DIR = Path(os.environ.get("DANE_WARSZAWA", Path.home() / "Downloads" / "dane_warszawa"))
CRS = "EPSG:2180"
RES = 8
MIN_AREA_M2 = 1000  # pomijamy okruchy na styku granicy, mniejsze niż ~30×30 m


def cell_polygon(cell: str) -> Polygon:
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def covering_cells(boundary) -> set[str]:
    """Wszystkie komórki H3 przecinające granicę (polyfill bufora + test przecięcia)."""
    buffered = gpd.GeoSeries([boundary.buffer(1000)], crs=CRS).to_crs("EPSG:4326").iloc[0]
    parts = getattr(buffered, "geoms", [buffered])
    cells = set()
    for p in parts:
        shape = h3.LatLngPoly([(y, x) for x, y in p.exterior.coords])
        cells |= set(h3.polygon_to_cells(shape, RES))
    return cells


def gus_population(hexes: gpd.GeoDataFrame, districts: gpd.GeoDataFrame) -> pd.DataFrame:
    """Ludność 2021 (NSP) i 2025 (kalibracja do dzielnic) dla każdego heksagonu."""
    grid = gpd.read_file(EXTERNAL_DIR / "warszawa_siatka_1km_NSP2021.geojson").to_crs(CRS)
    grid = grid[grid["RES_szac_w_warszawie"] > 0][["CODE", "RES_szac_w_warszawie", "geometry"]]
    pieces = gpd.overlay(grid, hexes[["h3", "kontur_dens", "geometry"]], how="intersection")
    pieces = gpd.overlay(pieces, districts[["dzielnica", "geometry"]], how="intersection")
    pieces["area"] = pieces.area
    pieces["w"] = pieces["area"] * pieces["kontur_dens"]
    tot_w = pieces.groupby("CODE")["w"].transform("sum")
    tot_a = pieces.groupby("CODE")["area"].transform("sum")
    share = (pieces["w"] / tot_w.where(tot_w > 0)).fillna(pieces["area"] / tot_a)
    pieces["p2021"] = pieces["RES_szac_w_warszawie"] * share

    d = districts.set_index("dzielnica")
    factor = d["ludnosc_gus_2025"] / pieces.groupby("dzielnica")["p2021"].sum()
    pieces["p2025"] = pieces["p2021"] * pieces["dzielnica"].map(factor)
    return pieces.groupby("h3")[["p2021", "p2025"]].sum()


def main() -> None:
    kontur = gpd.read_file(SRC)
    districts = gpd.read_file(EXTERNAL_DIR / "warszawa_dzielnice.geojson").to_crs(CRS)
    boundary = districts.union_all()

    cells = covering_cells(boundary) - set(kontur["h3"])
    full = gpd.GeoDataFrame({"h3": sorted(cells)}, geometry=[cell_polygon(c) for c in sorted(cells)],
                            crs="EPSG:4326").to_crs(CRS)
    clipped = full.copy()
    clipped["geometry"] = full.intersection(boundary)
    added = clipped[clipped.area >= MIN_AREA_M2].copy()

    # Dzielnica wg środka komórki H3 (jak w pliku Kontur); środek poza miastem -> najbliższa.
    centers = gpd.GeoDataFrame(
        {"h3": added["h3"].values},
        geometry=gpd.points_from_xy(*zip(*[h3.cell_to_latlng(c)[::-1] for c in added["h3"]])),
        crs="EPSG:4326",
    ).to_crs(CRS)
    added["dzielnica"] = gpd.sjoin_nearest(centers, districts[["dzielnica", "geometry"]]) \
        .groupby(level=0)["dzielnica"].first().values
    added["population"] = 0
    added["gestosc_os_km2"] = 0.0
    added["zrodlo"] = "dodany"

    kontur["zrodlo"] = "kontur"
    out = pd.concat([kontur, added.to_crs("EPSG:4326")], ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")

    # Ludność z GUS; gęstość Kontur (na pełną komórkę) służy tylko jako waga rozkładu.
    m = out.to_crs(CRS)
    m["kontur_dens"] = out["gestosc_os_km2"].values
    pop = gus_population(m, districts)
    out["population_kontur"] = out["population"].astype(int)
    out["population_2021"] = out["h3"].map(pop["p2021"]).fillna(0).round().astype(int)
    out["population"] = out["h3"].map(pop["p2025"]).fillna(0).round().astype(int)
    out["pow_km2"] = (m.area / 1e6).round(4).values
    out["gestosc_os_km2"] = (out["population"] / out["pow_km2"].clip(lower=0.01)).round(0)
    cols = ["h3", "population", "population_2021", "population_kontur", "gestosc_os_km2", "pow_km2",
            "dzielnica", "zrodlo", "geometry"]
    out = out[cols]

    out.to_file(OUT.with_suffix(".geojson"), driver="GeoJSON")
    csv = out.drop(columns="geometry")
    csv["lon"] = [round(h3.cell_to_latlng(c)[1], 6) for c in out["h3"]]
    csv["lat"] = [round(h3.cell_to_latlng(c)[0], 6) for c in out["h3"]]
    csv.to_csv(OUT.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    m = out.to_crs(CRS)
    print(f"== Kontur: {len(kontur)}, dodane: {len(added)}, razem: {len(out)} heksagonów")
    print(f"   pokrycie miasta: {m.union_all().intersection(boundary).area / 1e6:.1f} "
          f"z {boundary.area / 1e6:.1f} km²")
    print(f"   dodane wg dzielnic: {added['dzielnica'].value_counts().to_dict()}")
    check = out.groupby("dzielnica")[["population_kontur", "population_2021", "population"]].sum()
    check["gus_2025"] = districts.set_index("dzielnica")["ludnosc_gus_2025"]
    print(check.to_string())
    print(f"   razem: Kontur {out['population_kontur'].sum():,}, NSP 2021 {out['population_2021'].sum():,}, "
          f"2025 {out['population'].sum():,} (GUS 2025: {districts['ludnosc_gus_2025'].sum():,})")
    print(f"   dodane heksagony z ludnością wg GUS: {(out[out.zrodlo == 'dodany'].population > 0).sum()} "
          f"({out[out.zrodlo == 'dodany'].population.sum():,} os.)")
    print(f"   -> {OUT.with_suffix('.geojson').relative_to(ROOT)}, .csv")


if __name__ == "__main__":
    main()
