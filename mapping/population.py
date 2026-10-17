"""Łączy ludność (GUS NSP 2021 + Kontur 2023) z odcinkami ulic.

Wynik: data/segments_population.parquet — jeden wiersz na odcinek drogi
(kierunki u->v i v->u scalone), z cechami do modelu lokalizacji przystanków.

Metoda:
  1. Ludność kwadratów GUS 1 km (oficjalna) rozkładamy wewnątrz kwadratu według
     gęstości heksagonów Kontur — suma zgodna z GUS, Kontur tylko ją rozmieszcza.
  2. Każdy kawałek (kwadrat ∩ heksagon) oddaje ludność odcinkom ulic w nim,
     proporcjonalnie do ich długości. Kawałki bez ulic -> najbliższy odcinek.
  3. Ludność 2025 i prognoza: kwadraty × tempo zmian dzielnicy 2021→2025 (GUS BDL).
  4. Cechy: ludność w promieniu 400 m, odległość do słupka ZTM, ludność bez
     przystanku w zasięgu 400 m.

Uruchomienie (po build_graph.py):  .venv/bin/python mapping/population.py
"""

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import STRtree

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EXTERNAL_DIR = Path(os.environ.get("DANE_WARSZAWA", Path.home() / "Downloads" / "dane_warszawa"))
CRS = "EPSG:2180"  # PL-1992, metry

WALK_M = 400            # typowy zasięg dojścia do przystanku autobusowego
STOP_COUNT_RADIUS_M = 300
FORECAST_YEARS = [2028, 2030]
# Ludności nie przypisujemy do dróg ekspresowych i autostrad — nikt przy nich nie mieszka.
NO_RESIDENTS = {"motorway", "trunk", "motorway_link", "trunk_link"}


def load_segments() -> gpd.GeoDataFrame:
    e = gpd.read_parquet(DATA_DIR / "roads_edges.parquet")
    e["segment_id"] = [
        f"{min(u, v)}_{max(u, v)}_{round(length)}" for u, v, length in zip(e.u, e.v, e["length"])
    ]
    seg = (
        e.groupby("segment_id", sort=False)
        .agg(
            edge_ids=("edge_id", ";".join),
            name=("name", "first"),
            highway=("highway", "first"),
            length=("length", "first"),
            geometry=("geometry", "first"),
        )
        .reset_index()
    )
    seg = gpd.GeoDataFrame(seg, geometry="geometry", crs=e.crs).to_crs(CRS)
    kinds = seg["highway"].astype(str).str.split(";")
    seg["residential_access"] = ~kinds.map(lambda ks: all(k in NO_RESIDENTS for k in ks))
    return seg


def district_growth(districts: gpd.GeoDataFrame) -> pd.DataFrame:
    g = districts[["dzielnica", "ludnosc_gus_2021", "ludnosc_gus_2025"]].copy()
    g["growth_2021_2025"] = g["ludnosc_gus_2025"] / g["ludnosc_gus_2021"]
    g["cagr"] = g["growth_2021_2025"] ** (1 / 4) - 1
    return g.set_index("dzielnica")


def population_pieces(districts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Kawałki kwadrat GUS ∩ heksagon Kontur ∩ dzielnica z ludnością 2021 i 2025."""
    grid = gpd.read_file(EXTERNAL_DIR / "warszawa_siatka_1km_NSP2021.geojson").to_crs(CRS)
    hexes = gpd.read_file(EXTERNAL_DIR / "warszawa_h3_400m_kontur2023.geojson").to_crs(CRS)
    grid = grid[grid["RES_szac_w_warszawie"] > 0][["CODE", "RES_szac_w_warszawie", "geometry"]]

    pieces = gpd.overlay(grid, hexes[["h3", "gestosc_os_km2", "geometry"]], how="identity")
    # Cięcie granicami dzielnic, żeby żaden kawałek nie leżał w dwóch dzielnicach.
    pieces = gpd.overlay(pieces, districts[["dzielnica", "geometry"]], how="intersection")
    pieces["area"] = pieces.area
    pieces["w"] = pieces["area"] * pieces["gestosc_os_km2"].fillna(0)
    tot_w = pieces.groupby("CODE")["w"].transform("sum")
    tot_a = pieces.groupby("CODE")["area"].transform("sum")
    # Gdy Kontur nie ma nikogo w kwadracie (np. błąd na Woli), dzielimy po powierzchni.
    share = np.where(tot_w > 0, pieces["w"] / tot_w.replace(0, 1), pieces["area"] / tot_a)
    pieces["pop_2021"] = pieces["RES_szac_w_warszawie"] * share

    growth = district_growth(districts)
    pieces["cagr"] = pieces["dzielnica"].map(growth["cagr"]).fillna(0)
    # Kalibracja: kwadraty GUS nie pokrywają się z granicami dzielnic, więc
    # skalujemy kawałki tak, by suma w dzielnicy = oficjalna ludność GUS 2025.
    factor = growth["ludnosc_gus_2025"] / pieces.groupby("dzielnica")["pop_2021"].sum()
    pieces["pop_2025"] = pieces["pop_2021"] * pieces["dzielnica"].map(factor).fillna(1)

    pieces = pieces[pieces["pop_2021"] > 0].reset_index(drop=True)
    pieces["piece_id"] = pieces.index
    return pieces[["piece_id", "dzielnica", "cagr", "pop_2021", "pop_2025", "geometry"]]


def assign_to_segments(seg: gpd.GeoDataFrame, pieces: gpd.GeoDataFrame) -> pd.DataFrame:
    """Rozkłada ludność kawałków na odcinki proporcjonalnie do długości w kawałku."""
    roads = seg[seg["residential_access"]][["segment_id", "geometry"]]
    inter = gpd.overlay(roads, pieces[["piece_id", "geometry"]], how="intersection", keep_geom_type=True)
    inter["len_in"] = inter.length
    inter = inter[inter["len_in"] > 0]
    inter["share"] = inter["len_in"] / inter.groupby("piece_id")["len_in"].transform("sum")

    cols = ["pop_2021", "pop_2025"]
    assigned = inter[["segment_id", "piece_id", "share"]].merge(pieces[["piece_id"] + cols], on="piece_id")
    for c in cols:
        assigned[c] = assigned[c] * assigned["share"]

    # Kawałki bez żadnej ulicy (np. wnętrza dużych osiedli) -> najbliższy odcinek.
    orphans = pieces[~pieces["piece_id"].isin(inter["piece_id"])]
    if len(orphans):
        pts = gpd.GeoDataFrame(orphans[["piece_id"] + cols], geometry=orphans.representative_point(), crs=CRS)
        near = gpd.sjoin_nearest(pts, roads, how="left")[["segment_id"] + cols]
        assigned = pd.concat([assigned[["segment_id"] + cols], near])
    print(f"   kawałki bez ulicy dopięte do najbliższej: {len(orphans)} "
          f"({orphans['pop_2021'].sum():,.0f} os.)")
    return assigned.groupby("segment_id")[cols].sum()


def catchment(targets: gpd.GeoSeries, sources: gpd.GeoSeries, values, radius: float) -> np.ndarray:
    """Dla każdego punktu z `targets`: suma `values` źródeł w promieniu `radius`."""
    tree = STRtree(sources.values)
    t_idx, s_idx = tree.query(targets.values, predicate="dwithin", distance=radius)
    return np.bincount(t_idx, weights=np.asarray(values, dtype=float)[s_idx], minlength=len(targets))


def main() -> None:
    districts = gpd.read_file(EXTERNAL_DIR / "warszawa_dzielnice.geojson").to_crs(CRS)
    growth = district_growth(districts)

    print("== odcinki ulic")
    seg = load_segments()
    print("== kawałki ludności (GUS × Kontur)")
    pieces = population_pieces(districts)
    print("== przypisywanie ludności do odcinków")
    pop = assign_to_segments(seg, pieces)
    seg = seg.join(pop, on="segment_id")
    seg[["pop_2021", "pop_2025"]] = seg[["pop_2021", "pop_2025"]].fillna(0)

    mid = seg.geometry.interpolate(0.5, normalized=True)
    seg["dzielnica"] = gpd.sjoin_nearest(
        gpd.GeoDataFrame(geometry=mid, crs=CRS), districts[["dzielnica", "geometry"]], how="left"
    ).groupby(level=0)["dzielnica"].first().values
    cagr = seg["dzielnica"].map(growth["cagr"]).fillna(0)
    for year in FORECAST_YEARS:
        seg[f"pop_{year}_proj"] = seg["pop_2025"] * (1 + cagr) ** (year - 2025)
    seg["district_cagr"] = cagr
    seg["pop_per_100m"] = seg["pop_2025"] / seg["length"].clip(lower=1) * 100

    print("== przystanki i dostępność")
    stops = gpd.read_file(EXTERNAL_DIR / "warszawa_przystanki_ztm.geojson").to_crs(CRS)
    near = gpd.sjoin_nearest(
        gpd.GeoDataFrame(geometry=mid, crs=CRS), stops[["geometry"]], how="left", distance_col="d"
    )
    seg["dist_to_stop_m"] = near.groupby(level=0)["d"].min().values
    seg["stops_300m"] = catchment(mid, stops.geometry, np.ones(len(stops)), STOP_COUNT_RADIUS_M).astype(int)
    seg["served"] = seg["dist_to_stop_m"] <= WALK_M
    seg["pop_400m"] = catchment(mid, mid, seg["pop_2025"], WALK_M)
    # Ludność w zasięgu odcinka, która dziś nie ma słupka w promieniu 400 m —
    # główny sygnał "tu brakuje przystanku".
    seg["unserved_pop_400m"] = catchment(mid, mid, seg["pop_2025"].where(~seg["served"], 0), WALK_M)
    for year in FORECAST_YEARS:
        seg[f"unserved_pop_400m_{year}"] = catchment(
            mid, mid, seg[f"pop_{year}_proj"].where(~seg["served"], 0), WALK_M
        )

    out = seg.to_crs("EPSG:4326")
    out.to_parquet(DATA_DIR / "segments_population.parquet")

    total = seg["pop_2025"].sum()
    unserved = seg.loc[~seg["served"], "pop_2025"].sum()
    print(f"== zapisano {len(out)} odcinków -> data/segments_population.parquet")
    print(f"   ludność przypisana: 2021 {seg['pop_2021'].sum():,.0f}, 2025 {total:,.0f} "
          f"(GUS dzielnice 2025: {districts['ludnosc_gus_2025'].sum():,.0f})")
    print(f"   poza zasięgiem {WALK_M} m od słupka: {unserved:,.0f} os. ({unserved / total:.1%})")


if __name__ == "__main__":
    main()
