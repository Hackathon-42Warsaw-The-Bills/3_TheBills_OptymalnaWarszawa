"""Metro i kolej w klastrach H3: czy w klastrze jest stacja i jak daleko do najbliższej.

Wejście:  dane/klastry/kontury_400m/warszawa_h3_400m_pelna.geojson (mapping/h3_siatka.py)
          dane/kolej/stacje.geojson (mapping/kolej.py)
          data/hubs.parquet — planowane stacje z rokiem otwarcia (opcjonalnie, rail_plans.py)
Wynik:    dane/kolej/klastry_kolej.csv — jeden wiersz na klaster (klucz h3)

Kolumny:
  ma_metro, ma_kolej           — stacja metra / kolei (SKM, KM, WKD) w granicach klastra
  stacje_metro, stacje_kolej   — nazwy stacji w klastrze
  linie                        — linie zatrzymujące się w klastrze, np. "M1, S2, R1"
  dist_metro_m, dist_kolej_m   — od środka klastra do najbliższej stacji (w linii prostej)
  metro_800m, kolej_800m       — najbliższa stacja w zasięgu dojścia ~10 min
  metro_plan_rok               — najbliższy rok otwarcia planowanej stacji metra w klastrze

Uruchomienie (po kolej.py):  .venv/bin/python mapping/klastry_kolej.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HEX_PATH = ROOT / "dane" / "klastry" / "kontury_400m" / "warszawa_h3_400m_pelna.geojson"
STATIONS_PATH = ROOT / "dane" / "kolej" / "stacje.geojson"
PLANNED_PATH = ROOT / "data" / "hubs.parquet"
OUT = ROOT / "dane" / "kolej" / "klastry_kolej.csv"
CRS = "EPSG:2180"
WALK_RAIL_M = 800   # typowy zasięg dojścia do stacji szynowej (~10 min)


def join_list(values) -> str:
    return ", ".join(sorted({v for x in values for v in str(x).split(", ") if v}))


def nearest_dist(points: gpd.GeoSeries, targets: gpd.GeoDataFrame) -> np.ndarray:
    if targets.empty:
        return np.full(len(points), np.nan)
    j = gpd.sjoin_nearest(gpd.GeoDataFrame(geometry=points), targets[["geometry"]], distance_col="d")
    return j.groupby(level=0)["d"].min().reindex(range(len(points))).values


def main() -> None:
    hexes = gpd.read_file(HEX_PATH)[["h3", "dzielnica", "geometry"]].to_crs(CRS)
    stations = gpd.read_file(STATIONS_PATH).to_crs(CRS)
    stations["typ"] = np.where(stations["system"] == "metro", "metro", "kolej")

    inside = gpd.sjoin(stations, hexes[["h3", "geometry"]], predicate="within")
    out = hexes[["h3", "dzielnica"]].copy().set_index("h3")
    for typ in ("metro", "kolej"):
        g = inside[inside["typ"] == typ].groupby("h3")
        out[f"ma_{typ}"] = out.index.isin(g.groups)
        out[f"stacje_{typ}"] = g["nazwa"].agg(join_list).reindex(out.index).fillna("")
    out["linie"] = inside.groupby("h3")["linie"].agg(join_list).reindex(out.index).fillna("")

    centers = hexes.geometry.centroid.reset_index(drop=True)
    for typ in ("metro", "kolej"):
        d = nearest_dist(centers, stations[stations["typ"] == typ].reset_index(drop=True))
        out[f"dist_{typ}_m"] = np.round(d).astype(int)
        out[f"{typ}_800m"] = out[f"dist_{typ}_m"] <= WALK_RAIL_M

    out["metro_plan_rok"] = pd.NA
    if PLANNED_PATH.exists():
        plan = gpd.read_parquet(PLANNED_PATH).to_crs(CRS)
        plan = plan[(plan["kind"] == "metro") & (plan["status"] != "istniejący")]
        pin = gpd.sjoin(plan, hexes[["h3", "geometry"]], predicate="within")
        out["metro_plan_rok"] = pin.groupby("h3")["open_year"].min().reindex(out.index).astype("Int64")

    out.reset_index().to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"== {len(out)} klastrów -> {OUT.relative_to(ROOT)}")
    print(f"   ze stacją metra: {out['ma_metro'].sum()}, ze stacją kolei: {out['ma_kolej'].sum()}, "
          f"z jednym i drugim: {(out['ma_metro'] & out['ma_kolej']).sum()}")
    print(f"   metro ≤ {WALK_RAIL_M} m od środka: {out['metro_800m'].sum()}, kolej: {out['kolej_800m'].sum()}, "
          f"żadne: {(~out['metro_800m'] & ~out['kolej_800m']).sum()}")
    print(f"   z planowaną stacją metra: {out['metro_plan_rok'].notna().sum()} "
          f"({out['metro_plan_rok'].value_counts().sort_index().to_dict()})")


if __name__ == "__main__":
    main()
