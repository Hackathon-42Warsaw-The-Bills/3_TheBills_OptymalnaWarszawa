"""Prędkości autobusów ZTM na odcinkach ulic z rozkładu jazdy (GTFS) — wskaźnik korków.

ZTM układa rozkład według zmierzonych czasów przejazdu o różnych porach, więc rozkładowa
prędkość w szczycie vs poza szczytem pokazuje, gdzie autobusy stoją w korkach.
(Zastępstwo za APR ZDM, które nie ma otwartego API. Gdy będzie klucz do API miasta,
mapping/gps_ztm.py da prędkości zmierzone — ten sam format wyniku.)

Źródło: GTFS Warszawy (mkuran.pl/gtfs, generowany z danych ZTM), dzień roboczy SERVICE_DAY.
Metoda:
  1. Dla każdej pary kolejnych przystanków kursu autobusowego: odległość (shape_dist_traveled)
     i czas przejazdu. Mediana z kursów w SZCZYCIE i w PORZE_SWOBODNEJ.
  2. Geometria odcinka z shapes.txt próbkowana co SAMPLE_M; każdy punkt przypisujemy do
     najbliższego odcinka ulicy (≤ SNAP_M) z data/roads_edges.parquet.
  3. Na odcinek ulicy: średnia ważona liczbą kursów -> v_szczyt, v_swobodna, korek = 1 − v_szczyt / v_swobodna.

Wynik: data/predkosci_ulic.parquet — edge_id, v_szczyt_kmh, v_swobodna_kmh, korek, kursy_szczyt
Uruchomienie (po build_graph.py):  .venv/bin/python mapping/gtfs_predkosci.py
"""

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely import STRtree
from shapely.geometry import LineString
from shapely.ops import substring

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GTFS_DIR = DATA_DIR / "gtfs" / "warsaw"
GTFS_URL = "https://mkuran.pl/gtfs/warsaw.zip"
OUT = DATA_DIR / "predkosci_ulic.parquet"
CRS = "EPSG:2180"

SERVICE_SUFFIX = ":PcS"          # dni robocze pon–czw; bierzemy pierwszy dostępny (środa, jeśli jest)
PEAK = (7 * 3600, 9 * 3600)      # poranny szczyt
FREE = (20 * 3600, 22 * 3600)    # wieczór, ruch swobodny
SAMPLE_M = 25
SNAP_M = 15
MIN_SEG_M = 150                  # krótsze odcinki międzyprzystankowe — czas w rozkładzie za mało dokładny


def ensure_gtfs() -> None:
    if (GTFS_DIR / "stop_times.txt").exists():
        return
    GTFS_DIR.mkdir(parents=True, exist_ok=True)
    r = requests.get(GTFS_URL, timeout=600, headers={"User-Agent": "optymalna-warszawa-hackathon/1.0"})
    r.raise_for_status()
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(GTFS_DIR)


def to_sec(t: pd.Series) -> pd.Series:
    hms = t.str.split(":", expand=True).astype(int)
    return hms[0] * 3600 + hms[1] * 60 + hms[2]


def segment_speeds() -> pd.DataFrame:
    """Mediana czasu przejazdu między kolejnymi przystankami, szczyt vs pora swobodna."""
    routes = pd.read_csv(GTFS_DIR / "routes.txt", dtype=str)
    trips = pd.read_csv(GTFS_DIR / "trips.txt", dtype=str, usecols=["trip_id", "route_id", "service_id", "shape_id"])
    services = sorted(s for s in trips["service_id"].unique() if s.endswith(SERVICE_SUFFIX))
    day = services[min(2, len(services) - 1)]   # trzeci dzień roboczy w pliku = zwykle środa
    bus_routes = set(routes.loc[routes["route_type"] == "3", "route_id"])
    trips = trips[(trips["service_id"] == day) & trips["route_id"].isin(bus_routes)]
    print(f"   dzień rozkładu {day}: {len(trips)} kursów autobusowych")

    st = pd.read_csv(GTFS_DIR / "stop_times.txt", dtype={"trip_id": str, "stop_id": str},
                     usecols=["trip_id", "stop_sequence", "stop_id", "departure_time", "shape_dist_traveled"])
    st = st[st["trip_id"].isin(set(trips["trip_id"]))].sort_values(["trip_id", "stop_sequence"])
    st["t"] = to_sec(st["departure_time"])
    nxt = st.groupby("trip_id").shift(-1)
    seg = pd.DataFrame({"trip_id": st["trip_id"], "a": st["stop_id"], "b": nxt["stop_id"],
                        "t0": st["t"], "dt": nxt["t"] - st["t"],
                        "d0": st["shape_dist_traveled"], "d1": nxt["shape_dist_traveled"]}).dropna()
    seg["dist_m"] = (seg["d1"] - seg["d0"]) * 1000
    seg = seg[(seg["dist_m"] >= MIN_SEG_M) & (seg["dt"] > 0)]
    seg = seg.merge(trips[["trip_id", "shape_id"]], on="trip_id")

    def agg(window):
        w = seg[(seg["t0"] >= window[0]) & (seg["t0"] < window[1])]
        return w.groupby(["a", "b"]).agg(dt=("dt", "median"), dist_m=("dist_m", "median"), n=("dt", "size"),
                                         shape_id=("shape_id", "first"), d0=("d0", "first"), d1=("d1", "first"))
    peak, free = agg(PEAK), agg(FREE)
    out = peak.join(free[["dt"]], rsuffix="_free", how="inner")
    out["v_szczyt_kmh"] = out["dist_m"] / out["dt"] * 3.6
    out["v_swobodna_kmh"] = out["dist_m"] / out["dt_free"] * 3.6
    # Rozkład bywa „zaokrąglony” do minut — odcinki z absurdalną prędkością odrzucamy.
    out = out[out["v_szczyt_kmh"].between(3, 70) & out["v_swobodna_kmh"].between(3, 80)]
    return out.reset_index()


def segment_geometries(seg: pd.DataFrame) -> gpd.GeoDataFrame:
    shapes = pd.read_csv(GTFS_DIR / "shapes.txt", dtype={"shape_id": str})
    shapes = shapes[shapes["shape_id"].isin(set(seg["shape_id"]))].sort_values(["shape_id", "shape_pt_sequence"])
    lines = gpd.GeoSeries({sid: LineString(g[["shape_pt_lon", "shape_pt_lat"]].values)
                           for sid, g in shapes.groupby("shape_id") if len(g) > 1}, crs="EPSG:4326").to_crs(CRS)
    # shape_dist_traveled w km wg długości kształtu -> wycinek proporcjonalny
    km_total = shapes.groupby("shape_id")["shape_dist_traveled"].max()
    geoms = []
    for r in seg.itertuples():
        line = lines.get(r.shape_id)
        if line is None:
            geoms.append(None)
            continue
        f = line.length / max(km_total[r.shape_id], 1e-6)
        geoms.append(substring(line, r.d0 * f, r.d1 * f))
    return gpd.GeoDataFrame(seg, geometry=geoms, crs=CRS).dropna(subset=["geometry"])


def to_edges(seg: gpd.GeoDataFrame) -> pd.DataFrame:
    edges = gpd.read_parquet(DATA_DIR / "roads_edges.parquet").to_crs(CRS)[["edge_id", "geometry"]]
    tree = STRtree(edges.geometry.values)
    rows = []
    for r in seg.itertuples():
        g = r.geometry
        pts = [g.interpolate(d) for d in np.arange(SAMPLE_M / 2, g.length, SAMPLE_M)]
        if not pts:
            continue
        idx, dist = tree.query_nearest(pts, max_distance=SNAP_M, return_distance=True)
        for e in np.unique(idx[1]):
            rows.append((edges["edge_id"].iat[e], r.v_szczyt_kmh, r.v_swobodna_kmh, r.n))
    df = pd.DataFrame(rows, columns=["edge_id", "v_szczyt", "v_swobodna", "n"])
    w = df["n"]
    out = df.assign(vs=df["v_szczyt"] * w, vf=df["v_swobodna"] * w).groupby("edge_id").agg(
        vs=("vs", "sum"), vf=("vf", "sum"), kursy_szczyt=("n", "sum"))
    out["v_szczyt_kmh"] = (out["vs"] / out["kursy_szczyt"]).round(1)
    out["v_swobodna_kmh"] = (out["vf"] / out["kursy_szczyt"]).round(1)
    out["korek"] = (1 - out["v_szczyt_kmh"] / out["v_swobodna_kmh"]).clip(lower=0).round(3)
    return out[["v_szczyt_kmh", "v_swobodna_kmh", "korek", "kursy_szczyt"]].reset_index()


def main() -> None:
    ensure_gtfs()
    print("== odcinki międzyprzystankowe")
    seg = segment_speeds()
    print(f"   {len(seg)} par przystanków; mediana prędkości szczyt {seg['v_szczyt_kmh'].median():.1f} km/h, "
          f"wieczór {seg['v_swobodna_kmh'].median():.1f} km/h")
    print("== geometria i dopasowanie do ulic")
    out = to_edges(segment_geometries(seg))
    out.to_parquet(OUT)
    print(f"== {len(out)} odcinków ulic z prędkością -> {OUT.relative_to(ROOT)}")
    print(out[["v_szczyt_kmh", "v_swobodna_kmh", "korek"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
