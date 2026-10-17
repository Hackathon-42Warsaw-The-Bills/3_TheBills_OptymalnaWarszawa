"""Zmierzone prędkości autobusów z GPS ZTM — zamiennik prędkości rozkładowych (gtfs_predkosci.py).

API miasta (dane.um.warszawa.pl, get_ztm_lokalizacja_pojazdow) co ~10 s podaje pozycje wszystkich
autobusów. Klucz: ZTM_API_KEY w pliku .env (ten sam co dla frontend/serwer.py).

Dwa kroki:
  1. zbieraj — co POLL_S sekund zapisuje pozycje do data/gps/<data>.jsonl.
     Uruchom w dzień roboczy w szczycie (np. 7:00–9:00) i wieczorem (20:00–22:00):
        .venv/bin/python mapping/gps_ztm.py zbieraj 120     # minut
  2. licz — z kolejnych pozycji tego samego pojazdu liczy prędkość, przypisuje ją do
     najbliższego odcinka ulicy i zapisuje data/predkosci_ulic.parquet w TYM SAMYM formacie
     co gtfs_predkosci.py (v_szczyt_kmh, v_swobodna_kmh, korek, kursy_szczyt), więc
     export_symulator.py używa go bez zmian. Odcinki bez pomiarów zostają z rozkładu.
        .venv/bin/python mapping/gps_ztm.py licz
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely import STRtree
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GPS_DIR = DATA_DIR / "gps"
OUT = DATA_DIR / "predkosci_ulic.parquet"
GTFS_SPEEDS = DATA_DIR / "predkosci_ulic_gtfs.parquet"
CRS = "EPSG:2180"

sys.path.insert(0, str(ROOT / "frontend"))
import serwer  # noqa: E402 — to samo pobieranie co serwer frontendu (nowe/stare API, klucz z .env)
POLL_S = 15
PEAK_HOURS = range(7, 9)        # 7:00–8:59
FREE_HOURS = range(20, 22)      # 20:00–21:59
MIN_DT_S, MAX_DT_S = 10, 90     # pary pozycji: zbyt krótki odstęp = szum GPS, zbyt długi = postój/przerwa
MAX_KMH = 80
SNAP_M = 20
MIN_OBS = 5                     # minimum pomiarów na odcinek


def collect(minutes: float) -> None:
    if not serwer.API_KEY:
        sys.exit("Brak klucza: ustaw ZTM_API_KEY w .env (token z dane.um.warszawa.pl).")
    GPS_DIR.mkdir(parents=True, exist_ok=True)
    out = GPS_DIR / f"{datetime.now():%Y-%m-%d}.jsonl"
    end = time.time() + minutes * 60
    n = 0
    with open(out, "a") as f:
        while time.time() < end:
            try:
                rows = json.loads(serwer.fetch_vehicles())["pojazdy"]
                for lon, lat, line, brig, veh, typ, age in rows:
                    if typ == 1:  # tylko autobusy
                        t = (datetime.now().timestamp() - age)
                        f.write(json.dumps({"Lon": lon, "Lat": lat, "Lines": line, "Brigade": brig, "VehicleNumber": veh,
                                            "Time": datetime.fromtimestamp(t).isoformat(timespec="seconds")}) + "\n")
                        n += 1
                print(f"{datetime.now():%H:%M:%S}  {len(rows)} pojazdów (razem zapisanych {n})", flush=True)
            except (requests.RequestException, RuntimeError) as e:
                print(f"{datetime.now():%H:%M:%S}  błąd: {e}", flush=True)
            time.sleep(POLL_S)
    print(f"== zapisano {n} pozycji -> {out.relative_to(ROOT)}")


def load_positions() -> pd.DataFrame:
    rows = []
    for path in sorted(GPS_DIR.glob("*.jsonl")):
        with open(path) as f:
            rows += [json.loads(line) for line in f]
    df = pd.DataFrame(rows).drop_duplicates(["VehicleNumber", "Time"])
    df["t"] = pd.to_datetime(df["Time"])
    df = df[(df["Lat"].between(52.0, 52.5)) & (df["Lon"].between(20.7, 21.4))]
    return df.sort_values(["VehicleNumber", "t"])


def speeds_per_edge(df: pd.DataFrame) -> pd.DataFrame:
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["Lon"], df["Lat"]), crs="EPSG:4326").to_crs(CRS)
    pts["x"], pts["y"] = pts.geometry.x, pts.geometry.y
    nxt = pts.groupby("VehicleNumber")[["x", "y", "t"]].shift(-1)
    seg = pd.DataFrame({"x0": pts["x"], "y0": pts["y"], "x1": nxt["x"], "y1": nxt["y"],
                        "dt": (nxt["t"] - pts["t"]).dt.total_seconds(), "hour": pts["t"].dt.hour}).dropna()
    seg["dist"] = np.hypot(seg["x1"] - seg["x0"], seg["y1"] - seg["y0"])
    seg = seg[seg["dt"].between(MIN_DT_S, MAX_DT_S)]
    seg["kmh"] = seg["dist"] / seg["dt"] * 3.6
    seg = seg[seg["kmh"] <= MAX_KMH]
    seg["pora"] = np.where(seg["hour"].isin(PEAK_HOURS), "szczyt", np.where(seg["hour"].isin(FREE_HOURS), "swobodna", ""))
    seg = seg[seg["pora"] != ""]
    # Pomiar przypisujemy do odcinka ulicy najbliższego środkowi przejazdu.
    edges = gpd.read_parquet(DATA_DIR / "roads_edges.parquet").to_crs(CRS)[["edge_id", "geometry"]]
    mids = gpd.points_from_xy((seg["x0"] + seg["x1"]) / 2, (seg["y0"] + seg["y1"]) / 2, crs=CRS)
    idx, _ = STRtree(edges.geometry.values).query_nearest(mids, max_distance=SNAP_M, return_distance=True)
    seg = seg.iloc[idx[0]].assign(edge_id=edges["edge_id"].values[idx[1]])
    g = seg.groupby(["edge_id", "pora"])["kmh"].agg(["median", "size"]).unstack("pora")
    out = pd.DataFrame({"v_szczyt_kmh": g[("median", "szczyt")], "v_swobodna_kmh": g[("median", "swobodna")],
                        "kursy_szczyt": g[("size", "szczyt")]}).dropna(subset=["v_szczyt_kmh"])
    out = out[out["kursy_szczyt"] >= MIN_OBS]
    return out.reset_index()


def compute() -> None:
    df = load_positions()
    print(f"== {len(df)} pozycji GPS, {df['VehicleNumber'].nunique()} pojazdów")
    gps = speeds_per_edge(df)
    # Odcinki bez pomiaru (albo bez pomiaru wieczornego) uzupełniamy prędkościami z rozkładu.
    if not GTFS_SPEEDS.exists() and OUT.exists():
        pd.read_parquet(OUT).to_parquet(GTFS_SPEEDS)
    base = pd.read_parquet(GTFS_SPEEDS).set_index("edge_id") if GTFS_SPEEDS.exists() else pd.DataFrame()
    gps = gps.set_index("edge_id")
    gps["v_swobodna_kmh"] = gps["v_swobodna_kmh"].fillna(base["v_swobodna_kmh"] if len(base) else np.nan)
    gps["v_swobodna_kmh"] = gps["v_swobodna_kmh"].fillna(gps["v_szczyt_kmh"] * 1.15)
    merged = pd.concat([gps, base[~base.index.isin(gps.index)]]) if len(base) else gps
    merged["korek"] = (1 - merged["v_szczyt_kmh"] / merged["v_swobodna_kmh"]).clip(lower=0).round(3)
    merged.round(1).reset_index().to_parquet(OUT)
    print(f"== {len(gps)} odcinków z pomiarem GPS (+{len(merged) - len(gps)} z rozkładu) -> {OUT.relative_to(ROOT)}")
    print("   następnie: .venv/bin/python mapping/export_symulator.py")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "zbieraj":
        collect(float(sys.argv[2]) if len(sys.argv) > 2 else 120)
    elif len(sys.argv) >= 2 and sys.argv[1] == "licz":
        compute()
    else:
        sys.exit(__doc__)
