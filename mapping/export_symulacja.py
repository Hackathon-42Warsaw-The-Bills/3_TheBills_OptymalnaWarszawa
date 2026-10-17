"""Dane do animacji sieci według rozkładu (frontend/symulacja_dane.json).

Każdy kurs ZTM z dnia roboczego (autobus, tramwaj, metro, SKM) zapisany kompaktowo:
  shapes   — przebiegi (uproszczone do ~SIMPLIFY_M): współrzędne + odległość narastająca [m]
  profiles — profile kursu: pary (odległość wzdłuż przebiegu [m], czas od startu [s]) dla
             kolejnych przystanków; identyczne profile są zapisane raz
  trips    — [indeks przebiegu, indeks profilu, start [s od północy], indeks linii]
  routes   — numery linii i rodzaj (autobus / tramwaj / metro / kolej)

Metro jest w GTFS jako kursy wzorcowe z frequencies.txt — rozwijamy je na faktyczne odjazdy.
Przeglądarka liczy pozycję pojazdu: czas -> odcinek profilu -> odległość -> punkt na przebiegu.

Uruchomienie (po gtfs_predkosci.py, który pobiera GTFS):  .venv/bin/python mapping/export_symulacja.py
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parent.parent
GTFS = ROOT / "data" / "gtfs" / "warsaw"
OUT = ROOT / "frontend" / "symulacja_dane.json"
CRS = "EPSG:2180"
SIMPLIFY_M = 8
SERVICE_SUFFIX = ":PcS"
MODE = {"3": "autobus", "0": "tramwaj", "1": "metro", "2": "kolej"}


def to_sec(t: pd.Series) -> pd.Series:
    hms = t.str.split(":", expand=True).astype(int)
    return hms[0] * 3600 + hms[1] * 60 + hms[2]


def main() -> None:
    routes = pd.read_csv(GTFS / "routes.txt", dtype=str)[["route_id", "route_short_name", "route_type"]]
    trips = pd.read_csv(GTFS / "trips.txt", dtype=str, usecols=["trip_id", "route_id", "service_id", "shape_id"])
    day = sorted(s for s in trips["service_id"].unique() if s.endswith(SERVICE_SUFFIX))[2]   # środa
    keep = {day} | ({"PcM"} & set(trips["service_id"]))
    trips = trips[trips["service_id"].isin(keep)].merge(routes, on="route_id")
    trips = trips[trips["route_type"].isin(MODE)]
    print(f"== {day}: {len(trips)} kursów, {trips['shape_id'].nunique()} przebiegów")

    # --- przebiegi: uproszczone, z odległością narastającą
    sh = pd.read_csv(GTFS / "shapes.txt", dtype={"shape_id": str})
    sh = sh[sh["shape_id"].isin(set(trips["shape_id"]))].sort_values(["shape_id", "shape_pt_sequence"])
    shape_idx, shapes, scale = {}, [], {}
    for sid, g in sh.groupby("shape_id", sort=False):
        if len(g) < 2:
            continue
        line = gpd.GeoSeries([LineString(g[["shape_pt_lon", "shape_pt_lat"]].values)], crs=4326).to_crs(CRS).iloc[0]
        simple = line.simplify(SIMPLIFY_M)
        xy = np.array(simple.coords)
        cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
        ll = gpd.GeoSeries(gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=CRS).to_crs(4326)
        shape_idx[sid] = len(shapes)
        # shape_dist_traveled [km] -> metry na uproszczonym przebiegu
        scale[sid] = cum[-1] / max(g["shape_dist_traveled"].max() * 1000, 1)
        shapes.append({"c": [[round(p.x, 5), round(p.y, 5)] for p in ll], "d": [round(v) for v in cum]})

    # --- profile kursów (odległość, czas od startu) i odjazdy
    st = pd.read_csv(GTFS / "stop_times.txt", dtype={"trip_id": str},
                     usecols=["trip_id", "stop_sequence", "departure_time", "shape_dist_traveled"])
    st = st[st["trip_id"].isin(set(trips["trip_id"]))].sort_values(["trip_id", "stop_sequence"])
    st["t"] = to_sec(st["departure_time"])
    tinfo = trips.set_index("trip_id")
    route_list = sorted(trips["route_short_name"].unique(), key=lambda r: (len(r), r))
    route_idx = {r: i for i, r in enumerate(route_list)}
    route_typ = [MODE[trips.loc[trips["route_short_name"] == r, "route_type"].iloc[0]] for r in route_list]

    freq = pd.read_csv(GTFS / "frequencies.txt", dtype={"trip_id": str}) if (GTFS / "frequencies.txt").exists() else pd.DataFrame()
    freq = freq[freq["trip_id"].isin(set(trips["trip_id"]))] if len(freq) else freq

    profile_idx, profiles, out_trips = {}, [], []
    for tid, g in st.groupby("trip_id", sort=False):
        info = tinfo.loc[tid]
        sid = info["shape_id"]
        if sid not in shape_idx:
            continue
        t0 = int(g["t"].iloc[0])
        dist = (g["shape_dist_traveled"].fillna(0).values * 1000 * scale[sid]).round()
        prof = tuple(int(v) for pair in zip(dist, g["t"].values - t0) for v in pair)
        if prof not in profile_idx:
            profile_idx[prof] = len(profiles)
            profiles.append(list(prof))
        starts = [t0]
        fr = freq[freq["trip_id"] == tid] if len(freq) else []
        if len(fr):   # metro: odjazdy co headway w oknach czasowych
            starts = [int(s) for _, w in fr.iterrows()
                      for s in np.arange(to_sec(pd.Series([w["start_time"]]))[0], to_sec(pd.Series([w["end_time"]]))[0], w["headway_secs"])]
        for s0 in starts:
            out_trips.append([shape_idx[sid], profile_idx[prof], int(s0), route_idx[info["route_short_name"]]])
    out_trips.sort(key=lambda x: x[2])

    data = {"day": day, "routes": route_list, "route_typ": route_typ, "shapes": shapes, "profiles": profiles, "trips": out_trips}
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    n_pts = sum(len(s["c"]) for s in shapes)
    typ = pd.Series([route_typ[t[3]] for t in out_trips]).value_counts().to_dict()
    print(f"== {len(shapes)} przebiegów ({n_pts} punktów), {len(profiles)} profili, {len(out_trips)} kursów {typ}")
    print(f"== {OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
