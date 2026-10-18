"""Ranking linii ZTM: przepełnione, puste, w korkach, pogarszające się do 2036, najlepsze ogólnie.

Wskaźniki na linię (wszystkie rodzaje: autobus, tramwaj, metro, SKM):
  z rozkładu (GTFS, dzień roboczy):
    kursy_dzien, kursy_h_szczyt, dlugosc_km, przystanki,
    v_szczyt_kmh (7–9), v_wieczor_kmh (20–22), korki = 1 − v_szczyt / v_wieczor
  z symulacji pasażerów (dane/symulacja/odcinki_obciazenie.csv, poranny szczyt):
    max_wyk_2025/2036  — największe wykorzystanie pojemności na odcinku
    wypelnienie_2025/2036 — średnie wypełnienie = pasażerokm / miejscokm
    przepelnione_odcinki_2025/2036, zmiana_max_wyk (2036 − 2025)
  wynik 0–100 (najlepsze ogólnie) = WAGI: wypełnienie (percentyl wśród linii), brak przepełnienia (kara od 90%
    pojemności, pełna przy 150%), prędkość i brak korków (percentyl w obrębie rodzaju: autobus z autobusem…).

Wynik: dane/komunikacja/ranking_linii.csv, frontend/ranking_linii.json (z odcinkami do mapki linii, granicą i Wisłą)
Uruchomienie (po symulacja_pasazerow.py):  .venv/bin/python mapping/ranking_linii.py
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GTFS = ROOT / "data" / "gtfs" / "warsaw"
DANE = ROOT / "dane"
OUT_CSV = DANE / "komunikacja" / "ranking_linii.csv"
OUT_JSON = ROOT / "frontend" / "ranking_linii.json"
SERVICE_SUFFIX = ":PcS"
MODE = {"3": "autobus", "0": "tramwaj", "1": "metro", "2": "kolej"}
PEAK, EVENING = (7 * 3600, 9 * 3600), (20 * 3600, 22 * 3600)
WAGI = {"wypelnienie": 0.30, "bez_przepelnienia": 0.30, "predkosc": 0.20, "bez_korkow": 0.20}
OVER_FROM, OVER_MAX = 0.9, 1.5   # kara za przepełnienie: od 90% pojemności, pełna przy 150%


def to_sec(t: pd.Series) -> pd.Series:
    hms = t.str.split(":", expand=True).astype(int)
    return hms[0] * 3600 + hms[1] * 60 + hms[2]


def gtfs_stats() -> pd.DataFrame:
    routes = pd.read_csv(GTFS / "routes.txt", dtype=str)[["route_id", "route_short_name", "route_long_name", "route_type"]]
    trips = pd.read_csv(GTFS / "trips.txt", dtype=str, usecols=["trip_id", "route_id", "service_id", "direction_id"])
    day = sorted(s for s in trips["service_id"].unique() if s.endswith(SERVICE_SUFFIX))[2]
    keep = {day} | ({"PcM"} & set(trips["service_id"]))
    trips = trips[trips["service_id"].isin(keep)].merge(routes, on="route_id")
    trips = trips[trips["route_type"].isin(MODE)]
    st = pd.read_csv(GTFS / "stop_times.txt", dtype={"trip_id": str},
                     usecols=["trip_id", "stop_sequence", "departure_time", "shape_dist_traveled"])
    st = st[st["trip_id"].isin(set(trips["trip_id"]))]
    st["t"] = to_sec(st["departure_time"])
    g = st.groupby("trip_id").agg(t0=("t", "min"), t1=("t", "max"), km=("shape_dist_traveled", "max"), n=("stop_sequence", "size"))
    tr = trips.set_index("trip_id").join(g)
    tr["v"] = tr["km"] / ((tr["t1"] - tr["t0"]).clip(lower=60) / 3600)

    # metro: kursy wzorcowe z frequencies -> liczba kursów
    freq = pd.read_csv(GTFS / "frequencies.txt", dtype={"trip_id": str})
    freq = freq[freq["trip_id"].isin(set(tr.index))].copy()
    freq["s0"], freq["s1"] = to_sec(freq["start_time"]), to_sec(freq["end_time"])
    freq["runs"] = (freq["s1"] - freq["s0"]) / freq["headway_secs"]
    freq["runs_peak"] = ((np.minimum(freq["s1"], PEAK[1]) - np.maximum(freq["s0"], PEAK[0])).clip(lower=0)) / freq["headway_secs"]
    runs = freq.groupby("trip_id")[["runs", "runs_peak"]].sum()
    tr["kursy"] = tr.index.map(runs["runs"]).fillna(1.0)
    tr["kursy_peak"] = tr.index.map(runs["runs_peak"])
    tr["kursy_peak"] = tr["kursy_peak"].fillna(tr["t0"].between(*PEAK, inclusive="left").astype(float))

    def window_speed(d, w):
        x = d[d["t0"].between(w[0], w[1], inclusive="left")]
        return x["v"].median() if len(x) else np.nan

    rows = []
    for ref, d in tr.groupby("route_short_name"):
        is_metro = d["route_type"].iloc[0] == "1"
        rows.append({
            "linia": ref, "typ": MODE[d["route_type"].iloc[0]], "nazwa": d["route_long_name"].iloc[0],
            "kursy_dzien": round(d["kursy"].sum()), "kursy_h_szczyt": round(d["kursy_peak"].sum() / 2, 1),
            "dlugosc_km": round(d.groupby("direction_id")["km"].max().mean(), 1), "przystanki": int(d["n"].max()),
            "v_szczyt_kmh": round(d["v"].median() if is_metro else window_speed(d, PEAK), 1),
            "v_wieczor_kmh": round(d["v"].median() if is_metro else window_speed(d, EVENING), 1),
        })
    out = pd.DataFrame(rows)
    out["korki"] = (1 - out["v_szczyt_kmh"] / out["v_wieczor_kmh"]).clip(lower=0).round(3)
    return out


def load_stats(seg: pd.DataFrame) -> pd.DataFrame:
    xy0 = gpd.GeoSeries(gpd.points_from_xy(seg["lon_z"], seg["lat_z"]), crs=4326).to_crs(2180)
    xy1 = gpd.GeoSeries(gpd.points_from_xy(seg["lon_do"], seg["lat_do"]), crs=4326).to_crs(2180)
    seg = seg.assign(km=xy0.distance(xy1, align=False).values / 1000)
    rows = []
    for ref, d in seg.groupby("linia"):
        capkm = (d["pojemnosc_h"] * d["km"]).sum()
        r = {"linia": ref}
        for y in (2025, 2036):
            p = d[f"pasazerow_h_{y}"]
            r[f"max_wyk_{y}"] = round(d[f"wykorzystanie_{y}"].max(), 3) if p.notna().any() else np.nan
            r[f"wypelnienie_{y}"] = round((p.fillna(0) * d["km"]).sum() / capkm, 3) if capkm > 0 and p.notna().any() else np.nan
            r[f"przepelnione_odcinki_{y}"] = int((d[f"wykorzystanie_{y}"] > 1).sum())
            r[f"paskm_h_{y}"] = round((p.fillna(0) * d["km"]).sum())
        worst = d.loc[d["wykorzystanie_2036"].idxmax()]
        r["najgorszy_odcinek"] = f"{worst['z']} → {worst['do']}"
        rows.append(r)
    out = pd.DataFrame(rows)
    out["zmiana_max_wyk"] = (out["max_wyk_2036"] - out["max_wyk_2025"]).round(3)
    return out


def score(df: pd.DataFrame) -> pd.Series:
    # Model pasażerów zaniża liczby bezwzględne, więc wypełnienie oceniamy względnie (percentyl wśród linii);
    # przepełnienie karzemy osobno.
    fill = df["wypelnienie_2025"].rank(pct=True)
    over = 1 - ((df["max_wyk_2025"] - OVER_FROM) / (OVER_MAX - OVER_FROM)).clip(lower=0)
    speed = df.groupby("typ")["v_szczyt_kmh"].rank(pct=True)
    nojam = 1 - df.groupby("typ")["korki"].rank(pct=True)
    s = (WAGI["wypelnienie"] * fill.clip(0, 1) + WAGI["bez_przepelnienia"] * over.clip(0, 1)
         + WAGI["predkosc"] * speed + WAGI["bez_korkow"] * nojam)
    return (100 * s).round(1)


def main() -> None:
    seg = pd.read_csv(DANE / "symulacja" / "odcinki_obciazenie.csv", dtype={"linia": str, "kierunek": str})
    print("== rozkład")
    g = gtfs_stats()
    print("== obciążenie")
    lo = load_stats(seg)
    df = g.merge(lo, on="linia", how="left")
    df["wynik"] = score(df)
    df["ma_szczyt"] = df["max_wyk_2025"].notna()
    df.sort_values("wynik", ascending=False).to_csv(OUT_CSV, index=False, encoding="utf-8")

    # mapka linii: odcinki z obciążeniem (z symulacji), granica miasta i Wisła
    app = json.loads((ROOT / "frontend" / "symulator_dane.json").read_text(encoding="utf-8"))
    base = json.loads((ROOT / "frontend" / "warszawa.json").read_text(encoding="utf-8"))
    S = base["scale"]
    def decode(a):
        out, x, y = [], 0, 0
        for i in range(0, len(a), 2):
            x += a[i]; y += a[i + 1]; out.append([round(x / S, 4), round(y / S, 4)])
        return out
    segs = {}
    for r in seg.itertuples():
        segs.setdefault(r.linia, []).append([round(r.lon_z, 5), round(r.lat_z, 5), round(r.lon_do, 5), round(r.lat_do, 5),
                                             None if pd.isna(r.wykorzystanie_2025) else round(r.wykorzystanie_2025, 2),
                                             round(r.wykorzystanie_2036, 2), r.kierunek])
    cols = [c for c in df.columns if c != "ma_szczyt"]
    data = {
        "wagi": WAGI, "przepelnienie_od": OVER_FROM, "przepelnienie_max": OVER_MAX, "fields": cols,
        "linie": [[None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v) for v in row]
                  for row in df[cols].itertuples(index=False)],
        "odcinki": segs,
        "granica": [[[round(x, 4), round(y, 4)] for x, y in ring] for ring in app["boundary"]],
        "wisla": [decode(c) for c in base["river"]],
    }
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"== {len(df)} linii ({df['typ'].value_counts().to_dict()}), z obciążeniem w szczycie: {df['ma_szczyt'].sum()}")
    print(f"== {OUT_CSV.relative_to(ROOT)}, {OUT_JSON.relative_to(ROOT)} ({OUT_JSON.stat().st_size / 1e6:.1f} MB)")
    show = ["linia", "typ", "wynik", "wypelnienie_2025", "max_wyk_2025", "max_wyk_2036", "v_szczyt_kmh", "korki"]
    print("najlepsze:\n" + df.sort_values("wynik", ascending=False).head(6)[show].to_string(index=False))
    print("najgorsze:\n" + df[df["ma_szczyt"]].sort_values("wynik").head(6)[show].to_string(index=False))


if __name__ == "__main__":
    main()
