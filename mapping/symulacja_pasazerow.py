"""Symulacja pasażerów w porannym szczycie: kto skąd dokąd jedzie i jak obciąża linie (2025 vs 2036).

Klasyczny model czterostopniowy, uproszczony:
  1. Graf komunikacji z rozkładu ZTM (GTFS, dzień roboczy, 7:00–9:00):
     przystanek -> [czekanie = odstęp/2 + kara wsiadania] -> przystanek-linii -> [przejazd] -> … -> przystanek
     (przejazd metrem/SKM liczony × RAIL_FACTOR — premia kolejowa)
     + przesiadki pieszo między przystankami do TRANSFER_M, dojście z klastra do przystanków do ACCESS_M.
     Linia = najczęstszy przebieg danej linii i kierunku w szczycie; częstotliwość = wszystkie kursy w szczycie.
  2. Czasy podróży klaster -> klaster (848 × 848, Dijkstra).
  3. Podróże w godzinie szczytu: produkcja = mieszkańcy × PODROZE × udział komunikacji (malejący od centrum,
     średnio UDZIAL_ZTM) × SZCZYT,
     cele wg modelu grawitacyjnego (miejsca pracy, szkoły, usługi, handel, ludność) z exp(−β·czas);
     β dobrane tak, by średni czas podróży = SREDNI_CZAS_MIN.
  4. Przypisanie wszystko-albo-nic na najszybsze trasy -> obciążenie każdego odcinka linii
     vs pojemność (kursy/h × pojemność pojazdu wg typu taboru z rozkładu).

Scenariusze: 2025 (dzisiejsza sieć, ludność 2025) i 2036 (ludność 2036 + nowe metro M2 zachód i M3
z założoną częstotliwością i prędkością). Sieć autobusowo-tramwajowa w 2036 = dzisiejsza.

Wynik (dane/symulacja/):
  odcinki_obciazenie.csv — odcinek linii: kursy/h, pojemność/h, pasażerowie/h 2025 i 2036, wykorzystanie
  linie_obciazenie.csv   — linia: maksymalne wykorzystanie, pasażerokilometry, najbardziej obciążony odcinek
  klastry_podroze.csv    — klaster: podróże wyjeżdżające/przyjeżdżające, czas do centrum, dostępność
  przystanki_wsiadania.csv — wsiadający na przystanku
  frontend/pasazerowie_dane.json — to samo kompaktowo dla aplikacji (jeszcze niepodpięte)

Uruchomienie (po cechy_klastrow.py, export_symulator.py):  .venv/bin/python mapping/symulacja_pasazerow.py
"""

import json
from collections import Counter
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely import STRtree

ROOT = Path(__file__).resolve().parent.parent
GTFS = ROOT / "data" / "gtfs" / "warsaw"
DANE = ROOT / "dane"
OUT_DIR = DANE / "symulacja"
OUT_JSON = ROOT / "frontend" / "pasazerowie_dane.json"
CRS = "EPSG:2180"

PEAK = (7 * 3600, 9 * 3600)
SERVICE_SUFFIX = ":PcS"
WALK = 1.2                  # m/s (≈ 4,3 km/h)
ACCESS_M, TRANSFER_M = 600, 250
MAX_WAIT_S, BOARD_PEN_S = 600, 90
PODROZE, UDZIAL_ZTM, SZCZYT = 2.0, 0.45, 0.10   # podróży/os./dobę, średni udział komunikacji, udział godziny szczytu
# Udział komunikacji maleje z odległością od centrum (na obrzeżach częściej samochód):
# udział = UDZIAL_MAX − UDZIAL_SPADEK · km, w [UDZIAL_MIN, UDZIAL_MAX], potem skalowany do średniej UDZIAL_ZTM.
UDZIAL_MAX, UDZIAL_MIN, UDZIAL_SPADEK = 0.62, 0.22, 0.03
RAIL_FACTOR = 0.85          # premia kolejowa: czas jazdy metrem/SKM odczuwany jako 85% (szybkie, punktualne, bez korków)
SREDNI_CZAS_MIN = 35
CENTRUM = (21.0067, 52.2318)
CAPACITY = [("18m", 150), ("12m", 90), ("8-10m", 60), ("DUO", 230), ("2 wagony", 230), ("120N", 200),
            ("116N", 200), ("134N", 200), ("2x", 1200), ("WE", 600)]
CAP_DEFAULT = {"3": 100, "0": 200, "1": 1500, "2": 600}
MODE = {"3": "autobus", "0": "tramwaj", "1": "metro", "2": "kolej"}
# Nowe metro 2036 (z frontend/symulator_dane.json): odstęp w szczycie i prędkość handlowa (założenia)
NEW_METRO = {"M2 zachód": (2027, 240), "M3": (2033, 240)}
METRO_KMH, METRO_DWELL_S = 35, 30


def to_sec(t: pd.Series) -> pd.Series:
    hms = t.str.split(":", expand=True).astype(int)
    return hms[0] * 3600 + hms[1] * 60 + hms[2]


def capacity_of(fleet: str, rtype: str) -> int:
    for key, cap in CAPACITY:
        if isinstance(fleet, str) and key in fleet:
            return cap
    return CAP_DEFAULT[rtype]


# ---------- 1. linie w szczycie ----------

def peak_patterns():
    routes = pd.read_csv(GTFS / "routes.txt", dtype=str)[["route_id", "route_short_name", "route_type"]]
    trips = pd.read_csv(GTFS / "trips.txt", dtype=str, usecols=["trip_id", "route_id", "service_id", "direction_id", "fleet_type"])
    day = sorted(s for s in trips["service_id"].unique() if s.endswith(SERVICE_SUFFIX))[2]
    keep = {day} | ({"PcM"} & set(trips["service_id"]))
    trips = trips[trips["service_id"].isin(keep)].merge(routes, on="route_id")
    trips = trips[trips["route_type"].isin(MODE)]
    st = pd.read_csv(GTFS / "stop_times.txt", dtype={"trip_id": str, "stop_id": str},
                     usecols=["trip_id", "stop_sequence", "stop_id", "departure_time"])
    st = st[st["trip_id"].isin(set(trips["trip_id"]))].sort_values(["trip_id", "stop_sequence"])
    st["t"] = to_sec(st["departure_time"])

    freq = pd.read_csv(GTFS / "frequencies.txt", dtype={"trip_id": str})
    freq = freq[freq["trip_id"].isin(set(trips["trip_id"]))]
    freq["s0"], freq["s1"] = to_sec(freq["start_time"]), to_sec(freq["end_time"])
    metro_hw = {tid: g.loc[(g["s0"] <= 8 * 3600) & (g["s1"] > 8 * 3600), "headway_secs"].min() for tid, g in freq.groupby("trip_id")}

    first = st.groupby("trip_id").first()
    seqs = st.groupby("trip_id")["stop_id"].apply(tuple)
    tinfo = trips.set_index("trip_id").join(first["t"].rename("t0")).join(seqs.rename("seq"))
    patterns = []
    for (ref, dirn), g in tinfo.groupby(["route_short_name", "direction_id"]):
        rtype = g["route_type"].iloc[0]
        if rtype == "1":   # metro: kurs wzorcowy z frequencies
            tid = g.index[0]
            hw = metro_hw.get(tid)
            if not hw or np.isnan(hw):
                continue
            per_h, main = 3600 / hw, tid
            seq = g.loc[tid, "seq"]
            times = st.loc[st["trip_id"] == tid, "t"].values
            run = np.maximum(np.diff(times), 30)
        else:
            inpeak = g[(g["t0"] >= PEAK[0] - 1800) & (g["t0"] < PEAK[1] - 1800)]   # kursy w trasie w szczycie
            if len(inpeak) == 0:
                continue
            per_h = len(inpeak) / ((PEAK[1] - PEAK[0]) / 3600)
            seq = Counter(inpeak["seq"]).most_common(1)[0][0]
            ids = inpeak.index[inpeak["seq"] == seq]
            tt = st[st["trip_id"].isin(ids)].pivot_table(index="trip_id", columns="stop_sequence", values="t").values
            run = np.maximum(np.nanmedian(np.diff(tt, axis=1), axis=0), 20)
        fleet = g["fleet_type"].mode().iloc[0] if g["fleet_type"].notna().any() else None
        cap = capacity_of(fleet, rtype)
        patterns.append({"ref": ref, "dir": dirn, "typ": MODE[rtype], "per_h": per_h, "cap": cap,
                         "stops": list(seq), "run": [float(x) for x in run]})
    print(f"   linie w szczycie: {len(patterns)} przebiegów ({Counter(p['typ'] for p in patterns)})")
    return patterns


def new_metro_patterns(stops_xy: dict):
    """Nowe linie metra 2036 jako przebiegi z nowymi przystankami (współrzędne z danych aplikacji)."""
    d = json.loads((ROOT / "frontend" / "symulator_dane.json").read_text(encoding="utf-8"))
    out, extra = [], {}
    for m in d["planned_metro"]:
        if m["ref"] not in NEW_METRO:
            continue
        _, hw = NEW_METRO[m["ref"]]
        ids = []
        for lon, lat, name, ex in m["stacje"]:
            sid = f"NEW:{m['ref']}:{name}"
            extra[sid] = (lon, lat, name)
            ids.append(sid)
        xy = gpd.GeoSeries(gpd.points_from_xy([extra[i][0] for i in ids], [extra[i][1] for i in ids]), crs=4326).to_crs(CRS)
        dist = [xy.iloc[k].distance(xy.iloc[k + 1]) * 1.15 for k in range(len(ids) - 1)]
        run = [dd / (METRO_KMH / 3.6) + METRO_DWELL_S for dd in dist]
        for dirn, seq, r in (("0", ids, run), ("1", ids[::-1], run[::-1])):
            out.append({"ref": m["ref"], "dir": dirn, "typ": "metro", "per_h": 3600 / hw, "cap": 1500, "stops": seq, "run": r})
    return out, extra


# ---------- 2. graf ----------

class Network:
    def __init__(self, patterns, stops: pd.DataFrame, hexes: gpd.GeoDataFrame):
        self.patterns = patterns
        used = {s for p in patterns for s in p["stops"]}
        stops = stops[stops["stop_id"].isin(used)].reset_index(drop=True)
        self.stops = stops
        sidx = {s: i for i, s in enumerate(stops["stop_id"])}
        S, H = len(stops), len(hexes)
        rows, cols, w = [], [], []
        node = S + H
        self.seg = []            # (pattern, k, from_node, to_node)
        self.ls_stop = {}        # przystanek-linii -> przystanek (do liczenia wsiadających)
        for pi, p in enumerate(patterns):
            wait = min(1800 / p["per_h"], MAX_WAIT_S) + BOARD_PEN_S
            ls = list(range(node, node + len(p["stops"])))
            node += len(p["stops"])
            rf = RAIL_FACTOR if p["typ"] in ("metro", "kolej") else 1.0
            for k, (sid, n) in enumerate(zip(p["stops"], ls)):
                s = sidx[sid]
                self.ls_stop[n] = s
                if k < len(ls) - 1:
                    rows += [s]; cols += [n]; w += [wait]                 # wsiadanie
                    rows += [n]; cols += [ls[k + 1]]; w += [p["run"][k] * rf]  # przejazd
                    self.seg.append((pi, k, n, ls[k + 1]))
                if k > 0:
                    rows += [n]; cols += [s]; w += [1.0]                  # wysiadanie
        self.N = node
        xy = gpd.GeoSeries(gpd.points_from_xy(stops["lon"], stops["lat"]), crs=4326).to_crs(CRS)
        tree = STRtree(xy.values)
        a, b = tree.query(xy.values, predicate="dwithin", distance=TRANSFER_M)
        m = a != b
        d = np.hypot(xy.x.values[a[m]] - xy.x.values[b[m]], xy.y.values[a[m]] - xy.y.values[b[m]])
        rows += list(a[m]); cols += list(b[m]); w += list(d / WALK + 1)   # przesiadka pieszo
        hc = hexes.to_crs(CRS).centroid
        ha, hs = tree.query(hc.values, predicate="dwithin", distance=ACCESS_M)
        # klastry bez przystanku w ACCESS_M: dojście do najbliższego przystanku
        missing = sorted(set(range(H)) - set(ha))
        if missing:
            idx = tree.query_nearest(hc.values[missing])
            ha = np.concatenate([ha, np.array(missing)[idx[0]]]); hs = np.concatenate([hs, idx[1]])
        dd = np.hypot(hc.x.values[ha] - xy.x.values[hs], hc.y.values[ha] - xy.y.values[hs])
        rows += list(S + ha); cols += list(hs); w += list(dd / WALK + 1)   # dojście
        rows += list(hs); cols += list(S + ha); w += list(dd / WALK + 1)   # odejście
        self.S, self.H = S, H
        self.G = csr_matrix((w, (rows, cols)), shape=(self.N, self.N))
        self.seg_from = np.array([s[2] for s in self.seg]); self.seg_to = np.array([s[3] for s in self.seg])
        self.seg_of_to = np.full(self.N, -1); self.seg_of_to[self.seg_to] = np.arange(len(self.seg))
        self.is_ls = np.zeros(self.N, bool); self.is_ls[list(self.ls_stop)] = True
        print(f"   graf: {self.N} węzłów, {self.G.nnz} krawędzi, {len(self.seg)} odcinków linii, przystanków {S}")

    def skim(self):
        """Czasy podróży [min] klaster -> klaster i drzewa najkrótszych ścieżek."""
        origins = np.arange(self.S, self.S + self.H)
        dist, pred = dijkstra(self.G, indices=origins, return_predecessors=True)
        return dist[:, self.S:self.S + self.H] / 60, dist, pred


# ---------- 3. podróże ----------

def attractions(f: pd.DataFrame, pop) -> np.ndarray:
    jobs = f["praca_biura_uslugi_m2"] / 25 + f["praca_przemysl_m2"] / 100
    return (jobs + f["infra_szkola"] * 300 + f["infra_handel"] * 200 + f["infra_szpital"] * 500
            + f["infra_kultura"] * 50 + f["punkty_uslug"] * 2 + 0.25 * pop).values


def gravity(P, A, T):
    """Produkcja-ograniczony model grawitacyjny; β dobrane do średniego czasu podróży."""
    T = np.where(np.isfinite(T), T, np.inf)
    np.fill_diagonal(T, np.inf)          # podróże wewnątrz klastra — pieszo
    def run(beta):
        F = A[None, :] * np.exp(-beta * T)
        F /= F.sum(axis=1, keepdims=True) + 1e-12
        M = P[:, None] * F
        return M, (M * np.where(np.isfinite(T), T, 0)).sum() / M.sum()
    lo, hi = 0.0, 0.5
    for _ in range(40):
        beta = (lo + hi) / 2
        M, mean = run(beta)
        lo, hi = (beta, hi) if mean > SREDNI_CZAS_MIN else (lo, beta)
    print(f"   β = {beta:.4f}/min, średni czas podróży {mean:.1f} min, podróży w godz. szczytu {M.sum():,.0f}")
    return M, beta


# ---------- 4. przypisanie ----------

def assign(net: Network, M: np.ndarray, dist, pred):
    seg_flow = np.zeros(len(net.seg)); board = np.zeros(net.S)
    pred_is_stop = lambda p: (p >= 0) & (p < net.S)
    for o in range(net.H):
        d, pr = dist[o], pred[o]
        f = np.zeros(net.N)
        f[net.S:net.S + net.H] = M[o]
        order = np.argsort(-np.where(np.isfinite(d), d, -1))
        order = order[np.isfinite(d[order])]
        pl, fl = pr.tolist(), f
        for v in order.tolist():          # od liści do korzenia: przepływ = suma poddrzewa
            p = pl[v]
            if p >= 0:
                fl[p] += fl[v]
        nodes = np.nonzero(fl)[0]
        segs = net.seg_of_to[nodes]
        ok = (segs >= 0) & (pr[nodes] == net.seg_from[np.maximum(segs, 0)])
        np.add.at(seg_flow, segs[ok], fl[nodes[ok]])
        lsn = nodes[net.is_ls[nodes] & pred_is_stop(pr[nodes])]   # wsiadanie: przystanek -> przystanek-linii
        np.add.at(board, pr[lsn], fl[lsn])
    return seg_flow, board


def scenario(name, patterns, stops, hexes, feats, pop):
    print(f"== scenariusz {name}")
    net = Network(patterns, stops, hexes)
    T, dist, pred = net.skim()
    share = np.clip(UDZIAL_MAX - UDZIAL_SPADEK * feats["odl_centrum_km"].values, UDZIAL_MIN, UDZIAL_MAX)
    share *= UDZIAL_ZTM / np.average(share, weights=np.maximum(pop, 1e-9))   # średnia ważona ludnością = UDZIAL_ZTM
    P = pop * PODROZE * share * SZCZYT
    M, beta = gravity(P, attractions(feats, pop), T.copy())
    seg_flow, board = assign(net, M, dist, pred)
    return net, T, M, seg_flow, board


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hexes = gpd.read_file(DANE / "kontury_400m" / "warszawa_h3_400m_pelna.geojson")[["h3", "geometry"]]
    feats = pd.read_csv(DANE / "cechy_klastrow.csv").set_index("h3").reindex(hexes["h3"]).fillna(0)
    prog = pd.read_csv(DANE / "prognoza_ludnosci.csv").set_index("h3").reindex(hexes["h3"])
    stops = pd.read_csv(GTFS / "stops.txt", dtype={"stop_id": str}, usecols=["stop_id", "stop_name", "stop_code", "stop_lat", "stop_lon"])
    stops = stops.rename(columns={"stop_lat": "lat", "stop_lon": "lon"})

    print("== linie w szczycie")
    base = peak_patterns()
    new, extra = new_metro_patterns({})
    stops36 = pd.concat([stops, pd.DataFrame([{"stop_id": k, "stop_name": v[2], "stop_code": "", "lat": v[1], "lon": v[0]} for k, v in extra.items()])])

    n25, T25, M25, f25, b25 = scenario("2025", base, stops, hexes, feats, prog["pop_2025"].values.astype(float))
    n36, T36, M36, f36, b36 = scenario("2036", base + new, stops36, hexes, feats, prog["pop_2036"].values.astype(float))

    # ---- odcinki: 2036 zawiera wszystkie odcinki 2025 (+ nowe metro) w tej samej kolejności na początku
    st36 = n36.stops.set_index(np.arange(len(n36.stops)))
    rows = []
    flow25 = dict(zip([(n25.patterns[pi]["ref"], n25.patterns[pi]["dir"], k) for pi, k, _, _ in n25.seg], f25))
    for (pi, k, a, b), load36 in zip(n36.seg, f36):
        p = n36.patterns[pi]
        sa, sb = st36.loc[n36.ls_stop[a]], st36.loc[n36.ls_stop[b]]
        cap_h = p["per_h"] * p["cap"]
        load25 = flow25.get((p["ref"], p["dir"], k), np.nan)
        rows.append({"linia": p["ref"], "typ": p["typ"], "kierunek": p["dir"], "k": k,
                     "z": sa["stop_name"], "do": sb["stop_name"], "lon_z": sa["lon"], "lat_z": sa["lat"], "lon_do": sb["lon"], "lat_do": sb["lat"],
                     "kursy_h": round(p["per_h"], 1), "pojemnosc_h": round(cap_h), "pasazerow_h_2025": round(load25) if load25 == load25 else None,
                     "pasazerow_h_2036": round(load36)})
    seg = pd.DataFrame(rows)
    seg["wykorzystanie_2025"] = (seg["pasazerow_h_2025"] / seg["pojemnosc_h"]).round(3)
    seg["wykorzystanie_2036"] = (seg["pasazerow_h_2036"] / seg["pojemnosc_h"]).round(3)
    seg.to_csv(OUT_DIR / "odcinki_obciazenie.csv", index=False, encoding="utf-8")

    xy = gpd.GeoSeries(gpd.points_from_xy(seg["lon_z"], seg["lat_z"]), crs=4326).to_crs(CRS)
    xy2 = gpd.GeoSeries(gpd.points_from_xy(seg["lon_do"], seg["lat_do"]), crs=4326).to_crs(CRS)
    seg["km"] = xy.distance(xy2, align=False).values / 1000
    lin = seg.groupby(["linia", "typ"]).apply(lambda g: pd.Series({
        "max_wykorzystanie_2025": g["wykorzystanie_2025"].max(), "max_wykorzystanie_2036": g["wykorzystanie_2036"].max(),
        "paskm_h_2025": (g["pasazerow_h_2025"].fillna(0) * g["km"]).sum(), "paskm_h_2036": (g["pasazerow_h_2036"] * g["km"]).sum(),
        "najgorszy_odcinek_2036": f"{g.loc[g['wykorzystanie_2036'].idxmax(), 'z']} → {g.loc[g['wykorzystanie_2036'].idxmax(), 'do']}",
    }), include_groups=False).round(3).reset_index()
    lin.sort_values("max_wykorzystanie_2036", ascending=False).to_csv(OUT_DIR / "linie_obciazenie.csv", index=False, encoding="utf-8")

    # ---- klastry
    c = hexes.to_crs(CRS).centroid
    ci = int(np.argmin(np.hypot(c.x - gpd.GeoSeries(gpd.points_from_xy([CENTRUM[0]], [CENTRUM[1]]), crs=4326).to_crs(CRS).x[0],
                                c.y - gpd.GeoSeries(gpd.points_from_xy([CENTRUM[0]], [CENTRUM[1]]), crs=4326).to_crs(CRS).y[0])))
    jobs = attractions(feats, feats["population"])
    acc = lambda T: (np.where(T <= 45, 1, 0) * jobs[None, :]).sum(axis=1)
    kl = pd.DataFrame({"h3": hexes["h3"], "podroze_wyjazd_h_2025": M25.sum(1).round(0), "podroze_przyjazd_h_2025": M25.sum(0).round(0),
                       "podroze_wyjazd_h_2036": M36.sum(1).round(0), "podroze_przyjazd_h_2036": M36.sum(0).round(0),
                       "czas_do_centrum_min_2025": T25[:, ci].round(1), "czas_do_centrum_min_2036": T36[:, ci].round(1),
                       "cele_w_45min_2025": acc(T25).round(0), "cele_w_45min_2036": acc(T36).round(0)})
    kl.to_csv(OUT_DIR / "klastry_podroze.csv", index=False, encoding="utf-8")
    pd.DataFrame({"stop_id": n36.stops["stop_id"], "nazwa": n36.stops["stop_name"], "nr": n36.stops["stop_code"],
                  "wsiadajacy_h_2036": b36.round(0)}).merge(
        pd.DataFrame({"stop_id": n25.stops["stop_id"], "wsiadajacy_h_2025": b25.round(0)}), on="stop_id", how="left"
    ).to_csv(OUT_DIR / "przystanki_wsiadania.csv", index=False, encoding="utf-8")

    # ---- kompaktowo dla aplikacji (odcinki z obciążeniem)
    OUT_JSON.write_text(json.dumps({
        "opis": "Poranny szczyt (godzina), pasażerowie/h na odcinkach linii; model grawitacyjny + przypisanie najszybszą trasą",
        "fields": ["linia", "typ", "kier", "lon_z", "lat_z", "lon_do", "lat_do", "kursy_h", "poj_h", "p25", "p36"],
        "odcinki": [[r.linia, r.typ, r.kierunek, round(r.lon_z, 5), round(r.lat_z, 5), round(r.lon_do, 5), round(r.lat_do, 5),
                     r.kursy_h, r.pojemnosc_h, None if pd.isna(r.pasazerow_h_2025) else int(r.pasazerow_h_2025), int(r.pasazerow_h_2036)]
                    for r in seg.itertuples()],
    }, ensure_ascii=False, separators=(",", ":")))

    print("== wyniki")
    print(f"   wsiadających w godz. szczytu: 2025 {b25.sum():,.0f}, 2036 {b36.sum():,.0f}")
    over = lambda col: (seg[col] > 1).sum()
    print(f"   odcinki przepełnione (>100% pojemności): 2025 {over('wykorzystanie_2025')}, 2036 {over('wykorzystanie_2036')}")
    print("   linie najbardziej obciążone w 2036:")
    print(lin.sort_values("max_wykorzystanie_2036", ascending=False).head(12)[["linia", "typ", "max_wykorzystanie_2025", "max_wykorzystanie_2036", "najgorszy_odcinek_2036"]].to_string(index=False))
    print(f"   czas do centrum (mediana klastrów): 2025 {np.nanmedian(np.where(np.isfinite(T25[:, ci]), T25[:, ci], np.nan)):.0f} min, "
          f"2036 {np.nanmedian(np.where(np.isfinite(T36[:, ci]), T36[:, ci], np.nan)):.0f} min")
    print(f"== zapisano {OUT_DIR.relative_to(ROOT)}/ i {OUT_JSON.relative_to(ROOT)} ({OUT_JSON.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
