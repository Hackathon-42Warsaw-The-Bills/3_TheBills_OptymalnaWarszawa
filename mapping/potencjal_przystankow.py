"""Potencjał przystanku wzdłuż ulic autobusowych: gdzie nowej linii opłaca się postawić przystanek.

Dla przycisku „Dodaj przystanki automatycznie” w symulatorze (index.html): po istniejących przystankach ZTM
i Przystankach AI / top 30 leżących na narysowanej trasie przycisk dokłada miejsca z tego pliku, które są na trasie —
od największego zysku, ≥ 350 m od innych przystanków. Tam, gdzie wokół nic nie ma, przystanku nie będzie.

Model ten sam co w mapping/propozycje_przystankow.py (popyt, dzisiejsza dostępność, dojście pieszo po sieci chodników
z OSM z barierami — tory, Wisła, trasy ekspresowe):
  kandydaci — punkty co 100 m na ulicach przejezdnych dla autobusu, na których wolno postawić przystanek (stop_ok: bez
              wiaduktów, tuneli, łącznic i ekspresówek), ≥ 30 m od skrzyżowań; obie jezdnie / kierunki — jeden punkt.
  zysk      — 0,5·Σ w·max(0, a − A_dziś) + 0,5·Σ w·max(0, a − A_2036), gdzie a = f(dojście) × g(6 kursów/h — domyślnie
              „co 10 min” w symulatorze), w = mieszkańcy 2025 + 0,5·miejsca pracy + cele podróży (równoważnik osób),
              w 2036 r. także nowi mieszkańcy z pozwoleń GUNB. Kto ma już w zasięgu częsty przystanek, nie dodaje prawie
              nic — punkt tuż obok istniejącego przystanku ma zysk bliski zera.
  warto     — zysk ≥ 100 os.-równ. (ten sam próg co dla Przystanków AI na istniejących trasach, etap A).

Wynik: dane/przystanki/potencjal_przystankow.csv — tylko punkty, w których warto (zysk ≥ 100), malejąco po zysku:
punkt, ulica, zysk, mieszkańcy w zasięgu (w tym bez przystanku w standardzie ZTM dziś i z rzadkimi kursami), mieszkańcy
dalej, którym skróci się dojście, nowi mieszkańcy do 2036, miejsca pracy, cele podróży, najbliższy dzisiejszy przystanek
i uzasadnienie (pokazywane w dymku nad przystankiem).

Uruchomienie (po propozycje_przystankow.py — korzysta z jego cache w data/cache/):
    .venv/bin/python mapping/potencjal_przystankow.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import propozycje_przystankow as pp  # noqa: E402

OUT = pp.DANE / "przystanki" / "potencjal_przystankow.csv"
STEP_M = 100           # odstęp kandydatów wzdłuż ulicy (przystanki i tak stoją ≥ 350 m od siebie)
FREQ = 6.0             # kursy/h nowej linii — domyślne „co 10 min” w symulatorze
MIN_GAIN = 100         # próg „warto” — jak etap A Przystanków AI
BATCH = 16             # źródeł na jedno wywołanie dijkstry


def candidates(ed, cl):
    """Punkty co STEP_M m na ulicach, gdzie wolno postawić przystanek; obie jezdnie / kierunki — jeden na kratkę 10 m."""
    pts = []
    for r in ed[ed["stop_ok"] == 1].itertuples():
        line = shapely.LineString(np.column_stack(pp.TO_PL.transform(*np.array(r.coords).T)))
        for t in np.arange(pp.FROM_JUNCTION_M, line.length - pp.FROM_JUNCTION_M + 1e-6, STEP_M):
            p = line.interpolate(t)
            pts.append((r.Index, p.x, p.y))
    c = pd.DataFrame(pts, columns=["e", "x", "y"]).join(ed[["ulica"]], on="e")
    key = (c["x"] // 10).astype(int).astype(str) + "_" + (c["y"] // 10).astype(int).astype(str)
    c = c[~key.duplicated()].reset_index(drop=True)
    c["lon"], c["lat"] = pp.TO_WGS.transform(c["x"].values, c["y"].values)
    c["h3"] = pp.h3_of(c["lon"].values, c["lat"].values)
    return c[c["h3"].isin(cl.index)].reset_index(drop=True)


def upper_bound(d, c):
    """Odrzuca kandydatów, którzy nie mogą dać MIN_GAIN: górne ograniczenie zysku z popytu do poprawy ≤ LIMIT m w linii prostej."""
    imp = np.flatnonzero((d["A_dzis"].values < 0.999) | (d["A_2036"].values < 0.999))
    pot_w = (0.5 * d["w_dzis"].values * (1 - d["A_dzis"].values) + 0.5 * d["w_2036"].values * (1 - d["A_2036"].values))[imp]
    tree, xy, pot = cKDTree(d[["x", "y"]].values[imp]), c[["x", "y"]].values, np.zeros(len(c))
    for k in range(0, len(c), 2000):
        for j, n in enumerate(tree.query_ball_point(xy[k:k + 2000], r=pp.LIMIT)):
            pot[k + j] = pot_w[n].sum() if n else 0.0
    return c[pot >= MIN_GAIN].reset_index(drop=True)


def cel_text(types):
    """„szpital, szkoła ×2, przychodnia” — od najważniejszego celu."""
    vc = pd.Series(types).value_counts()
    vc = vc[sorted(vc.index, key=lambda k: -pp.DEST_EQ.get(k, 15))]
    return ", ".join(f"{pp.DEST_PL.get(k, k)} ×{v}" if v > 1 else pp.DEST_PL.get(k, k) for k, v in vc.items())


def gains(G, tree, ids, d, c):
    """Zysk dostępności i to, co jest w zasięgu dojścia, dla każdego kandydata (dijkstra po sieci pieszej do LIMIT m)."""
    c = c.copy()
    c["node"], c["off"] = pp.snap(tree, ids, c[["x", "y"]].values)
    order = np.argsort(d["node"].values, kind="stable")
    dn_sorted = d["node"].values[order]
    off_d, R = d["off"].values, d["R"].values
    w0, w1, A0, A1 = d["w_dzis"].values, d["w_2036"].values, d["A_dzis"].values, d["A_2036"].values
    m, nw, jobs, std = d["mieszk_2025"].values, d["nowi_2036"].values, d["praca"].values, d["w_std_dzis"].values
    is_cel, cel_typ = (d["rodzaj"] == "cel").values, d["cel_typ"].values
    g, off_c = float(pp.g_freq(FREQ)), c["off"].values
    cols = {k: np.zeros(len(c)) for k in ("zysk", "mieszk", "nowo", "rzadkie", "dalej", "nowi", "praca")}
    cele = np.full(len(c), "", dtype=object)
    by_node = pd.Series(np.arange(len(c))).groupby(c["node"].values).apply(list)
    nodes, t0 = by_node.index.values, time.time()
    for k in range(0, len(nodes), BATCH):
        batch = nodes[k:k + BATCH]
        D = dijkstra(G, directed=True, indices=batch, limit=pp.LIMIT)
        for r, nd in enumerate(batch):
            reach = np.flatnonzero(np.isfinite(D[r]))
            lo, hi = np.searchsorted(dn_sorted, reach, "left"), np.searchsorted(dn_sorted, reach, "right")
            cnt = hi - lo
            if not cnt.sum():
                continue
            # punkty popytu przy osiągniętych węzłach (wektorowo: ciągi lo..hi−1 dla każdego węzła)
            di = order[np.repeat(lo - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt) + np.arange(cnt.sum())]
            base = np.repeat(D[r][reach], cnt) + off_d[di]
            for ci in by_node[nd]:
                dist = base + off_c[ci]
                a = pp.f_bus(dist, R[di]) * g
                z = 0.5 * (w0[di] * np.clip(a - A0[di], 0, None)).sum() + 0.5 * (w1[di] * np.clip(a - A1[di], 0, None)).sum()
                cols["zysk"][ci] = z
                if z < MIN_GAIN:
                    continue
                inR, up = dist <= R[di], a - A0[di]
                cols["mieszk"][ci], cols["nowo"][ci] = m[di][inR].sum(), m[di][inR & ~std[di]].sum()
                # skąd jeszcze zysk: przystanek w standardzie, ale kursy rzadziej niż co 10 min; dalej niż próg — krótsze dojście
                cols["rzadkie"][ci], cols["dalej"][ci] = m[di][inR & std[di] & (up > 0.1)].sum(), m[di][~inR & (up > 0.05)].sum()
                cols["nowi"][ci], cols["praca"][ci] = nw[di][inR].sum(), jobs[di][inR].sum()
                sel = inR & is_cel[di]
                if sel.any():
                    cele[ci] = cel_text(cel_typ[di][sel])
        if (k // BATCH) % 250 == 0:
            pp.log(f"   {min(k + BATCH, len(nodes))}/{len(nodes)} węzłów ({time.time() - t0:.0f} s)")
    return c.assign(**cols, cele=cele)


def justification(r):
    f = pp.fmt
    tx = [f"{f(r.nowo)} z {f(r.mieszk)} mieszkańców w zasięgu {r.prog} m pieszo nie ma dziś przystanku w standardzie ZTM"
          if r.nowo >= 1 else f"{f(r.mieszk)} mieszkańców w zasięgu {r.prog} m pieszo"]
    if r.rzadkie >= 50:
        tx.append(f"{f(r.rzadkie)} z nich ma przystanek, ale z kursami rzadziej niż co 10 min")
    if r.dalej >= 50:
        tx.append(f"ok. {f(r.dalej)} mieszkańców dalej (do {r.prog + pp.DECAY} m) skróci dojście")
    if r.nowi >= 20:
        tx.append(f"do 2036 r. dojdzie ok. {f(r.nowi)} nowych mieszkańców")
    if r.praca >= 100:
        tx.append(f"ok. {f(r.praca)} miejsc pracy")
    if r.cele:
        tx.append(f"cele podróży: {r.cele}")
    tx.append(f"najbliższy przystanek dziś: {r.slupek}, {f(r.odl)} m")
    s = "; ".join(tx)
    return s[0].upper() + s[1:] + "."


def main() -> None:
    t0 = time.time()
    app = json.loads((pp.ROOT / "frontend" / "symulator_dane.json").read_text(encoding="utf-8"))
    cl = pp.clusters()
    G, xy, ids, tree = pp.load_walk_graph()
    d = pp.demand(cl, app)
    s, rail, plan = pp.supply(app)
    d = pp.access(G, tree, ids, d, s, rail, plan)
    ed, _ = pp.bus_graph(app)
    c = candidates(ed, cl)
    n_all = len(c)
    c = upper_bound(d, c)
    pp.log(f"kandydaci: {n_all} punktów co {STEP_M} m na ulicach dla autobusu; z możliwym zyskiem ≥ {MIN_GAIN}: {len(c)}")
    c = gains(G, tree, ids, d, c)
    res = c[c["zysk"] >= MIN_GAIN].copy()
    k = cl.loc[res["h3"]]
    dist, j = cKDTree(s[["x", "y"]].values).query(res[["x", "y"]].values)
    res = res.assign(strefa=k["strefa"].values, prog=k["R"].values.astype(int), odl=dist, slupek=s["nazwa"].values[j])
    res["uzasadnienie"] = [justification(r) for r in res.itertuples()]
    out = pd.DataFrame({
        "lon": res["lon"].round(6), "lat": res["lat"].round(6), "ulica": res["ulica"], "h3": res["h3"],
        "strefa_ztm": res["strefa"], "prog_dojscia_m": res["prog"], "zysk_dostepnosci": res["zysk"].round().astype(int),
        "mieszkancy_2025_w_zasiegu": res["mieszk"].round().astype(int),
        "mieszkancy_2025_bez_przystanku_w_standardzie": res["nowo"].round().astype(int),
        "mieszkancy_2025_z_rzadkimi_kursami": res["rzadkie"].round().astype(int),
        "mieszkancy_2025_dalej_krotsze_dojscie": res["dalej"].round().astype(int),
        "nowi_mieszkancy_2036_w_zasiegu": res["nowi"].round().astype(int), "miejsca_pracy_w_zasiegu": res["praca"].round().astype(int),
        "cele_w_zasiegu": res["cele"], "najblizszy_przystanek": res["slupek"], "odl_najblizszy_przystanek_m": res["odl"].round().astype(int),
        "uzasadnienie": res["uzasadnienie"],
    }).sort_values("zysk_dostepnosci", ascending=False)
    out.to_csv(OUT, index=False, encoding="utf-8")
    q = out["zysk_dostepnosci"].quantile([0.5, 0.9]).round().astype(int).tolist()
    pp.log(f"== {OUT.relative_to(pp.ROOT)}: {len(out)} z {len(c)} sprawdzonych punktów warto (zysk ≥ {MIN_GAIN}; mediana {q[0]}, "
           f"90. percentyl {q[1]}, maks. {out['zysk_dostepnosci'].max()}); {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
