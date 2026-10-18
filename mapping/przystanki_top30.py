"""Gdzie dokładnie w klastrze postawić przystanek — 30 klastrów priorytetowych z modelu ML (dane/wskaznik_wykluczenia/output_run3/lista_heksagonow_top30.csv).

1. Popyt w punktach (nie w całym heksagonie):
   - mieszkańcy 2025: ludność klastra (prognoza_ludnosci.csv) rozłożona na budynki mieszkalne z OSM
     proporcjonalnie do powierzchni użytkowej (obrys × kondygnacje); building=yes liczy się z wagą 0,5,
   - nowi mieszkańcy do 2036: przyrost klastra (dP_2036) rozłożony na pozwolenia GUNB od 2019 r.
     proporcjonalnie do liczby mieszkań (bez pozwoleń — na budynki, jak wyżej),
   - miejsca pracy: biura/usługi z OSM, 1 etat na 30 m² powierzchni,
   - cele podróży z warszawa_infrastruktura (szkoły, szpitale, handel…) jako równoważnik osób.
   Popyt w zasięgu istniejącego przystanku ZTM (≤ 300 m) liczy się tylko w 30% — i tak ma już dojście.
2. Kandydaci: punkty co 25 m wzdłuż ulic przejezdnych dla autobusu i dopuszczonych pod przystanek
   (graf symulatora, stop_ok), ≥ 30 m od skrzyżowania, wewnątrz klastra — tylko na głównej sieci drogowej
   bez ślepych odnóg (drogi_przejezdne: 2-rdzeń grafu ulic + największa spójna składowa); na drodze,
   która się kończy, autobus musiałby zawracać. Każdy kandydat wie, które
   linie ZTM jadą już tą ulicą (≤ 15 m od trasy).
3. Ocena kandydata = popyt w zasięgu dojścia: 300 m w linii prostej ≈ 400 m pieszo (współczynnik krętości 1,33),
   waga 1 do 150 m, potem liniowo do 0,4 na 300 m; premia za przesiadkę na kolej/metro (≤ 250 m)
   i Veturilo (≤ 150 m).
4. Wybór: zachłanne pokrycie — najlepszy punkt, potem kolejne ≥ 350 m dalej, jeśli dokładają ≥ 40%
   tego co pierwszy (popyt już obsłużony przez wybrany przystanek się nie liczy). Maks. 3 na klaster.
   Osobno: najlepszy punkt na każdej linii ZTM przejeżdżającej przez klaster (dołożenie przystanku do linii).

Wynik:
  dane/przystanki_top30.csv          — proponowane przystanki (1–3 na klaster)
  dane/przystanki_top30_na_liniach.csv — najlepszy punkt na każdej istniejącej linii w klastrze
  data/cache/budynki_mieszkalne.parquet — budynki z OSM w Warszawie (cache, pierwszy przebieg kilka minut)

Uruchomienie:  .venv/bin/python mapping/przystanki_top30.py
"""

import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import h3
import networkx as nx
import numpy as np
import osmium
import pandas as pd
import shapely
from shapely import wkb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cechy_klastrow import PBF, WORK_BUILDINGS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DANE, DATA = ROOT / "dane", ROOT / "data"
ML_DIR = DANE / "wskaznik_wykluczenia" / "output_run3"   # wyniki modelu wskaźnika wykluczenia (lista top 30)
CRS = "EPSG:2180"
CACHE = DATA / "cache" / "budynki_mieszkalne.parquet"

RESIDENTIAL = {"apartments": 4, "residential": 3, "house": 2, "detached": 2, "semidetached_house": 2,
               "terrace": 2, "dormitory": 4, "bungalow": 1}
YES_WEIGHT, YES_AREA = 0.5, (40, 3000)     # building=yes: zwykle dom lub blok bez tagu — połowa wagi
PERMITS_FROM = 2019
M2_PER_JOB = 30
INFRA_EQ = {"szkola": 300, "szpital": 400, "centrum_handlowe": 400, "urzad_dzielnicy": 200, "targowisko": 150,
            "park": 80, "kino": 80, "teatr": 80, "muzeum": 50, "miejsce_kultu": 50}
COVERED_WEIGHT, COVER_R = 0.3, 300
R_FULL, R_MAX, W_EDGE = 150, 300, 0.4      # zasięg dojścia (m, w linii prostej)
SAMPLE_M, FROM_JUNCTION_M = 25, 30
ON_ROUTE_M = 15
BONUS_RAIL, BONUS_RAIL_R = 500, 250
BONUS_BIKE, BONUS_BIKE_R = 100, 150
MIN_SPACING, MIN_GAIN, MAX_STOPS = 350, 0.4, 3
# sieć do wykrywania ślepych odnóg: ulice, po których może jechać autobus, + trunk/busway jako łączniki
# (bez nich część dzielnic połączonych tylko drogą krajową wypadłaby z głównej spójnej sieci)
THROUGH_HIGHWAYS = {"primary", "secondary", "tertiary", "unclassified", "residential", "primary_link", "secondary_link",
                    "tertiary_link", "trunk", "trunk_link", "busway"}
BORDER_M = 150   # węzeł grafu tak blisko granicy miasta = droga jedzie dalej poza Warszawę


def levels(tags, default: float) -> float:
    m = re.match(r"\d+(\.\d+)?", tags.get("building:levels") or "")
    return float(m.group()) if m else default


def osm_buildings(bbox) -> gpd.GeoDataFrame:
    """Budynki mieszkalne i budynki pracy (punkt + powierzchnia użytkowa m²) w prostokącie."""
    if CACHE.exists():
        return gpd.read_parquet(CACHE)
    minx, miny, maxx, maxy = bbox
    fab = osmium.geom.WKBFactory()
    rows = []
    for o in osmium.FileProcessor(str(PBF)).with_areas().with_filter(osmium.filter.KeyFilter("building")):
        if not o.is_area():
            continue
        b = o.tags.get("building")
        kind = "dom" if b in RESIDENTIAL else "praca" if b in WORK_BUILDINGS else "yes" if b == "yes" else None
        if not kind:
            continue
        try:
            g = wkb.loads(fab.create_multipolygon(o), hex=True)
        except Exception:  # noqa: BLE001 — uszkodzona geometria w OSM
            continue
        c = g.representative_point()
        if minx <= c.x <= maxx and miny <= c.y <= maxy:
            lv = levels(o.tags, RESIDENTIAL.get(b) or WORK_BUILDINGS.get(b) or 2)
            rows.append((c.x, c.y, kind, g, lv))
    b = gpd.GeoDataFrame(rows, columns=["lon", "lat", "kind", "geometry", "lv"], crs=4326)
    b["pow_zabudowy"] = b.to_crs(CRS).area
    b["m2"] = b["pow_zabudowy"] * b["lv"]
    b = gpd.GeoDataFrame(b.drop(columns="geometry"), geometry=gpd.points_from_xy(b["lon"], b["lat"]), crs=4326)
    b.to_parquet(CACHE)
    return b


def demand_points(hexes: gpd.GeoDataFrame, area_hex: set) -> gpd.GeoDataFrame:
    """Punkty popytu (EPSG:2180) w klastrach area_hex: mieszkańcy, nowi mieszkańcy, praca, cele."""
    b = osm_buildings(tuple(hexes.total_bounds)).copy()   # cała Warszawa — cache przyda się też dla innych klastrów
    b["h3"] = [h3.latlng_to_cell(y, x, 8) for x, y in zip(b["lon"], b["lat"])]
    b = b[b["h3"].isin(area_hex)]
    prog = pd.read_csv(DANE / "prognoza_ludnosci.csv", encoding="utf-8-sig").set_index("h3")

    # mieszkańcy 2025 na budynki mieszkalne
    res = b[(b["kind"] == "dom") | ((b["kind"] == "yes") & b["pow_zabudowy"].between(*YES_AREA))].copy()
    res["w"] = res["m2"] * np.where(res["kind"] == "yes", YES_WEIGHT, 1.0)
    res["share"] = res["w"] / res.groupby("h3")["w"].transform("sum")
    res["osoby"] = res["share"] * res["h3"].map(prog["pop_2025"]).fillna(0)
    parts = [pd.DataFrame({"lon": res["lon"], "lat": res["lat"], "h3": res["h3"], "rodzaj": "mieszkancy", "osoby": res["osoby"]})]
    # klastry z ludnością, ale bez budynków w OSM — środek klastra
    miss = [h for h in area_hex if h not in set(res["h3"]) and prog["pop_2025"].get(h, 0) > 0]
    parts.append(pd.DataFrame([{"lon": h3.cell_to_latlng(h)[1], "lat": h3.cell_to_latlng(h)[0], "h3": h,
                                "rodzaj": "mieszkancy", "osoby": prog["pop_2025"][h]} for h in miss]))

    # nowi mieszkańcy 2036 na pozwolenia GUNB
    dp = prog["dP_2036"].clip(lower=0)
    per = gpd.read_parquet(DATA / "gunb_permits.parquet")
    per = per[(per["rok"] >= PERMITS_FROM) & per.geometry.notna()].to_crs(4326)
    per = pd.DataFrame({"lon": per.geometry.x, "lat": per.geometry.y, "w": per["n_flats"].astype(float)})
    per["h3"] = [h3.latlng_to_cell(y, x, 8) for x, y in zip(per["lon"], per["lat"])]
    per = per[per["h3"].isin(area_hex)]
    per["osoby"] = per["w"] / per.groupby("h3")["w"].transform("sum") * per["h3"].map(dp)
    rest = res[~res["h3"].isin(set(per["h3"]))]
    rest_osoby = rest["share"] * rest["h3"].map(dp).fillna(0)
    parts.append(pd.DataFrame({"lon": per["lon"], "lat": per["lat"], "h3": per["h3"], "rodzaj": "nowi_2036", "osoby": per["osoby"]}))
    parts.append(pd.DataFrame({"lon": rest["lon"], "lat": rest["lat"], "h3": rest["h3"], "rodzaj": "nowi_2036", "osoby": rest_osoby}))

    # praca i cele podróży
    work = b[b["kind"] == "praca"]
    parts.append(pd.DataFrame({"lon": work["lon"], "lat": work["lat"], "h3": work["h3"], "rodzaj": "praca", "osoby": work["m2"] / M2_PER_JOB}))
    inf = gpd.read_file(DANE / "warszawa_infrastruktura.geojson")
    inf = inf[inf["h3"].isin(area_hex)]
    parts.append(pd.DataFrame({"lon": inf["lon"], "lat": inf["lat"], "h3": inf["h3"], "rodzaj": "cele",
                               "osoby": inf["kategoria"].map(INFRA_EQ).fillna(50)}))

    d = pd.concat(parts, ignore_index=True)
    d = d[d["osoby"] > 0]
    return gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d["lon"], d["lat"]), crs=4326).to_crs(CRS)


def drogi_przejezdne(app: dict) -> set:
    """edge_id krawędzi głównej, spójnej sieci drogowej bez ślepych odnóg.

    Sieć: krawędzie grafu symulatora z ulic klas THROUGH_HIGHWAYS albo z bus/psv=yes|designated (tagi OSM
    z data/roads_edges.parquet); bez access=no|private (chyba że wolno autobusom) i bez bus=no. Graf nieskierowany
    bez dubli i pętli → 2-rdzeń (znikają wszystkie drogi, które się kończą, razem z całymi ślepymi „drzewkami”
    osiedlowymi) → największa spójna składowa (bez odciętych kawałków sieci). Graf symulatora jest przycięty do
    granicy Warszawy, więc węzły ≤ BORDER_M od granicy łączymy z wirtualnym węzłem „poza miastem” — inaczej drogi
    wylotowe (np. Radiowa, Chełmżyńska) kończyłyby się sztucznie na granicy."""
    g = app["graph"]
    f = {k: i for i, k in enumerate(g["fields"])}
    ed = pd.DataFrame({"u": [e[f["u"]] for e in g["edges"]], "v": [e[f["v"]] for e in g["edges"]],
                       "len_m": [e[f["len_m"]] for e in g["edges"]], "edge_id": [e[f["edge_id"]] for e in g["edges"]]})
    osm = pd.read_parquet(DATA / "roads_edges.parquet", columns=["edge_id", "highway", "access", "bus", "psv"])
    ed = ed.merge(osm, on="edge_id", how="left")
    tok = lambda v: set() if pd.isna(v) else set(str(v).split(";"))  # noqa: E731
    psv_ok = ed["bus"].astype(str).isin(["yes", "designated"]) | ed["psv"].astype(str).isin(["yes", "designated"])
    closed = ed["access"].map(lambda v: bool(tok(v) & {"no", "private"})) & ~psv_ok
    siec = (ed["highway"].map(lambda v: bool(tok(v) & THROUGH_HIGHWAYS)) | psv_ok) & ~closed & ~ed["bus"].astype(str).eq("no")
    und = nx.Graph()
    und.add_edges_from(zip(ed.loc[siec, "u"], ed.loc[siec, "v"]))
    und.remove_edges_from(nx.selfloop_edges(und))
    nodes = np.array(list(und.nodes))
    pts = gpd.GeoSeries(gpd.points_from_xy(*np.array(g["nodes"])[nodes].T), crs=4326).to_crs(CRS)
    for ring in app.get("boundary") or []:
        border = gpd.GeoSeries([shapely.LineString(ring)], crs=4326).to_crs(CRS).iloc[0]
        und.add_edges_from((n, "poza_miastem") for n in nodes[pts.distance(border).values <= BORDER_M])
    core = nx.k_core(und, 2)
    main = max(nx.connected_components(core), key=len)
    ok = {frozenset(e) for e in core.subgraph(main).edges() if "poza_miastem" not in e}
    przejezdna = siec & pd.Series([frozenset((u, v)) in ok for u, v in zip(ed["u"], ed["v"])], index=ed.index)
    km = lambda m: f"{ed.loc[m, 'len_m'].sum() / 2000:.0f} km"  # noqa: E731 — oba kierunki jezdni → /2
    print(f"   drogi przejezdne: {km(przejezdna)} z {km(siec)} sieci ulic dla autobusu; "
          f"odrzucone ślepe odnogi i odcięte kawałki: {km(siec & ~przejezdna)}")
    return set(ed.loc[przejezdna, "edge_id"])


def candidates(app: dict, polys: gpd.GeoDataFrame, routes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Punkty co 25 m na ulicach dopuszczonych pod przystanek i przejezdnych (bez ślepych odnóg), w klastrach top 30."""
    g = app["graph"]
    f = {k: i for i, k in enumerate(g["fields"])}
    names = g["names"]
    ok = drogi_przejezdne(app)
    rows = [(e[f["edge_id"]], names[e[f["name"]]] if e[f["name"]] is not None and e[f["name"]] >= 0 else "",
             shapely.LineString(e[f["coords"]])) for e in g["edges"] if e[f["stop_ok"]] and e[f["edge_id"]] in ok]
    e = gpd.GeoDataFrame(rows, columns=["edge_id", "ulica", "geometry"], crs=4326).to_crs(CRS)
    e = e[e.intersects(polys.union_all())]
    pts = []
    for r in e.itertuples():
        L = r.geometry.length
        for s in np.arange(FROM_JUNCTION_M, L - FROM_JUNCTION_M + 1e-6, SAMPLE_M):
            pts.append((r.edge_id, r.ulica, r.geometry.interpolate(s)))
    c = gpd.GeoDataFrame(pts, columns=["edge_id", "ulica", "geometry"], crs=CRS)
    c = gpd.sjoin(c, polys[["h3", "geometry"]], predicate="within").drop(columns="index_right")
    # dwa kierunki jezdni dają te same punkty — jeden wystarczy
    c["key"] = (c.geometry.x // 5).astype(int).astype(str) + "_" + (c.geometry.y // 5).astype(int).astype(str)
    c = c.drop_duplicates("key").drop(columns="key").reset_index(drop=True)
    # linie ZTM jadące tą ulicą
    hit = gpd.sjoin(c[["geometry"]], routes[["route_short_name", "geometry"]], predicate="dwithin", distance=ON_ROUTE_M)
    lines = hit.groupby(level=0)["route_short_name"].agg(lambda s: sorted(set(s), key=lambda x: (len(x), x)))
    c["linie_ztm"] = [lines.get(i, []) for i in c.index]
    return c


def weight(d: np.ndarray) -> np.ndarray:
    return np.where(d <= R_FULL, 1.0, 1 - (1 - W_EDGE) * (d - R_FULL) / (R_MAX - R_FULL)) * (d <= R_MAX)


def main() -> None:
    top = pd.read_csv(ML_DIR / "lista_heksagonow_top30.csv")
    hexes = gpd.read_file(DANE / "kontury_400m" / "warszawa_h3_400m_pelna.geojson")[["h3", "dzielnica", "geometry"]]
    polys = hexes[hexes["h3"].isin(top["h3"])].to_crs(CRS)
    # popyt także z sąsiednich klastrów — zasięg dojścia wychodzi poza heksagon
    area = set().union(*(h3.grid_disk(h, 1) for h in top["h3"])) & set(hexes["h3"])
    app = json.loads((ROOT / "frontend" / "symulator_dane.json").read_text(encoding="utf-8"))
    routes = gpd.read_file(DANE / "warszawa_linie_trasy.geojson")
    routes = routes[routes["typ"].isin(["autobus", "tramwaj"])].to_crs(CRS)   # przystanek dokładamy do autobusu/tramwaju

    print("== popyt (budynki OSM, GUNB, infrastruktura)")
    dem = demand_points(hexes, area)
    ztm = gpd.GeoSeries(gpd.points_from_xy([s[0] for s in app["stops"]["ztm"]], [s[1] for s in app["stops"]["ztm"]]), crs=4326).to_crs(CRS)
    rail = gpd.GeoSeries(gpd.points_from_xy([s[0] for s in app["stops"]["rail"]], [s[1] for s in app["stops"]["rail"]]), crs=4326).to_crs(CRS)
    bike = pd.read_csv(DANE / "veturilo_stacje.csv")
    bike = gpd.GeoSeries(gpd.points_from_xy(bike["lon"], bike["lat"]), crs=4326).to_crs(CRS)
    covered = np.zeros(len(dem), bool)
    covered[np.unique(shapely.STRtree(ztm.values).query(dem.geometry.values, predicate="dwithin", distance=COVER_R)[0])] = True
    dem["w"] = dem["osoby"] * np.where(covered, COVERED_WEIGHT, 1.0)

    print("== kandydaci")
    c = candidates(app, polys, routes)
    tree = shapely.STRtree(dem.geometry.values)
    ci, di = tree.query(c.geometry.values, predicate="dwithin", distance=R_MAX)
    dist = shapely.distance(c.geometry.values[ci], dem.geometry.values[di])
    wt = weight(dist)
    for col, kind in [("mieszkancy_2025", "mieszkancy"), ("nowi_2036", "nowi_2036"), ("miejsca_pracy", "praca"), ("cele", "cele")]:
        m = (dem["rodzaj"].values[di] == kind)
        c[col] = np.bincount(ci, weights=wt * dem["osoby"].values[di] * m, minlength=len(c)).round()
    c["popyt"] = np.bincount(ci, weights=wt * dem["w"].values[di], minlength=len(c))
    near = lambda pts, r: np.isin(np.arange(len(c)), shapely.STRtree(pts.values).query(c.geometry.values, predicate="dwithin", distance=r)[0])  # noqa: E731
    c["przesiadka_kolej"] = near(rail, BONUS_RAIL_R)
    c["przy_veturilo"] = near(bike, BONUS_BIKE_R)
    c["ocena"] = c["popyt"] + BONUS_RAIL * c["przesiadka_kolej"] + BONUS_BIKE * c["przy_veturilo"]
    c["odl_do_przystanku_ztm_m"] = shapely.distance(c.geometry.values, ztm.values[shapely.STRtree(ztm.values).nearest(c.geometry.values)]).round()

    print("== wybór punktów")
    out, per_line = [], []
    for t in top.itertuples():
        cc = c[c["h3"] == t.h3]
        if cc.empty:
            out.append({"h3": t.h3, "rank": t.rank, "dzielnica": t.dzielnica, "nr": 0, "uwaga": "brak ulicy przejezdnej dla autobusu w klastrze"})
            continue
        idx = cc.index.values
        pairs = np.isin(ci, idx)
        served = np.zeros(len(dem))   # najlepsza dotychczasowa waga dojścia dla każdego punktu popytu
        chosen, first_gain = [], None
        for k in range(MAX_STOPS):
            gain = np.bincount(ci[pairs], weights=np.clip(wt[pairs] - served[di[pairs]], 0, None) * dem["w"].values[di[pairs]], minlength=len(c))[idx]
            gain = gain + (BONUS_RAIL * cc["przesiadka_kolej"] + BONUS_BIKE * cc["przy_veturilo"]).values * (k == 0)
            ok = np.ones(len(idx), bool)
            for j in chosen:
                ok &= cc.geometry.distance(c.geometry[j]).values >= MIN_SPACING
            if not ok.any():
                break
            best = int(np.argmax(np.where(ok, gain, -1)))
            if first_gain is None:
                first_gain = gain[best]
            elif gain[best] < MIN_GAIN * first_gain:
                break
            j = idx[best]
            chosen.append(j)
            m = pairs & (ci == j)
            served[di[m]] = np.maximum(served[di[m]], wt[m])
            r = c.loc[j]
            p = gpd.GeoSeries([r.geometry], crs=CRS).to_crs(4326).iloc[0]
            out.append({"h3": t.h3, "rank": t.rank, "dzielnica": t.dzielnica, "nr": k + 1,
                        "lon": round(p.x, 6), "lat": round(p.y, 6), "ulica": r.ulica, "ocena": round(r.ocena),
                        "nowo_obsluzeni": round(gain[best]), "mieszkancy_2025": r.mieszkancy_2025, "nowi_2036": r.nowi_2036,
                        "miejsca_pracy": r.miejsca_pracy, "cele": r.cele, "odl_do_przystanku_ztm_m": r.odl_do_przystanku_ztm_m,
                        "linie_ztm_na_ulicy": " ".join(r.linie_ztm), "przesiadka_kolej": r.przesiadka_kolej,
                        "przy_veturilo": r.przy_veturilo, "uwaga": ""})
        # najlepszy punkt na każdej linii ZTM w klastrze
        ex = cc.explode("linie_ztm").dropna(subset=["linie_ztm"])
        for line, g in ex.groupby("linie_ztm"):
            r = g.loc[g["ocena"].idxmax()]
            p = gpd.GeoSeries([r.geometry], crs=CRS).to_crs(4326).iloc[0]
            per_line.append({"h3": t.h3, "rank": t.rank, "dzielnica": t.dzielnica, "linia": line,
                             "lon": round(p.x, 6), "lat": round(p.y, 6), "ulica": r.ulica, "ocena": round(r.ocena),
                             "odl_do_przystanku_ztm_m": r.odl_do_przystanku_ztm_m})

    o = pd.DataFrame(out)
    o.to_csv(DANE / "przystanki_top30.csv", index=False, encoding="utf-8")
    pl = pd.DataFrame(per_line).sort_values(["rank", "ocena"], ascending=[True, False])
    pl.to_csv(DANE / "przystanki_top30_na_liniach.csv", index=False, encoding="utf-8")
    print(f"== {len(c)} kandydatów, {int((o['nr'] > 0).sum())} przystanków w {o.loc[o['nr'] > 0, 'h3'].nunique()} klastrach; "
          f"klastry bez ulicy dla autobusu: {int((o['nr'] == 0).sum())}; punktów na liniach: {len(pl)}")
    show = ["rank", "dzielnica", "nr", "ulica", "ocena", "nowo_obsluzeni", "mieszkancy_2025", "nowi_2036", "odl_do_przystanku_ztm_m", "linie_ztm_na_ulicy"]
    print(o[show].to_string(index=False))


if __name__ == "__main__":
    main()
