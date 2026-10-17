"""Z deficytu na heksagonach do konkretnych przystanków i linii dowozowych.

Wejście: data/deficit_h3.parquet (model/deficit.py), data/roads_edges.parquet,
data/roads_nodes.parquet, data/hubs.parquet.

Wynik:
  - data/recommended_stops.parquet  — przystanki (punkt po prawej stronie jezdni,
                                      edge_id krawędzi skierowanej, po której jedzie autobus)
  - data/recommended_lines.parquet  — linie: geometria w stronę węzła, długość, obsłużony deficyt
  - viz/recommendations.json        — to samo dla wizualizacji

Metoda:
  1. Filtr techniczny: autobus 12 m (18 t, 3,6 m wysokości, jezdnia ≥ 5,5 m tam, gdzie OSM
     zna szerokość). Bez dróg serwisowych i stref zamieszkania, chyba że mają bus/psv=yes.
     Przystanek tylko na ulicy (nie na wiadukcie, w tunelu, łącznicy ani ekspresówce).
  2. Kandydaci co CANDIDATE_STEP m na dopuszczonych odcinkach w obszarach deficytu.
  3. Maximal covering: zachłannie wybieramy kandydata, który pokrywa (WALK_M) najwięcej
     jeszcze niepokrytego deficytu, z minimalnym rozstawem MIN_SPACING.
  4. Każdy przystanek -> najbliższy w sieci węzeł przesiadkowy otwarty w danym roku.
     W grupie węzła budujemy linie: najdalszy przystanek + ścieżka do węzła, potem
     wstawianie kolejnych przystanków najtańszym objazdem (limity długości i objazdu).
  5. Trasa w grafie skierowanym (jednokierunkowe ulice), przystanek na krawędzi
     skierowanej faktycznie przejeżdżanej w danym kierunku, odsunięty na prawą stronę.

Uruchomienie:  .venv/bin/python model/routes.py [rok]   (domyślnie ostatni horyzont)
"""

import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, substring

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
VIZ_DIR = ROOT / "viz"
CRS = "EPSG:2180"

# Autobus przegubowy / standardowy
BUS_WEIGHT_T = 18
BUS_HEIGHT_M = 3.6
MIN_WIDTH_M = 5.5

WALK_M = 400
CANDIDATE_STEP = 200
MIN_SPACING = 350
MIN_STOP_GAIN = 120        # os. deficytu, poniżej których nie stawiamy kolejnego przystanku
FOCUS_MIN_DEFICIT = 40     # heksagon wchodzi do obszaru deficytu od tylu osób
MAX_LINE_M = 9000
MAX_DETOUR = 0.35          # względny wzrost długości linii przy dołożeniu przystanku
MIN_LINE_DEFICIT = 600     # linia ma sens, gdy obsłuży tylu ludzi z deficytem
HUB_SNAP_M = 350
# Kara (w metrach „kosztu”) za dowóz do tramwaju zamiast do metra/kolei.
HUB_PENALTY = {"rail": 0, "tram": 1200}
SIDE_OFFSET_M = 4

ALWAYS_OK = {"yes", "designated"}
NO_STOPS = {"motorway", "trunk", "motorway_link", "trunk_link", "primary_link",
            "secondary_link", "tertiary_link"}
BUS_HIGHWAYS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
                "residential", "busway", "motorway_link", "trunk_link", "primary_link",
                "secondary_link", "tertiary_link"}
# Kara za klasę drogi — wolimy prowadzić linie ulicami zbiorczymi, nie osiedlowymi.
CLASS_COST = {"trunk": 0.9, "primary": 0.95, "secondary": 1.0, "tertiary": 1.05, "busway": 0.8,
              "unclassified": 1.2, "residential": 1.35}


def num(value) -> float | None:
    m = re.search(r"\d+(?:[.,]\d+)?", str(value)) if value is not None else None
    return float(m.group().replace(",", ".")) if m else None


def bus_ok(row) -> bool:
    kinds = set(str(row["highway"]).split(";"))
    if str(row.get("bus")) in ALWAYS_OK or str(row.get("psv")).split(";")[0] in ALWAYS_OK or "busway" in kinds:
        return True
    if not kinds & BUS_HIGHWAYS:
        return False
    w, mw, mh = num(row.get("width")), num(row.get("maxweight")), num(row.get("maxheight"))
    if w is not None and w < MIN_WIDTH_M:
        return False
    if mw is not None and mw < BUS_WEIGHT_T:
        return False
    if mh is not None and str(row.get("maxheight")) not in ("default", "none") and mh < BUS_HEIGHT_M:
        return False
    if str(row.get("maxheight")) == "below_default":
        return False
    return True


def stop_ok(row) -> bool:
    kinds = set(str(row["highway"]).split(";"))
    flat = str(row.get("bridge")) in ("<NA>", "nan", "None") and str(row.get("tunnel")) in ("nan", "None", "<NA>", "building_passage")
    return row["bus_ok"] and not kinds & NO_STOPS and flat and row["length"] >= 40


def edge_cost(row) -> float:
    kinds = str(row["highway"]).split(";")
    return row["length"] * min(CLASS_COST.get(k.removesuffix("_link"), 1.3) for k in kinds)


def load_network() -> tuple[gpd.GeoDataFrame, nx.DiGraph, gpd.GeoDataFrame]:
    e = gpd.read_parquet(DATA_DIR / "roads_edges.parquet").to_crs(CRS)
    e["bus_ok"] = e.apply(bus_ok, axis=1)
    e["stop_ok"] = e.apply(stop_ok, axis=1)
    e["cost"] = e.apply(edge_cost, axis=1)
    ok = e[e["bus_ok"]].sort_values("cost").drop_duplicates(["u", "v"])
    g = nx.DiGraph()
    for r in ok.itertuples():
        g.add_edge(r.u, r.v, cost=r.cost, length=r.length, edge_id=r.edge_id)
    # Zostawiamy największy silnie spójny kawałek — autobus musi umieć wrócić.
    g = g.subgraph(max(nx.strongly_connected_components(g), key=len)).copy()
    nodes = gpd.read_parquet(DATA_DIR / "roads_nodes.parquet").to_crs(CRS).set_index("osmid")
    nodes = nodes[nodes.index.isin(g.nodes)]
    ok = ok[ok["u"].isin(g.nodes) & ok["v"].isin(g.nodes)]
    print(f"   graf autobusowy: {g.number_of_nodes()} węzłów, {g.number_of_edges()} krawędzi "
          f"({ok['length'].sum() / 1000:.0f} km z {e.drop_duplicates(['u', 'v'])['length'].sum() / 1000:.0f} km)")
    return ok.set_index("edge_id", drop=False), g, nodes


def candidates(edges: gpd.GeoDataFrame, focus: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Punkty co CANDIDATE_STEP m na odcinkach dopuszczonych do przystanków, blisko deficytu."""
    seg = edges[edges["stop_ok"]].copy()
    seg["pair"] = [tuple(sorted(p)) for p in zip(seg.u, seg.v)]
    seg = seg.drop_duplicates("pair")
    near = seg.sindex.query(focus.geometry.buffer(WALK_M).union_all(), predicate="intersects")
    seg = seg.iloc[np.unique(near)]
    rows = []
    for r in seg.itertuples():
        n = max(1, int(r.length // CANDIDATE_STEP))
        for k in range(n):
            pos = (k + 0.5) / n
            rows.append((r.edge_id, pos, r.geometry.interpolate(pos, normalized=True), r.name))
    return gpd.GeoDataFrame(rows, columns=["edge_id", "pos", "geometry", "street"], crs=CRS)


def choose_stops(cand: gpd.GeoDataFrame, demand: gpd.GeoDataFrame, col: str) -> gpd.GeoDataFrame:
    tree = STRtree(demand.geometry.values)
    c_idx, d_idx = tree.query(cand.geometry.values, predicate="dwithin", distance=WALK_M)
    cover = pd.Series(d_idx).groupby(c_idx).apply(np.array).to_dict()
    remaining = demand[col].values.astype(float).copy()
    chosen, pts = [], []
    while True:
        gains = {c: remaining[d].sum() for c, d in cover.items()}
        if not gains:
            break
        best = max(gains, key=gains.get)
        if gains[best] < MIN_STOP_GAIN:
            break
        p = cand.geometry.iloc[best]
        if any(p.distance(q) < MIN_SPACING for q in pts):
            cover.pop(best)
            continue
        chosen.append((best, gains[best]))
        pts.append(p)
        remaining[cover.pop(best)] = 0
    out = cand.iloc[[c for c, _ in chosen]].copy()
    out["covered_deficit"] = [g for _, g in chosen]
    return out.reset_index(drop=True)


def snap_hubs(hubs: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    tree = STRtree(nodes.geometry.values)
    idx = tree.nearest(hubs.geometry.values)
    hubs = hubs.copy()
    hubs["node"] = nodes.index.values[idx]
    hubs["snap_m"] = hubs.geometry.distance(nodes.geometry.iloc[idx].reset_index(drop=True), align=False).values
    return hubs[hubs["snap_m"] <= HUB_SNAP_M]


def build_lines(stops: gpd.GeoDataFrame, edges, g: nx.DiGraph, hubs: gpd.GeoDataFrame) -> list[dict]:
    ug = g.to_undirected(as_view=False)
    # Wirtualne źródło na każdy węzeł z karą zależną od rodzaju (metro/kolej vs tramwaj).
    best = hubs.assign(pen=hubs["mode"].map(HUB_PENALTY)).sort_values("pen").drop_duplicates("node")
    for node, pen in zip(best["node"], best["pen"]):
        ug.add_edge(("HUB", node), node, cost=pen)
    dist, paths = nx.multi_source_dijkstra(ug, {("HUB", n) for n in best["node"]}, weight="cost")
    for node in best["node"]:
        ug.remove_node(("HUB", node))
    stops["anchor"] = edges.loc[stops["edge_id"], "u"].values
    stops = stops[stops["anchor"].isin(dist)].copy()
    stops["hub_node"] = [paths[a][1] for a in stops["anchor"]]
    stops["hub_dist"] = [dist[a] for a in stops["anchor"]]
    for n in best["node"]:
        dist[n] = 0.0

    # Odległości między przystankami (i do węzłów) liczymy raz, z obcięciem na długości linii.
    anchors_all = set(stops["anchor"]) | set(best["node"])
    sp = {a: nx.single_source_dijkstra_path_length(ug, a, cutoff=MAX_LINE_M, weight="cost")
          for a in set(stops["anchor"])}

    def d(a, b):
        if a in sp:
            return sp[a].get(b, np.inf)
        return sp[b].get(a, np.inf) if b in sp else np.inf

    # Linie budujemy globalnie: zaczynamy od najdalszego przystanku, a po drodze do węzła
    # zbieramy przystanki z innych obszarów, jeśli objazd jest mały.
    lines = []
    todo = stops.sort_values("hub_dist", ascending=False)
    while len(todo):
        first = todo.iloc[0]
        seq = [first.name]          # kolejność przystanków od końca linii do węzła
        anchors = [first["anchor"], first["hub_node"]]
        legs = [d(first["anchor"], first["hub_node"])]
        for sid, s in todo.iloc[1:].iterrows():
            cand = None
            for i in range(len(anchors) - 1):
                a, b = anchors[i], anchors[i + 1]
                da, db = d(a, s["anchor"]), d(s["anchor"], b)
                extra = da + db - legs[i]
                if sum(legs) + extra <= MAX_LINE_M and extra <= MAX_DETOUR * sum(legs) + 300:
                    if cand is None or extra < cand[0]:
                        cand = (extra, i, da, db)
            if cand:
                _, i, da, db = cand
                anchors.insert(i + 1, s["anchor"])
                legs[i:i + 1] = [da, db]
                seq.insert(i + 1, sid)
        todo = todo.drop(seq)
        lines.append({"hub_node": first["hub_node"], "stop_ids": seq, "anchors": anchors})
    return lines


def directed_route(g: nx.DiGraph, edges, stops, line: dict) -> tuple[list[str], LineString, list, list]:
    """Trasa w stronę węzła w grafie skierowanym.

    Dla każdego przystanku wybieramy kierunek krawędzi (u->v albo v->u), który daje krótszy
    dojazd — przystanek stoi na krawędzi faktycznie przejeżdżanej w tym kierunku.
    """
    def cost(a, b):
        try:
            return nx.shortest_path_length(g, a, b, weight="cost")
        except nx.NetworkXNoPath:
            return np.inf

    route_edges, stop_edges, stop_pos = [], [], []
    cur = None
    for sid in line["stop_ids"]:
        eid = stops.loc[sid, "edge_id"]
        u, v = edges.loc[eid, "u"], edges.loc[eid, "v"]
        pos = stops.loc[sid, "pos"]
        options = [(u, v, pos)] + ([(v, u, 1 - pos)] if g.has_edge(v, u) else [])
        if cur is None:
            nxt = line["hub_node"]
            a, b, ps = min(options, key=lambda o: cost(o[1], nxt))
        else:
            a, b, ps = min(options, key=lambda o: cost(cur, o[0]))
            if cur != a:
                path = nx.shortest_path(g, cur, a, weight="cost")
                route_edges += [g[x][y]["edge_id"] for x, y in zip(path, path[1:])]
        route_edges.append(g[a][b]["edge_id"])
        stop_edges.append(g[a][b]["edge_id"])
        stop_pos.append(ps)
        cur = b
    path = nx.shortest_path(g, cur, line["hub_node"], weight="cost")
    route_edges += [g[x][y]["edge_id"] for x, y in zip(path, path[1:])]
    geom = linemerge([edges.loc[e, "geometry"] for e in route_edges])
    if geom.geom_type != "LineString":
        geom = LineString([c for part in geom.geoms for c in part.coords])
    return route_edges, geom, stop_edges, stop_pos


def right_side_point(edge_geom: LineString, pos: float) -> Point:
    """Punkt przystanku odsunięty na prawą stronę jezdni (ruch prawostronny)."""
    L = edge_geom.length
    a = edge_geom.interpolate(max(0, pos * L - 2))
    b = edge_geom.interpolate(min(L, pos * L + 2))
    dx, dy = b.x - a.x, b.y - a.y
    n = (dx * dx + dy * dy) ** 0.5 or 1
    p = edge_geom.interpolate(pos * L)
    return Point(p.x + dy / n * SIDE_OFFSET_M, p.y - dx / n * SIDE_OFFSET_M)


def main() -> None:
    hexes = gpd.read_parquet(DATA_DIR / "deficit_h3.parquet").to_crs(CRS)
    years = sorted(int(c.split("_")[1]) for c in hexes.columns if c.startswith("deficit_"))
    year = int(sys.argv[1]) if len(sys.argv) > 1 else years[-1]
    col = f"deficit_{year}"
    print(f"== horyzont {year}")

    print("== sieć autobusowa")
    edges, g, nodes = load_network()

    demand = gpd.GeoDataFrame(hexes[["h3", "dzielnica", col]], geometry=hexes.centroid, crs=CRS)
    focus = demand[demand[col] >= FOCUS_MIN_DEFICIT]
    print(f"   obszary deficytu: {len(focus)} heksagonów, {focus[col].sum():,.0f} os.")

    print("== przystanki (maximal covering)")
    cand = candidates(edges, focus)
    stops = choose_stops(cand, focus, col)
    print(f"   {len(cand)} kandydatów -> {len(stops)} przystanków, pokrywają {stops['covered_deficit'].sum():,.0f} os.")

    print("== linie do węzłów przesiadkowych")
    all_hubs = gpd.read_parquet(DATA_DIR / "hubs.parquet").to_crs(CRS)
    hubs = snap_hubs(all_hubs[all_hubs["open_year"] <= year], nodes)
    lines = build_lines(stops, edges, g, hubs)

    rank = hubs["mode"].map(HUB_PENALTY)
    hub_by_node = hubs.assign(r=rank).sort_values(["r", "open_year"]).drop_duplicates("node").set_index("node")
    tree = STRtree(demand.geometry.values)
    rows, stop_rows = [], []
    for line in lines:
        served = stops.loc[line["stop_ids"], "covered_deficit"].sum()
        if served < MIN_LINE_DEFICIT:
            continue
        route_edges, geom, stop_edges, stop_pos = directed_route(g, edges, stops, line)
        if geom.length > MAX_LINE_M * 1.6:
            continue  # jednokierunkowe ulice wymusiłyby absurdalny objazd
        lid = f"N{len(rows) + 1:02d}"
        hub = hub_by_node.loc[line["hub_node"]]
        pts = [right_side_point(edges.loc[e, "geometry"], ps) for e, ps in zip(stop_edges, stop_pos)]
        covered = np.unique(np.concatenate([tree.query(p, predicate="dwithin", distance=WALK_M) for p in pts]))
        cov = demand.iloc[covered]
        by_year = {t: float(hexes.iloc[covered][f"deficit_{t}"].sum()) for t in years}
        needed = next((t for t in years if by_year[t] >= MIN_LINE_DEFICIT), year)
        growth = float(hexes.iloc[covered][f"growth_{year}"].sum())
        districts = cov["dzielnica"].value_counts().index.tolist()
        streets = [s for s in dict.fromkeys(edges.loc[route_edges, "name"].dropna().astype(str).str.split(";").str[0])]
        why = (f"{served:,.0f} os. bez dostępu do transportu w {year} r. (w tym przyrost {growth:,.0f} os. z nowej zabudowy); "
               f"dowóz do: {hub['name']} ({hub['kind']}, od {int(hub['open_year'])}).").replace(",", " ")
        rows.append({
            "line_id": lid, "kind": "linia" if len(pts) > 1 else "przystanek / wydłużenie linii", "hub": hub["name"], "hub_kind": hub["kind"], "hub_open_year": int(hub["open_year"]),
            "n_stops": len(pts), "length_m": round(geom.length), "served_deficit": round(served),
            "growth_people": round(growth), "needed_from": needed,
            **{f"deficit_{t}": round(v) for t, v in by_year.items()},
            "districts": ", ".join(districts[:3]), "streets": " – ".join(streets[:8]),
            "justification": why, "edge_ids": ";".join(route_edges), "geometry": geom,
        })
        for k, (sid, eid, p) in enumerate(zip(line["stop_ids"], stop_edges, pts)):
            stop_rows.append({"line_id": lid, "seq": k + 1, "edge_id": eid, "street": stops.loc[sid, "street"],
                              "covered_deficit": round(stops.loc[sid, "covered_deficit"]), "geometry": p})

    lines_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS).to_crs("EPSG:4326")
    stops_gdf = gpd.GeoDataFrame(stop_rows, geometry="geometry", crs=CRS).to_crs("EPSG:4326")
    lines_gdf.to_parquet(DATA_DIR / "recommended_lines.parquet")
    stops_gdf.to_parquet(DATA_DIR / "recommended_stops.parquet")
    print(f"== {len(lines_gdf)} linii, {len(stops_gdf)} przystanków -> data/recommended_*.parquet")
    print(lines_gdf[["line_id", "hub", "n_stops", "length_m", "served_deficit", "needed_from", "districts"]]
          .sort_values("served_deficit", ascending=False).to_string(index=False))

    export(hexes, years, lines_gdf, stops_gdf, all_hubs)


def export(hexes, years, lines, stops, hubs) -> None:
    hexes = hexes.to_crs("EPSG:4326")
    keep = hexes[hexes[[f"deficit_{t}" for t in years]].max(axis=1) > 5]
    r = lambda x: round(float(x), 5)
    data = {
        "years": years,
        "summary": {t: {"deficit": round(float(hexes[f"deficit_{t}"].sum())),
                        "growth": round(float(hexes[f"growth_{t}"].clip(lower=0).sum()))} for t in years},
        "hexes": [{"h3": row.h3, "d": row.dzielnica, "pop": round(row.pop_2025),
                   **{f"def{t}": round(getattr(row, f"deficit_{t}")) for t in years},
                   **{f"pop{t}": round(getattr(row, f"pop_{t}")) for t in years},
                   "ring": [[r(x), r(y)] for x, y in row.geometry.exterior.coords]}
                  for row in keep.itertuples()],
        "lines": [{k: (v if not isinstance(v, (np.integer, np.floating)) else v.item())
                   for k, v in row.items() if k not in ("geometry", "edge_ids")}
                  | {"coords": [[r(x), r(y)] for x, y in row["geometry"].coords]}
                  for _, row in lines.iterrows()],
        "stops": [{"line": s.line_id, "seq": int(s.seq), "street": s.street if isinstance(s.street, str) else "",
                   "deficit": int(s.covered_deficit), "edge_id": s.edge_id, "xy": [r(s.geometry.x), r(s.geometry.y)]}
                  for s in stops.itertuples()],
        "hubs": [{"name": h["name"], "kind": h["kind"], "mode": h["mode"], "status": h["status"],
                  "open": int(h["open_year"]), "project": h["project"], "xy": [r(p.x), r(p.y)]}
                 for (_, h), p in zip(hubs.iterrows(), hubs.to_crs("EPSG:4326").geometry)
                 if h["status"] != "istniejący" or h["mode"] == "rail"],
    }
    out = VIZ_DIR / "recommendations.json"
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"== {out}: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
