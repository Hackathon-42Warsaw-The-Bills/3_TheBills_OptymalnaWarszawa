"""Buduje graf sieci ulic i torowisk tramwajowych Warszawy z OpenStreetMap.

Źródłem jest snapshot województwa mazowieckiego z Geofabrik (plik PBF), a nie
Overpass API — wynik jest powtarzalny i nie zależy od dostępności serwera.

Wynik (katalog data/):
  - roads_edges.parquet / roads_nodes.parquet  — odcinki ulic (skrzyżowanie -> skrzyżowanie)
  - trams_edges.parquet / trams_nodes.parquet  — odcinki torowisk
  - roads.graphml / trams.graphml              — pełne grafy do wczytania przez osmnx
  - warszawa_boundary.geojson                  — granica miasta
  - preview.html                               — podgląd na mapie

Każdy odcinek ma stabilne `edge_id` = "<u>_<v>_<key>" (u, v to ID węzłów OSM),
do którego później dowiązujemy wagi przewidywane w czasie.

Uruchomienie:  .venv/bin/python mapping/build_graph.py
"""

from pathlib import Path

import folium
import geopandas as gpd
import osmium
import osmnx as ox
import requests

PLACE = "Warszawa, Polska"
PBF_URL = "https://download.geofabrik.de/europe/poland/mazowieckie-latest.osm.pbf"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OSM_DIR = DATA_DIR / "osm"
PBF_PATH = OSM_DIR / "mazowieckie-latest.osm.pbf"
BOUNDARY_PATH = DATA_DIR / "warszawa_boundary.geojson"

# Drogi, po których może jechać autobus. service bez parkingów i podjazdów,
# bo pętle autobusowe są w OSM często oznaczone jako service.
BUS_HIGHWAYS = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "living_street", "busway", "service", "motorway_link",
    "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}
EXCLUDED_SERVICE = {"parking_aisle", "driveway", "drive-through", "emergency_access"}
EXCLUDED_ACCESS = {"private", "no"}
PSV_ALLOWED = {"yes", "designated"}

# Dodatkowe tagi OSM przydatne do oceny przejezdności dla autobusu.
EXTRA_WAY_TAGS = [
    "bus", "psv", "busway", "lanes:bus", "lanes:psv", "turn:lanes",
    "maxweight", "maxheight", "maxwidth", "width", "surface", "service",
    "railway", "gauge",
]

EDGE_COLUMNS = [
    "edge_id", "u", "v", "key", "osmid", "name", "ref", "highway", "railway",
    "service", "length", "lanes", "maxspeed", "oneway", "reversed", "bridge",
    "tunnel", "junction", "access", "bus", "psv", "busway", "lanes:bus",
    "lanes:psv", "turn:lanes", "maxweight", "maxheight", "maxwidth", "width",
    "surface", "geometry",
]


def is_bus_road(tags) -> bool:
    if tags.get("bus") in PSV_ALLOWED or tags.get("psv") in PSV_ALLOWED:
        return "highway" in tags
    return (
        tags.get("highway") in BUS_HIGHWAYS
        and tags.get("area") != "yes"
        and tags.get("service") not in EXCLUDED_SERVICE
        and tags.get("access") not in EXCLUDED_ACCESS
    )


def is_tram(tags) -> bool:
    return tags.get("railway") == "tram"


def configure_osmnx() -> None:
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(DATA_DIR / "cache")
    ox.settings.log_console = False
    ox.settings.useful_tags_way = list(
        dict.fromkeys(ox.settings.useful_tags_way + EXTRA_WAY_TAGS)
    )


def ensure_pbf() -> None:
    if PBF_PATH.exists():
        return
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"== pobieranie {PBF_URL}")
    with requests.get(PBF_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(PBF_PATH, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def load_boundary() -> gpd.GeoDataFrame:
    if not BOUNDARY_PATH.exists():
        ox.geocode_to_gdf(PLACE).to_file(BOUNDARY_PATH, driver="GeoJSON")
    return gpd.read_file(BOUNDARY_PATH)


def extract_osm(name: str, predicate, bbox) -> Path:
    """Wycina z PBF drogi spełniające `predicate` w obrębie bbox i zapisuje do .osm."""
    out = OSM_DIR / f"{name}.osm"
    if out.exists():
        return out
    min_lon, min_lat, max_lon, max_lat = bbox
    box = osmium.osm.Box(osmium.osm.Location(min_lon, min_lat), osmium.osm.Location(max_lon, max_lat))

    def in_box(way) -> bool:
        return any(n.location.valid() and box.contains(n.location) for n in way.nodes)

    print(f"== {name}: wycinanie z {PBF_PATH.name}")
    with osmium.BackReferenceWriter(str(out), ref_src=str(PBF_PATH), overwrite=True) as writer:
        for obj in osmium.FileProcessor(str(PBF_PATH), osmium.osm.NODE | osmium.osm.WAY).with_locations():
            if obj.is_way() and predicate(obj.tags) and in_box(obj):
                writer.add_way(obj)
    return out


def to_tables(graph) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes, edges = ox.graph_to_gdfs(graph)
    edges = edges.reset_index()
    edges["edge_id"] = (
        edges["u"].astype(str) + "_" + edges["v"].astype(str) + "_" + edges["key"].astype(str)
    )
    edges = edges[[c for c in EDGE_COLUMNS if c in edges.columns]]

    # Po uproszczeniu grafu atrybuty scalonych odcinków są listami. Parquet
    # wymaga jednego typu na kolumnę, więc listy zapisujemy jako "a;b".
    for df in (nodes, edges):
        for col in df.columns.drop("geometry"):
            if df[col].dtype == object:
                df[col] = df[col].map(
                    lambda x: ";".join(map(str, x)) if isinstance(x, list) else x
                ).astype("string")
    return nodes.reset_index(), edges


def build(name: str, predicate, boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    osm_file = extract_osm(name, predicate, boundary.total_bounds)
    print(f"== {name}: budowanie grafu")
    graph = ox.graph_from_xml(osm_file, simplify=False, retain_all=True)
    graph = ox.truncate.truncate_graph_polygon(
        graph, boundary.union_all(), truncate_by_edge=True
    )
    graph = ox.truncate.largest_component(graph, strongly=False)
    graph = ox.simplify_graph(graph)

    ox.save_graphml(graph, DATA_DIR / f"{name}.graphml")
    nodes, edges = to_tables(graph)
    nodes.to_parquet(DATA_DIR / f"{name}_nodes.parquet")
    edges.to_parquet(DATA_DIR / f"{name}_edges.parquet")
    print(
        f"== {name}: {len(nodes)} węzłów, {len(edges)} odcinków, "
        f"{edges['length'].sum() / 1000:.0f} km"
    )
    return edges


def save_preview(roads: gpd.GeoDataFrame, trams: gpd.GeoDataFrame) -> None:
    roads = roads[["edge_id", "name", "highway", "geometry"]].copy()
    roads["geometry"] = roads.geometry.simplify(0.00002)
    m = roads.explore(
        column="highway", tiles="CartoDB positron", tooltip=["edge_id", "name", "highway"],
        style_kwds={"weight": 1.5}, legend=True, name="Ulice",
    )
    # Torowiska w OSM zwykle nie mają nazw, więc kolumny "name" może nie być.
    trams[["edge_id", "geometry"]].explore(
        m=m, color="red", tooltip=["edge_id"], style_kwds={"weight": 2.5},
        name="Tramwaje",
    )
    folium.LayerControl().add_to(m)
    m.save(DATA_DIR / "preview.html")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    configure_osmnx()
    ensure_pbf()
    boundary = load_boundary()
    roads = build("roads", is_bus_road, boundary)
    trams = build("trams", is_tram, boundary)
    save_preview(roads, trams)
    print(f"== Gotowe: {DATA_DIR}")


if __name__ == "__main__":
    main()
