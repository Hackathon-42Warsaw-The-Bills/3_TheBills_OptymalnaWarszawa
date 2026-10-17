"""Historia ludności klastrów H3: spisy GUS 2011 i 2021 oraz GHSL 2000–2030.

Wejście:
  - dane/kontury_400m/warszawa_h3_400m_pelna.geojson   — siatka klastrów (mapping/h3_siatka.py)
  - GUS NSP 2011, siatka 1 km (INSPIRE, EPSG:3035)      — pobierana do data/gus2011/
  - GUS NSP 2021, siatka 1 km (dane_warszawa)
  - GHSL GHS-POP R2023A, 100 m, epoki 2000–2030         — czytane zdalnie z mirrora COG

Metoda (na pikselach GHSL 100 m):
  Każdy piksel ma kwadrat GUS 1 km i heksagon H3. Ludność kwadratu GUS dzielimy na jego
  piksele proporcjonalnie do GHSL z epoki najbliższej spisowi (2011 <- GHSL 2010,
  2021 <- GHSL 2020), więc zmiana 2011→2021 nie jest zafałszowana rozkładem z innego roku.
  Piksele poza Warszawą odpadają (kwadraty brzegowe liczone są w całości, jak w GUS).
  GHSL (model JRC; 2025 i 2030 to projekcje) sumujemy do heksagonów bez zmian.

Uwaga: NSP 2011 to ludność wg definicji krajowej, NSP 2021 — rezydenci. Dla Warszawy
różnica definicji jest mała w sumie, ale porównując klastry patrzymy głównie na zmiany względne.

Wynik: dane/historia_ludnosci.csv — h3, gus_2011, gus_2021, ghsl_<rok>, zmiany.

Uruchomienie (po h3_siatka.py):  .venv/bin/python mapping/historia_ludnosci.py
"""

import io
import os
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GUS2011_DIR = DATA_DIR / "gus2011"
HEX_PATH = ROOT / "dane" / "kontury_400m" / "warszawa_h3_400m_pelna.geojson"
OUT = ROOT / "dane" / "historia_ludnosci.csv"
EXTERNAL_DIR = Path(os.environ.get("DANE_WARSZAWA", Path.home() / "Downloads" / "dane_warszawa"))

GUS2011_TILES = {  # kafelki 100 km (N) × 1000 km (E) z usługi ATOM GUS, PD.GRID.2011
    "N32E5": "https://geo.stat.gov.pl/atom-web/download/?fileId=99715f15ab46c320900aac757b9ac077&name=StatisticalDistribution_GRID_N32E5.zip",
    "N33E5": "https://geo.stat.gov.pl/atom-web/download/?fileId=851a4a8d407a6bfabb30264220ea92af&name=StatisticalDistribution_GRID_N33E5.zip",
}
GHSL_URL = ("https://data.source.coop/nlebovits/ghsl/pop-{y}/pop-{y}-R3_C20/"
            "GHS_POP_E{y}_GLOBE_R2023A_54009_100_V1_0_R3_C20.tif")
GHSL_EPOCHS = [2000, 2005, 2010, 2015, 2020, 2025, 2030]
CENSUS_WEIGHTS = {2011: 2010, 2021: 2020}   # spis -> epoka GHSL używana do rozkładu
PAD_M = 2000                                # zapas wokół miasta (pełne kwadraty brzegowe)

PD = "{http://inspire.ec.europa.eu/schemas/pd/4.0}"
XL = "{http://www.w3.org/1999/xlink}"


def gus_2011() -> pd.Series:
    """Ludność ogółem 2011 per kwadrat 1 km, indeks (N_km, E_km) w EPSG:3035."""
    cache = GUS2011_DIR / "grid2011_total.csv"
    if not cache.exists():
        GUS2011_DIR.mkdir(parents=True, exist_ok=True)
        rows = []
        for name, url in GUS2011_TILES.items():
            print(f"   pobieranie GUS 2011 {name}")
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    for _, el in ET.iterparse(f):
                        if el.tag != PD + "StatisticalValue":
                            continue
                        # Bez wymiaru tematycznego (płeć/wiek) = ludność ogółem.
                        if next(el.iter(PD + "thematic"), None) is None:
                            cell = el.find(f".//{PD}spatial").get(XL + "href").rsplit("-1km", 1)[-1]
                            rows.append((int(cell[:4]), int(cell[4:]), float(el.find(PD + "value").text)))
                        el.clear()
        pd.DataFrame(rows, columns=["N", "E", "pop_2011"]).to_csv(cache, index=False)
    t = pd.read_csv(cache)
    return t.set_index(["N", "E"])["pop_2011"]


def gus_2021() -> pd.Series:
    g = pd.read_csv(EXTERNAL_DIR / "warszawa_siatka_1km_NSP2021.csv", encoding="utf-8-sig")
    code = g["CODE"].str.extract(r"N(\d+)E(\d+)").astype(int) // 1000
    return pd.Series(g["RES"].values, index=pd.MultiIndex.from_arrays([code[0], code[1]], names=["N", "E"]))


def ghsl_pixels(boundary_ll) -> pd.DataFrame:
    """Piksele GHSL w okolicy Warszawy: środek (lon/lat, EPSG:3035), wartości dla epok."""
    to_moll = Transformer.from_crs(4326, "ESRI:54009", always_xy=True)
    minx, miny, maxx, maxy = boundary_ll.bounds
    xs, ys = to_moll.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
    bounds = (min(xs) - PAD_M, min(ys) - PAD_M, max(xs) + PAD_M, max(ys) + PAD_M)
    data, transform = {}, None
    for y in GHSL_EPOCHS:
        with rasterio.open(GHSL_URL.format(y=y)) as src:
            win = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=win).astype(float)
            arr[arr < 0] = 0
            data[y] = arr
            transform = src.window_transform(win)
        print(f"   GHSL {y}: {data[y].sum():,.0f} os. w oknie")
    rows, cols = np.indices(data[GHSL_EPOCHS[0]].shape)
    mx, my = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    lon, lat = Transformer.from_crs("ESRI:54009", 4326, always_xy=True).transform(mx, my)
    ex, ny = Transformer.from_crs(4326, 3035, always_xy=True).transform(lon, lat)
    px = pd.DataFrame({"lon": lon, "lat": lat, "N": (np.asarray(ny) // 1000).astype(int),
                       "E": (np.asarray(ex) // 1000).astype(int)})
    for y in GHSL_EPOCHS:
        px[f"ghsl_{y}"] = data[y].ravel()
    return px


def main() -> None:
    hexes = gpd.read_file(HEX_PATH)
    boundary = hexes.union_all()

    print("== spisy GUS")
    census = {2011: gus_2011(), 2021: gus_2021()}
    print("== GHSL")
    px = ghsl_pixels(boundary)

    pts = gpd.GeoSeries(gpd.points_from_xy(px["lon"], px["lat"]), crs="EPSG:4326")
    px["in_city"] = pts.within(boundary).values
    px["h3"] = [h3.latlng_to_cell(la, lo, 8) for la, lo in zip(px["lat"], px["lon"])]

    for year, pop in census.items():
        w = px[f"ghsl_{CENSUS_WEIGHTS[year]}"]
        key = pd.MultiIndex.from_arrays([px["N"], px["E"]])
        tot_w = w.groupby(key).transform("sum")
        n_px = px.groupby(["N", "E"])["lon"].transform("size")
        # Kwadrat bez ludności w GHSL -> równo na piksele.
        share = np.where(tot_w > 0, w / tot_w.replace(0, 1), 1 / n_px)
        px[f"gus_{year}"] = pop.reindex(key).fillna(0).values * share

    city = px[px["in_city"]]
    cols = [f"gus_{y}" for y in census] + [f"ghsl_{y}" for y in GHSL_EPOCHS]
    agg = city.groupby("h3")[cols].sum().reindex(hexes["h3"]).fillna(0)
    out = agg.round(0).astype(int)
    out.insert(0, "dzielnica", hexes.set_index("h3")["dzielnica"])
    out["zmiana_2011_2021"] = out["gus_2021"] - out["gus_2011"]
    out["zmiana_2011_2021_proc"] = (100 * out["zmiana_2011_2021"] / out["gus_2011"].where(out["gus_2011"] >= 100)).round(1)
    out["ghsl_zmiana_2010_2020"] = out["ghsl_2020"] - out["ghsl_2010"]
    out["ghsl_zmiana_2020_2030"] = out["ghsl_2030"] - out["ghsl_2020"]
    out.reset_index().to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"== zapisano {len(out)} heksagonów -> {OUT.relative_to(ROOT)}")
    print("   sumy dla Warszawy:", {c: f"{out[c].sum():,}" for c in cols})


if __name__ == "__main__":
    main()
