"""Stacje roweru miejskiego Veturilo (Nextbike, otwarty feed GBFS) — lista stacji i zestawienie na klastry H3.

Źródło: https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_vw/gbfs.json  (system „VETURILO 3.0”)
  station_information — nazwa, numer, położenie, pojemność (liczba stojaków)
  station_status      — rowery i wolne stojaki w chwili pobrania (migawka, zmienia się co minutę)

Wynik:
  dane/veturilo_stacje.csv       — jedna stacja na wiersz (tylko w granicach Warszawy), z h3 i dzielnicą
  dane/veturilo_w_klastrach.csv  — indeks h3, wszystkie 848 klastrów:
      stacje_veturilo, stojaki_veturilo, rowery_teraz, ma_veturilo, odl_do_veturilo_m (od środka klastra)

Uruchomienie:  .venv/bin/python mapping/veturilo.py
"""

from datetime import datetime
from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DANE = ROOT / "dane"
GBFS = "https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_vw/gbfs.json"
HEADERS = {"User-Agent": "optymalna-warszawa-hackathon/1.0"}
CRS = "EPSG:2180"


def feed(name: str) -> dict:
    root = requests.get(GBFS, headers=HEADERS, timeout=30).json()["data"]
    url = next(f["url"] for f in next(iter(root.values()))["feeds"] if f["name"] == name)
    return requests.get(url, headers=HEADERS, timeout=30).json()["data"]


def main() -> None:
    info = pd.DataFrame(feed("station_information")["stations"])
    status = pd.DataFrame(feed("station_status")["stations"])
    st = info.merge(status[["station_id", "num_bikes_available", "num_docks_available"]], on="station_id", how="left")
    st = st[~st.get("is_virtual_station", False).fillna(False).astype(bool)]

    hexes = gpd.read_file(DANE / "kontury_400m" / "warszawa_h3_400m_pelna.geojson")[["h3", "dzielnica", "geometry"]]
    pts = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st["lon"], st["lat"]), crs=4326)
    pts = gpd.sjoin(pts, hexes, predicate="within", how="inner").drop(columns="index_right")
    pts["h3"] = [h3.latlng_to_cell(y, x, 8) for x, y in zip(pts["lon"], pts["lat"])]
    stamp = datetime.now().isoformat(timespec="minutes")
    out = pd.DataFrame({
        "station_id": pts["station_id"], "nazwa": pts["name"], "numer": pts["short_name"],
        "lon": pts["lon"].round(6), "lat": pts["lat"].round(6), "stojaki": pts["capacity"],
        "rowery_teraz": pts["num_bikes_available"], "wolne_stojaki_teraz": pts["num_docks_available"],
        "h3": pts["h3"], "dzielnica": pts["dzielnica"], "pobrano": stamp,
    }).sort_values(["dzielnica", "nazwa"])
    out.to_csv(DANE / "veturilo_stacje.csv", index=False, encoding="utf-8")

    # Zestawienie na wszystkie klastry (indeks h3), odległość od środka klastra do najbliższej stacji
    g = out.groupby("h3").agg(stacje_veturilo=("station_id", "size"), stojaki_veturilo=("stojaki", "sum"),
                              rowery_teraz=("rowery_teraz", "sum"))
    k = hexes.set_index("h3")[[]].join(g).fillna(0).astype(int)
    k["ma_veturilo"] = k["stacje_veturilo"] > 0
    c = hexes.to_crs(CRS).centroid
    sp = gpd.GeoSeries(gpd.points_from_xy(out["lon"], out["lat"]), crs=4326).to_crs(CRS)
    near = gpd.sjoin_nearest(gpd.GeoDataFrame(geometry=c.values, crs=CRS), gpd.GeoDataFrame(geometry=sp.values, crs=CRS), distance_col="d")
    k["odl_do_veturilo_m"] = near.groupby(level=0)["d"].min().round().astype(int).values
    k.index.name = "h3"
    k.to_csv(DANE / "veturilo_w_klastrach.csv", encoding="utf-8")

    print(f"== {len(info)} stacji w feedzie, w granicach Warszawy: {len(out)}, stojaków: {out['stojaki'].sum()}, "
          f"rowerów teraz ({stamp}): {int(out['rowery_teraz'].sum())}")
    print(f"== klastry ze stacją: {k['ma_veturilo'].sum()} z {len(k)}; stacja ≤ 500 m od środka: {(k['odl_do_veturilo_m'] <= 500).sum()}")
    print(out.groupby("dzielnica").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
