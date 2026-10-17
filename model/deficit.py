"""Wskaźnik deficytu transportowego na siatce H3 dla horyzontów 2026–2036.

Wejście (po mapping/population.py i mapping/rail_plans.py):
  - data/segments_population.parquet  — ludność 2025 na odcinkach ulic
  - data/hubs.parquet                  — węzły szynowe (dziś + planowane, z rokiem otwarcia)
  - data/gunb_permits.parquet          — pozwolenia na budowę (opcjonalnie, mapping/gunb.py)
  - dane_warszawa: przystanki ZTM, dzielnice

Wynik: data/deficit_h3.parquet — jeden wiersz na heksagon (H3 res 9, ~0,1 km²),
kolumny `<miara>_<rok>` dla każdego roku z HORIZONS.

Metoda (wszystko w osobach, żeby dało się to uczciwie wytłumaczyć):
  1. Ludność 2025 z odcinków -> heksagony (środek odcinka).
  2. Przyrost do roku t:
       - pozwolenia GUNB: mieszkania × PEOPLE_PER_FLAT, zasiedlone BUILD_LAG lat po decyzji;
       - po wyczerpaniu się pozwoleń: średnie roczne tempo pozwoleń dzielnicy kontynuowane
         tam, gdzie deweloperzy budowali ostatnio (front zabudowy);
       - bez pliku GUNB: tempo zmian ludności dzielnicy 2021→2025 (GUS).
  3. Dostępność w roku t (0..1): kolej/metro (stacje otwarte do roku t), tramwaj, autobus.
     Zasięgi maleją liniowo od „pełnego” do „zerowego” dystansu.
  4. Deficyt(t) = ludność 2025 × (1 − dostępność_dziś)       # dziś bez przystanku
                + przyrost(t) × (1 − dostępność_szynowa(t))   # nowi mieszkańcy, których nie
                                                              # przejmie kolej ani tramwaj —
                                                              # dzisiejsze linie autobusowe
                                                              # są skrojone pod rok 2025
     Indeks 0–100 = deficyt / 99. percentyl dla roku 2036.

Uruchomienie:  .venv/bin/python model/deficit.py
"""

import os
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EXTERNAL_DIR = Path(os.environ.get("DANE_WARSZAWA", Path.home() / "Downloads" / "dane_warszawa"))
CRS = "EPSG:2180"

H3_RES = 9
BASE_YEAR = 2025
HORIZONS = [2026, 2028, 2031, 2033, 2036]

PEOPLE_PER_FLAT = 2.0   # średnia wielkość gospodarstwa w nowych mieszkaniach w Warszawie (~1,9–2,1)
BUILD_LAG = 3           # lat od pozwolenia do zasiedlenia
TREND_YEARS = 5         # z ilu ostatnich lat pozwoleń liczymy tempo dzielnicy

# (dystans pełnej dostępności, dystans zerowej dostępności, waga) w metrach
RAIL_ACCESS = (800, 1500, 1.0)
TRAM_ACCESS = (500, 900, 0.8)
BUS_ACCESS = (400, 700, 1.0)   # dziś: czy jest jakikolwiek przystanek


def hexes_of(points: gpd.GeoSeries) -> np.ndarray:
    ll = points.to_crs("EPSG:4326")
    return np.array([h3.latlng_to_cell(y, x, H3_RES) for x, y in zip(ll.x, ll.y)])


def hex_frame(cells) -> gpd.GeoDataFrame:
    cells = sorted(set(cells))
    polys = [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in cells]
    g = gpd.GeoDataFrame({"h3": cells}, geometry=polys, crs="EPSG:4326").to_crs(CRS)
    return g.set_index("h3")


def ramp(dist: np.ndarray, full: float, zero: float, weight: float) -> np.ndarray:
    return weight * np.clip((zero - dist) / (zero - full), 0, 1)


def nearest_dist(targets: gpd.GeoSeries, sources: gpd.GeoSeries) -> np.ndarray:
    if len(sources) == 0:
        return np.full(len(targets), np.inf)
    tree = STRtree(sources.values)
    idx = tree.nearest(targets.values)
    return targets.distance(sources.iloc[idx].reset_index(drop=True), align=False).values


def base_population(districts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    seg = gpd.read_parquet(DATA_DIR / "segments_population.parquet").to_crs(CRS)
    mid = seg.geometry.interpolate(0.5, normalized=True)
    seg["h3"] = hexes_of(mid)
    pop = seg.groupby("h3")[["pop_2025"]].sum()
    hexes = hex_frame(pop.index).join(pop)
    c = gpd.GeoDataFrame(geometry=hexes.centroid, crs=CRS)
    hexes["dzielnica"] = gpd.sjoin_nearest(c, districts[["dzielnica", "geometry"]]).groupby(level=0)["dzielnica"].first()
    return hexes


def growth_from_permits(hexes: gpd.GeoDataFrame, permits: gpd.GeoDataFrame) -> pd.DataFrame:
    """Przyrost ludności [heksagon × rok] z pozwoleń GUNB i kontynuacji trendu."""
    permits = permits.to_crs(CRS)
    permits = permits[permits["n_flats"] > 0].copy()
    permits["h3"] = hexes_of(permits.geometry)
    permits["occupied"] = permits["rok"] + BUILD_LAG
    # Zasiedlone przed 2025 są już w ludności GUS 2025 (choć rozłożone wg dzielnicy).
    permits = permits[permits["occupied"] > BASE_YEAR]

    last_year = int(permits["rok"].max())
    recent = permits[permits["rok"] > last_year - TREND_YEARS]
    c = gpd.GeoDataFrame(geometry=recent.geometry, crs=CRS)
    recent = recent.assign(dzielnica=gpd.sjoin_nearest(c, hexes[["dzielnica", "geometry"]].reset_index(drop=True))
                           .groupby(level=0)["dzielnica"].first())
    per_year = recent.groupby("dzielnica")["n_flats"].sum() / TREND_YEARS
    # Front zabudowy: udział heksagonu w ostatnich pozwoleniach dzielnicy.
    share = recent.groupby(["dzielnica", "h3"])["n_flats"].sum()
    share = share / share.groupby(level=0).transform("sum")

    out = pd.DataFrame(0.0, index=hexes.index, columns=HORIZONS)
    for t in HORIZONS:
        pipeline = permits[permits["occupied"] <= t].groupby("h3")["n_flats"].sum()
        out[t] = out[t].add(pipeline.reindex(out.index).fillna(0) * PEOPLE_PER_FLAT)
        trend_years = max(0, t - (last_year + BUILD_LAG))
        if trend_years:
            trend = (share * share.index.get_level_values(0).map(per_year).values).groupby(level=1).sum()
            out[t] += trend.reindex(out.index).fillna(0) * trend_years * PEOPLE_PER_FLAT
    return out


def growth_from_districts(hexes: gpd.GeoDataFrame, districts: gpd.GeoDataFrame) -> pd.DataFrame:
    g = districts.set_index("dzielnica")
    cagr = (g["ludnosc_gus_2025"] / g["ludnosc_gus_2021"]) ** (1 / 4) - 1
    rate = hexes["dzielnica"].map(cagr).fillna(0)
    return pd.DataFrame({t: hexes["pop_2025"] * ((1 + rate) ** (t - BASE_YEAR) - 1) for t in HORIZONS})


def access(hexes: gpd.GeoDataFrame, hubs: gpd.GeoDataFrame, stops: gpd.GeoDataFrame) -> dict:
    c = hexes.centroid
    bus = ramp(nearest_dist(c, stops.geometry), *BUS_ACCESS)
    tram_today = stops[stops["linie_tram"].fillna("") != ""]
    out = {}
    for t in [BASE_YEAR] + HORIZONS:
        h = hubs[hubs["open_year"] <= t]
        rail = ramp(nearest_dist(c, h[h["mode"] == "rail"].geometry), *RAIL_ACCESS)
        tram_pts = pd.concat([tram_today.geometry, h[h["mode"] == "tram"].geometry])
        tram = ramp(nearest_dist(c, tram_pts), *TRAM_ACCESS)
        out[t] = {"rail_tram": np.maximum(rail, tram), "all": np.maximum.reduce([rail, tram, bus])}
    return out


def main() -> None:
    districts = gpd.read_file(EXTERNAL_DIR / "warszawa_dzielnice.geojson").to_crs(CRS)
    stops = gpd.read_file(EXTERNAL_DIR / "warszawa_przystanki_ztm.geojson").to_crs(CRS)
    hubs = gpd.read_parquet(DATA_DIR / "hubs.parquet").to_crs(CRS)

    print("== ludność 2025 -> H3")
    hexes = base_population(districts)
    permits_path = DATA_DIR / "gunb_permits.parquet"
    if permits_path.exists():
        print("== przyrost: pozwolenia GUNB")
        growth = growth_from_permits(hexes, gpd.read_parquet(permits_path))
        hexes["growth_source"] = "gunb"
    else:
        print("== przyrost: brak data/gunb_permits.parquet -> tempo dzielnic GUS")
        growth = growth_from_districts(hexes, districts)
        hexes["growth_source"] = "gus_district"
    # Nowe osiedla mogą powstać tam, gdzie dziś nikt nie mieszka — dokładamy heksagony.
    missing = growth.index.difference(hexes.index)
    if len(missing):
        extra = hex_frame(missing)
        extra["pop_2025"] = 0.0
        hexes = pd.concat([hexes, extra])
    growth = growth.reindex(hexes.index).fillna(0)

    print("== dostępność")
    acc = access(hexes, hubs, stops)
    hexes["access_2025"] = acc[BASE_YEAR]["all"]
    for t in HORIZONS:
        hexes[f"pop_{t}"] = hexes["pop_2025"] + growth[t].clip(lower=0)
        hexes[f"growth_{t}"] = growth[t]
        hexes[f"access_rail_{t}"] = acc[t]["rail_tram"]
        # Dzisiejsza luka znika, gdy w roku t dojdzie tam kolej lub tramwaj.
        today_gap = hexes["pop_2025"].values * (1 - np.maximum(acc[BASE_YEAR]["all"], acc[t]["rail_tram"]))
        hexes[f"deficit_{t}"] = today_gap + growth[t].clip(lower=0).values * (1 - acc[t]["rail_tram"])
    ref = np.percentile(hexes[f"deficit_{HORIZONS[-1]}"], 99)
    for t in HORIZONS:
        hexes[f"index_{t}"] = (100 * hexes[f"deficit_{t}"] / ref).clip(upper=100).round(1)

    out = hexes.reset_index().to_crs("EPSG:4326")
    out.to_parquet(DATA_DIR / "deficit_h3.parquet")
    print(f"== zapisano {len(out)} heksagonów -> data/deficit_h3.parquet")
    # Uwaga: przyrost to nowi mieszkańcy nowych osiedli — część przeprowadza się w obrębie
    # Warszawy, więc suma nie jest prognozą ludności miasta, tylko lokalnego popytu.
    for t in HORIZONS:
        print(f"   {t}: ludność {hexes[f'pop_{t}'].sum():>10,.0f}  przyrost {growth[t].sum():>8,.0f}  "
              f"deficyt {hexes[f'deficit_{t}'].sum():>8,.0f} os.")
    top = hexes.nlargest(10, f"deficit_{HORIZONS[-1]}")[["dzielnica", "pop_2025", f"pop_{HORIZONS[-1]}", f"deficit_{HORIZONS[-1]}"]]
    print(top.round(0).to_string())


if __name__ == "__main__":
    main()
