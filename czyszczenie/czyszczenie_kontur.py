import argparse
import sys
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd

import utils
from config import (
    CLIPPED_BELOW,
    COLUMN_NAMES,
    CRS_GPS,
    CRS_METERS,
    DECIMALS,
    DISTRICTS,
    FEW_PEOPLE,
    FILE_NAME,
    H3_RESOLUTION,
    INPUT_DIR,
    OUTPUT_COLUMNS,
    OUTPUT_DIR,
    POPULATION_COLUMNS,
    SOURCES,
    WARSAW_BOX,
)


def load_table(path):
    table = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    table.columns = table.columns.str.strip().str.lower()

    missing = set(COLUMN_NAMES) - set(table.columns)
    if missing:
        raise ValueError(f"W pliku CSV brakuje kolumn: {sorted(missing)}")

    return table[list(COLUMN_NAMES)].rename(columns=COLUMN_NAMES)


def load_shapes(path):
    shapes = gpd.read_file(path)
    shapes.columns = [c.strip().lower() for c in shapes.columns]
    shapes = shapes.rename(columns={"h3": "h3_index"})[["h3_index", "geometry"]]
    if shapes.crs is None:
        shapes = shapes.set_crs(CRS_GPS)
    return shapes.to_crs(CRS_GPS).drop_duplicates(subset="h3_index")


def clean_values(table, report):
    table = table.copy()
    table["h3_index"] = table["h3_index"].str.strip().str.lower()
    table["district"] = table["district"].map(utils.tidy_text).map(utils.official_district_name)
    table["source"] = table["source"].map(utils.tidy_text).str.lower()
    for column in POPULATION_COLUMNS + ["density_src_per_km2", "area_src_km2", "lon", "lat"]:
        table[column] = utils.to_number(table[column])

    is_valid_h3 = table["h3_index"].map(lambda x: isinstance(x, str) and h3.is_valid_cell(x))
    report.add("usunięte: błędny identyfikator H3", (~is_valid_h3).sum())
    table = table[is_valid_h3]

    is_wrong_size = table["h3_index"].map(h3.get_resolution) != H3_RESOLUTION
    report.add(f"usunięte: rozdzielczość H3 inna niż {H3_RESOLUTION}", is_wrong_size.sum())
    table = table[~is_wrong_size]

    is_unknown_district = table["district"].isna()
    report.add("usunięte: nieznana dzielnica", is_unknown_district.sum())
    table = table[~is_unknown_district]

    is_unknown_source = ~table["source"].isin(SOURCES)
    report.add("usunięte: nieznane źródło", is_unknown_source.sum())
    table = table[~is_unknown_source]

    is_bad_population = table["population"].isna() | (table["population"] < 0)
    report.add("usunięte: brak lub ujemna liczba mieszkańców", is_bad_population.sum())
    table = table[~is_bad_population]

    is_duplicate = table.duplicated(subset="h3_index", keep="first")
    report.add("usunięte: powtórzony heksagon", is_duplicate.sum())
    table = table[~is_duplicate]

    for column in POPULATION_COLUMNS:
        table[column] = table[column].clip(lower=0).round().astype("Int64")
    table["population"] = table["population"].astype("int64")
    return table


def add_shapes(table, shapes, report):
    data = gpd.GeoDataFrame(table.merge(shapes, on="h3_index", how="left"), geometry="geometry", crs=CRS_GPS)

    is_missing = data.geometry.isna() | data.geometry.is_empty
    report.add("odtworzone: brakujący kształt", is_missing.sum())
    if is_missing.any():
        data.loc[is_missing, "geometry"] = data.loc[is_missing, "h3_index"].map(utils.hexagon_shape)

    is_broken = ~data.geometry.is_valid
    report.add("naprawione: uszkodzony kształt", is_broken.sum())
    if is_broken.any():
        data.loc[is_broken, "geometry"] = data.loc[is_broken, "geometry"].make_valid()
    return data


def add_areas(data, report):
    full_hexagons = gpd.GeoSeries(data["h3_index"].map(utils.hexagon_shape), crs=CRS_GPS)
    data["area_km2"] = utils.area_km2(data.geometry)
    data["hex_area_km2"] = utils.area_km2(full_hexagons)
    data["coverage_ratio"] = (data["area_km2"] / data["hex_area_km2"]).clip(upper=1.0)
    data["is_clipped"] = data["coverage_ratio"] < CLIPPED_BELOW
    report.add("oznaczone: heksagon przycięty granicą miasta", data["is_clipped"].sum())
    return data


def fix_coordinates(data, report):
    lon_min, lat_min, lon_max, lat_max = WARSAW_BOX
    is_bad = (
        data["lon"].isna()
        | data["lat"].isna()
        | ~data["lon"].between(lon_min, lon_max)
        | ~data["lat"].between(lat_min, lat_max)
    )
    report.add("zastąpione: współrzędne puste lub poza Warszawą", is_bad.sum())

    centers = data.geometry.to_crs(CRS_METERS).centroid.to_crs(CRS_GPS)
    data["lon"] = data["lon"].where(~is_bad, centers.x)
    data["lat"] = data["lat"].where(~is_bad, centers.y)

    hex_centers = data["h3_index"].map(h3.cell_to_latlng)
    data["hex_lat"] = hex_centers.map(lambda point: point[0])
    data["hex_lon"] = hex_centers.map(lambda point: point[1])
    return data


def add_density(data, report):
    data["density_per_km2"] = data["population"] / data["area_km2"]

    differs = (data["density_src_per_km2"] - data["density_per_km2"]).abs() > data["density_per_km2"] * 0.1 + 1
    report.add("uwaga: gęstość ze źródła różni się o >10%", differs.sum())

    area_differs = (data["area_src_km2"] - data["area_km2"]).abs() > data["area_km2"] * 0.02
    report.add("uwaga: powierzchnia ze źródła różni się o >2%", area_differs.sum())
    return data


def flag_suspicious(data, report):
    data["low_population"] = data["population"] < FEW_PEOPLE
    report.add(f"oznaczone: mniej niż {FEW_PEOPLE} mieszkańców", data["low_population"].sum())

    log_density = np.log1p(data["density_per_km2"])
    q1, q3 = log_density.quantile([0.25, 0.75])
    spread = q3 - q1
    data["density_outlier"] = ~log_density.between(q1 - 3 * spread, q3 + 3 * spread)
    report.add("oznaczone: skrajnie nietypowa gęstość", data["density_outlier"].sum())
    return data


def add_district_info(data):
    data["district_teryt"] = data["district"].map(DISTRICTS)
    data["district_slug"] = data["district"].map(utils.without_polish_chars)
    district_total = data.groupby("district")["population"].transform("sum")
    data["population_share_district"] = data["population"] / district_total
    density_rank = data["density_per_km2"].rank(method="first")
    data["density_quintile"] = pd.qcut(density_rank, 5, labels=[1, 2, 3, 4, 5]).astype("int8")
    return data


def tidy_output(data):
    data = data.round({
        "lon": DECIMALS, "lat": DECIMALS, "hex_lon": DECIMALS, "hex_lat": DECIMALS,
        "area_km2": 4, "area_src_km2": 4, "hex_area_km2": 4, "coverage_ratio": 4,
        "density_per_km2": 1, "density_src_per_km2": 1,
        "population_share_district": 6,
    })
    return data[OUTPUT_COLUMNS].sort_values(["district", "h3_index"]).reset_index(drop=True)


def check_result(data):
    assert data["h3_index"].is_unique, "heksagony się powtarzają"
    assert data["district"].isin(DISTRICTS).all(), "nieznana dzielnica"
    assert (data["population"] >= 0).all(), "ujemna liczba mieszkańców"
    assert (data["area_km2"] > 0).all(), "zerowa powierzchnia"
    assert data.geometry.is_valid.all(), "uszkodzony kształt"
    assert data[["lon", "lat"]].notna().all().all(), "brak współrzędnych"


def district_summary(data):
    summary = data.groupby(["district", "district_teryt"], as_index=False).agg(
        cells=("h3_index", "size"),
        population=("population", "sum"),
        area_km2=("area_km2", "sum"),
        clipped_cells=("is_clipped", "sum"),
    )
    summary["density_per_km2"] = (summary["population"] / summary["area_km2"]).round(1)
    summary["area_km2"] = summary["area_km2"].round(3)
    return summary.sort_values("population", ascending=False)


def save(data, summary, folder):
    folder.mkdir(parents=True, exist_ok=True)
    data.drop(columns="geometry").to_csv(folder / f"{FILE_NAME}_clean.csv", index=False, encoding="utf-8")
    data.to_file(folder / f"{FILE_NAME}_clean.geojson", driver="GeoJSON", COORDINATE_PRECISION=DECIMALS)
    summary.to_csv(folder / f"{FILE_NAME}_dzielnice.csv", index=False, encoding="utf-8")
    try:
        data.to_parquet(folder / f"{FILE_NAME}_clean.parquet", index=False)
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Czyszczenie danych o zaludnieniu Warszawy (Kontur 2023, H3)")
    parser.add_argument("--csv", type=Path, default=INPUT_DIR / f"{FILE_NAME}.csv")
    parser.add_argument("--geojson", type=Path, default=INPUT_DIR / f"{FILE_NAME}.geojson")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    report = utils.Report()

    table = load_table(args.csv)
    report.add("wiersze na wejściu", len(table))
    table = clean_values(table, report)

    data = add_shapes(table, load_shapes(args.geojson), report)
    data = add_areas(data, report)
    data = fix_coordinates(data, report)

    data = add_density(data, report)
    data = flag_suspicious(data, report)
    data = add_district_info(data)

    data = tidy_output(data)
    check_result(data)
    report.add("wiersze na wyjściu", len(data))

    summary = district_summary(data)
    save(data, summary, args.out)

    report.print()
    print()
    print(f"Ludność łącznie:      {data['population'].sum():,}".replace(",", " "))
    print(f"Powierzchnia łącznie: {data['area_km2'].sum():.2f} km²")
    print()
    print(summary.to_string(index=False))
    print(f"\nZapisano w folderze: {args.out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
