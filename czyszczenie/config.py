from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "dane" / "kontury_400m"
OUTPUT_DIR = ROOT / "dane" / "kontury_400m_clean"
FILE_NAME = "warszawa_h3_400m_pelna"

H3_RESOLUTION = 8
CRS_GPS = "EPSG:4326"
CRS_METERS = "EPSG:2180"
DECIMALS = 6
CLIPPED_BELOW = 0.98
FEW_PEOPLE = 10
WARSAW_BOX = (20.84, 52.08, 21.30, 52.38)
SOURCES = {"kontur", "dodany"}
POPULATION_COLUMNS = ["population", "population_2021", "population_kontur"]

DISTRICTS = {
    "Bemowo": "146502",
    "Białołęka": "146503",
    "Bielany": "146504",
    "Mokotów": "146505",
    "Ochota": "146506",
    "Praga-Południe": "146507",
    "Praga-Północ": "146508",
    "Rembertów": "146509",
    "Śródmieście": "146510",
    "Targówek": "146511",
    "Ursus": "146512",
    "Ursynów": "146513",
    "Wawer": "146514",
    "Wesoła": "146515",
    "Wilanów": "146516",
    "Włochy": "146517",
    "Wola": "146518",
    "Żoliborz": "146519",
}

COLUMN_NAMES = {
    "h3": "h3_index",
    "population": "population",
    "population_2021": "population_2021",
    "population_kontur": "population_kontur",
    "gestosc_os_km2": "density_src_per_km2",
    "pow_km2": "area_src_km2",
    "dzielnica": "district",
    "zrodlo": "source",
    "lon": "lon",
    "lat": "lat",
}

OUTPUT_COLUMNS = [
    "h3_index",
    "district",
    "district_slug",
    "district_teryt",
    "source",
    "population",
    "population_2021",
    "population_kontur",
    "population_share_district",
    "area_km2",
    "area_src_km2",
    "hex_area_km2",
    "coverage_ratio",
    "is_clipped",
    "density_per_km2",
    "density_src_per_km2",
    "density_quintile",
    "low_population",
    "density_outlier",
    "lon",
    "lat",
    "hex_lon",
    "hex_lat",
    "geometry",
]
