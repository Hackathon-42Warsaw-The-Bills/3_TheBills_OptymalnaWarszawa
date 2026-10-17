import unicodedata

import h3
import pandas as pd
from shapely.geometry import Polygon

from config import CRS_METERS, DISTRICTS


class Report:
    def __init__(self):
        self.lines = []

    def add(self, label, count):
        self.lines.append((label, int(count)))

    def print(self):
        width = max(len(label) for label, _ in self.lines)
        for label, count in self.lines:
            print(f"{label:<{width}}  {count:>6}")


def tidy_text(value):
    if pd.isna(value):
        return pd.NA
    text = unicodedata.normalize("NFC", str(value))
    text = " ".join(text.split())
    text = "-".join(part.strip() for part in text.split("-"))
    return text or pd.NA


def without_polish_chars(name):
    name = name.replace("ł", "l").replace("Ł", "L")
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return ascii_name.lower().replace(" ", "_").replace("-", "_")


def official_district_name(value):
    if pd.isna(value):
        return pd.NA
    for name in DISTRICTS:
        if value.casefold() == name.casefold() or without_polish_chars(value) == without_polish_chars(name):
            return name
    return pd.NA


def to_number(column):
    text = column.str.strip().str.replace(",", ".", regex=False).str.replace(r"\s", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def hexagon_shape(h3_index):
    return Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(h3_index)])


def area_km2(shapes):
    return shapes.to_crs(CRS_METERS).area.values / 1_000_000
