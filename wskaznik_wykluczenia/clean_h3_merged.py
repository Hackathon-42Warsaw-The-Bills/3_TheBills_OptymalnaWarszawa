"""Tworzy h3_merged_clean.csv: kopia h3_merged.csv bez wierszy,
w ktorych gestosc_os_km2, pracownicy_szac, przystanki_tramwajowe,
przystanki_autobusowe i czy_metro sa jednoczesnie rowne 0."""
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "h3_merged.csv"
DST = BASE / "h3_merged_clean.csv"
ZERO_COLS = ["gestosc_os_km2", "pracownicy_szac", "przystanki_tramwajowe",
             "przystanki_autobusowe", "czy_metro"]


def is_zero(value):
    try:
        return float(str(value).strip()) == 0.0
    except (ValueError, TypeError):
        return False


def main():
    with open(SRC, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if not all(is_zero(r[c]) for c in ZERO_COLS)]
    with open(DST, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(kept)
    print(f"wejscie: {len(rows)} | usunieto: {len(rows) - len(kept)} "
          f"| zapisano: {len(kept)} -> {DST.name}")


if __name__ == "__main__":
    main()
