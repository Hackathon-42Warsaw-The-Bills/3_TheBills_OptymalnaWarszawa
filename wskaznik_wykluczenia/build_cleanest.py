"""Buduje h3_merged_cleanest.csv: baza h3_merged_clean.csv + pakiet
podstawowy i rozszerzony z innych plikow w dane/ (join 1:1 po h3),
plus wsiadajacy_h_2025 zagregowani ze slupkow do heksagonow."""
import csv
from collections import defaultdict
from pathlib import Path

try:
    import h3
except ImportError:
    h3 = None

DANE = Path(__file__).resolve().parent.parent
WK = DANE / "wskaznik_wykluczenia"
H3RES = 8


def load(path, key="h3"):
    with open(DANE / path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r[key]: r for r in rows}


def main():
    base = load(WK.relative_to(DANE) / "h3_merged_clean.csv")
    cechy = load("cechy_klastrow.csv")
    kolej = load("kolej/klastry_kolej.csv")
    vet = load("veturilo_w_klastrach.csv")
    pod = load("symulacja/klastry_podroze.csv")
    pot = load("potrzeba_komunikacji.csv")
    prog = load("prognoza_ludnosci.csv")

    CECHY_COLS = ["kursy_dzien_autobus", "kursy_dzien_tramwaj", "kursy_dzien_metro",
                 "kursy_dzien_kolej", "kursy_szczyt_bus_tram", "linie_ztm",
                 "v_autobus_szczyt_kmh", "udzial_65_", "odl_centrum_km",
                 "infra_handel", "infra_kult", "infra_kultura", "infra_park",
                 "infra_szkola", "infra_szpital", "infra_targowisko", "infra_urzad"]
    NEW_COLS = (CECHY_COLS + ["dist_metro_m", "dist_kolej_m", "metro_plan_rok"]
                + ["odl_do_veturilo_m", "ma_veturilo"]
                + ["czas_do_centrum_min_2025"] + ["roznica_dzis"]
                + ["pop_2031"] + ["wsiadajacy_h_2025"])

    # agregacja wsiadan ze slupkow do heksagonow
    wsiad = defaultdict(float)
    nomap = 0
    with open(DANE / "symulacja/przystanki_wsiadania.csv", encoding="utf-8-sig") as f:
        wsiad_rows = list(csv.DictReader(f))
    with open(DANE / "linie_dostepne_na_przystanku.csv", encoding="utf-8-sig") as f:
        coords = {r["stop_id"]: (float(r["stop_lat"]), float(r["stop_lon"]))
                  for r in csv.DictReader(f)}
    for r in wsiad_rows:
        sid = r["stop_id"]
        if sid not in coords:
            nomap += 1
            continue
        lat, lon = coords[sid]
        wsiad[h3.latlng_to_cell(lat, lon, H3RES)] += float(r["wsiadajacy_h_2025"] or 0)
    print(f"slupki z wsiadaniami: {len(wsiad_rows)} | bez wspolrzednych: {nomap} "
          f"| heksow z wsiadaniami: {len(wsiad)}")

    ZERO_FILL = set(CECHY_COLS + ["odl_do_veturilo_m", "ma_veturilo",
                                  "czas_do_centrum_min_2025", "roznica_dzis",
                                  "pop_2031", "wsiadajacy_h_2025"])
    miss = defaultdict(int)
    out = []
    for hx, r in base.items():
        nr = dict(r)
        c = cechy.get(hx, {})
        k = kolej.get(hx, {})
        v = vet.get(hx, {})
        p = pod.get(hx, {})
        pt = pot.get(hx, {})
        pg = prog.get(hx, {})
        vals = ({col: c.get(col, "") for col in CECHY_COLS}
                | {"dist_metro_m": k.get("dist_metro_m", ""),
                   "dist_kolej_m": k.get("dist_kolej_m", ""),
                   "metro_plan_rok": k.get("metro_plan_rok", ""),
                   "odl_do_veturilo_m": v.get("odl_do_veturilo_m", ""),
                   "ma_veturilo": v.get("ma_veturilo", ""),
                   "czas_do_centrum_min_2025": p.get("czas_do_centrum_min_2025", ""),
                   "roznica_dzis": pt.get("roznica_dzis", ""),
                   "pop_2031": pg.get("pop_2031", ""),
                   "wsiadajacy_h_2025": round(wsiad.get(hx, 0.0), 1)})
        for col, val in vals.items():
            if val == "" or val is None:
                miss[col] += 1
                nr[col] = 0 if col in ZERO_FILL else ""
            else:
                nr[col] = val
        out.append(nr)

    dst = WK / "h3_merged_cleanest.csv"
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(base[next(iter(base))].keys()) + NEW_COLS)
        w.writeheader()
        w.writerows(out)
    print(f"wierszy: {len(out)} | zapisano: {dst.name}")
    print("braki (wypelnione 0/puste):",
          {c: n for c, n in sorted(miss.items()) if n} or "brak")


if __name__ == "__main__":
    main()
