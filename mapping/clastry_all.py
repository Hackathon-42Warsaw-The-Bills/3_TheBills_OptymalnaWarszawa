"""Jedna tabela ze wszystkim, co wiemy o klastrach H3 (848 wierszy, indeks h3).

Łączy wszystkie pliki z dane/ z kolumną h3 (jeden wiersz na klaster), a z plików z wieloma wierszami
na klaster (przystanki top 30, przystanki ZTM, wsiadający) robi podsumowania. Wskaźnik wykluczenia
komunikacyjnego (pred_wykluczenie, rank_wykluczenie, top30) pochodzi z ML_DIR/heatmapa_wykluczenia.geojson. Same statystyki: listy
(np. linie, stacje) zamienione na liczbę elementów, tak/nie na 1/0; z tekstu zostaje tylko dzielnica.
Kolumna o tej samej nazwie w kilku plikach: jeśli wartości są takie same — zostaje jedna, jeśli różne —
druga dostaje przyrostek ze źródłem (np. population__cechy). Brak obiektów w klastrze = 0 (liczniki).

Wynik:
  dane/klastry/clastry_all.csv          — h3 + wszystkie kolumny
  dane/klastry/clastry_all_kolumny.csv  — słownik: kolumna -> plik źródłowy

Uruchomienie (po skryptach, które tworzą pliki źródłowe):  .venv/bin/python mapping/clastry_all.py
"""

from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DANE = ROOT / "dane"
ML_DIR = DANE / "wskaznik_wykluczenia" / "output_run3"   # wyniki modelu wskaźnika wykluczenia (heatmapa + lista top 30)
OUT, OUT_COLS = DANE / "klastry" / "clastry_all.csv", DANE / "klastry" / "clastry_all_kolumny.csv"

# (plik, krótka nazwa źródła, kolumny do pominięcia, czy brak wiersza = 0)
TABLES = [
    ("klastry/kontury_400m/warszawa_h3_400m_pelna.csv", "kontury", [], False),
    ("klastry/kontury_400m_clean/warszawa_h3_400m_pelna_clean.csv", "kontury_clean", ["district", "district_slug"], False),
    ("ludnosc/historia_ludnosci.csv", "historia", [], False),
    ("ludnosc/prognoza_ludnosci.csv", "prognoza", [], False),
    ("ludnosc/wspolczynnik_przyrostu_ludnosci.csv", "wspolczynnik", [], False),
    ("kolej/klastry_kolej.csv", "kolej", [], False),
    ("klastry/cechy_klastrow.csv", "cechy", [], False),
    ("infrastruktura/infrastruktura_w_klastrach.csv", "infrastruktura", [], True),
    ("infrastruktura/veturilo_w_klastrach.csv", "veturilo", [], False),
    ("komunikacja/potrzeba_komunikacji.csv", "potrzeba", [], False),
    ("symulacja/klastry_podroze.csv", "symulacja", [], False),
]


def read(path: str) -> pd.DataFrame:
    d = pd.read_csv(DANE / path, encoding="utf-8-sig", low_memory=False)
    return d.rename(columns={"h3_index": "h3"})


def same(a: pd.Series, b: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return bool(np.allclose(a.astype(float), b.astype(float), rtol=1e-6, atol=1e-6, equal_nan=True))
    return a.fillna("").astype(str).equals(b.fillna("").astype(str))


def derived() -> list[tuple[str, pd.DataFrame]]:
    """Podsumowania z plików, w których klaster ma wiele wierszy."""
    out = []
    # wskaźnik wykluczenia komunikacyjnego (model MLP) dla wszystkich klastrów — z heatmapy (gałąź main)
    heat = gpd.read_file(ML_DIR / "heatmapa_wykluczenia.geojson").set_index("h3")
    top = read(ML_DIR / "lista_heksagonow_top30.csv").set_index("h3")
    analiza = heat["pred"] >= 0   # pred = rank = −1: klaster poza analizą modelu → brak wartości
    out.append(("wykluczenie", pd.DataFrame({"pred_wykluczenie": heat["pred"].where(analiza), "rank_wykluczenie": heat["rank"].where(analiza),
                                             "top30": heat["top30"], "top30_etykieta": top["etykieta_final"]})))

    p = read("przystanki/przystanki_top30.csv")
    p = p[p["nr"] > 0]
    best = p.sort_values("nr").drop_duplicates("h3").set_index("h3")
    agg = p.groupby("h3").agg(nowe_przystanki_liczba=("nr", "size"), nowe_przystanki_obsluzeni=("nowo_obsluzeni", "sum"))
    agg["nowy_przystanek_lon"], agg["nowy_przystanek_lat"] = best["lon"], best["lat"]
    agg["nowy_przystanek_ocena"] = best["ocena"]
    agg["nowy_przystanek_odl_ztm_m"] = best["odl_do_przystanku_ztm_m"]
    pl = read("przystanki/przystanki_top30_na_liniach.csv")
    agg["linie_ztm_do_dolozenia_przystanku"] = pl.groupby("h3")["linia"].nunique()
    out.append(("przystanki_top30", agg))

    # wsiadający w porannym szczycie (symulacja) — przystanki przypisane do klastra po położeniu
    st = read("komunikacja/linie_dostepne_na_przystanku.csv").drop_duplicates("stop_id")
    st["h3"] = [h3.latlng_to_cell(la, lo, 8) for la, lo in zip(st["stop_lat"], st["stop_lon"])]
    w = read("symulacja/przystanki_wsiadania.csv").merge(st[["stop_id", "h3"]], on="stop_id")
    lines = read("komunikacja/linie_dostepne_na_przystanku.csv").astype({"linia": str})
    lines["h3"] = lines["stop_id"].map(st.set_index("stop_id")["h3"])
    wagg = w.groupby("h3")[["wsiadajacy_h_2025", "wsiadajacy_h_2036"]].sum().round()
    wagg["slupki_ztm"] = st.groupby("h3").size()
    wagg["linie_ztm_na_przystankach"] = lines.groupby("h3")["linia"].nunique()
    out.append(("symulacja_wsiadania", wagg))
    return out


COUNTS_ZERO = ["nowe_przystanki_liczba", "nowe_przystanki_obsluzeni", "linie_ztm_do_dolozenia_przystanku",
               "wsiadajacy_h_2025", "wsiadajacy_h_2036", "slupki_ztm", "linie_ztm_na_przystankach"]
TEXT_KEEP = {"dzielnica"}
LIST_COLS = {"stacje_metro": "stacje_metro_liczba", "stacje_kolej": "stacje_kolej_liczba", "linie": "linie_kolej_metro_liczba"}
LIST_SEP = r"\s*[,;|]\s*"


def to_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Same liczby: listy -> liczba elementów, bool -> 0/1, tekst opisowy (źródło, ulica…) usunięty."""
    df = df.copy()
    for c in COUNTS_ZERO:
        if c in df:
            df[c] = df[c].fillna(0)
    for c in list(df.columns):
        s = df[c]
        if c in TEXT_KEEP or pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_bool_dtype(s) or set(s.dropna().astype(str).unique()) <= {"True", "False"}:
            df[c] = s.map({True: 1, False: 0, "True": 1, "False": 0})
            continue
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() == s.notna().sum():
            df[c] = num
        elif c in LIST_COLS:   # lista, np. "M1, M2" -> 2
            df[LIST_COLS[c]] = s.fillna("").astype(str).str.split(LIST_SEP).map(lambda xs: len([x for x in xs if x]))
            df = df.drop(columns=c)
        else:
            df = df.drop(columns=c)
    return df


def main() -> None:
    base = read(TABLES[0][0])
    df = base.set_index("h3")
    src = {c: TABLES[0][0] for c in df.columns}
    conflicts, cols = [], {}
    tables = [(f, name, skip, zero, read(f).set_index("h3").drop(columns=skip)) for f, name, skip, zero in TABLES[1:]]
    tables += [(f"(wyliczone) {name}", name, [], False, d) for name, d in derived()]
    for f, name, _, zero, d in tables:
        d = d[~d.index.duplicated()].reindex(df.index)
        if zero:
            d = d.fillna(0)
        for c in d.columns:
            prev = df[c] if c in df.columns else cols.get(c)
            if prev is not None:
                if same(prev, d[c]):
                    continue
                conflicts.append(c)
                new = f"{c}__{name}"
            else:
                new = c
            cols[new] = d[c]
            src[new] = f
    df = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)
    src = {c: src.get(c, src.get(next((k for k, v in LIST_COLS.items() if v == c), ""))) for c in to_stats(df).columns}
    df = to_stats(df).reset_index()
    df.to_csv(OUT, index=False, encoding="utf-8")
    pd.DataFrame({"kolumna": list(src), "zrodlo": list(src.values())}).to_csv(OUT_COLS, index=False, encoding="utf-8")
    print(f"== {OUT.relative_to(ROOT)}: {len(df)} klastrów × {df.shape[1]} kolumn; słownik: {OUT_COLS.relative_to(ROOT)}")
    if conflicts:
        print(f"   różne wartości pod tą samą nazwą (zostawione obie, z przyrostkiem źródła): {sorted(set(conflicts))}")


if __name__ == "__main__":
    main()
