"""Prognoza ludności klastrów H3 rok po roku (2025–2036).

Model dla klastra h w dzielnicy d, rok t:

    P_h(t) = P_h(2025) · (1 + dryf_d)^(t − 2025)                       zabudowa istniejąca
           + Σ_pozwolenia mieszkania · realizacja · OS_NA_MIESZKANIE · udział zasiedlenia(t)
           + kontynuacja: tempo pozwoleń z ostatnich lat po wyczerpaniu się rejestru

Kalibracja (calibrate) na spisach NSP 2011 → 2021, czyli na faktycznych mieszkańcach:
    zmiana_d = a · mieszkania_oddane_d(2011–2021) + b · ludność_d(2011)     (18 dzielnic)
  a = osoby na nowe mieszkanie, b = spadek ludności zabudowy istniejącej (mniejsze
  gospodarstwa, przeprowadzki do nowych mieszkań). Dryf dzielnicy = reszta modelu,
  ściągnięta w 3/4 do średniej miasta (DRIFT_SHRINK) — pojedyncze dzielnice są zaszumione.
  Realizacja = mieszkania oddane / mieszkania z pozwoleniami sprzed REALIZATION_LAG lat (GUS).

Dlaczego spisy, a nie roczne dane GUS: roczna ludność GUS opiera się na meldunkach —
w latach 2021–2025 oddano ~74 tys. mieszkań, a „ludność” wzrosła o ~4 tys. Najemcy bez
meldunku i migranci nie są tam widoczni, a dla transportu to realni pasażerowie.
Prognoza GUS (P4359) służy jako kontrola sumy miasta, nie jako cel kalibracji.

Wariant porównawczy A (forecast_top_down): przyrost z prognozy GUS rozłożony na klastry
proporcjonalnie do nowych mieszkań.

Uruchomienie:  .venv/bin/python mapping/prognoza_ludnosci.py   (notebook: przewidywania-ludnosci.ipynb)
Wynik:         dane/prognoza_ludnosci.csv (z dP_2031_norm, dP_2036_norm: przyrost znormalizowany do 0–1)
"""

import json
import os
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
HEX_PATH = ROOT / "dane" / "kontury_400m" / "warszawa_h3_400m_pelna.csv"
HIST_PATH = ROOT / "dane" / "historia_ludnosci.csv"
PERMITS_PATH = DATA_DIR / "gunb_permits.parquet"
GUS_CACHE = DATA_DIR / "gus_bdl_cache.json"
OUT = ROOT / "dane" / "prognoza_ludnosci.csv"
EXTERNAL_DIR = Path(os.environ.get("DANE_WARSZAWA", Path.home() / "Downloads" / "dane_warszawa"))

BASE_YEAR = 2025
YEARS = list(range(2025, 2037))
OCCUPANCY_LAG = {2: 0.25, 3: 0.50, 4: 0.25}   # lata od pozwolenia -> udział zasiedlenia
REALIZATION_LAG = 3
TREND_YEARS = range(2022, 2026)                 # pełne, kompletne lata rejestru
TREND_DECAY = 0.90                              # kontynuacja słabnie o 10% rocznie
DRIFT_SHRINK = 0.25                             # waga dryfu dzielnicy vs średnia miasta (LOO R² dzielnic niskie)
SCENARIOS = {                                    # mnożniki parametrów z kalibracji
    "niski": {"realizacja": 0.85, "os_na_mieszkanie": 0.85},
    "bazowy": {"realizacja": 1.00, "os_na_mieszkanie": 1.00},
    "wysoki": {"realizacja": 1.05, "os_na_mieszkanie": 1.15},
}

BDL = "https://bdl.stat.gov.pl/api/v1"
WARSAW_UNIT = "071412865000"
WARSAW_TOTAL = "M.st.Warszawa od 2002"
BDL_PERMIT_VARS = [747635, 747636, 747638, 747639]   # P3816: mieszkania z pozwoleniami (nieindywidualne)
BDL_COMPLETED_VAR = 748601                            # P3824: mieszkania oddane do użytkowania, ogółem
BDL_PROJECTION_SUBJECT = "P4359"                      # prognoza ludności 2023–2060 (NSP 2021)
HEADERS = {"User-Agent": "optymalna-warszawa-hackathon/1.0"}


# ---------- dane ----------

def _bdl(path: str, params) -> dict:
    cache = json.loads(GUS_CACHE.read_text()) if GUS_CACHE.exists() else {}
    key = path + "?" + "&".join(f"{k}={v}" for k, v in params)
    if key not in cache:
        r = requests.get(f"{BDL}/{path}", params=list(params) + [("format", "json")], headers=HEADERS, timeout=60)
        r.raise_for_status()
        cache[key] = r.json()
        GUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        GUS_CACHE.write_text(json.dumps(cache))
    return cache[key]


def gus_projection() -> pd.Series:
    """Prognoza GUS ludności m.st. Warszawy, interpolowana na każdy rok z YEARS."""
    ids, page = {}, 0
    while True:
        v = _bdl("variables", [("subject-id", BDL_PROJECTION_SUBJECT), ("page-size", 100), ("page", page)])
        ids |= {int(x["n1"].split()[-1]): x["id"] for x in v["results"] if x.get("n2") == x.get("n3") == "ogółem"}
        if "next" not in v.get("links", {}):
            break
        page += 1
    years = sorted(y for y in ids if y <= max(YEARS) + 5)
    data = _bdl(f"data/by-unit/{WARSAW_UNIT}", [("var-id", ids[y]) for y in years])
    by_id = {r["id"]: r["values"][0]["val"] for r in data["results"]}
    s = pd.Series({y: by_id[ids[y]] for y in years}, dtype=float)
    return s.reindex(range(min(years), max(years) + 1)).interpolate().reindex(YEARS)


def gus_permitted_flats() -> pd.Series:
    data = _bdl(f"data/by-unit/{WARSAW_UNIT}", [("var-id", v) for v in BDL_PERMIT_VARS])
    total = pd.Series(dtype=float)
    for r in data["results"]:
        total = total.add(pd.Series({int(v["year"]): v["val"] for v in r["values"]}), fill_value=0)
    return total


def gus_completed_flats() -> pd.DataFrame:
    """Mieszkania oddane do użytkowania: dzielnica × rok (+ wiersz WARSAW_TOTAL)."""
    data = _bdl(f"data/by-variable/{BDL_COMPLETED_VAR}",
                [("unit-parent-id", WARSAW_UNIT), ("unit-level", 6), ("page-size", 50)])
    rows = [(r["name"].replace(" - dzielnica", ""), int(v["year"]), v["val"])
            for r in data["results"] for v in r["values"]]
    return pd.DataFrame(rows, columns=["dzielnica", "rok", "n"]).pivot(index="dzielnica", columns="rok", values="n")


def district_population() -> pd.DataFrame:
    d = gpd.read_file(EXTERNAL_DIR / "warszawa_dzielnice.geojson")
    return d.set_index("dzielnica")[["ludnosc_gus_2021", "ludnosc_gus_2025"]]


def load_inputs() -> dict:
    hexes = pd.read_csv(HEX_PATH, encoding="utf-8-sig")
    hist = pd.read_csv(HIST_PATH, encoding="utf-8-sig")[["h3", "gus_2011", "gus_2021"]]
    hexes = hexes.merge(hist, on="h3", how="left")

    permits = gpd.read_parquet(PERMITS_PATH)
    permits["h3"] = [h3.latlng_to_cell(y, x, 8) for x, y in zip(permits.geometry.x, permits.geometry.y)]
    permits = permits[permits["h3"].isin(hexes["h3"])]
    # Rejestr z pierwszych lat jest niekompletny — waga roku = GUS / suma szacunków z rejestru.
    gus_permits = gus_permitted_flats()
    per_year = permits.groupby("rok")["n_flats"].sum()
    scale = (gus_permits / per_year).reindex(per_year.index).fillna(1.0).clip(lower=1.0, upper=2.0)
    permits["n_flats_adj"] = permits["n_flats"] * permits["rok"].map(scale)
    # Ostatni rok rejestru jest niepełny (dane do dnia pobrania) — przeliczamy na cały rok.
    last = permits["rok"].max()
    last_day = pd.to_datetime(permits.loc[permits["rok"] == last, "data_decyzji"]).max()
    year_frac = (last_day.dayofyear) / (366 if last_day.is_leap_year else 365)
    permits.loc[permits["rok"] == last, "n_flats_adj"] /= year_frac
    inputs = {"hexes": hexes, "permits": pd.DataFrame(permits.drop(columns="geometry")),
              "permit_scale": scale, "gus_permits": gus_permits, "completed": gus_completed_flats(),
              "gus_projection": gus_projection(), "districts": district_population()}
    inputs["calibration"] = calibrate(inputs)
    return inputs


# ---------- kalibracja ----------

def census_table(inputs: dict) -> pd.DataFrame:
    """Dzielnice: ludność NSP 2011 i 2021 (z klastrów) i mieszkania oddane między spisami."""
    c = inputs["hexes"].groupby("dzielnica")[["gus_2011", "gus_2021"]].sum()
    o = inputs["completed"]
    # Spisy na 31 marca: 3/4 roku 2011, lata 2012–2020, 1/4 roku 2021.
    c["oddane"] = o.loc[c.index, range(2012, 2021)].sum(axis=1) + 0.75 * o.loc[c.index, 2011] + 0.25 * o.loc[c.index, 2021]
    c["zmiana"] = c["gus_2021"] - c["gus_2011"]
    return c


def fit(c: pd.DataFrame) -> tuple[float, float]:
    X = np.c_[c["oddane"], c["gus_2011"]]
    (a, b), *_ = np.linalg.lstsq(X, c["zmiana"], rcond=None)
    return a, b


def calibrate(inputs: dict) -> dict:
    c = census_table(inputs)
    a, b = fit(c)
    pred = a * c["oddane"] + b * c["gus_2011"]
    # Leave-one-out: każda dzielnica prognozowana z parametrów dopasowanych bez niej.
    loo = pd.Series({d: (lambda ab: ab[0] * c.loc[d, "oddane"] + ab[1] * c.loc[d, "gus_2011"])(fit(c.drop(d)))
                     for d in c.index})
    ss = ((c["zmiana"] - c["zmiana"].mean()) ** 2).sum()
    city_drift = (1 + b) ** 0.1 - 1
    resid = ((c["zmiana"] - a * c["oddane"]) / c["gus_2011"] + 1) ** 0.1 - 1
    drift = DRIFT_SHRINK * resid + (1 - DRIFT_SHRINK) * city_drift

    o = inputs["completed"].loc[WARSAW_TOTAL]
    yrs = [y for y in range(2014, 2026) if y + REALIZATION_LAG in o.index and y in inputs["gus_permits"].index]
    realization = o.loc[[y + REALIZATION_LAG for y in yrs]].sum() / inputs["gus_permits"].loc[yrs].sum()

    c["pred"], c["pred_loo"], c["dryf_roczny"] = pred, loo, drift
    return {"os_na_mieszkanie": float(a), "dryf_miasta": float(city_drift), "dryf_dzielnic": drift,
            "realizacja": float(min(realization, 1.0)), "r2": float(1 - ((c["zmiana"] - pred) ** 2).sum() / ss),
            "r2_loo": float(1 - ((c["zmiana"] - loo) ** 2).sum() / ss), "tabela": c}


# ---------- składniki modelu ----------

def occupied_share(decision_year: np.ndarray, year: int, after: int) -> np.ndarray:
    """Jaka część mieszkań z pozwolenia zostaje zasiedlona w latach (after, year]."""
    out = np.zeros(len(decision_year))
    for lag, w in OCCUPANCY_LAG.items():
        occ = decision_year + lag
        out += w * ((occ > after) & (occ <= year))
    return out


def new_flats(permits: pd.DataFrame, hexes: pd.DataFrame, year: int, after: int) -> pd.Series:
    """Mieszkania (po skalowaniu rejestru) zasiedlone w klastrach w latach (after, year]."""
    w = occupied_share(permits["rok"].values, year, after)
    return (permits["n_flats_adj"] * w).groupby(permits["h3"]).sum().reindex(hexes["h3"]).fillna(0)


def trend_flats(permits: pd.DataFrame, hexes: pd.DataFrame, year: int) -> pd.Series:
    """Mieszkania z kontynuacji trendu zasiedlone do roku `year`.

    Po ostatnim roku rejestru zakładamy „wirtualne pozwolenia”: co rok tyle mieszkań, ile
    średnio w TREND_YEARS w danym klastrze, słabnące o TREND_DECAY rocznie, zasiedlane
    z tym samym opóźnieniem co prawdziwe pozwolenia."""
    last = int(permits["rok"].max())
    recent = permits[permits["rok"].isin(TREND_YEARS)]
    per_hex_year = recent.groupby("h3")["n_flats_adj"].sum() / len(TREND_YEARS)
    virtual_years = np.arange(last + 1, year + 1)
    if len(virtual_years) == 0:
        return pd.Series(0.0, index=hexes["h3"])
    weights = TREND_DECAY ** (virtual_years - last) * occupied_share(virtual_years, year, BASE_YEAR)
    return (per_hex_year * weights.sum()).reindex(hexes["h3"]).fillna(0)


# ---------- prognoza ----------

def forecast(inputs: dict, scenario: str = "bazowy") -> pd.DataFrame:
    """Ludność [klaster × rok] oraz składowe: zabudowa istniejąca i nowe budynki."""
    hexes, permits, cal = inputs["hexes"], inputs["permits"], inputs["calibration"]
    sc = SCENARIOS[scenario]
    ppl = cal["realizacja"] * sc["realizacja"] * cal["os_na_mieszkanie"] * sc["os_na_mieszkanie"]
    base = hexes.set_index("h3")["population"].astype(float)
    drift = hexes["dzielnica"].map(cal["dryf_dzielnic"]).fillna(cal["dryf_miasta"]).values
    cols = {}
    for t in YEARS:
        stock = base * (1 + drift) ** (t - BASE_YEAR)
        new = (new_flats(permits, hexes, t, BASE_YEAR) + trend_flats(permits, hexes, t)) * ppl
        cols[t] = stock + new.values
    return pd.DataFrame(cols, index=base.index)


def forecast_top_down(inputs: dict) -> pd.DataFrame:
    """Wariant porównawczy A: przyrost miasta z prognozy GUS rozłożony na klastry
    proporcjonalnie do mieszkań zasiedlanych do roku t; zabudowa istniejąca bez zmian."""
    hexes, permits = inputs["hexes"], inputs["permits"]
    base = hexes.set_index("h3")["population"].astype(float)
    gus = inputs["gus_projection"]
    cols = {}
    for t in YEARS:
        flats = new_flats(permits, hexes, t, BASE_YEAR) + trend_flats(permits, hexes, t)
        share = flats / flats.sum() if flats.sum() > 0 else flats
        cols[t] = base + share.values * (gus[t] - base.sum())
    return pd.DataFrame(cols, index=base.index)


NORM_QUANTILE = 0.99   # przyrost na tym percentylu (dla NORM_REF_COL) = 1
NORM_REF_COL = "dP_2036"


def normalize_growth(dp: pd.Series, ref: float) -> pd.Series:
    """Przyrost ludności -> 0..1: spadki = 0, potem √ΔP / √ref, powyżej ref = 1.

    Pierwiastek tłumi kilka rekordowych klastrów, ale zachowuje różnice między dużymi
    wzrostami. Wspólne `ref` dla wszystkich horyzontów, żeby wartości były porównywalne w czasie."""
    return (np.sqrt(dp.clip(lower=0)) / np.sqrt(ref)).clip(upper=1).round(4)


def build_output(inputs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    hexes = inputs["hexes"].set_index("h3")
    out = hexes[["dzielnica", "population"]].rename(columns={"population": "pop_2025"})
    totals = {}
    for sc in SCENARIOS:
        pop = forecast(inputs, sc)
        totals[sc] = pop.sum()
        suffix = "" if sc == "bazowy" else f"_{sc}"
        for t in YEARS[1:]:
            out[f"pop_{t}{suffix}"] = pop[t].round().astype(int)
    top_down = forecast_top_down(inputs)
    totals["wariant GUS (A)"] = top_down.sum()
    totals["prognoza GUS"] = inputs["gus_projection"]
    out["pop_2036_wariant_gus"] = top_down[YEARS[-1]].round().astype(int)
    permits = inputs["permits"]
    out["mieszkania_w_budowie"] = new_flats(permits, inputs["hexes"], 2100, BASE_YEAR).values.round().astype(int)
    out["dryf_roczny_proc"] = (100 * hexes["dzielnica"].map(inputs["calibration"]["dryf_dzielnic"])
                               .fillna(inputs["calibration"]["dryf_miasta"])).round(2).values
    out["dP_2031"] = out["pop_2031"] - out["pop_2025"]
    out["dP_2036"] = out["pop_2036"] - out["pop_2025"]
    ref = out[NORM_REF_COL].clip(lower=0).quantile(NORM_QUANTILE)
    for col in ("dP_2031", "dP_2036"):
        out[f"{col}_norm"] = normalize_growth(out[col], ref)
    return out.reset_index(), pd.DataFrame(totals).round().astype(int)


def main() -> None:
    inputs = load_inputs()
    cal = inputs["calibration"]
    print(f"== kalibracja na spisach 2011→2021: {cal['os_na_mieszkanie']:.2f} os./nowe mieszkanie, "
          f"dryf miasta {100 * cal['dryf_miasta']:.2f}%/rok, realizacja pozwoleń {cal['realizacja']:.2f}, "
          f"R² {cal['r2']:.2f} (leave-one-out {cal['r2_loo']:.2f})")
    out, totals = build_output(inputs)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("== suma miasta wg wariantu:")
    print(totals.loc[[2025, 2028, 2031, 2033, 2036]].to_string())
    print(f"== {len(out)} klastrów -> {OUT.relative_to(ROOT)}")
    print(out.groupby("dzielnica")[["pop_2025", "pop_2031", "pop_2036"]].sum()
          .assign(zmiana_proc=lambda d: (100 * (d.pop_2036 / d.pop_2025 - 1)).round(1))
          .sort_values("zmiana_proc").to_string())


if __name__ == "__main__":
    main()
