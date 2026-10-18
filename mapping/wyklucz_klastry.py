"""Usuwa wskazane klastry H3 z wyników modelu wskaźnika wykluczenia (heatmapa + lista top 30).

Model (notebooki/colab_ww_hackathon.ipynb) liczy wskaźnik dla każdego klastra z mieszkańcami, także
dla takich, gdzie wynik nie ma sensu — np. teren lotniska z kilkunastoma mieszkańcami trafiał do top 30.
Tu klastry z WYKLUCZONE dostają status jak pozostałe spoza analizy (pred = rank = −1, top30 = 0); miejsca
w rankingu pozostałych klastrów są przeliczane, a lista top 30 uzupełniana kolejnymi klastrami z rankingu.
Skrypt można uruchamiać wielokrotnie (już wykluczonych nie rusza) — po każdym nowym runie modelu.

Wejście/wyjście (w miejscu): ML_DIR/heatmapa_wykluczenia.geojson, ML_DIR/lista_heksagonow_top30.csv
Potem: mapping/przystanki_top30.py i mapping/clastry_all.py (korzystają z listy top 30 i heatmapy).

Uruchomienie:  .venv/Scripts/python mapping/wyklucz_klastry.py
"""

import json
from pathlib import Path

import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "dane" / "wskaznik_wykluczenia" / "output_run3"
TOP_N = 30
STATUS = "poza analiza (wykluczony ręcznie)"

# h3 → powód
WYKLUCZONE = {
    "881f5226adfffff": "Włochy, lotnisko Chopina — 12 mieszkańców, wskaźnik bez sensu",
}


def main() -> None:
    heat_path, top_path = ML_DIR / "heatmapa_wykluczenia.geojson", ML_DIR / "lista_heksagonow_top30.csv"
    heat = json.loads(heat_path.read_text(encoding="utf-8"))
    top = pd.read_csv(top_path)
    props = {f["properties"]["h3"]: f for f in heat["features"]}

    for h3, why in WYKLUCZONE.items():
        f = props.get(h3)
        if f is None:
            print(f"{h3}: brak w heatmapie — pomijam")
            continue
        p = f["properties"]
        print(f"{h3} ({why}): pred {p['pred']} → −1, było miejsce {p['rank']}")
        p.update(pred=-1.0, rank=-1, top30=0, status=STATUS)

    # ranking od nowa wśród klastrów z analizy (malejąco po pred, remisy jak w poprzedniej kolejności)
    ok = sorted((f for f in heat["features"] if f["properties"]["pred"] >= 0),
                key=lambda f: (-f["properties"]["pred"], f["properties"]["rank"]))
    for i, f in enumerate(ok, 1):
        f["properties"]["rank"] = i
        f["properties"]["top30"] = int(i <= TOP_N)

    # lista top 30: dotychczasowe wiersze (z etykietą modelu) + uzupełnienie z heatmapy
    old = top.set_index("h3")
    rows = []
    for f in ok[:TOP_N]:
        p = f["properties"]
        if p["h3"] in old.index:
            r = old.loc[p["h3"]].to_dict()
        else:
            c = shapely.geometry.shape(f["geometry"]).centroid
            r = {"dzielnica": p["dzielnica"], "lon": round(c.x, 6), "lat": round(c.y, 6), "etykieta_final": None,
                 "pred_wykluczenie": p["pred"]}
            print(f"top 30 uzupełnione: {p['h3']} ({p['dzielnica']}), pred {p['pred']}")
        r.update(h3=p["h3"], rank=p["rank"])
        rows.append(r)
    out = pd.DataFrame(rows)[top.columns]

    heat_path.write_text(json.dumps(heat, ensure_ascii=False), encoding="utf-8")
    out.to_csv(top_path, index=False, encoding="utf-8")
    print(f"== {heat_path.relative_to(ROOT)}: {len(ok)} klastrów w analizie; {top_path.relative_to(ROOT)}: {len(out)} wierszy")


if __name__ == "__main__":
    main()
