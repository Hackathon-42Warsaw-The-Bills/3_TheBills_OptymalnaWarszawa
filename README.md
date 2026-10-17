# 3_TheBills_OptymalnaWarszawa

## Aplikacja: Symulator linii Warszawa 2036

`index.html` w katalogu głównym — mapa prognozy ludności klastrów 2025–2036, piaskownica do rysowania
linii autobusowych z oceną na żywo (nowy popyt, korki, tabor, przesiadki), linie ZTM z filtrem.
Dane aplikacji: `frontend/symulator_dane.json` i `frontend/warszawa.json`.

**GitHub Pages:** Settings → Pages → *Deploy from a branch* → gałąź, folder `/ (root)`.
Wszystko działa statycznie oprócz pojazdów na żywo (API ZTM wymaga tokenu i nie pozwala na zapytania z przeglądarki).

**Lokalnie (z pojazdami na żywo):**

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # raz
echo "ZTM_API_KEY=token-z-dane.um.warszawa.pl" > .env                 # .env jest w .gitignore
.venv/bin/python frontend/serwer.py                                   # http://localhost:8765/
```

`http://localhost:8765/#przyklad` otwiera aplikację z przykładową linią.

**Odświeżenie danych aplikacji** (po zmianach w `dane/` lub prognozie):

```bash
.venv/bin/python mapping/build_graph.py        # graf ulic z OSM (raz, ~5 min)
.venv/bin/python mapping/gtfs_predkosci.py     # prędkości autobusów z rozkładu ZTM
.venv/bin/python mapping/prognoza_ludnosci.py  # prognoza ludności klastrów
.venv/bin/python mapping/export_symulator.py   # -> frontend/symulator_dane.json
.venv/bin/python mapping/veturilo.py           # -> dane/veturilo_stacje.csv, veturilo_w_klastrach.csv (GBFS Nextbike)
```

Po zmianie samego grafu ulic (np. filtra przejezdności dla autobusu) wystarczy:

```bash
.venv/bin/python mapping/build_graph.py
.venv/bin/python mapping/export_symulator.py --tylko-graf   # podmienia tylko "graph" w symulator_dane.json
```
