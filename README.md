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
.venv/bin/python mapping/veturilo.py           # -> dane/infrastruktura/veturilo_stacje.csv, veturilo_w_klastrach.csv (GBFS Nextbike)
```

Po zmianie samego grafu ulic (np. filtra przejezdności dla autobusu) wystarczy:

```bash
.venv/bin/python mapping/build_graph.py
.venv/bin/python mapping/export_symulator.py --tylko-graf   # podmienia tylko "graph" w symulator_dane.json
```

## Dane (`dane/`)

| Folder | Co zawiera |
|---|---|
| `klastry/` | siatka H3 (`kontury_400m/`, `kontury_400m_clean/`), cechy klastrów, tabela zbiorcza `clastry_all.csv` ze słownikiem kolumn |
| `ludnosc/` | historia ludności (spisy GUS, GHSL), prognoza 2025–2036, współczynnik przyrostu |
| `komunikacja/` | przystanki ZTM z liniami, trasy linii, potrzeba komunikacji, ranking linii |
| `kolej/` | stacje i linie metra/SKM/KM/WKD, metro w klastrach, dane o pasażerach metra |
| `infrastruktura/` | cele podróży (szkoły, szpitale, handel…), miejsca pożytku publicznego, stacje Veturilo |
| `przystanki/` | propozycje nowych przystanków (cała Warszawa) i przystanki w 30 klastrach priorytetowych |
| `symulacja/` | symulacja pasażerów w szczycie: obciążenie odcinków i linii, wsiadający, walidacja metra |
| `wskaznik_wykluczenia/` | model wskaźnika wykluczenia: dane wejściowe i wyniki przebiegów (`output_run3/` używa aplikacja) |
| `dokumentacja/` | źródła danych zewnętrznych i pochodzenie informacji w panelu klastra |

## Notebooki (`notebooki/`)

Pierwsza komórka kodu każdego notebooka przechodzi do katalogu głównego repo, więc ścieżki `dane/...` i import
modułów z `mapping/` działają niezależnie od tego, z którego katalogu uruchomiono Jupytera.
`colab_ww_hackathon.ipynb` (model wskaźnika wykluczenia) działa w Google Colab i pobiera dane z GitHuba.
