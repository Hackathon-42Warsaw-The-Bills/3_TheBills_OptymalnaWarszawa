# 3_TheBills_OptymalnaWarszawa
## Mapa Warszawy (graf odcinków)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python mapping/build_graph.py
```

Skrypt pobiera snapshot OSM województwa mazowieckiego z Geofabrik, wycina granice
Warszawy i buduje dwa grafy: ulice przejezdne dla autobusów (`roads`) oraz torowiska
tramwajowe (`trams`). Wynik trafia do `data/` (poza repo):

- `*_edges.parquet` / `*_nodes.parquet` — odcinki skrzyżowanie→skrzyżowanie z atrybutami OSM
- `*.graphml` — grafy do wczytania przez `osmnx.load_graphml`
- `preview.html` — podgląd na mapie

Klucz odcinka to `edge_id = "<u>_<v>_<key>"` (ID węzłów OSM). Do niego dowiązujemy wagi.
Aby odświeżyć mapę, usuń `data/osm/` i uruchom skrypt ponownie.

## Ludność na odcinkach ulic

```bash
.venv/bin/python mapping/population.py   # po build_graph.py; ~10 s
```

Wejście: zbiór `dane_warszawa` (domyślnie `~/Downloads/dane_warszawa`, zmienna `DANE_WARSZAWA`).
Wynik: `data/segments_population.parquet` — jeden wiersz na odcinek (oba kierunki scalone, `edge_ids`).

Ludność z siatki GUS 1 km (NSP 2021) jest rozkładana wewnątrz kwadratów według gęstości Kontur H3,
cięta granicami dzielnic, kalibrowana do ludności dzielnic GUS 2025 i przypisywana do odcinków
proporcjonalnie do długości. Ekspresówki i autostrady nie dostają mieszkańców.

| Kolumna | Znaczenie |
|---|---|
| `pop_2021`, `pop_2025` | mieszkańcy przypisani do odcinka |
| `pop_2028_proj`, `pop_2030_proj` | prognoza wg tempa zmian dzielnicy 2021→2025 |
| `pop_per_100m` | gęstość zamieszkania wzdłuż odcinka |
| `pop_400m` | mieszkańcy w promieniu 400 m od środka odcinka |
| `dist_to_stop_m`, `stops_300m`, `served` | dostęp do słupków ZTM (`served` = słupek ≤ 400 m) |
| `unserved_pop_400m[_2028/_2030]` | mieszkańcy w promieniu 400 m bez słupka w zasięgu — sygnał braku przystanku |
