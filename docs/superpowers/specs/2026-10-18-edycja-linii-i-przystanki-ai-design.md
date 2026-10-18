# Przystanki AI i tryb edycji istniejącej linii

Zastępuje specyfikację `2026-10-18-przedluzanie-linii-design.md` (tryb „Przedłuż linię” → „Edytuj linię”).

## 1. Przystanki AI (`dane/propozycje_przystankow.geojson`, `mapping/propozycje_przystankow.py`)

- Zastępują propozycje z `przystanki_top30.csv` (warstwa `prop-st`, romb z numerem = `priorytet` 1–3).
- Nazwa w UI: **„Przystanki AI”** (przełącznik w „Problem” i „Mapa”, panel klastra).
- Klikalna lista w zakładce „Problem” (jak lista top 30): sortowanie po priorytecie, potem `zysk_dostepnosci` malejąco;
  pozycja: romb z priorytetem, `nazwa_robocza`, dzielnica, „+zysk os.”, `id`. Klik → przybliżenie do punktu i szczegóły:
  `rodzaj`, `uzasadnienie`, przycisk **„Edytuj linię X”** dla pierwszej linii z `linia_rekomendowana`.
- Podpowiedź na mapie: id, nazwa, priorytet, etap, zysk, dojście dziś → po, proponowana obsługa.
- „Najlepsze miejsca na istniejących liniach” (`przystanki_top30_na_liniach.csv`) bez zmian.

## 2. Tryb „Edytuj linię” (`data-mode="edit"`)

**Wczytanie linii:** klik ≤ 14 px od przebiegu linii autobusowej (kilka linii → lista). Kierunek = przebieg bliższy kliknięciu.
Przebieg dopasowany do grafu: punkty co 150 m, odcinek ≤ 30 m o zgodnym kierunku (cos ≥ 0,5), łańcuch najkrótszych
ścieżek; punkt pomijany, gdy dojazd > 2 × odległość w linii prostej + 250 m. Jakość = |długość w grafie / długość przebiegu − 1|;
> 10% → ostrzeżenie „przebieg odtworzony w przybliżeniu”. Przystanki: słupki linii ≤ 35 m od przebiegu, jeden na nazwę,
w kolejności wzdłuż przebiegu. „Kursy co” = `clamp(round(1140 / kursy), 2, 60)`.

**Stan:** `LE = { ref, nazwa, kier, kursy, quality, poly, path: [edge], old: [stop + removed], added: [stop], draft, base }`.
Piaskownica jest zsynchronizowana: `route.legs = [path]`, `stops = aktywne stare (≤ 40 m od trasy, nieusunięte) + dodane`.

**Menu edycji** (w miejscu narzędzi): tytuł linii, „Przedłuż / objazd”, „Przystanki”, „Podgląd”; „Zakończ odcinek”,
„Cofnij”, „Przywróć oryginał”, „Zamknij edycję”.

- **Przedłuż / objazd:** klik w przystanek linii = start odcinka. Pierwszy aktywny przystanek → przedłużenie od początku
  (dopisywane przed trasą), ostatni → przedłużenie od końca (dopisywane za trasą), oba stosowane na bieżąco.
  Przystanek w środku → objazd: podgląd przerywaną linią; klik w przystanek tej linii dalej w kierunku jazdy zamyka objazd
  i zastępuje fragment trasy między przystankami (przystanki na pominiętym fragmencie wypadają). „Zakończ odcinek”:
  kończy przedłużenie, porzuca niezamknięty objazd.
- **Przystanki:** klik ≤ 40 m od istniejącego przystanku linii → usuń / przywróć; klik ≤ 40 m od dodanego → usuń;
  inaczej nowy przystanek przy trasie (jak dotąd, ≤ 80 m).
- **Cofnij:** migawka stanu przed każdą zmianą. **Przywróć oryginał:** stan z wczytania.
- **Zamknij edycję** / „Usuń linię” / „Resetuj”: koniec trybu, czysta piaskownica.

**Mapa:** oryginalny przebieg zawsze pogrubiony (`ext-line`), przystanki linii (aktywne / usunięte / pominięte),
podgląd objazdu (przerywany), bieżąca trasa jak dotąd (kolory korków).

**Ocena:** cała linia po zmianach + tabela Przed / Po / Zmiana: długość, czas w szczycie, przystanki, mieszkańcy w zasięgu,
nowy popyt, pojazdy w szczycie, mieszkańcy w zasięgu na km (efektywność).

**JSON:** `edycja_linii: { linia, kierunek, usuniete: ["nazwa nr"], dodane: [[lon, lat]] }` + `krawedzie` (cała trasa).
Wczytanie odtwarza linię, trasę z krawędzi, usunięte i dodane przystanki. Asystent: `edycja_linii` z porównaniem.

## Testy

Przystanki AI (lista, klik, szczegóły, „Edytuj linię X”), wczytanie linii (jedna / kilka), pogrubiony oryginał,
przedłużenie od końca, objazd zamykany przystankiem, usuwanie / przywracanie przystanku, cofnij, przywróć oryginał,
tabela przed/po, JSON tam i z powrotem, zamknięcie edycji; regresja (`suite`, `tabs`, `simreset`, `prop`).
