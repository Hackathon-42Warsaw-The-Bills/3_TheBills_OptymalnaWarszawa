# Przedłużanie i odgałęzianie istniejącej linii autobusowej

## Cel

W zakładce „Projektuj” można wziąć istniejącą linię autobusową ZTM i poprowadzić ją nowym odcinkiem
od dowolnego jej przystanku (końcówka = przedłużenie, przystanek w środku = odgałęzienie),
używając istniejących narzędzi „Rysuj trasę” i „Przystanki”. Ocena pokazuje **tylko przyrost**.

## Przebieg

1. Czwarte narzędzie **„Przedłuż linię”** (`data-mode="extend"`).
2. Klik na mapie ≤ 14 px od przebiegu linii autobusowej (`D.routes`, `typ: "autobus"`):
   jedna linia → wybór od razu; kilka → lista przy kursorze (numer · nazwa), oba kierunki tej samej linii to jedna pozycja.
   Klik w miejscu bez linii autobusowej → komunikat w podpowiedzi.
3. Wybrana linia: przebieg (oba kierunki) wyróżniony, jej przystanki (`D.stops.ztm`, linia w liście linii) jako punkty.
4. Klik ≤ 14 px od przystanku tej linii → start: trasa zaczyna się w węźle grafu najbliższym przystankowi,
   tryb przełącza się na „Rysuj trasę”. Przystanek startowy nie jest nowym przystankiem.
5. „Kursy co [min]” = `clamp(round(1140 / kursy_w_kierunku), 2, 60)` (doba ≈ 19 h kursowania), do zmiany.

## Ocena (przyrost)

- Nagłówek: „Przedłużenie linii 523 od przystanku X” / „Odgałęzienie linii 523 od przystanku X” (końcówka = pierwszy lub ostatni przystanek przebiegu w którymkolwiek kierunku).
- Popyt, pasażerowie, długość, czas, korki, przesiadki, cele podróży: jak dziś, liczone dla nowego odcinka i nowych przystanków.
- Pojazdy do dołożenia: `ceil(2 · czas_przejazdu_odcinka / (kursy_co · 60))` — bez postoju na pętli (pętla już istnieje).
- Znacznik „Przedłużenie 523 · ×” — „×” lub „Usuń linię” kończy tryb przedłużenia.

## Dane i integracje

- Stan: `EXT = { ref, nazwa, stop: {name, nr, lon, lat}, end: boolean }` albo `null`.
- JSON: `currentProposal()` dodaje `przedluzenie: { linia, przystanek: [lon, lat], nazwa_przystanku }`; `loadProposal()` odtwarza `EXT`.
- Asystent (`describe`): pole `przedluzenie_linii`. Symulacja: jak nowa linia (bez zmian).
- Tylko linie autobusowe (graf ulic jest autobusowy).

## Testy

Wybór linii (jedna / kilka), wybór przystanku (końcówka, środek), start trasy w przystanku, domyślna częstotliwość,
ocena bez postoju na pętli, JSON tam i z powrotem, anulowanie, regresja (`suite.js`, `tabs.js`, `simreset.js`).
