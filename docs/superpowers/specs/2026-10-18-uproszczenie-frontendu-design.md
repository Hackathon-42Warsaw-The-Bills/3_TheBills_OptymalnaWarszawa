# Uproszczenie frontendu symulatora — etap 1 (UI w jednym pliku)

## Cel

`index.html` ma pozostać w pełni funkcjonalny, ale być czytelniejszy dla **jury na prezentacji**:
jasna ścieżka „problem → zaprojektuj linię → ocena”, krótki panel zamiast jednej kolumny ~2900 px
z 12 sekcjami. Etap 2 (później, osobna specyfikacja): rozbicie pliku na CSS i moduły JS.

## Zakres

- Przebudowa panelu bocznego na zakładki, skrócenie tekstów, porządek w kodzie **w obrębie `index.html`**.
- Bez zmian: mapa i nakładki po prawej (lista dzielnic, przyciski powiększania, panel klastra),
  logika obliczeń (trasowanie, ocena, symulacja, model pasażerów), dane, `linie.html`, `frontend/index.html`.
- Żadna funkcja nie znika — schowane funkcje są w zwijanych sekcjach.

## Układ panelu

**Nagłówek (przyklejony u góry panelu):** tytuł + przycisk motywu, jedno zdanie podtytułu, link do rankingu linii,
rok prognozy + scenariusz (wpływają na wszystkie zakładki), pasek zakładek.

| Zakładka | Zawartość |
|---|---|
| **Problem** (domyślna) | warstwa klastrów (Mieszkańcy / Przyrost / Podróże ZTM / Cele) z legendą; priorytety modelu: top 30, proponowane przystanki, miejsca na liniach, przycisk heatmapy |
| **Projektuj** | tryby Przeglądaj / Rysuj trasę / Przystanki + Cofnij / Usuń linię / Przykładowa linia / Resetuj; ocena na żywo; asystent; zwinięte: „Parametry linii”, „Dla zespołu: wymiana JSON z algorytmem” |
| **Mapa** | filtr linii, Zaznacz/Odznacz wszystko, grupy: Sieć, Planowane metro, Cele podróży; zwinięte „O danych” (dotychczasowa stopka) |
| **Symulacja** | zegar, Start/Pauza, prędkość; zwinięte: „Model pasażerów (szczyt 7–8)” (obciążenie linii, wsiadający), „Pojazdy na żywo” |

## Zachowanie

- Ostatnio otwarta zakładka pamiętana w `localStorage` (w try/catch; bez dostępu — zakładka „Problem”).
- `#przyklad` otwiera zakładkę „Projektuj” z przykładową linią; `#linia-…` otwiera „Mapa” z filtrem.
- Przełączenie trybu na Rysuj trasę / Przystanki (przyciskiem lub skrótem) pokazuje zakładkę „Projektuj”.
- Zakładki to przyciski z `role="tab"`, `aria-selected`, obsługa strzałek ←/→; panele `role="tabpanel"`.
- Wszystkie identyfikatory elementów (`id`) używane w JS zostają bez zmian — przenoszone są całe bloki HTML.

## Treść

- „Prawdopodobieństwo wykluczenia” → **„Wskaźnik wykluczenia”** (lista top 30, panel klastra, heatmapa, odznaki),
  z jednym zdaniem: wskaźnik 0–100 z modelu na podstawie gęstości, miejsc pracy, przyrostu i dostępu do przystanków.
- Długie opisy metod skrócone do jednego zdania; szczegóły w zwijanych „Jak to liczymy?”.
- „Cele podróży” domyślnie wyłączone (najwięcej ikon na starcie); pozostałe domyślne warstwy bez zmian.

## Kod (w obrębie pliku)

- CSS pogrupowany w opisane sekcje (podstawy, panel i zakładki, kontrolki, mapa i nakładki, motyw).
- JS: spis treści na początku skryptu, sekcje w kolejności zakładek, usunięcie martwego kodu i nieużywanych stylów.
- Bez zmian w logice — tylko przenoszenie, nazwy sekcji, usuwanie nieużywanego.

## Testy

- Istniejące testy przeglądarkowe (trasowanie, przystanki, filtry, dzielnica, symulacja, panel klastra, warstwy z nowych danych)
  muszą przechodzić bez zmian w asercjach (poza nazwą „Wskaźnik wykluczenia”).
- Nowe: przełączanie zakładek (klik i strzałki), zapamiętywanie zakładki, `#przyklad` → „Projektuj”,
  tryb rysowania → „Projektuj”, każda kontrolka dostępna w jednej z zakładek, brak błędów JS, brak przewijania w poziomie,
  zrzuty panelu w obu motywach.
