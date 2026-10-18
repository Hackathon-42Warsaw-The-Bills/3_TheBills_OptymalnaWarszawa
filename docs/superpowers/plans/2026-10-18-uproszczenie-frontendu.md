# Uproszczenie frontendu — etap 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Panel `index.html` w 4 zakładkach (Problem · Projektuj · Mapa · Symulacja) z przyklejonym nagłówkiem (rok, scenariusz), krótszymi tekstami i uporządkowanym kodem — bez zmiany logiki.

**Architecture:** Jeden plik `index.html`. Istniejące bloki HTML przenoszone w całości (wszystkie `id` bez zmian) do czterech `<div role="tabpanel">`. Nowy mały moduł JS `TABS` (pokaż zakładkę, zapamiętaj w `localStorage`, strzałki, hash). Schowane funkcje w `<details>`.

**Tech Stack:** HTML/CSS/vanilla JS, MapLibre GL 4.7.1 (bez zmian). Testy: Edge headless + lokalny serwer `serve.py` (scratchpad `routetest/`), który wstrzykuje `window.__T()` i skrypt testu.

## Global Constraints

- Wszystkie `id` elementów używanych w JS zostają bez zmian.
- Mapa i nakładki po prawej (lista dzielnic, zoom, panel klastra) bez zmian.
- Żadna funkcja nie znika; schowane = `<details>` (domyślnie zwinięte).
- Nazwa w UI: „Wskaźnik wykluczenia” zamiast „Prawdopodobieństwo wykluczenia”.
- `localStorage` tylko w try/catch; brak dostępu → zakładka „Problem”.
- Zakładki: `role="tablist"`/`"tab"`/`"tabpanel"`, `aria-selected`, `aria-controls`, strzałki ←/→.

---

### Task 1: Zestaw testów regresji (stan wyjściowy)

**Files:**
- Create: `<scratchpad>/routetest/suite.js` (test przeglądarkowy, poza repo)
- Modify: `<scratchpad>/routetest/serve.py` (już wystawia `window.__T()`)

**Interfaces:**
- Produces: `suite.js` — lista asercji `ok(nazwa, warunek, info)`, wynik w `<pre id="out">`; uruchamiany komendą z Kroku 2.

- [ ] **Step 1:** Połączyć dotychczasowe testy (trasowanie: klik w łącznicę, cofanie, JSON tam i z powrotem; przystanki tylko przy trasie; filtr linii M1/S2/527; dzielnica w symulacji; panel klastra Mokotów #1; warstwy propozycji, MPP, wsiadający; Veturilo) w jeden `suite.js`. Asercje odnoszą się do `id` elementów i źródeł mapy — nie do położenia w panelu.
- [ ] **Step 2:** Uruchomić na obecnym `index.html`:
  `python -I serve.py <repo>` + `msedge --headless=new --virtual-time-budget=300000 --dump-dom http://127.0.0.1:8799/__test.html`
  Expected: wszystkie PASS (stan wyjściowy).

### Task 2: Nagłówek i zakładki

**Files:** Modify: `index.html` (CSS przy `.panel`, HTML `<aside class="panel">`, JS przed `wireUI`)

**Interfaces:**
- Produces: `showTab(name: "problem"|"design"|"map"|"sim", {focus?: boolean})`; stała `TAB_KEY = "zakladka"`.

- [ ] **Step 1: Test (failing):** w `tabs.js`: istnieje `[role=tablist]` z 4 zakładkami; klik „Mapa” → `#tab-map` widoczny, inne `hidden`; `localStorage.zakladka === "map"`; strzałka → przenosi na „Symulacja”; `#year` i `#scenario` są poza panelami zakładek.
- [ ] **Step 2:** Uruchomić — FAIL (brak tablist).
- [ ] **Step 3: Implementacja.** Nagłówek:

```html
<header class="head">
  <button class="theme-btn" id="theme" type="button"></button>
  <h1>Symulator linii Warszawa 2036</h1>
  <div class="sub">Zobacz, gdzie brakuje komunikacji, narysuj linię i sprawdź, kogo obsłuży. <a href="linie.html">Ranking linii ZTM →</a></div>
  <div class="year"><b id="year-label">2036</b><input type="range" id="year" min="2025" max="2036" step="1" value="2036" aria-label="Rok prognozy"></div>
  <div class="seg" id="scenario"> …3 przyciski bez zmian… </div>
  <div class="tabs" role="tablist" aria-label="Sekcje panelu">
    <button role="tab" id="t-problem" aria-controls="tab-problem" data-tab="problem">Problem</button>
    <button role="tab" id="t-design" aria-controls="tab-design" data-tab="design">Projektuj</button>
    <button role="tab" id="t-map" aria-controls="tab-map" data-tab="map">Mapa</button>
    <button role="tab" id="t-sim" aria-controls="tab-sim" data-tab="sim">Symulacja</button>
  </div>
</header>
<div class="tabpanel" role="tabpanel" id="tab-problem" aria-labelledby="t-problem">…</div>
```

JS:

```js
const TAB_KEY = "zakladka", TABS = ["problem", "design", "map", "sim"];
function showTab(name, { focus = false } = {}) {
  if (!TABS.includes(name)) name = "problem";
  for (const t of TABS) {
    const on = t === name, b = $(`t-${t}`);
    b.setAttribute("aria-selected", on); b.tabIndex = on ? 0 : -1; $(`tab-${t}`).hidden = !on;
    if (on && focus) b.focus();
  }
  try { localStorage.setItem(TAB_KEY, name); } catch {}
}
```
Klik → `showTab(dataset.tab)`; `keydown` ←/→/Home/End na tablist → sąsiednia zakładka z `focus: true`. Start: `#przyklad` → „design”, `#linia-…` → „map”, inaczej zapamiętana lub „problem”. CSS: `.head { position: sticky; top: 0; z-index: 2; background: var(--paper) }`, zakładki jak `.seg`.
- [ ] **Step 4:** `tabs.js` PASS, `suite.js` PASS.

### Task 3: Rozmieszczenie sekcji w zakładkach i zwijane „Więcej”

**Files:** Modify: `index.html` (tylko HTML panelu + tekst opisów)

- [ ] **Step 1: Test (failing):** każda kontrolka z listy (`metric, top-on, top-list, prop-on, propl-on, heat-btn` → problem; `tools, undo, clear, reset, sample-line, headway…share, eval, ai-btn, json` → design; `line-filter, layers-all, layers, pmetro-layers, infra-layers` → map; `sim-play, sim-time, load-on, board-on, live` → sim) leży w odpowiednim `#tab-*`; `#headway`, `#json`, `#load-on`, `#live` są w zwiniętym `<details>`.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** Przenieść bloki 1:1 (bez zmiany `id`):
  - Problem: „Warstwa klastrów” (z legendą), „Priorytety” (top 30, propozycje, heatmapa).
  - Projektuj: narzędzia + `#hint` + akcje, „Ocena”, „Asystent”, `<details><summary>Parametry linii</summary>…</details>`, `<details><summary>Dla zespołu: wymiana JSON z algorytmem</summary>…</details>`.
  - Mapa: filtr linii + przyciski + grupy warstw, `<details><summary>O danych</summary>…stopka…</details>`.
  - Symulacja: zegar/start/prędkość, `<details><summary>Model pasażerów (szczyt 7–8)</summary>` obciążenie + wsiadający `</details>`, `<details><summary>Pojazdy na żywo</summary>`… + „działa tylko lokalnie (frontend/serwer.py)” `</details>`.
  - Opisy skrócone do 1 zdania; długie metody w `<details class="how"><summary>Jak to liczymy?</summary>`.
- [ ] **Step 4:** test PASS, `suite.js` PASS.

### Task 4: Zachowanie zakładek przy pracy z mapą + nazewnictwo + domyślne cele podróży

**Files:** Modify: `index.html` (`setMode`, `loadProposal` przy `#przyklad`, top-list click, teksty „wykluczenia”, `INFRA_ON`)

- [ ] **Step 1: Test (failing):** `setMode("route")` (klik narzędzia lub `#przyklad`) → zakładka „design”; tekst „Prawdopodobieństwo” nie występuje w panelu ani w panelu klastra, jest „Wskaźnik wykluczenia”; przy pierwszym wejściu wszystkie `#infra-layers input` odznaczone.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** W `setMode(m)`: `if (m !== "view") showTab("design")`. Zamiana tekstów w: `renderTopList` (notka), `#heat-note`, odznaka panelu klastra, wiersz „Prawdopodobieństwo wykluczenia” → „Wskaźnik wykluczenia”. `INFRA_ON = new Set()`.
- [ ] **Step 4:** PASS (w `suite.js` zaktualizować oczekiwaną etykietę wiersza na „Wskaźnik wykluczenia”).

### Task 5: Porządek w kodzie

**Files:** Modify: `index.html`

- [ ] **Step 1:** CSS w sekcjach z nagłówkami `/* === Podstawy === */`, `/* === Panel i zakładki === */`, `/* === Kontrolki === */`, `/* === Mapa i nakładki === */`, `/* === Telefon === */`; usunąć reguły bez użycia (sprawdzenie: każda klasa z CSS występuje w HTML lub JS).
- [ ] **Step 2:** JS: spis treści na początku skryptu (lista sekcji `// ---------- … ----------` w kolejności), usunąć nieużywane funkcje/zmienne (sprawdzenie: grep nazwy = 1 wystąpienie).
- [ ] **Step 3:** `suite.js` + `tabs.js` PASS, brak błędów JS.

### Task 6: Weryfikacja końcowa

- [ ] **Step 1:** Pełny `suite.js` + `tabs.js` + test braku przewijania w poziomie (1400 px i 390 px szerokości).
- [ ] **Step 2:** Zrzuty każdej zakładki w obu motywach; przegląd wzrokowy.
- [ ] **Step 3:** Bez commita — użytkownik commituje sam; podać propozycję wiadomości.
