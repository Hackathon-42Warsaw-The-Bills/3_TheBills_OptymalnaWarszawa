"""Serwer lokalny: aplikacja (index.html w katalogu repo, jak na GitHub Pages) + pośrednik do API ZTM
(autobusy i tramwaje na żywo) i do API Anthropic (ocena linii przez asystenta AI) — tego GitHub Pages nie potrafi.

API miasta nie pozwala na zapytania wprost z przeglądarki (brak CORS), a kluczy nie wolno
wkładać do kodu strony — dlatego zapytania idą przez ten serwer. Klucze w .env:
  ZTM_API_KEY        — token JWT z dane.um.warszawa.pl (nowe API, get_ztm_lokalizacja_pojazdow, nagłówek Authorization)
                       albo klucz UUID ze starego api.um.warszawa.pl (busestrams_get) — rozpoznawany automatycznie;
  ANTHROPIC_API_KEY  — klucz z console.anthropic.com do oceny przez asystenta AI; model w ANTHROPIC_MODEL
                       (domyślnie claude-opus-5-5), adres API w ANTHROPIC_BASE_URL (domyślnie api.anthropic.com).

Uruchomienie:
    .venv/bin/python frontend/serwer.py      # http://localhost:8765/ ; klucze z .env
    .venv/bin/python frontend/serwer.py 9000                            # inny port; bez kluczy działa wszystko
                                                                        # oprócz „na żywo” i asystenta AI
Endpointy:
    GET  /api/status   -> {"live": true|false, "ai": true|false}
    GET  /api/pojazdy  -> {"czas": ..., "pojazdy": [[lon, lat, linia, brygada, nr_pojazdu, typ, sek_temu], ...]}
                          typ: 1 = autobus, 2 = tramwaj; odpowiedź z ZTM trzymana w pamięci CACHE_S sekund
    POST /api/ocena    {"prompt": "..."} -> tekst odpowiedzi modelu strumieniem (text/plain, kawałek po kawałku);
                          błąd: {"blad": "..."} z kodem 4xx/5xx
"""

import json
import os
import sys
import threading
import time
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent          # frontend/
SITE = ROOT.parent                              # katalog repo — to samo co GitHub Pages
# Czego serwer nie wydaje (sekrety i dane robocze): ścieżki z segmentem zaczynającym się od kropki
# (.env, .git, .venv) oraz katalog data/.
BLOCKED_TOP = {"data"}


def load_dotenv() -> None:
    """Zmienne z pliku .env (katalog repo lub frontend/); zmienne ustawione w powłoce mają pierwszeństwo."""
    for path in (ROOT.parent / ".env", ROOT / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()
API_URL = "https://dane.um.warszawa.pl/api/action/get_ztm_lokalizacja_pojazdow"
OLD_API_URL = "https://api.um.warszawa.pl/api/action/busestrams_get/"
OLD_RESOURCE_ID = "f2e5503e-927d-4ad3-9500-4ab9e55deb59"
API_KEY = os.environ.get("ZTM_API_KEY", "").strip()
NEW_API = "." in API_KEY   # token JWT (trzy części z kropkami) -> nowe API; UUID -> stare
CACHE_S = 10          # ZTM odświeża pozycje mniej więcej co 10 s
MAX_AGE_S = 300       # pozycje starsze niż 5 min (pojazd zjechał, nadajnik milczy) pomijamy
AI_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
AI_MODEL = os.environ.get("ANTHROPIC_MODEL", "").strip() or "claude-opus-5-5"
AI_URL = (os.environ.get("ANTHROPIC_BASE_URL", "").strip() or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
AI_MAX_PROMPT = 20000  # znaków — strona wysyła polecenie i kilka KB danych linii

_cache = {"t": 0.0, "body": b""}
_lock = threading.Lock()


def fetch_vehicles() -> bytes:
    now = datetime.now()
    out = []
    for typ in (1, 2):
        if NEW_API:
            r = requests.post(API_URL, json={"type": typ}, timeout=20,
                              headers={"Authorization": API_KEY, "Content-Type": "application/json"})
        else:
            r = requests.get(OLD_API_URL, params={"resource_id": OLD_RESOURCE_ID, "apikey": API_KEY, "type": typ}, timeout=20)
        try:
            body = r.json()
        except ValueError:
            raise RuntimeError(f"API ZTM: HTTP {r.status_code}, odpowiedź nie jest JSON-em")
        rows = body.get("result", []) if isinstance(body, dict) else body
        if not isinstance(rows, list):        # przy błędzie (zły klucz, limit) API zwraca tekst
            msg = body.get("error") or body.get("message") or rows if isinstance(body, dict) else rows
            raise RuntimeError(f"API ZTM (HTTP {r.status_code}): {msg}")
        for v in rows:
            try:
                age = (now - datetime.fromisoformat(v["Time"])).total_seconds()
            except (KeyError, ValueError):
                continue
            if 0 <= age <= MAX_AGE_S or -60 < age < 0:
                out.append([round(v["Lon"], 5), round(v["Lat"], 5), str(v["Lines"]).strip(), str(v.get("Brigade", "")),
                            str(v.get("VehicleNumber", "")), typ, int(max(age, 0))])
    return json.dumps({"czas": now.isoformat(timespec="seconds"), "pojazdy": out}, ensure_ascii=False).encode()


class Handler(SimpleHTTPRequestHandler):
    def send_json(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/status":
            return self.send_json(200, json.dumps({"live": bool(API_KEY), "ai": bool(AI_KEY)}).encode())
        if path == "/api/pojazdy":
            if not API_KEY:
                return self.send_json(503, json.dumps({"blad": "Brak klucza ZTM_API_KEY na serwerze."}, ensure_ascii=False).encode())
            with _lock:
                if time.time() - _cache["t"] > CACHE_S:
                    try:
                        _cache["body"], _cache["t"] = fetch_vehicles(), time.time()
                    except Exception as e:  # noqa: BLE001 — każdy błąd upstreamu zwracamy stronie
                        return self.send_json(502, json.dumps({"blad": str(e)}, ensure_ascii=False).encode())
                body = _cache["body"]
            return self.send_json(200, body)
        parts = [p for p in path.split("/") if p]
        if any(p.startswith(".") for p in parts) or (parts and parts[0] in BLOCKED_TOP) or path.endswith(".py"):
            return self.send_error(404, "File not found")
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] != "/api/ocena":
            return self.send_error(404, "File not found")
        if not AI_KEY:
            return self.send_json(503, json.dumps({"blad": "Brak klucza ANTHROPIC_API_KEY na serwerze."}, ensure_ascii=False).encode())
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            prompt = str(body.get("prompt", ""))[:AI_MAX_PROMPT]
        except (ValueError, AttributeError):
            prompt = ""
        if not prompt.strip():
            return self.send_json(400, json.dumps({"blad": "Oczekuję JSON-a z polem prompt."}, ensure_ascii=False).encode())
        try:
            r = requests.post(AI_URL, stream=True, timeout=(10, 120),
                              headers={"x-api-key": AI_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                              json={"model": AI_MODEL, "max_tokens": 1024, "stream": True, "messages": [{"role": "user", "content": prompt}]})
        except requests.RequestException as e:
            return self.send_json(502, json.dumps({"blad": f"API Anthropic: {e}"}, ensure_ascii=False).encode())
        if r.status_code != 200:
            try:
                msg = r.json().get("error", {}).get("message") or r.text
            except ValueError:
                msg = r.text
            return self.send_json(502, json.dumps({"blad": f"API Anthropic (HTTP {r.status_code}): {msg[:300]}"}, ensure_ascii=False).encode())
        # Odpowiedź modelu (zdarzenia SSE) przekazujemy stronie jako zwykły tekst, kawałek po kawałku — strona dopisuje
        # go na bieżąco. Koniec tekstu = zamknięcie połączenia (HTTP/1.0, bez Content-Length).
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        r.encoding = "utf-8"
        try:
            for line in r.iter_lines(chunk_size=64, decode_unicode=True):   # małe porcje — tekst pojawia się płynnie
                if not line or not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:])
                except ValueError:
                    continue
                if ev.get("type") == "content_block_delta" and ev.get("delta", {}).get("type") == "text_delta":
                    self.wfile.write(ev["delta"]["text"].encode("utf-8"))
                    self.wfile.flush()
                elif ev.get("type") == "error":
                    self.wfile.write(f"\n\n[Błąd API Anthropic: {ev.get('error', {}).get('message', '')}]".encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            pass  # strona przerwała ocenę („Zatrzymaj”)
        except requests.RequestException as e:
            self.wfile.write(f"\n\n[Przerwane połączenie z API Anthropic: {e}]".encode("utf-8"))
        finally:
            r.close()
            self.close_connection = True

    def end_headers(self):
        # Pliki danych zawsze z rewalidacją — po przeliczeniu prognozy przeglądarka nie użyje starej kopii.
        if self.path.split("?")[0].endswith((".json", ".html")) or self.path.split("?")[0] == "/":
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        if args and "/api/" in str(args[0]):
            return  # bez zalewania konsoli co 15 s
        super().log_message(fmt, *args)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(SITE)))
    print(f"Aplikacja: http://localhost:{port}/")
    print("Pojazdy na żywo: " + (("włączone (" + ("dane.um.warszawa.pl" if NEW_API else "api.um.warszawa.pl") + ")")
                                 if API_KEY else "wyłączone — ustaw ZTM_API_KEY w .env"))
    print("Asystent AI: " + (f"włączony ({AI_MODEL})" if AI_KEY else "wyłączony — ustaw ANTHROPIC_API_KEY w .env"))
    server.serve_forever()


if __name__ == "__main__":
    main()
