"""Serwer frontendu: pliki z frontend/ + pośrednik do API ZTM (autobusy i tramwaje na żywo).

API miasta nie pozwala na zapytania wprost z przeglądarki (brak CORS), a klucza nie wolno
wkładać do kodu strony — dlatego zapytania idą przez ten serwer. Klucz w ZTM_API_KEY (.env):
  - token JWT z dane.um.warszawa.pl (nowe API, get_ztm_lokalizacja_pojazdow, nagłówek Authorization),
  - albo klucz UUID ze starego api.um.warszawa.pl (busestrams_get) — rozpoznawany automatycznie.

Uruchomienie:
    .venv/bin/python frontend/serwer.py      # klucz z .env (ZTM_API_KEY=...) lub zmiennej środowiskowej
    .venv/bin/python frontend/serwer.py 9000                            # inny port; bez klucza
                                                                        # działa wszystko oprócz „na żywo”
Endpointy:
    GET /api/status    -> {"live": true|false}
    GET /api/pojazdy   -> {"czas": ..., "pojazdy": [[lon, lat, linia, brygada, nr_pojazdu, typ, sek_temu], ...]}
                          typ: 1 = autobus, 2 = tramwaj; odpowiedź z ZTM trzymana w pamięci CACHE_S sekund
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

ROOT = Path(__file__).resolve().parent


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
            return self.send_json(200, json.dumps({"live": bool(API_KEY)}).encode())
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
        return super().do_GET()

    def log_message(self, fmt, *args):
        if args and "/api/" in str(args[0]):
            return  # bez zalewania konsoli co 15 s
        super().log_message(fmt, *args)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT)))
    print(f"Frontend: http://localhost:{port}/symulator.html   (mapa ulic: /index.html)")
    print("Pojazdy na żywo: " + (("włączone (" + ("dane.um.warszawa.pl" if NEW_API else "api.um.warszawa.pl") + ")")
                                 if API_KEY else "wyłączone — ustaw ZTM_API_KEY w .env"))
    server.serve_forever()


if __name__ == "__main__":
    main()
