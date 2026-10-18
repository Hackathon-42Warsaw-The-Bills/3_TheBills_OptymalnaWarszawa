#!/usr/bin/env python3
"""Oblicza kolumne etykieta_target dla h3_merged_clean.csv.

Wzor (ustalony):
    P = norm(log1p(gestosc_os_km2))
    J = norm(log1p(pracownicy_szac))
    D = `wspolczynnik przyrostu` (juz w 0-1, wprost)
    Z = norm(log1p(w_tram*n_tram + w_bus*n_bus + w_metro*czy_metro + w_kolej*czy_kolej))
    M = alpha + alpha*D
    raw = M*(P+J) - Z
    etykieta_target = clip((raw - p1(raw)) / (p99(raw) - p1(raw)), 0, 1)

Normalizacja P/J/Z: min-max na wartosciach log1p (odporne na zera dzieki log1p).
Normalizacja koncowa: percentyle p1/p99 (odpornosc na outliery).
Idempotentne: ponowne uruchomienie nadpisuje kolumne etykieta_target.

Uzycie:
    python3 compute_target.py [--input h3_merged_clean.csv] [--alpha 0.3] [--w-kolej 2.5]
"""
import argparse
import csv
import math
import os
import sys


def percentile(sorted_vals, q):
    """q w 0-100, interpolacja liniowa. sorted_vals musi byc posortowane."""
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("pusta lista do percentyla")
    if n == 1:
        return sorted_vals[0]
    rank = q / 100.0 * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def fnum(v, default=0.0):
    try:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return default
        return float(str(v).replace(",", "."))
    except (ValueError, TypeError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                    "dane", "wskaznik_wykluczenia", "h3_merged_clean.csv"))
    ap.add_argument("--output", default=None, help="domyslnie nadpisuje --input")
    ap.add_argument("--target-col", default="etykieta_target")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--w-tram", type=float, default=1.5)
    ap.add_argument("--w-bus", type=float, default=1.0)
    ap.add_argument("--w-metro", type=float, default=3.0)
    ap.add_argument("--w-kolej", type=float, default=2.5)
    a = ap.parse_args()

    out_path = a.output or a.input
    if not (0 < a.alpha <= 2.0):
        print(f"UWAGA: alpha={a.alpha} poza zalecanym (0, 2]. Kontynuuje.", file=sys.stderr)

    with open(a.input, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)

    req = ["gestosc_os_km2", "pracownicy_szac", "przystanki_tramwajowe",
           "przystanki_autobusowe", "czy_metro", "wspolczynnik przyrostu"]
    missing = [c for c in req if c not in cols]
    if missing:
        sys.exit(f"BLAD: brak kolumn w {a.input}: {missing}. Kolumny: {cols}")

    lp, lj, lz, ds = [], [], [], []
    cache = []
    for r in rows:
        gest = max(fnum(r["gestosc_os_km2"]), 0.0)
        prac = max(fnum(r["pracownicy_szac"]), 0.0)
        tram = max(fnum(r["przystanki_tramwajowe"]), 0.0)
        bus = max(fnum(r["przystanki_autobusowe"]), 0.0)
        metro = fnum(r["czy_metro"])
        metro = 1.0 if metro >= 0.5 else 0.0
        kolej = fnum(r.get("czy_kolej", 0.0))
        kolej = 1.0 if kolej >= 0.5 else 0.0
        d = min(max(fnum(r["wspolczynnik przyrostu"]), 0.0), 1.0)
        v_p = math.log1p(gest)
        v_j = math.log1p(prac)
        v_z = math.log1p(a.w_tram * tram + a.w_bus * bus + a.w_metro * metro + a.w_kolej * kolej)
        lp.append(v_p)
        lj.append(v_j)
        lz.append(v_z)
        ds.append(d)
        cache.append((v_p, v_j, v_z, d))

    def norm_minmax(vals):
        lo, hi = min(vals), max(vals)
        span = hi - lo
        if span <= 0:
            return [0.0] * len(vals), lo, hi
        return [(v - lo) / span for v in vals], lo, hi

    pn, lo_p, hi_p = norm_minmax(lp)
    jn, lo_j, hi_j = norm_minmax(lj)
    zn, lo_z, hi_z = norm_minmax(lz)

    raws = []
    for p, j, z, d in zip(pn, jn, zn, ds):
        m = a.alpha + a.alpha * d
        raws.append(m * (p + j) - z)

    s = sorted(raws)
    p1 = percentile(s, 1)
    p99 = percentile(s, 99)
    span = p99 - p1
    if span <= 0:
        labels = [0.0] * len(raws)
    else:
        labels = [min(max((v - p1) / span, 0.0), 1.0) for v in raws]

    for r, lab in zip(rows, labels):
        r[a.target_col] = f"{lab:.6f}"

    out_cols = cols + ([] if a.target_col in cols else [a.target_col])
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out_path)

    lab_sorted = sorted(labels)
    n_kolej = sum(1 for r in rows if fnum(r.get("czy_kolej", 0.0)) >= 0.5)
    print(f"wiersze: {len(rows)} (czy_kolej=1: {n_kolej})")
    print(f"P log1p: min={lo_p:.4f} max={hi_p:.4f}")
    print(f"J log1p: min={lo_j:.4f} max={hi_j:.4f}")
    print(f"Z log1p: min={lo_z:.4f} max={hi_z:.4f}")
    print(f"raw: min={min(raws):.4f} p1={p1:.4f} med={percentile(s,50):.4f} p99={p99:.4f} max={max(raws):.4f}")
    print(f"{a.target_col}: min={min(labels):.4f} med={percentile(lab_sorted,50):.4f} max={max(labels):.4f}")
    print(f"zapisano: {out_path} (alpha={a.alpha} w_tram={a.w_tram} w_bus={a.w_bus} w_metro={a.w_metro} w_kolej={a.w_kolej})")


if __name__ == "__main__":
    main()
