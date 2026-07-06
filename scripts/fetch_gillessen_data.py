"""Fetch the Gillessen et al. (2017, ApJ 837, 30) S-star orbital data from
VizieR.

Source catalog: J/ApJ/837/30, table5 ("Orbital data used in this work") — the
machine-readable version of the paper's Table 5. Rows are observation epochs
with a Flag ('a...' astrometry epoch, 'rv...' radial-velocity epoch); each
star contributes columns oRA-<star>/oDE-<star> (offsets from Sgr A* in mas,
with e_ uncertainties) and RV-<star>/e_RV-<star> (heliocentric km/s), empty
where the star was not measured at that epoch.

For every requested star this writes
    data/gillessen_2017/<star>_astrometry.csv       epoch_yr, ra_mas, dec_mas, sig_ra_mas, sig_dec_mas
    data/gillessen_2017/<star>_radial_velocity.csv  epoch_yr, vr_kms, sig_vr_kms
(lower-case star name; S2's files keep their original names so the whole
Phase-4/5 pipeline is unchanged).

Run:  py -3.11 scripts/fetch_gillessen_data.py [--stars S2 S1 S8 S12 S13]
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import urllib.request

# Primary CDS host first, then mirrors (the CDS host rate-limits bursts).
URLS = [
    ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?"
     "-source=J/ApJ/837/30/table5&-out.max=100000&-out.all"),
    ("https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?"
     "-source=J/ApJ/837/30/table5&-out.max=100000&-out.all"),
]

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "gillessen_2017"


def fetch() -> str:
    last: Exception | None = None
    for url in URLS:
        try:
            req = urllib.request.Request(url,
                                         headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                raw = r.read().decode("utf-8", "replace")
            if "oRA-S2" in raw:
                return raw
            last = RuntimeError(f"unexpected response from {url}")
        except Exception as e:  # noqa: BLE001 — try the next mirror
            print(f"  [retry] {url.split('/')[2]}: {e}")
            last = e
    raise last or RuntimeError("no VizieR mirror reachable")


def parse_rows(raw: str) -> tuple[dict[str, int], list[list[str]]]:
    header: list[str] | None = None
    data_rows: list[list[str]] = []
    for l in raw.splitlines():
        if l.startswith("#") or not l.strip():
            continue
        if header is None:
            header = l.split("\t")
            continue
        if set(l.replace("\t", "").strip()) <= {"-"}:
            continue
        first = l.split("\t")[0].strip()
        try:
            float(first)
        except ValueError:
            continue  # units row etc.
        data_rows.append(l.split("\t"))
    assert header is not None, "no header found in VizieR response"
    return {name.strip(): k for k, name in enumerate(header)}, data_rows


def extract_star(star: str, idx: dict[str, int],
                 rows: list[list[str]]) -> tuple[int, int]:
    for required in (f"oRA-{star}", f"oDE-{star}", f"e_oRA-{star}",
                     f"e_oDE-{star}", f"RV-{star}", f"e_RV-{star}"):
        assert required in idx, f"star {star}: missing column {required}"

    def field(row: list[str], name: str) -> str:
        k = idx[name]
        return row[k].strip() if k < len(row) else ""

    base = star.lower()
    ast_path = OUT_DIR / f"{base}_astrometry.csv"
    rv_path = OUT_DIR / f"{base}_radial_velocity.csv"
    n_ast = n_rv = 0
    with open(ast_path, "w", newline="") as fa, \
         open(rv_path, "w", newline="") as fv:
        wa = csv.writer(fa)
        wv = csv.writer(fv)
        wa.writerow(["epoch_yr", "ra_mas", "dec_mas", "sig_ra_mas",
                     "sig_dec_mas"])
        wv.writerow(["epoch_yr", "vr_kms", "sig_vr_kms"])
        for row in rows:
            date = field(row, "Date")
            flag = field(row, "Flag")
            if not date:
                continue
            if flag.startswith("a"):
                ra, de = field(row, f"oRA-{star}"), field(row, f"oDE-{star}")
                era = field(row, f"e_oRA-{star}")
                ede = field(row, f"e_oDE-{star}")
                if ra and de and era and ede:
                    wa.writerow([date, ra, de, era, ede])
                    n_ast += 1
            elif flag.startswith("rv"):
                rv, erv = field(row, f"RV-{star}"), field(row, f"e_RV-{star}")
                if rv and erv:
                    wv.writerow([date, rv, erv])
                    n_rv += 1
    print(f"{star}: {n_ast} astrometry + {n_rv} RV points")
    return n_ast, n_rv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stars", nargs="+",
                    default=["S2", "S1", "S8", "S12", "S13"])
    args = ap.parse_args()

    raw = fetch()
    idx, rows = parse_rows(raw)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for star in args.stars:
        n_ast, n_rv = extract_star(star, idx, rows)
        assert n_ast > 20, f"{star}: unexpectedly few astrometric points"

    # S2's lower-cased filenames coincide with the legacy Phase-4/5 names
    # (s2_astrometry.csv / s2_radial_velocity.csv) — assert they stay rich.
    if "S2" in args.stars:
        n2 = sum(1 for _ in open(OUT_DIR / "s2_astrometry.csv")) - 1
        assert n2 > 100, "S2: unexpectedly few astrometric points"


if __name__ == "__main__":
    main()
