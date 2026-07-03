"""Fetch the Gillessen et al. (2017, ApJ 837, 30) S2 orbital data from VizieR.

Source catalog: J/ApJ/837/30, table5 ("Orbital data used in this work"),
i.e. the published machine-readable version of the paper's Table 5. Extracts
the S2 columns (offsets relative to Sgr A* in mas + uncertainties; heliocentric
radial velocities in km/s + uncertainties) into:

    data/gillessen_2017/s2_astrometry.csv       epoch_yr, ra_mas, dec_mas, sig_ra_mas, sig_dec_mas
    data/gillessen_2017/s2_radial_velocity.csv  epoch_yr, vr_kms, sig_vr_kms

Run:  py -3.11 scripts/fetch_gillessen_data.py
"""
from __future__ import annotations

import csv
import io
import pathlib
import urllib.request

URL = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?"
       "-source=J/ApJ/837/30/table5&-out.max=unlimited&-out.all")

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "gillessen_2017"


def fetch() -> str:
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8", "replace")


def main() -> None:
    raw = fetch()
    lines = [l for l in raw.splitlines()]

    # Locate the header row (tab-separated column names, after the '#' block).
    header = None
    data_rows: list[list[str]] = []
    for i, l in enumerate(lines):
        if l.startswith("#") or not l.strip():
            continue
        if header is None:
            header = l.split("\t")
            # next lines: units row and ----- separator row, then data
            continue
        if set(l.replace("\t", "").strip()) <= {"-"}:
            continue
        if header is not None and l.split("\t")[0].strip().replace(".", "").isdigit() is False:
            # units row like 'yr' — skip anything whose first field isn't numeric
            first = l.split("\t")[0].strip()
            try:
                float(first)
            except ValueError:
                continue
        data_rows.append(l.split("\t"))

    assert header is not None, "no header found in VizieR response"
    idx = {name.strip(): k for k, name in enumerate(header)}
    for required in ("Date", "Flag", "oRA-S2", "oDE-S2", "e_oRA-S2",
                     "e_oDE-S2", "RV-S2", "e_RV-S2"):
        assert required in idx, f"missing column {required}: {sorted(idx)[:10]}..."

    def field(row: list[str], name: str) -> str:
        k = idx[name]
        return row[k].strip() if k < len(row) else ""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_ast = n_rv = 0
    with open(OUT_DIR / "s2_astrometry.csv", "w", newline="") as fa, \
         open(OUT_DIR / "s2_radial_velocity.csv", "w", newline="") as fv:
        wa = csv.writer(fa)
        wv = csv.writer(fv)
        wa.writerow(["epoch_yr", "ra_mas", "dec_mas", "sig_ra_mas", "sig_dec_mas"])
        wv.writerow(["epoch_yr", "vr_kms", "sig_vr_kms"])
        for row in data_rows:
            date = field(row, "Date")
            flag = field(row, "Flag")
            if not date:
                continue
            if flag.startswith("a"):
                ra, de = field(row, "oRA-S2"), field(row, "oDE-S2")
                era, ede = field(row, "e_oRA-S2"), field(row, "e_oDE-S2")
                if ra and de and era and ede:
                    wa.writerow([date, ra, de, era, ede])
                    n_ast += 1
            elif flag.startswith("rv"):
                rv, erv = field(row, "RV-S2"), field(row, "e_RV-S2")
                if rv and erv:
                    wv.writerow([date, rv, erv])
                    n_rv += 1

    print(f"S2 astrometry points: {n_ast}")
    print(f"S2 radial velocity points: {n_rv}")
    assert n_ast > 100, "unexpectedly few astrometric points"
    assert n_rv > 30, "unexpectedly few RV points"


if __name__ == "__main__":
    main()
