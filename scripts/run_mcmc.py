"""MCMC posteriors for the S2 fit (paper/THESIS_PLAN.md §2.1 [CORE]).

Samples the 13-parameter (Kerr) and 14-parameter (no-hair, + eps_q) posteriors
with the affine-invariant ensemble sampler of Goodman & Weare (emcee),
evaluating chi^2 through the persistent `fit_s2_tool --chi2` server (the full
GR geodesic forward model — the SAME code path as the least-squares fit, so
the posterior and the Ceres solution are directly comparable). The chi^2
already contains the Plewa+2015 reference-frame priors; within the broad
physical bounds below the posterior is exp(-chi^2/2).

Outputs (validation/thesis/):
    mcmc_s2_chain.npz / mcmc_s2.json          Kerr chain + percentile summary
    mcmc_nohair_chain.npz / mcmc_nohair.json  no-hair chain + summary

Run: py -3.11 scripts/run_mcmc.py [kerr|nohair|both] [--steps N]
     [--walkers N] [--workers K] [--seed S]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

try:
    import emcee
except ImportError:
    sys.exit("emcee is required: py -3.11 -m pip install emcee")

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = ROOT / "build/sci-rel/science/orbit_fitter/fit_s2_tool.exe"
DATA = ROOT / "data/gillessen_2017"
OUT = ROOT / "validation/thesis"

NAMES = ["M_msun", "R0_pc", "a_mas", "e", "incl_deg", "Omega_deg",
         "omega_deg", "t_peri_yr", "x0_mas", "y0_mas", "vx_masyr",
         "vy_masyr", "vz_kms"]
# Broad physical bounds (flat priors; the frame priors live inside chi^2).
BOUNDS = {
    "M_msun": (3.0e6, 6.0e6),
    "R0_pc": (7000.0, 9600.0),
    "a_mas": (110.0, 140.0),
    "e": (0.80, 0.95),
    "incl_deg": (120.0, 150.0),
    "Omega_deg": (210.0, 245.0),
    "omega_deg": (50.0, 82.0),
    "t_peri_yr": (2001.8, 2002.8),
    "x0_mas": (-3.0, 3.0),
    "y0_mas": (-3.0, 3.0),
    "vx_masyr": (-1.0, 1.0),
    "vy_masyr": (-1.0, 1.0),
    "vz_kms": (-60.0, 60.0),
    "eps_q": (-1.5e6, 1.5e6),
}
# Walker-ball scales when the covariance is unavailable for a parameter.
FALLBACK_SCALE = {"x0_mas": 0.05, "y0_mas": 0.05, "vx_masyr": 0.02,
                  "vy_masyr": 0.02, "vz_kms": 3.0, "eps_q": 1.5e5}


class Chi2Pool:
    """K persistent `fit_s2_tool --chi2` workers; evaluates a whole batch of
    parameter vectors by round-robin distribution (each worker is FIFO)."""

    def __init__(self, mode: str, workers: int):
        args = [str(TOOL), "--chi2", str(DATA), mode]
        self.procs = [subprocess.Popen(args, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, text=True,
                                       bufsize=1) for _ in range(workers)]

    def chi2_batch(self, X: np.ndarray) -> np.ndarray:
        n, k = len(X), len(self.procs)
        for i, x in enumerate(X):
            p = self.procs[i % k]
            p.stdin.write(" ".join(repr(v) for v in x) + "\n")
        for p in self.procs:
            p.stdin.flush()
        out = np.empty(n)
        for i in range(n):
            out[i] = float(self.procs[i % k].stdout.readline())
        return out

    def close(self) -> None:
        for p in self.procs:
            p.stdin.close()
            p.wait()


def run(mode: str, steps: int, nwalkers: int, workers: int,
        seed: int) -> None:
    nohair = mode == "nohair"
    names = NAMES + (["eps_q"] if nohair else [])
    ndim = len(names)

    fit = json.loads((ROOT / "validation/phase4/s2_fit.json").read_text())
    centre, scale = [], []
    for n in names:
        if n == "eps_q":
            nh = json.loads((ROOT / "validation/phase5/s2_fit.json").read_text())
            centre.append(nh["nohair"]["eps_q_best"])
            scale.append(max(nh["nohair"]["eps_q_sigma"], 1e4))
            continue
        centre.append(fit[n])
        sig = fit.get("sigma_" + n, 0.0)
        scale.append(sig if sig > 0 else FALLBACK_SCALE.get(n, 1e-3))
    centre, scale = np.array(centre), np.array(scale)
    lo = np.array([BOUNDS[n][0] for n in names])
    hi = np.array([BOUNDS[n][1] for n in names])

    pool = Chi2Pool(mode, workers)

    def log_prob(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        bad = np.any((X < lo) | (X > hi), axis=1)
        lp = np.full(len(X), -np.inf)
        if np.any(~bad):
            c2 = pool.chi2_batch(X[~bad])
            lp[~bad] = -0.5 * c2
        return lp

    rng = np.random.default_rng(seed)
    p0 = centre + 0.5 * scale * rng.standard_normal((nwalkers, ndim))
    p0 = np.clip(p0, lo + 1e-12, hi - 1e-12)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, vectorize=True)
    # Seed emcee's internal RandomState too: default_rng above only seeds the
    # walker ball, and an unseeded sampler makes every `make reproduce` emit a
    # different chain (breaking the numbers.tex reproducibility contract).
    sampler.random_state = np.random.RandomState(seed).get_state()
    t0 = time.time()
    print(f"[{mode}] sampling {nwalkers} walkers x {steps} steps "
          f"({ndim} params, {workers} chi2 workers) ...")
    sampler.run_mcmc(p0, steps, progress=False)
    print(f"[{mode}] done in {time.time() - t0:.0f} s; "
          f"acceptance = {sampler.acceptance_fraction.mean():.3f}")

    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_max = float(np.nanmax(tau))
    except Exception:
        tau_max = steps / 50.0
    burn = int(min(4 * tau_max, steps * 0.5))
    thin = max(1, int(tau_max / 2))
    samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    print(f"[{mode}] tau_max = {tau_max:.0f}, burn = {burn}, thin = {thin}, "
          f"{len(samples)} posterior samples")

    OUT.mkdir(parents=True, exist_ok=True)
    tag = "nohair" if nohair else "s2"
    np.savez_compressed(OUT / f"mcmc_{tag}_chain.npz",
                        samples=samples, names=np.array(names))
    summary = {n: list(np.percentile(samples[:, i], [16, 50, 84]))
               for i, n in enumerate(names)}
    summary.update({
        "n_steps": steps, "n_walkers": nwalkers, "burn": burn, "thin": thin,
        "n_samples": int(len(samples)), "autocorr_max": tau_max,
        "acceptance": float(sampler.acceptance_fraction.mean()),
        "seed": seed, "mode": mode,
    })
    (OUT / f"mcmc_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"[{mode}] wrote mcmc_{tag}_chain.npz + mcmc_{tag}.json")
    pool.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="both",
                    choices=["kerr", "nohair", "both"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--walkers", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.mode in ("kerr", "both"):
        run("kerr", a.steps, a.walkers, a.workers, a.seed)
    if a.mode in ("nohair", "both"):
        run("nohair", a.steps, a.walkers, a.workers, a.seed + 1)


if __name__ == "__main__":
    main()
