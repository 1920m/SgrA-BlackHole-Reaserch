"""Auto-generate every figure in the paper from simulation output
(no hand-drawn figures; `make figures` calls this).

Inputs (produced by the tools in validation/ and science/):
    validation/reference_images/{schwarzschild,kerr}_cpu*.json  shadow metrology
    validation/phase3/brightness_profile.csv                    disk beaming
    validation/phase3/subring_report.json + subring_n*.png      photon ring
    validation/phase4/s2_fit.json, s2_residuals.csv,
                      s2_model_track.csv, s2_redshift_curve.csv S2 fit
    validation/phase5/s2_fit.json                               no-hair profile
    validation/phase10/sweep.csv                                spin-incl sweep
    validation/phase7/*.compare.json                            CPU/GPU diff
    data/gillessen_2017/*.csv                                   observed data

Outputs: paper/figures/*.pdf (vector) + *.png for raster panels.
Missing inputs are skipped with a warning so the script can run at any phase.

Run: py -3.11 scripts/make_figures.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.linewidth": 0.7,
    "figure.dpi": 150,
})


def warn_skip(name: str, missing: pathlib.Path) -> None:
    print(f"  [skip] {name}: missing {missing.relative_to(ROOT)}")


def read_csv(path: pathlib.Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, list] = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}


def fig_s2_orbit() -> None:
    fit = ROOT / "validation/phase4"
    track_p = fit / "s2_model_track.csv"
    ast_p = ROOT / "data/gillessen_2017/s2_astrometry.csv"
    if not track_p.exists():
        return warn_skip("s2_orbit", track_p)
    track = read_csv(track_p)
    ast = read_csv(ast_p)

    figr, ax = plt.subplots(figsize=(3.5, 4.2))
    ax.plot(track["ra_mas"].astype(float), track["dec_mas"].astype(float),
            "-", color="0.3", lw=0.8, label="GR model (this work)")
    ax.errorbar(ast["ra_mas"].astype(float), ast["dec_mas"].astype(float),
                xerr=ast["sig_ra_mas"].astype(float),
                yerr=ast["sig_dec_mas"].astype(float),
                fmt="o", ms=2, lw=0.5, color="tab:blue", alpha=0.75,
                label="Gillessen+17 astrometry")
    ax.plot([0], [0], "*", ms=10, color="k", label="Sgr A*")
    ax.set_xlabel(r"$\Delta$RA [mas] (East $\rightarrow$)")
    ax.set_ylabel(r"$\Delta$Dec [mas]")
    ax.invert_xaxis()  # astronomical convention: East left
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")
    figr.tight_layout()
    figr.savefig(FIG / "fig_s2_orbit.pdf")
    plt.close(figr)
    print("  wrote fig_s2_orbit.pdf")


def fig_s2_rv() -> None:
    fit = ROOT / "validation/phase4"
    track_p = fit / "s2_model_track.csv"
    rv_p = ROOT / "data/gillessen_2017/s2_radial_velocity.csv"
    if not track_p.exists():
        return warn_skip("s2_rv", track_p)
    track = read_csv(track_p)
    rv = read_csv(rv_p)

    figr, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.plot(track["epoch_yr"].astype(float), track["vr_kms"].astype(float),
            "-", color="0.3", lw=0.8, label="GR model")
    ax.errorbar(rv["epoch_yr"].astype(float), rv["vr_kms"].astype(float),
                yerr=rv["sig_vr_kms"].astype(float), fmt="o", ms=2.5, lw=0.6,
                color="tab:red", label="Gillessen+17 RV")
    ax.set_xlabel("epoch [yr]")
    ax.set_ylabel(r"$v_r$ [km/s]")
    ax.legend(fontsize=7)
    figr.tight_layout()
    figr.savefig(FIG / "fig_s2_rv.pdf")
    plt.close(figr)
    print("  wrote fig_s2_rv.pdf")


def fig_s2_residuals() -> None:
    p = ROOT / "validation/phase4/s2_residuals.csv"
    if not p.exists():
        return warn_skip("s2_residuals", p)
    r = read_csv(p)
    epoch = r["epoch_yr"].astype(float)
    nres = r["norm_resid"].astype(float)
    types = r["type"]

    figr, axes = plt.subplots(3, 1, figsize=(4.6, 4.2), sharex=True)
    for ax, t, color, lab in zip(axes, ["ra", "dec", "rv"],
                                 ["tab:blue", "tab:green", "tab:red"],
                                 [r"RA/$\sigma$", r"Dec/$\sigma$", r"$v_r/\sigma$"]):
        m = types == t
        ax.axhline(0, color="0.6", lw=0.5)
        ax.plot(epoch[m], nres[m], "o", ms=2, color=color)
        ax.set_ylabel(lab)
        ax.set_ylim(-4, 4)
    axes[-1].set_xlabel("epoch [yr]")
    figr.tight_layout()
    figr.savefig(FIG / "fig_s2_residuals.pdf")
    plt.close(figr)
    print("  wrote fig_s2_residuals.pdf")


def fig_redshift_excess() -> None:
    p = ROOT / "validation/phase4/s2_redshift_curve.csv"
    if not p.exists():
        return warn_skip("redshift_excess", p)
    r = read_csv(p)
    t = r["epoch_yr"].astype(float)

    figr, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.plot(t, r["v_excess_kms"].astype(float), "-", color="k", lw=1.2,
            label=r"GR excess $c(z_{\rm tot}-\beta_{\rm los})$")
    ax.plot(t, r["v_trans_kms"].astype(float), "--", color="tab:blue", lw=0.8,
            label=r"transverse Doppler $c(\gamma-1)$")
    ax.plot(t, r["v_grav_kms"].astype(float), ":", color="tab:red", lw=0.8,
            label=r"gravitational $c(1/\alpha-1)$")
    ax.axvline(2018.379, color="0.7", lw=0.5)
    ax.annotate("pericenter\n2018.38", (2018.42, 60), fontsize=7, color="0.4")
    ax.set_xlabel("epoch [yr]")
    ax.set_ylabel("velocity equivalent [km/s]")
    ax.legend(fontsize=7)
    figr.tight_layout()
    figr.savefig(FIG / "fig_redshift_excess.pdf")
    plt.close(figr)
    print("  wrote fig_redshift_excess.pdf")


def fig_shadow_boundary() -> None:
    ks = ROOT / "validation/reference_images/kerr_cpu_a94.json"
    ss = ROOT / "validation/reference_images/schwarzschild_cpu.json"
    if not ks.exists():
        return warn_skip("shadow_boundary", ks)
    kerr = json.loads(ks.read_text())
    schw = json.loads(ss.read_text()) if ss.exists() else None

    figr, ax = plt.subplots(figsize=(3.4, 3.4))
    psi = np.array([b["psi_rad"] for b in kerr["boundary"]])
    alp = np.array([b["alpha_rad"] for b in kerr["boundary"]])
    # angular radius → impact parameter in M
    rc = kerr["r_cam_M"]
    b_M = np.sin(alp) * rc / np.sqrt(1.0 - 2.0 / rc)
    ax.plot(np.append(b_M * np.cos(psi), b_M[0] * np.cos(psi[0])),
            np.append(b_M * np.sin(psi), b_M[0] * np.sin(psi[0])),
            "-o", ms=2, lw=0.9, color="tab:purple",
            label=rf"Kerr $a_*={kerr['a_star']}$ (traced)")
    if schw:
        b0 = np.sqrt(27.0)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(b0 * np.cos(th), b0 * np.sin(th), "--", lw=0.7, color="0.5",
                label=r"Schwarzschild $\sqrt{27}\,M$")
    ax.set_xlabel(r"$b_x$ [$M$]")
    ax.set_ylabel(r"$b_y$ [$M$]")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower left")
    figr.tight_layout()
    figr.savefig(FIG / "fig_shadow_boundary.pdf")
    plt.close(figr)
    print("  wrote fig_shadow_boundary.pdf")


def fig_brightness_profile() -> None:
    p = ROOT / "validation/phase3/brightness_profile.csv"
    if not p.exists():
        return warn_skip("brightness_profile", p)
    r = read_csv(p)
    x = r["x_norm"].astype(float)
    I = r["I_bol"].astype(float)
    g = r["g"].astype(float)

    figr, (ax, ax2) = plt.subplots(2, 1, figsize=(4.6, 3.6), sharex=True,
                                   height_ratios=[2, 1])
    ax.semilogy(x, I, ".", ms=1.5, color="tab:orange")
    ax.set_ylabel(r"$I_{\rm bol}$ [arb.]")
    ax2.plot(x, g, ".", ms=1.5, color="tab:blue")
    ax2.axhline(1.0, color="0.6", lw=0.5)
    ax2.set_ylabel(r"$g=\nu_{\rm obs}/\nu_{\rm em}$")
    ax2.set_xlabel("image-plane position [normalised]")
    figr.tight_layout()
    figr.savefig(FIG / "fig_brightness_profile.pdf")
    plt.close(figr)
    print("  wrote fig_brightness_profile.pdf")


def fig_nohair_profile() -> None:
    p = ROOT / "validation/phase5/s2_fit.json"
    if not p.exists():
        return warn_skip("nohair_profile", p)
    j = json.loads(p.read_text())
    if "nohair" not in j:
        return warn_skip("nohair_profile", p)
    prof = j["nohair"]["profile"]
    eq = np.array([q["eps_q"] for q in prof])
    c2 = np.array([q["chi2"] for q in prof])

    figr, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.plot(eq / 1e5, c2 - c2.min(), "o-", ms=3, lw=0.9, color="tab:purple")
    ax.axhline(3.84, color="0.5", lw=0.6, ls="--")
    ax.annotate(r"$\Delta\chi^2 = 3.84$ (95%)", (eq.min() / 1e5, 4.1),
                fontsize=7, color="0.4")
    ax.set_xlabel(r"$\epsilon_q\ [\times 10^5]$")
    ax.set_ylabel(r"$\Delta\chi^2$")
    figr.tight_layout()
    figr.savefig(FIG / "fig_nohair_profile.pdf")
    plt.close(figr)
    print("  wrote fig_nohair_profile.pdf")


def fig_subring_panels() -> None:
    base = ROOT / "validation/phase3"
    paths = [base / f"subring_n{n}.png" for n in range(3)]
    if not paths[0].exists():
        return warn_skip("subring_panels", paths[0])
    from PIL import Image
    figr, axes = plt.subplots(1, 3, figsize=(6.8, 2.4))
    for ax, p, n in zip(axes, paths, range(3)):
        ax.imshow(np.asarray(Image.open(p)))
        ax.set_title(f"$n={n}$", fontsize=9)
        ax.axis("off")
    figr.tight_layout()
    figr.savefig(FIG / "fig_subrings.pdf", dpi=200)
    plt.close(figr)
    print("  wrote fig_subrings.pdf")


def fig_disk_render() -> None:
    p = ROOT / "validation/phase3/disk_nt.png"
    k = ROOT / "validation/reference_images/kerr_cpu_a94.png"
    if not p.exists() or not k.exists():
        return warn_skip("disk_render", p)
    from PIL import Image
    figr, axes = plt.subplots(1, 2, figsize=(6.8, 3.4))
    for ax, path, title in zip(
            axes, [k, p],
            [r"Kerr $a_*=0.94$, band disk", "Novikov–Thorne + beaming"]):
        ax.imshow(np.asarray(Image.open(path)))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    figr.tight_layout()
    figr.savefig(FIG / "fig_disk_render.pdf", dpi=200)
    plt.close(figr)
    print("  wrote fig_disk_render.pdf")


def fig_shadow_sweep() -> None:
    p = ROOT / "validation/phase10/sweep.csv"
    if not p.exists():
        return warn_skip("shadow_sweep", p)
    r = read_csv(p)
    a = r["a_star"].astype(float)
    inc = r["incl_deg"].astype(float)
    d = r["diameter_uas"].astype(float)
    na, ni = len(np.unique(a)), len(np.unique(inc))
    A = a.reshape(na, ni)
    I = inc.reshape(na, ni)
    D = d.reshape(na, ni)

    figr, ax = plt.subplots(figsize=(4.6, 3.4))
    pc = ax.pcolormesh(A, I, D, shading="auto", cmap="magma")
    cs = ax.contour(A, I, D, levels=[48.7 - 7, 48.7, 48.7 + 7],
                    colors=["w", "c", "w"], linewidths=[0.6, 1.2, 0.6])
    ax.clabel(cs, fontsize=6, fmt="%.1f")
    figr.colorbar(pc, ax=ax, label=r"shadow diameter [$\mu$as]")
    ax.plot([0.94], [30.0], "w*", ms=8)
    ax.set_xlabel(r"$a_*$")
    ax.set_ylabel(r"inclination [deg]")
    figr.tight_layout()
    figr.savefig(FIG / "fig_shadow_sweep.pdf")
    plt.close(figr)
    print("  wrote fig_shadow_sweep.pdf")


def main() -> None:
    print("generating paper figures ...")
    fig_s2_orbit()
    fig_s2_rv()
    fig_s2_residuals()
    fig_redshift_excess()
    fig_shadow_boundary()
    fig_brightness_profile()
    fig_nohair_profile()
    fig_subring_panels()
    fig_disk_render()
    fig_shadow_sweep()
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
