"""Auto-generate every figure in the paper and thesis from simulation output
(paper/THESIS_PLAN.md §2.2: no hand-drawn figures;
`make figures` calls this).

Inputs (produced by the tools in validation/ and science/):
    validation/reference_images/{schwarzschild,kerr}_cpu*.json  shadow metrology
    validation/phase3/brightness_profile.csv                    disk beaming
    validation/phase3/subring_report.json + subring_n*.png      photon ring
    validation/phase4/s2_fit.json, s2_residuals.csv,
                      s2_model_track.csv, s2_redshift_curve.csv S2 fit
    validation/phase5/s2_fit.json                               no-hair profile
    validation/phase10/sweep.csv                                spin-incl sweep
    validation/phase7/*.compare.json                            CPU/GPU diff
    validation/phase11/bench.json                               benchmarks
    validation/thesis/*.csv                                     thesis data tools
    validation/thesis/mcmc_*.npz                                MCMC chains
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figstyle import OKABE, CYCLE, apply_style

apply_style()

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
THESIS = ROOT / "validation" / "thesis"


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


def save(figr, name: str, **kw) -> None:
    figr.tight_layout()
    figr.savefig(FIG / name, **kw)
    plt.close(figr)
    print(f"  wrote {name}")


# =====================================================================
# The ten original paper figures, restyled for the full-width report
# (THESIS_PLAN §2.2a: shared palette, larger fonts, pedagogical
# annotations; data unchanged).
# =====================================================================

def fig_s2_orbit() -> None:
    fit = ROOT / "validation/phase4"
    track_p = fit / "s2_model_track.csv"
    ast_p = ROOT / "data/gillessen_2017/s2_astrometry.csv"
    if not track_p.exists():
        return warn_skip("s2_orbit", track_p)
    track = read_csv(track_p)
    ast = read_csv(ast_p)
    ep = track["epoch_yr"].astype(float)
    ra = track["ra_mas"].astype(float)
    dec = track["dec_mas"].astype(float)

    figr, ax = plt.subplots(figsize=(4.6, 5.4))
    ax.plot(ra, dec, "-", color="0.3", lw=1.0, label="GR model (this work)")
    ax.errorbar(ast["ra_mas"].astype(float), ast["dec_mas"].astype(float),
                xerr=ast["sig_ra_mas"].astype(float),
                yerr=ast["sig_dec_mas"].astype(float),
                fmt="o", ms=2.5, lw=0.6, color=OKABE["blue"], alpha=0.8,
                label="Gillessen+17 astrometry")
    ax.plot([0], [0], "*", ms=12, color="k", label="Sgr A*")
    # pericentre passages (2002.33, 2018.38)
    for t_peri, dy in [(2002.33, 8), (2018.379, -10)]:
        k = int(np.argmin(np.abs(ep - t_peri)))
        ax.annotate(f"pericentre\n{t_peri:.2f}", (ra[k], dec[k]),
                    xytext=(ra[k] - 6, dec[k] + dy), fontsize=8,
                    color=OKABE["vermillion"],
                    arrowprops=dict(arrowstyle="->", lw=0.7,
                                    color=OKABE["vermillion"]))
    ax.set_xlabel(r"$\Delta$RA [mas] (East $\rightarrow$)")
    ax.set_ylabel(r"$\Delta$Dec [mas]")
    ax.invert_xaxis()  # astronomical convention: East left
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    save(figr, "fig_s2_orbit.pdf")


def fig_s2_rv() -> None:
    fit = ROOT / "validation/phase4"
    track_p = fit / "s2_model_track.csv"
    rv_p = ROOT / "data/gillessen_2017/s2_radial_velocity.csv"
    if not track_p.exists():
        return warn_skip("s2_rv", track_p)
    track = read_csv(track_p)
    rv = read_csv(rv_p)

    figr, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(track["epoch_yr"].astype(float), track["vr_kms"].astype(float),
            "-", color="0.3", lw=1.0, label="GR model")
    ax.errorbar(rv["epoch_yr"].astype(float), rv["vr_kms"].astype(float),
                yerr=rv["sig_vr_kms"].astype(float), fmt="o", ms=3, lw=0.7,
                color=OKABE["vermillion"], label="Gillessen+17 RV")
    for t_peri in (2002.33, 2018.379):
        ax.axvline(t_peri, color="0.75", lw=0.6, zorder=0)
    ax.annotate("pericentre passages", (2002.6, 3300), fontsize=8, color="0.4")
    ax.set_xlabel("epoch [yr]")
    ax.set_ylabel(r"$v_r$ [km/s]")
    ax.legend()
    save(figr, "fig_s2_rv.pdf")


def fig_s2_residuals() -> None:
    p = ROOT / "validation/phase4/s2_residuals.csv"
    if not p.exists():
        return warn_skip("s2_residuals", p)
    r = read_csv(p)
    epoch = r["epoch_yr"].astype(float)
    nres = r["norm_resid"].astype(float)
    types = r["type"]

    figr, axes = plt.subplots(3, 1, figsize=(5.6, 4.6), sharex=True)
    for ax, t, color, lab in zip(
            axes, ["ra", "dec", "rv"],
            [OKABE["blue"], OKABE["green"], OKABE["vermillion"]],
            [r"RA/$\sigma$", r"Dec/$\sigma$", r"$v_r/\sigma$"]):
        m = types == t
        ax.axhline(0, color="0.6", lw=0.5)
        ax.plot(epoch[m], nres[m], "o", ms=2.5, color=color)
        ax.set_ylabel(lab)
        ax.set_ylim(-4, 4)
    axes[-1].set_xlabel("epoch [yr]")
    save(figr, "fig_s2_residuals.pdf")


def fig_redshift_excess() -> None:
    p = ROOT / "validation/phase4/s2_redshift_curve.csv"
    if not p.exists():
        return warn_skip("redshift_excess", p)
    r = read_csv(p)
    t = r["epoch_yr"].astype(float)

    figr, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(t, r["v_excess_kms"].astype(float), "-", color="k", lw=1.3,
            label=r"GR excess $c(z_{\rm tot}-\beta_{\rm los})$")
    ax.plot(t, r["v_trans_kms"].astype(float), "--", color=OKABE["blue"],
            lw=0.9, label=r"transverse Doppler $c(\gamma-1)$")
    ax.plot(t, r["v_grav_kms"].astype(float), ":", color=OKABE["vermillion"],
            lw=1.1, label=r"gravitational $c(1/\alpha-1)$")
    ax.axvline(2018.379, color="0.7", lw=0.6)
    ax.annotate("pericentre\n2018.38", (2018.42, 60), fontsize=8, color="0.4")
    ax.set_xlabel("epoch [yr]")
    ax.set_ylabel("velocity equivalent [km/s]")
    ax.legend()
    save(figr, "fig_redshift_excess.pdf")


def fig_shadow_boundary() -> None:
    ks = ROOT / "validation/reference_images/kerr_cpu_a94.json"
    ss = ROOT / "validation/reference_images/schwarzschild_cpu.json"
    if not ks.exists():
        return warn_skip("shadow_boundary", ks)
    kerr = json.loads(ks.read_text())
    schw = json.loads(ss.read_text()) if ss.exists() else None

    figr, ax = plt.subplots(figsize=(4.4, 4.4))
    psi = np.array([b["psi_rad"] for b in kerr["boundary"]])
    alp = np.array([b["alpha_rad"] for b in kerr["boundary"]])
    rc = kerr["r_cam_M"]
    b_M = np.sin(alp) * rc / np.sqrt(1.0 - 2.0 / rc)
    ax.plot(np.append(b_M * np.cos(psi), b_M[0] * np.cos(psi[0])),
            np.append(b_M * np.sin(psi), b_M[0] * np.sin(psi[0])),
            "-o", ms=3, lw=1.0, color=OKABE["purple"],
            label=rf"Kerr $a_*={kerr['a_star']}$ (traced)")
    # Analytic Bardeen (1973) critical curve at the render's inclination:
    # alpha = -xi/sin i, beta = ±sqrt(eta + a^2 cos^2 i - xi^2 cot^2 i).
    # Same orientation convention as the camera (image up = spin axis), so
    # the traced points must land on it side for side — the check the
    # handedness correction restored.
    a = kerr["a_star"]
    inc = np.radians(kerr.get("incl_deg", 90.0))
    rp = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(-a)))
    rr = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(+a)))
    rt = np.linspace(rp * (1 + 1e-9), rr * (1 - 1e-9), 4000)
    xi = (rt**2 * (3.0 - rt) - a**2 * (rt + 1.0)) / (a * (rt - 1.0))
    eta = rt**3 * (4.0 * a**2 - rt * (rt - 3.0) ** 2) / (a**2 * (rt - 1.0) ** 2)
    beta2 = eta + a**2 * np.cos(inc) ** 2 - xi**2 / np.tan(inc) ** 2
    ok = beta2 >= 0.0
    al, be = -xi[ok] / np.sin(inc), np.sqrt(beta2[ok])
    ax.plot(np.concatenate([al, al[::-1]]), np.concatenate([be, -be[::-1]]),
            "-", lw=0.9, color=OKABE["orange"], zorder=1,
            label=rf"Bardeen (1973), $i={np.degrees(inc):.0f}^\circ$")
    if schw:
        b0 = np.sqrt(27.0)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(b0 * np.cos(th), b0 * np.sin(th), "--", lw=0.8, color="0.5",
                label=r"Schwarzschild $\sqrt{27}\,M$")
    ax.annotate("prograde side\n(flattened)", (min(b_M * np.cos(psi)), 0),
                xytext=(-4.7, 3.6), fontsize=8, color="0.35",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="0.35"))
    ax.set_xlabel(r"$b_x$ [$M$]")
    ax.set_ylabel(r"$b_y$ [$M$]")
    ax.set_aspect("equal")
    ax.legend(loc="lower left")
    save(figr, "fig_shadow_boundary.pdf")


def fig_brightness_profile() -> None:
    p = ROOT / "validation/phase3/brightness_profile.csv"
    if not p.exists():
        return warn_skip("brightness_profile", p)
    r = read_csv(p)
    x = r["x_norm"].astype(float)
    I = r["I_bol"].astype(float)
    g = r["g"].astype(float)

    figr, (ax, ax2) = plt.subplots(2, 1, figsize=(5.6, 4.2), sharex=True,
                                   height_ratios=[2, 1])
    ax.semilogy(x, I, ".", ms=2, color=OKABE["orange"])
    # Every sample on the major axis is a direct (n = 0) image at this
    # inclination — the n = 1 ring lies above/below the shadow (Fig.
    # subrings) — so label what is actually plotted, at data positions:
    # the beamed approaching limb (image left, spin axis up), the receding
    # limb, and the shadow gap between the steep inner edges.
    pos = I > 0
    xl = x[pos & (x < 0)][np.argmax(I[pos & (x < 0)])]
    xr = x[pos & (x > 0)][np.argmax(I[pos & (x > 0)])]
    ax.annotate("approaching\nlimb", (xl, I.max()), xytext=(xl - 0.05, I.max() * 0.05),
                fontsize=8, color="0.35", ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.35"))
    ax.annotate("receding\nlimb", (xr, I[pos & (x > 0)].max()),
                xytext=(xr + 0.08, I[pos & (x > 0)].max() * 0.02),
                fontsize=8, color="0.35", ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.35"))
    gap = 0.5 * (x[pos & (x < 0)].max() + x[pos & (x > 0)].min())
    ax.text(gap, I[pos].min() * 3, "shadow", fontsize=8, color="0.35", ha="center")
    ax.set_ylabel(r"$I_{\rm bol}$ [arb.]")
    ax2.plot(x, g, ".", ms=2, color=OKABE["blue"])
    ax2.axhline(1.0, color="0.6", lw=0.5)
    ax2.set_ylabel(r"$g=\nu_{\rm obs}/\nu_{\rm em}$")
    ax2.set_xlabel("image-plane position [normalised]")
    save(figr, "fig_brightness_profile.pdf")


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

    figr, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(eq / 1e5, c2 - c2.min(), "o-", ms=4, lw=1.0, color=OKABE["purple"])
    for lev, lab in [(1.0, r"$\Delta\chi^2=1$ (68%)"),
                     (3.84, r"$\Delta\chi^2=3.84$ (95%)")]:
        ax.axhline(lev, color="0.5", lw=0.6, ls="--")
        ax.annotate(lab, (eq.min() / 1e5, lev + 0.15), fontsize=8, color="0.4")
    ax.set_xlabel(r"$\epsilon_q\ [\times 10^5]$")
    ax.set_ylabel(r"$\Delta\chi^2$")
    save(figr, "fig_nohair_profile.pdf")


def fig_subring_panels() -> None:
    base = ROOT / "validation/phase3"
    paths = [base / f"subring_n{n}.png" for n in range(3)]
    if not paths[0].exists():
        return warn_skip("subring_panels", paths[0])
    from PIL import Image
    figr, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for ax, p, n in zip(axes, paths, range(3)):
        ax.imshow(np.asarray(Image.open(p)))
        ax.set_title(f"$n={n}$", fontsize=10)
        ax.axis("off")
    save(figr, "fig_subrings.pdf", dpi=300)


def fig_disk_render() -> None:
    p = ROOT / "validation/phase3/disk_nt.png"
    k = ROOT / "validation/reference_images/kerr_cpu_a94.png"
    if not p.exists() or not k.exists():
        return warn_skip("disk_render", p)
    from PIL import Image
    figr, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    for ax, path, title in zip(
            axes, [k, p],
            [r"Kerr $a_*=0.94$, band disk", "Novikov–Thorne + beaming"]):
        ax.imshow(np.asarray(Image.open(path)))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    save(figr, "fig_disk_render.pdf", dpi=300)


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

    figr, ax = plt.subplots(figsize=(5.6, 4.0))
    pc = ax.pcolormesh(A, I, D, shading="auto", cmap="magma")
    # EHT photon-ring band 51.8 ± 2.3 μas
    cs = ax.contour(A, I, D, levels=[51.8 - 2.3, 51.8, 51.8 + 2.3],
                    colors=["w", "c", "w"], linewidths=[0.6, 1.4, 0.6])
    ax.clabel(cs, fontsize=7, fmt="%.1f")
    figr.colorbar(pc, ax=ax, label=r"ring diameter [$\mu$as]")
    ax.plot([0.94], [30.0], "w*", ms=10)
    ax.annotate("EHT best geometry", (0.94, 30), xytext=(0.55, 12),
                fontsize=8, color="w",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="w"))
    ax.set_xlabel(r"$a_*$")
    ax.set_ylabel(r"inclination [deg]")
    save(figr, "fig_shadow_sweep.pdf")


# =====================================================================
# New thesis figures (THESIS_PLAN §2.2) — Kerr geometry (Ch 2)
# =====================================================================

def fig_radii_vs_spin() -> None:
    p = THESIS / "radii_vs_spin.csv"
    if not p.exists():
        return warn_skip("radii_vs_spin", p)
    r = read_csv(p)
    a = r["a_star"].astype(float)

    figr, ax = plt.subplots(figsize=(5.6, 4.0))
    series = [
        ("r_isco_retro", r"$r_{\rm ISCO}^-$ (retrograde)", OKABE["vermillion"], "--"),
        ("r_isco_pro", r"$r_{\rm ISCO}^+$ (prograde)", OKABE["vermillion"], "-"),
        ("r_mb_retro", r"$r_{\rm mb}^-$", OKABE["green"], "--"),
        ("r_mb_pro", r"$r_{\rm mb}^+$", OKABE["green"], "-"),
        ("r_ph_retro", r"$r_{\rm ph}^-$", OKABE["blue"], "--"),
        ("r_ph_pro", r"$r_{\rm ph}^+$", OKABE["blue"], "-"),
        ("r_ergo_eq", r"$r_E(\pi/2)$ ergosphere", OKABE["orange"], "-."),
        ("r_plus", r"$r_+$ horizon", "k", "-"),
    ]
    for key, lab, c, ls in series:
        ax.plot(a, r[key].astype(float), ls, color=c, lw=1.1, label=lab)
    ax.fill_between(a, 0, r["r_plus"].astype(float), color="0.85")
    ax.set_xlabel(r"$a_*$")
    ax.set_ylabel(r"radius [$M$]")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 9.5)
    ax.legend(ncol=2, loc="upper right")
    save(figr, "fig_radii_vs_spin.pdf")


def fig_efficiency_vs_spin() -> None:
    p = THESIS / "efficiency_vs_spin.csv"
    if not p.exists():
        return warn_skip("efficiency_vs_spin", p)
    r = read_csv(p)
    a = r["a_star"].astype(float)

    figr, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(a, 100 * r["eta_pro"].astype(float), "-", color=OKABE["blue"],
            lw=1.3, label=r"prograde $\eta = 1-E_{\rm ISCO}$")
    ax.plot(a, 100 * r["eta_retro"].astype(float), "--",
            color=OKABE["vermillion"], lw=1.1, label="retrograde")
    ax.axhline(100 * (1 - np.sqrt(8.0 / 9.0)), color="0.6", lw=0.6, ls=":")
    ax.annotate(r"Schwarzschild 5.72%", (0.02, 6.1), fontsize=8, color="0.4")
    ax.set_xlabel(r"$a_*$")
    ax.set_ylabel(r"accretion efficiency $\eta$ [%]")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left")
    save(figr, "fig_efficiency_vs_spin.pdf")


def fig_energetics() -> None:
    p = THESIS / "energetics_r.csv"
    if not p.exists():
        return warn_skip("energetics", p)
    r = read_csv(p)
    a = r["a_star"].astype(float)
    figr, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
    for spin, c in zip(np.unique(a), CYCLE):
        m = a == spin
        rr = r["r"].astype(float)[m]
        axes[0].plot(rr, r["E"].astype(float)[m], color=c, lw=1.1,
                     label=rf"$a_*={spin:g}$")
        axes[1].plot(rr, r["L"].astype(float)[m], color=c, lw=1.1)
        axes[2].plot(rr, r["Omega"].astype(float)[m], color=c, lw=1.1)
    axes[0].set_ylabel(r"$E(r)$")
    axes[1].set_ylabel(r"$L(r)$ [$M$]")
    axes[2].set_ylabel(r"$\Omega(r)$ [$M^{-1}$]")
    for ax in axes:
        ax.set_xlabel(r"$r$ [$M$]")
    axes[0].legend(fontsize=7)
    save(figr, "fig_energetics.pdf")


def fig_timelike_potential() -> None:
    p = THESIS / "timelike_potential.csv"
    if not p.exists():
        return warn_skip("timelike_potential", p)
    r = read_csv(p)
    a = r["a_star"].astype(float)
    L = r["L"].astype(float)
    figr, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, spin in zip(axes, [0.0, 0.94]):
        m0 = np.abs(a - spin) < 1e-9
        for Lv, c in zip(np.unique(L[m0]), CYCLE):
            m = m0 & (np.abs(L - Lv) < 1e-12)
            ax.plot(r["r"].astype(float)[m], r["Veff"].astype(float)[m],
                    color=c, lw=1.1, label=rf"$L={Lv:.2f}M$")
        ax.axhline(1.0, color="0.7", lw=0.5)
        ax.set_title(rf"$a_*={spin:g}$ (prograde)", fontsize=10)
        ax.set_xlabel(r"$r$ [$M$]")
        ax.set_xlim(0, 40)
        ax.legend(fontsize=7)
    axes[0].set_ylabel(r"$V_{\rm eff}(r) = E_+(r)$")
    axes[0].set_ylim(0.88, 1.06)
    save(figr, "fig_timelike_potential.pdf")


# =====================================================================
# Geodesic structure (Ch 3)
# =====================================================================

def fig_photon_potential() -> None:
    p = THESIS / "photon_potential.csv"
    if not p.exists():
        return warn_skip("photon_potential", p)
    r = read_csv(p)
    a = r["a_star"].astype(float)
    b = r["b"].astype(float)
    figr, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, spin, title in zip(axes, [0.0, 0.94],
                               [r"Schwarzschild", r"Kerr $a_*=0.94$"]):
        m0 = np.abs(a - spin) < 1e-9
        for bv, c in zip(np.unique(b[m0]), CYCLE):
            m = m0 & (np.abs(b - bv) < 1e-12)
            ax.plot(r["r"].astype(float)[m], r["R_over_E2"].astype(float)[m],
                    color=c, lw=1.1, label=rf"$b={bv:.3f}M$")
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$r$ [$M$]")
        ax.set_ylim(-400, 900)
        ax.legend(fontsize=7)
    axes[0].set_ylabel(r"$R(r)/E^2$")
    save(figr, "fig_photon_potential.pdf")


def fig_photon_trajectories() -> None:
    p = THESIS / "trajectories_photon.csv"
    if not p.exists():
        return warn_skip("photon_trajectories", p)
    r = read_csv(p)
    lab = r["label"]
    b = r["b"].astype(float)
    rr = r["r"].astype(float)
    ph = r["phi"].astype(float)

    figr, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
    for ax, sel, a_star, title in [
            (axes[0], lab == "schw", 0.0, "Schwarzschild"),
            (axes[1], (lab == "kerr_pro") | (lab == "kerr_retro"), 0.94,
             r"Kerr $a_*=0.94$ (equatorial)")]:
        rh = 1.0 + np.sqrt(1.0 - a_star**2)
        for bv, c in zip(np.unique(b[sel]), CYCLE + CYCLE):
            m = sel & (b == bv)
            ax.plot(rr[m] * np.cos(ph[m]), rr[m] * np.sin(ph[m]), lw=0.9,
                    color=c)
        ax.add_patch(plt.Circle((0, 0), rh, color="k", zorder=5))
        if a_star == 0.0:
            ax.add_patch(plt.Circle((0, 0), 3.0, fill=False, color="0.6",
                                    lw=0.6, ls="--", zorder=4))
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$x$ [$M$]")
    axes[0].set_ylabel(r"$y$ [$M$]")
    save(figr, "fig_photon_trajectories.pdf")


def fig_rosette() -> None:
    p = THESIS / "trajectory_rosette.csv"
    if not p.exists():
        return warn_skip("rosette", p)
    r = read_csv(p)
    rr = r["r"].astype(float)
    ph = r["phi"].astype(float)
    nrm = r["norm"].astype(float)

    figr, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.4),
                                   width_ratios=[1.2, 1])
    ax.plot(rr * np.cos(ph), rr * np.sin(ph), lw=0.7, color=OKABE["blue"])
    ax.add_patch(plt.Circle((0, 0), 2.0, color="k", zorder=5))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [$M$]")
    ax.set_ylabel(r"$y$ [$M$]")
    ax.set_title(r"bound orbit $r\in[8,25]\,M$: rosette", fontsize=10)
    ax2.semilogy(r["tau"].astype(float), np.abs(nrm + 1.0) + 1e-17,
                 lw=0.7, color=OKABE["vermillion"])
    ax2.set_xlabel(r"proper time $\tau$ [$M$]")
    ax2.set_ylabel(r"$|g_{\mu\nu}u^\mu u^\nu + 1|$")
    ax2.set_title("normalisation drift", fontsize=10)
    save(figr, "fig_rosette.pdf")


# =====================================================================
# Numerics evidence (Ch 4)
# =====================================================================

def fig_projection_boundary() -> None:
    p = THESIS / "projection_boundary.csv"
    if not p.exists():
        return warn_skip("projection_boundary", p)
    r = read_csv(p)
    tol = r["tol"].astype(float)
    proj = r["projection"].astype(int)
    dev = np.abs(r["rel_dev"].astype(float))

    figr, ax = plt.subplots(figsize=(5.2, 3.4))
    for pv, c, lab, mk in [(1, OKABE["blue"], "projection ON", "o"),
                           (0, OKABE["vermillion"], "projection OFF", "s")]:
        m = proj == pv
        ax.loglog(tol[m], dev[m] + 1e-16, mk + "-", color=c, ms=4, lw=1.0,
                  label=lab)
    ax.set_xlabel("integrator tolerance (atol = rtol)")
    ax.set_ylabel(r"$|\Delta\alpha_{\rm sh}|/\alpha_{\rm sh}$ vs Synge")
    ax.legend()
    save(figr, "fig_projection_boundary.pdf")


def fig_drift_vs_tol() -> None:
    p = THESIS / "drift_vs_tol.csv"
    if not p.exists():
        return warn_skip("drift_vs_tol", p)
    r = read_csv(p)
    tol = r["tol"].astype(float)
    proj = r["projection"].astype(int)

    figr, ax = plt.subplots(figsize=(5.2, 3.4))
    for key, c, lab in [("dE_rel", OKABE["blue"], r"$|\Delta E|/E$"),
                        ("dLz_rel", OKABE["green"], r"$|\Delta L_z|/L_z$"),
                        ("dQ_rel", OKABE["vermillion"], r"$|\Delta Q|/Q$")]:
        for pv, ls in [(1, "-"), (0, "--")]:
            m = proj == pv
            ax.loglog(tol[m], r[key].astype(float)[m] + 1e-18, ls, color=c,
                      lw=1.0, marker="o" if pv else "s", ms=3,
                      label=lab + (" (proj)" if pv else " (no proj)"))
    ax.axhline(1e-6, color="0.6", lw=0.6, ls=":")
    ax.annotate("health-check gate $10^{-6}$", (2e-13, 1.5e-6), fontsize=7,
                color="0.4")
    ax.set_xlabel("integrator tolerance (atol = rtol)")
    ax.set_ylabel("max relative drift along grazing ray")
    ax.legend(fontsize=6.5, ncol=2)
    save(figr, "fig_drift_vs_tol.pdf")


def fig_step_size() -> None:
    p = THESIS / "step_size.csv"
    if not p.exists():
        return warn_skip("step_size", p)
    r = read_csv(p)
    tau = r["tau"].astype(float)
    figr, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.semilogy(tau, r["h"].astype(float), lw=0.8, color=OKABE["blue"],
                label=r"accepted step $h$")
    ax2 = ax.twinx()
    ax2.plot(tau, r["r"].astype(float), lw=0.8, color=OKABE["vermillion"],
             label=r"$r(\tau)$")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$r$ [$M$] (dashed)", color=OKABE["vermillion"])
    ax.set_xlabel(r"Mino time $\tau$")
    ax.set_ylabel(r"step size $h$", color=OKABE["blue"])
    save(figr, "fig_step_size.pdf")


# =====================================================================
# Shadow / photon shell (Ch 5, Ch 7)
# =====================================================================

def fig_critical_curves() -> None:
    # Analytic Bardeen critical curves, edge-on observer: alpha = -xi,
    # beta = ±sqrt(eta). Pure closed-form evaluation (Bardeen 1973).
    figr, ax = plt.subplots(figsize=(4.8, 4.4))
    for a, c in zip([1e-4, 0.5, 0.94, 0.998], CYCLE):
        rp = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(-a)))
        rr = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(+a)))
        rt = np.linspace(rp * (1 + 1e-9), rr * (1 - 1e-9), 800)
        xi = (rt**2 * (3.0 - rt) - a**2 * (rt + 1.0)) / (a * (rt - 1.0))
        eta = rt**3 * (4.0 * a**2 - rt * (rt - 3.0) ** 2) / (a**2 * (rt - 1.0) ** 2)
        ok = eta >= 0.0
        alpha, beta = -xi[ok], np.sqrt(eta[ok])
        lab = rf"$a_*={a:.3g}$" if a > 1e-3 else r"$a_*=0$"
        ax.plot(np.concatenate([alpha, alpha[::-1]]),
                np.concatenate([beta, -beta[::-1]]), color=c, lw=1.1,
                label=lab)
    ax.set_aspect("equal")
    # alpha = -xi: prograde photons (xi > 0) appear on the image LEFT for an
    # observer whose image "up" is the spin axis (thesis appendix "Image
    # handedness"; the label used to point the other way).
    ax.set_xlabel(r"$\alpha$ [$M$] ($\leftarrow$ prograde)")
    ax.set_ylabel(r"$\beta$ [$M$]")
    ax.legend(loc="upper right", fontsize=7)
    save(figr, "fig_critical_curves.pdf")


def fig_spherical_orbits() -> None:
    p = THESIS / "spherical_orbits.csv"
    if not p.exists():
        return warn_skip("spherical_orbits", p)
    r = read_csv(p)
    a = r["a_star"].astype(float)
    figr, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for spin, c in zip(np.unique(a), CYCLE):
        m = a == spin
        rt = r["r_tilde"].astype(float)[m]
        axes[0].plot(rt, r["xi"].astype(float)[m], color=c, lw=1.1,
                     label=rf"$a_*={spin:g}$")
        axes[1].plot(rt, r["gamma"].astype(float)[m], color=c, lw=1.1)
    axes[0].axhline(0, color="0.7", lw=0.5)
    axes[0].set_ylabel(r"$\xi(\tilde r)$ [$M$]")
    axes[1].set_ylabel(r"Lyapunov $\gamma(\tilde r)$")
    axes[1].axhline(np.pi, color="0.7", lw=0.5, ls=":")
    for ax in axes:
        ax.set_xlabel(r"photon-shell radius $\tilde r$ [$M$]")
    axes[0].legend(fontsize=7)
    save(figr, "fig_spherical_orbits.pdf")


def fig_lyapunov_vs_spin() -> None:
    p = THESIS / "lyapunov_vs_spin.csv"
    if not p.exists():
        return warn_skip("lyapunov_vs_spin", p)
    r = read_csv(p)
    a = r["a_star"].astype(float)
    figr, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(a, r["gamma_polar"].astype(float), "-", color=OKABE["blue"],
            lw=1.2, label=r"polar orbit ($\xi=0$)")
    ax.plot(a, r["gamma_pro"].astype(float), "--", color=OKABE["green"],
            lw=1.0, label="equatorial prograde")
    ax.plot(a, r["gamma_retro"].astype(float), ":", color=OKABE["vermillion"],
            lw=1.2, label="equatorial retrograde")
    ax.axhline(np.pi, color="0.7", lw=0.6)
    ax.annotate(r"$\gamma=\pi$ (Schwarzschild)", (0.03, np.pi + 0.02),
                fontsize=8, color="0.4")
    ax.set_xlabel(r"$a_*$")
    ax.set_ylabel(r"Lyapunov exponent $\gamma$ per half-orbit")
    ax.legend()
    save(figr, "fig_lyapunov_vs_spin.pdf")


def fig_subring_sweep() -> None:
    p = THESIS / "subring_sweep.csv"
    if not p.exists():
        return warn_skip("subring_sweep", p)
    r = read_csv(p)
    a = r["a_star"].astype(float)
    figr, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax.plot(a, r["ratio_n0_n1"].astype(float), "o-", color=OKABE["blue"],
            lw=1.1, ms=4, label=r"$F_0/F_1$")
    f2 = r["flux_n2"].astype(float)
    ok = f2 > 1e-6
    ax.plot(a[ok], r["ratio_n1_n2"].astype(float)[ok], "s--",
            color=OKABE["vermillion"], lw=1.1, ms=4, label=r"$F_1/F_2$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$a_*$")
    ax.set_ylabel("sub-ring flux ratio")
    ax.legend()
    for key, c, lab in [("width_n1_psi0_rad", OKABE["blue"], r"$\psi=0$"),
                        ("width_n1_psi90_rad", OKABE["green"], r"$\psi=90°$"),
                        ("width_n1_psi270_rad", OKABE["purple"], r"$\psi=270°$")]:
        w = r[key].astype(float) * 1e3
        m = w > 0
        ax2.plot(a[m], w[m], "o-", color=c, lw=1.0, ms=4, label=lab)
    ax2.set_xlabel(r"$a_*$")
    ax2.set_ylabel(r"$n=1$ band width [mrad]")
    ax2.legend(fontsize=7)
    save(figr, "fig_subring_sweep.pdf")


# =====================================================================
# S-star forward model + fit diagnostics (Ch 8, Ch 9)
# =====================================================================

def _kepler_solve(Manom: np.ndarray, e: float) -> np.ndarray:
    E = Manom.copy()
    for _ in range(60):
        E = E - (E - e * np.sin(E) - Manom) / (1.0 - e * np.cos(E))
    return E


def fig_romer_delay() -> None:
    p = ROOT / "validation/phase4/s2_fit.json"
    if not p.exists():
        return warn_skip("romer_delay", p)
    j = json.loads(p.read_text())
    a_mas, e = j["a_mas"], j["e"]
    inc = np.radians(j["incl_deg"])
    om = np.radians(j["omega_deg"])
    P, tp, R0 = j["period_yr"], j["t_peri_yr"], j["R0_pc"]
    a_au = (a_mas / 1000.0) * R0  # angular semimajor axis → AU
    t = np.linspace(tp, tp + P, 2000)
    Manom = 2 * np.pi * (t - tp) / P
    E = _kepler_solve(Manom, e)
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                        np.sqrt(1 - e) * np.cos(E / 2))
    r_au = a_au * (1 - e * np.cos(E))
    z_au = r_au * np.sin(nu + om) * np.sin(inc)  # line-of-sight displacement
    delay_days = z_au * 499.004784 / 86400.0     # 1 AU = 499 light-seconds

    figr, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(t, delay_days, lw=1.2, color=OKABE["blue"])
    ax.axhline(0, color="0.7", lw=0.5)
    ax.axvline(tp + P / 2 if e < 0.5 else tp, color="0.85", lw=0.5)
    ax.annotate("pericentre", (tp + 0.1, delay_days.min() * 0.9), fontsize=8,
                color="0.4")
    ax.set_xlabel("epoch [yr]")
    ax.set_ylabel(r"R{\o}mer delay $-z\sin i/c$ [light-days]")
    save(figr, "fig_romer_delay.pdf")


def fig_precession_decomposition() -> None:
    p = ROOT / "validation/phase4/s2_fit.json"
    if not p.exists():
        return warn_skip("precession_decomposition", p)
    j = json.loads(p.read_text())
    # Fitted semi-major axis in gravitational radii: a_AU / (M · r_g(1 M_sun))
    # with r_g(1 M_sun) = GM_sun/c² = 1.4766 km = 9.8708e-9 AU.
    a_au = (j["a_mas"] / 1000.0) * j["R0_pc"]
    aM = a_au / (j["M_msun"] * 9.8708e-9)
    e = j["e"]
    p_slr = aM * (1 - e * e)
    arcmin = 3437.746770784939
    spins = np.linspace(0, 0.998, 100)
    dw_s = 6 * np.pi / p_slr * np.ones_like(spins) * arcmin
    dw_lt = -8 * np.pi * spins * (1.0 / p_slr) ** 1.5 * arcmin
    dw_q = -3 * np.pi * (-(spins**2)) / p_slr**2 * arcmin  # Kerr q̃ = −a*²

    figr, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.semilogy(spins, dw_s, "-", color=OKABE["blue"], lw=1.2,
                label=r"Schwarzschild $6\pi M/p$")
    ax.semilogy(spins, np.abs(dw_lt), "--", color=OKABE["green"], lw=1.1,
                label=r"|Lense–Thirring| $8\pi a_*(M/p)^{3/2}$")
    ax.semilogy(spins, np.abs(dw_q) + 1e-12, ":", color=OKABE["vermillion"],
                lw=1.2, label=r"|quadrupole| $3\pi a_*^2 (M/p)^2$")
    ax.set_xlabel(r"$a_*$")
    ax.set_ylabel("pericentre advance [arcmin/orbit]")
    ax.set_title("S2 orbit: relativistic precession hierarchy", fontsize=10)
    ax.legend()
    save(figr, "fig_precession_decomposition.pdf")


def fig_o_minus_c() -> None:
    p = ROOT / "validation/phase4/s2_residuals.csv"
    if not p.exists():
        return warn_skip("o_minus_c", p)
    r = read_csv(p)
    epoch = r["epoch_yr"].astype(float)
    obs = r["obs"].astype(float)
    mod = r["model"].astype(float)
    sig = r["sigma"].astype(float)
    types = r["type"]

    figr, axes = plt.subplots(3, 1, figsize=(5.6, 5.2), sharex=True)
    for ax, t, c, lab in zip(
            axes, ["ra", "dec", "rv"],
            [OKABE["blue"], OKABE["green"], OKABE["vermillion"]],
            [r"O$-$C RA [mas]", r"O$-$C Dec [mas]", r"O$-$C $v_r$ [km/s]"]):
        m = types == t
        ax.axhline(0, color="0.6", lw=0.5)
        ax.errorbar(epoch[m], (obs - mod)[m], yerr=sig[m], fmt="o", ms=2.5,
                    lw=0.5, color=c)
        ax.set_ylabel(lab)
    axes[-1].set_xlabel("epoch [yr]")
    save(figr, "fig_o_minus_c.pdf")


def fig_residual_quiver() -> None:
    p = ROOT / "validation/phase4/s2_residuals.csv"
    if not p.exists():
        return warn_skip("residual_quiver", p)
    r = read_csv(p)
    types = r["type"]
    epoch = r["epoch_yr"].astype(float)
    obs = r["obs"].astype(float)
    mod = r["model"].astype(float)
    mra, mdec = types == "ra", types == "dec"
    # pair RA and Dec rows by epoch order (written interleaved by the tool)
    ra_o, ra_m, ep = obs[mra], mod[mra], epoch[mra]
    de_o, de_m = obs[mdec], mod[mdec]

    figr, ax = plt.subplots(figsize=(5.0, 5.6))
    ax.plot(ra_m, de_m, ".", ms=2, color="0.7", label="model positions")
    q = ax.quiver(ra_m, de_m, (ra_o - ra_m), (de_o - de_m), ep,
                  angles="xy", scale_units="xy", scale=0.02, width=0.004,
                  cmap="viridis")
    figr.colorbar(q, ax=ax, label="epoch [yr]", shrink=0.8)
    ax.plot([0], [0], "k*", ms=10)
    ax.annotate(r"arrows $\times 50$", (0.03, 0.03), xycoords="axes fraction",
                fontsize=8, color="0.4")
    ax.set_xlabel(r"$\Delta$RA [mas]")
    ax.set_ylabel(r"$\Delta$Dec [mas]")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    save(figr, "fig_residual_quiver.pdf")


def fig_redshift_decomposition() -> None:
    p = ROOT / "validation/phase4/s2_redshift_curve.csv"
    if not p.exists():
        return warn_skip("redshift_decomposition", p)
    r = read_csv(p)
    t = r["epoch_yr"].astype(float)
    vt = r["v_trans_kms"].astype(float)
    vg = r["v_grav_kms"].astype(float)

    figr, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.fill_between(t, 0, vt, color=OKABE["blue"], alpha=0.55,
                    label=r"transverse Doppler $c(\gamma-1)$")
    ax.fill_between(t, vt, vt + vg, color=OKABE["vermillion"], alpha=0.55,
                    label=r"gravitational $c\,z_{\rm grav}$")
    ax.plot(t, r["v_excess_kms"].astype(float), "k-", lw=1.0,
            label="total GR excess (model)")
    ax.axvline(2018.379, color="0.7", lw=0.6)
    ax.set_xlabel("epoch [yr]")
    ax.set_ylabel("velocity equivalent [km/s]")
    ax.legend(loc="upper left")
    save(figr, "fig_redshift_decomposition.pdf")


# =====================================================================
# Calibration maps (Ch 12) + GPU/perf (Ch 11)
# =====================================================================

def fig_axis_ratio_map() -> None:
    p = ROOT / "validation/phase10/sweep.csv"
    if not p.exists():
        return warn_skip("axis_ratio_map", p)
    r = read_csv(p)
    if "disp_x_M" not in r:
        return warn_skip("axis_ratio_map (needs regenerated sweep.csv)", p)
    a = r["a_star"].astype(float)
    inc = r["incl_deg"].astype(float)
    na, ni = len(np.unique(a)), len(np.unique(inc))
    A, I = a.reshape(na, ni), inc.reshape(na, ni)

    figr, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, key, lab, cmap in [
            (axes[0], "axis_ratio", r"axis ratio $d_x/d_y$", "viridis"),
            (axes[1], "disp_x_M", r"centroid shift $\Delta x$ [$M$]", "coolwarm")]:
        V = r[key].astype(float).reshape(na, ni)
        pc = ax.pcolormesh(A, I, V, shading="auto", cmap=cmap)
        figr.colorbar(pc, ax=ax, label=lab)
        ax.plot([0.94], [30.0], "w*", ms=8)
        ax.set_xlabel(r"$a_*$")
        ax.set_ylabel("inclination [deg]")
    save(figr, "fig_shadow_shape_maps.pdf")


def fig_taylor_trig_error() -> None:
    # Truncation error of the GPU double-precision sin/cos cores
    # (engine/shaders/geodesic.comp.glsl dsin_core/dcos_core): range-reduced
    # Taylor polynomials of degree 11 / 10 on |x| ≤ π/4, evaluated exactly as
    # in the shader.
    x = np.linspace(-np.pi / 4, np.pi / 4, 4001)
    x2 = x * x
    s = x * (1 + x2 * (-1 / 6 + x2 * (1 / 120 + x2 * (-1 / 5040 +
        x2 * (1 / 362880 + x2 * (-1 / 39916800))))))
    c = 1 + x2 * (-0.5 + x2 * (1 / 24 + x2 * (-1 / 720 +
        x2 * (1 / 40320 + x2 * (-1 / 3628800)))))
    figr, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.semilogy(x, np.abs(s - np.sin(x)) + 1e-18, lw=0.9, color=OKABE["blue"],
                label=r"$|\mathrm{dsin\_core}(x) - \sin x|$")
    ax.semilogy(x, np.abs(c - np.cos(x)) + 1e-18, lw=0.9,
                color=OKABE["vermillion"],
                label=r"$|\mathrm{dcos\_core}(x) - \cos x|$")
    ax.axhline(2.2e-16, color="0.6", lw=0.6, ls=":")
    ax.annotate("double-precision ulp", (0, 3e-16), fontsize=7, color="0.4",
                ha="center")
    ax.set_xlabel(r"$x$ (post reduction, $|x| \leq \pi/4$)")
    ax.set_ylabel("absolute error")
    ax.legend()
    save(figr, "fig_taylor_trig_error.pdf")


def fig_bench() -> None:
    p = ROOT / "validation/phase11/bench.json"
    if not p.exists():
        return warn_skip("bench", p)
    j = json.loads(p.read_text())
    figr, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    keys = ["cpu_null_steps_per_s", "cpu_timelike_steps_per_s",
            "cpu_christoffel_per_s"]
    labs = ["null RK45\nsteps/s", "timelike RK45\nsteps/s",
            "Christoffel\nevals/s"]
    ax.bar(labs, [j[k] for k in keys],
           color=[OKABE["blue"], OKABE["green"], OKABE["orange"]])
    ax.set_yscale("log")
    ax.set_ylabel("per second per core (CPU)")
    px = np.array([1280 * 720, 1920 * 1080])
    fps = np.array([j["gpu_fps_hd"], j["gpu_fps_fullhd"]])
    ax2.plot(px / 1e6, fps, "o-", color=OKABE["vermillion"], lw=1.1)
    ax2.plot(px / 1e6, fps[0] * px[0] / px, ":", color="0.6", lw=0.9,
             label=r"$\propto 1/N_{\rm px}$")
    for x, f, lab in zip(px / 1e6, fps, ["720p", "1080p"]):
        ax2.annotate(f"{lab}\n{f:.1f} fps", (x, f + 2), fontsize=8,
                     ha="center")
    ax2.set_xlabel("megapixels")
    ax2.set_ylabel("fps (GTX 1050, fp32 live path)")
    ax2.set_ylim(0, 60)
    ax2.legend()
    save(figr, "fig_bench.pdf")


# =====================================================================
# Disk radiative structure (Ch 6)
# =====================================================================

def fig_disk_profiles() -> None:
    p = THESIS / "disk_profiles.csv"
    if not p.exists():
        return warn_skip("disk_profiles", p)
    r = read_csv(p)
    a = r["a_star"].astype(float)
    figr, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for spin, c in zip(np.unique(a), CYCLE):
        m = a == spin
        rr = r["r"].astype(float)[m]
        axes[0].plot(rr, r["flux_rel"].astype(float)[m], color=c, lw=1.1,
                     label=rf"$a_*={spin:g}$")
        axes[1].plot(rr, r["T_over_Tpeak"].astype(float)[m], color=c, lw=1.1)
    axes[0].set_ylabel(r"$F(r)/F_{\rm max}$")
    axes[1].set_ylabel(r"$T_{\rm eff}(r)/T_{\rm peak}$")
    for ax in axes:
        ax.set_xlabel(r"$r$ [$M$]")
        ax.set_xlim(0, 30)
    axes[0].legend(fontsize=7)
    save(figr, "fig_disk_profiles.pdf")


def fig_cie_locus() -> None:
    try:
        import planck_cie as gl
    except Exception:
        return warn_skip("cie_locus", ROOT / "scripts/planck_cie.py")
    T = np.geomspace(1000.0, 40000.0, 300)
    xyz = np.array([gl.planck_xyz(t) for t in T])
    xy = xyz[:, :2] / xyz.sum(axis=1, keepdims=True)
    rgb = np.array([np.clip(gl.planck_rgb(t), 0, 1) ** (1 / 2.2) for t in T])

    figr, (ax, ax2) = plt.subplots(2, 1, figsize=(5.2, 4.6),
                                   height_ratios=[3, 1])
    for i in range(len(T) - 1):
        ax.plot(xy[i:i + 2, 0], xy[i:i + 2, 1], color=rgb[i], lw=2.2)
    for tmark in (1500, 3000, 6500, 12000, 30000):
        k = int(np.argmin(np.abs(T - tmark)))
        ax.annotate(f"{tmark} K", xy[k], xytext=(xy[k][0] + 0.02, xy[k][1]),
                    fontsize=7, color="0.3")
    ax.set_xlabel("CIE x")
    ax.set_ylabel("CIE y")
    ax.set_title("Planckian locus (Wyman–Sloan–Shirley CMF fits)", fontsize=10)
    ax2.imshow(rgb[np.newaxis, :, :], aspect="auto",
               extent=[np.log10(T[0]), np.log10(T[-1]), 0, 1])
    ax2.set_yticks([])
    ax2.set_xlabel(r"$\log_{10} T$ [K] $\rightarrow$ display colour")
    save(figr, "fig_cie_locus.pdf")


# =====================================================================
# MCMC posteriors (Ch 9 / Ch 10; needs validation/thesis/mcmc_*.npz)
# =====================================================================

_CORNER_LABELS = {
    "M_msun": r"$M$ [$10^6 M_\odot$]",
    "R0_pc": r"$R_0$ [kpc]",
    "a_mas": r"$a$ [mas]",
    "e": r"$e$",
    "incl_deg": r"$i$ [deg]",
    "Omega_deg": r"$\Omega$ [deg]",
    "omega_deg": r"$\omega$ [deg]",
    "t_peri_yr": r"$t_p$ [yr]",
    "eps_q": r"$\epsilon_q$ [$10^5$]",
}
_CORNER_SCALE = {"M_msun": 1e-6, "R0_pc": 1e-3, "eps_q": 1e-5}


def _corner(samples: np.ndarray, names: list[str], out: str) -> None:
    keep = [i for i, n in enumerate(names) if n in _CORNER_LABELS]
    s = samples[:, keep].copy()
    labs = []
    for k, i in enumerate(keep):
        n = names[i]
        s[:, k] *= _CORNER_SCALE.get(n, 1.0)
        labs.append(_CORNER_LABELS[n])
    nd = s.shape[1]
    figr, axes = plt.subplots(nd, nd, figsize=(1.05 * nd, 1.05 * nd))
    for i in range(nd):
        for jj in range(nd):
            ax = axes[i, jj]
            if jj > i:
                ax.axis("off")
                continue
            if i == jj:
                ax.hist(s[:, i], bins=40, color=OKABE["blue"],
                        histtype="stepfilled", alpha=0.7)
                q = np.percentile(s[:, i], [16, 50, 84])
                for qq in q:
                    ax.axvline(qq, color="0.4", lw=0.5,
                               ls="--" if qq != q[1] else "-")
            else:
                ax.hist2d(s[:, jj], s[:, i], bins=45, cmap="Blues")
            if i < nd - 1:
                ax.set_xticklabels([])
            if jj > 0 or i == 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=5)
            if i == nd - 1:
                ax.set_xlabel(labs[jj], fontsize=6)
            if jj == 0 and i > 0:
                ax.set_ylabel(labs[i], fontsize=6)
    figr.subplots_adjust(hspace=0.08, wspace=0.08)
    figr.savefig(FIG / out, bbox_inches="tight")
    plt.close(figr)
    print(f"  wrote {out}")


def fig_corner_s2() -> None:
    p = THESIS / "mcmc_s2_chain.npz"
    if not p.exists():
        return warn_skip("corner_s2", p)
    d = np.load(p, allow_pickle=True)
    _corner(d["samples"], list(d["names"]), "fig_corner_s2.pdf")


def fig_nohair_contours() -> None:
    p = THESIS / "mcmc_nohair_chain.npz"
    if not p.exists():
        return warn_skip("nohair_contours", p)
    d = np.load(p, allow_pickle=True)
    names = list(d["names"])
    s = d["samples"]
    iM, iq = names.index("M_msun"), names.index("eps_q")

    figr, ax = plt.subplots(figsize=(4.6, 3.6))
    h, xe, ye = np.histogram2d(s[:, iq] / 1e5, s[:, iM] / 1e6, bins=60)
    h = h.T
    hs = np.sort(h.ravel())[::-1]
    cdf = np.cumsum(hs) / hs.sum()
    lev = sorted({hs[np.searchsorted(cdf, q)] for q in (0.68, 0.95)})
    ax.contourf(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), h,
                levels=[*lev, h.max()], colors=["#c6dbef", "#4292c6"],
                alpha=0.85)
    ax.axvline(0.0, color=OKABE["vermillion"], lw=0.9, ls="--")
    ax.annotate("Kerr value", (0.02, 0.93), xycoords="axes fraction",
                fontsize=8, color=OKABE["vermillion"])
    ax.set_xlabel(r"$\epsilon_q$ [$\times 10^5$]")
    ax.set_ylabel(r"$M$ [$10^6\,M_\odot$]")
    save(figr, "fig_nohair_contours.pdf")


# =====================================================================
# Cluster fit (Ch 9.5; needs validation/phase13)
# =====================================================================

def fig_cluster_orbits() -> None:
    p = ROOT / "validation/phase13/cluster_fit.json"
    if not p.exists():
        return warn_skip("cluster_orbits", p)
    j = json.loads(p.read_text())
    figr, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6),
                                   width_ratios=[1.1, 1])
    for star, c in zip(j["stars"], CYCLE + CYCLE):
        tr = ROOT / f"validation/phase13/{star}_track.csv"
        if tr.exists():
            t = read_csv(tr)
            ax.plot(t["ra_mas"].astype(float), t["dec_mas"].astype(float),
                    lw=0.9, color=c, label=star)
        ap = ROOT / f"data/gillessen_2017/{star}_astrometry.csv"
        if ap.exists():
            d = read_csv(ap)
            ax.plot(d["ra_mas"].astype(float), d["dec_mas"].astype(float),
                    "o", ms=1.5, color=c, alpha=0.6)
    ax.plot([0], [0], "k*", ms=10)
    ax.set_xlabel(r"$\Delta$RA [mas]")
    ax.set_ylabel(r"$\Delta$Dec [mas]")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.legend(fontsize=6)
    # per-star vs joint mass
    stars = j["per_star"]
    names = [s["star"] for s in stars]
    M = np.array([s["M_msun"] for s in stars]) / 1e6
    sM = np.array([s["sigma_M_msun"] for s in stars]) / 1e6
    y = np.arange(len(names))
    ax2.errorbar(M, y, xerr=sM, fmt="o", ms=4, color=OKABE["blue"], lw=1.0)
    ax2.axvspan(j["M_msun"] / 1e6 - j["sigma_M_msun"] / 1e6,
                j["M_msun"] / 1e6 + j["sigma_M_msun"] / 1e6,
                color=OKABE["orange"], alpha=0.35, label="joint fit")
    ax2.axvline(4.297, color="0.4", lw=0.8, ls="--", label="GRAVITY 2022")
    ax2.set_yticks(y, names)
    ax2.set_xlabel(r"$M$ [$10^6\,M_\odot$]")
    ax2.legend(fontsize=7)
    save(figr, "fig_cluster_orbits.pdf")


def main() -> None:
    print("generating paper + thesis figures ...")
    # original ten (restyled)
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
    # thesis: Kerr geometry
    fig_radii_vs_spin()
    fig_efficiency_vs_spin()
    fig_energetics()
    fig_timelike_potential()
    # thesis: geodesics
    fig_photon_potential()
    fig_photon_trajectories()
    fig_rosette()
    # thesis: numerics
    fig_projection_boundary()
    fig_drift_vs_tol()
    fig_step_size()
    # thesis: shadow & photon shell
    fig_critical_curves()
    fig_spherical_orbits()
    fig_lyapunov_vs_spin()
    fig_subring_sweep()
    # thesis: disk
    fig_disk_profiles()
    fig_cie_locus()
    # thesis: S-star model & fit
    fig_romer_delay()
    fig_precession_decomposition()
    fig_o_minus_c()
    fig_residual_quiver()
    fig_redshift_decomposition()
    # thesis: calibration + engineering
    fig_axis_ratio_map()
    fig_taylor_trig_error()
    fig_bench()
    # thesis: MCMC + cluster (skipped until their tools have run)
    fig_corner_s2()
    fig_nohair_contours()
    fig_cluster_orbits()
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
