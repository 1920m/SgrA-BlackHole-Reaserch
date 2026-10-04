"""Generate every figure of the extension paper (paper/extension/figures/)
from simulation output — plots from validation/extension/*.json|csv, images
from the engine renders of scripts/render_ext_gallery.py and the validation
runs (figures come from code).

Run: py -3.11 scripts/make_figures_ext.py
"""
from __future__ import annotations

import json
import math
import pathlib
import shutil
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from figstyle import OKABE, apply_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXT = ROOT / "validation" / "extension"
REN = EXT / "renders"
P14 = ROOT / "validation" / "phase14"
FIG = ROOT / "paper" / "extension" / "figures"


def load(name: str) -> dict:
    return json.loads((EXT / name).read_text())


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def copy(src: pathlib.Path, name: str) -> None:
    if src.exists():
        shutil.copyfile(src, FIG / name)
        print(f"  {name}")
    else:
        print(f"  [missing] {src}")


# ---- x02: spectral rendering ---------------------------------------------------
def fig_contrast() -> None:
    sp = load("spectral.json")
    c = sp["contrast"]
    grid = sorted(c[4:], key=lambda d: d["T_peak_K"])
    T = np.array([d["T_peak_K"] for d in grid])
    vis = np.array([d["visual_left_over_right"] for d in grid])
    bolo = c[0]["bolometric_left_over_right"]
    tp = [d["T_peak_K"] for d in sp["thin_disk"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.axvspan(min(tp), max(tp), color=OKABE["sky"], alpha=0.18,
               label=r"$T_{\rm peak}$ at Sgr A*'s $\dot M$ range")
    ax.loglog(T, vis, color=OKABE["vermillion"], label="visual (CIE $Y$ of $B_\\nu(gT)$)")
    ax.axhline(bolo, color=OKABE["blue"], ls="--", label=r"bolometric ($g^4$)")
    for d in c[1:4]:
        ax.plot(d["T_peak_K"], d["visual_left_over_right"], "o", color=OKABE["vermillion"], ms=4)
    ax.set_xlabel(r"disk peak temperature $T_{\rm peak}$ [K]")
    ax.set_ylabel("approaching / receding")
    ax.legend(loc="upper right")
    ax.set_xlim(T.min(), T.max())
    save(fig, "fig_x_contrast.pdf")


def fig_images_spectral() -> None:
    copy(P14 / "g13_cpu.png", "x_g13_cpu.png")
    copy(P14 / "g13_gpu.png", "x_g13_gpu.png")
    for n in ("preset0_physical", "preset1_earth", "preset2_close", "preset3_lowmdot",
              "preset4_highmdot", "preset0_bolometric"):
        copy(REN / f"{n}.png", f"x_{n}.png")


# ---- x04: observers --------------------------------------------------------------
def fig_observers() -> None:
    for n in ("obs_zamo", "obs_infall", "obs_orbit", "ergo_crescent"):
        copy(REN / f"{n}.png", f"x_{n}.png")
    ob = load("observers.json")
    gp = load("g17_observers_gpu.json")
    gpu = {(c["observer"], c["a"], c["r"], c["incl_deg"]): c for c in gp["cases"]}
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    labels, k = [], 0
    for s in ob["shadow_edges"]:
        if not s.get("valid"):
            continue
        key = (s["observer"], s["a"], s["r"], s["incl_deg"])
        y = k
        ax.plot([math.degrees(math.atan(s["x_left"])), math.degrees(math.atan(s["x_right"]))],
                [y, y], color=OKABE["blue"] if s["a"] > 0 else OKABE["green"], lw=3,
                solid_capstyle="butt")
        if key in gpu:
            g = gpu[key]["gpu"]
            ax.plot([math.degrees(math.atan(g[0])), math.degrees(math.atan(g[1]))], [y, y],
                    "|", color=OKABE["vermillion"], ms=9, mew=1.6)
        labels.append(f"{s['observer']}, $a_*$={s['a']:g}, r={s['r']:g}, i={s['incl_deg']:g}°")
        k += 1
    ax.set_yticks(range(k))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.axvline(0.0, color="0.6", lw=0.6)
    ax.set_xlabel("shadow along the image's horizontal axis [deg from the look direction]")
    ax.plot([], [], color=OKABE["green"], lw=3, label="CPU fp64, $a_*=0$")
    ax.plot([], [], color=OKABE["blue"], lw=3, label="CPU fp64, $a_*=0.94$")
    ax.plot([], [], "|", color=OKABE["vermillion"], ms=9, mew=1.6, label="GPU live edges")
    ax.legend(loc="lower right", fontsize=6.5)
    save(fig, "fig_x_observer_edges.pdf")


# ---- x05: flares ---------------------------------------------------------------
CASES = [("g18_R7_i160", "GRAVITY 2018 fit: $R=7M$, $i=160°$", OKABE["vermillion"]),
         ("g20_R9_i140", "GRAVITY 2020 fit: $R=9M$, $i=140°$", OKABE["blue"]),
         ("p40_i160", "$P = 40$ min: $R=6.64M$, $i=160°$", OKABE["green"])]


def lowpass(v: np.ndarray, K: int = 12) -> np.ndarray:
    f = np.fft.rfft(v)
    f[K + 1:] = 0
    return np.fft.irfft(f, len(v))


def fig_flare_curves() -> None:
    fl = load("flares.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for name, lab, col in CASES:
        d = np.genfromtxt(EXT / f"flare_{name}.csv", delimiter=",", names=True)
        F = lowpass(d["flux"])
        a1.plot(d["t_min"], F / F.mean(), color=col, label=lab)
        n1 = lowpass(d["flux_n1"]) / F.mean()
        a1.plot(d["t_min"], n1, color=col, ls=":", lw=1)
        cx, cy = lowpass(d["cx_uas"]), lowpass(d["cy_uas"])
        a2.plot(np.append(cx, cx[0]), np.append(cy, cy[0]), color=col)
        a2.plot(cx[0], cy[0], "o", color=col, ms=3)
        k = len(cx) // 16
        a2.annotate("", xy=(cx[k + 1], cy[k + 1]), xytext=(cx[k], cy[k]),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1))
    a1.set_xlabel("observing time [min]")
    a1.set_ylabel("flux / mean  (dotted: lensed $n\\geq1$)")
    a1.legend(fontsize=6.5, loc="upper right")
    a2.plot(0, 0, "k+", ms=8)
    # cited reference scale (GRAVITY 2018): ~120 uas of centroid motion in ~30 min
    a2.plot([-60, 60], [-62, -62], color="0.3", lw=2)
    a2.text(0, -58, r"GRAVITY 2018: $\sim$120 $\mu$as in $\sim$30 min", ha="center",
            fontsize=6.5, color="0.3")
    a2.set_aspect("equal")
    a2.set_xlim(-70, 70)
    a2.set_ylim(-70, 70)
    a2.set_xlabel(r"centroid $x$ [$\mu$as]")
    a2.set_ylabel(r"centroid $y$ [$\mu$as]")
    fig.tight_layout()
    save(fig, "fig_x_flare_curves.pdf")


def fig_echo() -> None:
    fl = load("flares.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    d = np.genfromtxt(EXT / "flare_g18_R7_i160_flash0.csv", delimiter=",", names=True)
    m = max(d["flux_n0"].max(), 1e-30)
    a1.plot(d["t_min"], d["flux_n0"] / m, color=OKABE["vermillion"], label="direct image ($n=0$)")
    a1.plot(d["t_min"], 10 * d["flux_n1"] / m, color=OKABE["blue"],
            label="first lensed image ($n=1$, $\\times10$)")
    a1.set_xlabel("observing time [min] (arbitrary origin)")
    a1.set_ylabel("flux / direct peak")
    a1.legend(fontsize=6.5)
    a1.set_xlim(d["t_min"][np.argmax(d["flux_n0"])] - 4, d["t_min"][np.argmax(d["flux_n1"])] + 6)
    cases = {c["name"]: c for c in fl["cases"]}
    for name, lab, col in CASES:
        e = cases[name]["echo"]
        ph = [x["phase_deg"] for x in e]
        a2.plot(ph, [x["delay_min"] for x in e], "o-", color=col, ms=3, label=lab)
    a2.set_xlabel("orbital phase of the flash [deg] (0: toward the camera)")
    a2.set_ylabel("echo delay $n=1$ vs $n=0$ [min]")
    a2.set_xticks(range(0, 360, 90))
    a2.legend(fontsize=6.0)
    fig.tight_layout()
    save(fig, "fig_x_echo.pdf")


def fig_flare_images() -> None:
    fl = load("flares.json")
    n = next(c for c in fl["cases"] if c["name"] == "g18_R7_i160")["n_pixels"]
    hw = next(c for c in fl["cases"] if c["name"] == "g18_R7_i160")["half_width_M"]
    upm = fl["uas_per_M"]
    fig, axs = plt.subplots(1, 5, figsize=(9.0, 2.1))
    imgs = [np.fromfile(EXT / f"flare_g18_R7_i160_k{k}.f32", dtype=np.float32).reshape(n, n)
            for k in (0, 64, 128, 192)]
    imgs.append(np.fromfile(EXT / "flare_g18_R7_i160_mean.f32", dtype=np.float32).reshape(n, n))
    vmax = max(i.max() for i in imgs[:4])
    ext = [-hw * upm, hw * upm, -hw * upm, hw * upm]
    titles = ["0", "P/4", "P/2", "3P/4", "orbit mean"]
    for ax, im, t in zip(axs, imgs, titles):
        ax.imshow(np.log10(np.maximum(im / vmax, 1e-4)), extent=ext, cmap="inferno",
                  vmin=-4, vmax=0, origin="upper")
        ax.set_title(t, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].plot([-ext[1] * 0.85, -ext[1] * 0.85 + 50], [ext[2] * 0.85] * 2, color="w", lw=1.5)
    axs[0].text(-ext[1] * 0.85 + 25, ext[2] * 0.72, r"50 $\mu$as", color="w", fontsize=6,
                ha="center")
    fig.tight_layout()
    save(fig, "fig_x_flare_images.pdf")
    for k in range(4):
        copy(REN / f"flare_{k:04d}.png", f"x_flare_{k}.png")


# ---- x06: the hot accretion flow and its polarisation ----------------------------
def fig_riaf() -> None:
    ri = load("riaf.json")
    m = ri["main"]
    n, half, upm = m["n_pixels"], m["half_width_M"], ri["uas_per_M"]
    nu = ri["nu_obs_hz"]
    tb = 2.99792458e10 ** 2 / (2 * 1.380649e-16 * nu * nu)
    raw = np.fromfile(EXT / "riaf_main.f32", dtype=np.float32).reshape(n, n) * tb
    blr = np.fromfile(EXT / "riaf_main_blur.f32", dtype=np.float32).reshape(n, n) * tb
    ext = [-half * upm, half * upm, -half * upm, half * upm]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.5))
    for ax, im, title in ((axs[0], raw, "model"), (axs[1], blr, f"blurred, {ri['beam_uas']:.0f} $\\mu$as beam")):
        h = ax.imshow(im / 1e9, extent=ext, cmap="afmhot", origin="upper")
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(25.9 * np.cos(th), 25.9 * np.sin(th), color="w", ls="--", lw=0.8)  # EHT 51.8 uas
        c = m["ring_raw"]
        ax.plot(c["centre_uas"][0] + 0.5 * c["d_uas"] * np.cos(th),
                c["centre_uas"][1] + 0.5 * c["d_uas"] * np.sin(th), color="c", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"$\mu$as")
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.03, label=r"$T_b$ [$10^9$ K]")
    axs[0].set_ylabel(r"$\mu$as")
    fig.tight_layout()
    save(fig, "fig_x_riaf_images.pdf")

    # visibility amplitudes along 12 position angles (direct Fourier sum)
    rad = 2 * half / n * upm * np.pi / 180 / 3600e6
    x = (np.arange(n) + 0.5 - n / 2) * rad
    X, Y = np.meshgrid(x, -x)
    us = np.linspace(0, 10e9, 161)
    img = raw / raw.sum()
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for pa in np.arange(0, 180, 15):
        c, s = np.cos(np.radians(pa)), np.sin(np.radians(pa))
        V = np.array([abs((img * np.exp(-2j * np.pi * u * (c * X + s * Y))).sum()) for u in us])
        ax.semilogy(us / 1e9, V, color=OKABE["blue"], lw=0.6, alpha=0.6)
    va = np.genfromtxt(EXT / "riaf_main_vis.csv", delimiter=",", names=True)
    ax.semilogy(va["u_glambda"], va["amp"], color="k", lw=1.4, label="mean over position angle")
    for u0 in (3.0, 6.5):
        ax.axvline(u0, color=OKABE["vermillion"], ls=":", lw=1)
    ax.text(3.05, 1.3e-3, "EHT minima", color=OKABE["vermillion"], fontsize=7)
    ax.set_ylim(1e-3, 1.2)
    ax.set_xlim(0, 10)
    ax.set_xlabel(r"baseline $|u|$ [G$\lambda$]")
    ax.set_ylabel(r"$|V|/V(0)$")
    ax.legend(fontsize=7, loc="upper right")
    save(fig, "fig_x_riaf_vis.pdf")


def fig_pol() -> None:
    ri = load("riaf.json")
    pol = {(e["field"], int(e["incl_deg"])): e for e in ri["polarisation"]}
    half, upm = ri["main"]["half_width_M"], ri["uas_per_M"]
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.7))
    zoom = 42.0                                   # µas half-width shown
    p0 = ri["pol_fraction"]
    for ax, fld in zip(axs, ("toroidal", "radial", "vertical")):
        e = pol[(fld, 150)]
        n = e["n_pixels"]
        tag = f"riaf_pol_{fld}_i150"
        I = np.fromfile(EXT / f"{tag}_I.f32", dtype=np.float32).reshape(n, n)
        Q = np.fromfile(EXT / f"{tag}_Q.f32", dtype=np.float32).reshape(n, n)
        U = np.fromfile(EXT / f"{tag}_U.f32", dtype=np.float32).reshape(n, n)
        ext = [-half * upm, half * upm, -half * upm, half * upm]
        ax.imshow(I / I.max(), extent=ext, cmap="gray", origin="upper", vmin=0, vmax=1)
        step = max(1, n // 22)
        b = half * upm * (2 * (np.arange(n) + 0.5) / n - 1)
        P = np.hypot(Q, U)
        Pm = P.max()
        for j in range(step // 2, n, step):
            for i in range(step // 2, n, step):
                if I[j, i] < 0.12 * I.max() or abs(b[i]) > zoom or abs(b[j]) > zoom:
                    continue
                chi = 0.5 * np.arctan2(U[j, i], Q[j, i])   # from image-up toward image-left
                L = 1.1 * step * 2 * half * upm / n * np.sqrt(P[j, i] / Pm)
                dx, dy = -np.sin(chi) * L / 2, np.cos(chi) * L / 2
                frac = P[j, i] / max(I[j, i], 1e-30)
                ax.plot([b[i] - dx, b[i] + dx], [-b[j] - dy, -b[j] + dy],
                        color=plt.cm.plasma(min(frac / p0, 1.0)), lw=2.0,
                        solid_capstyle="round")
        ang = e["blurred"]["beta2_arg_deg"]
        ax.set_title(f"{fld} field: $\\angle\\beta_2$ = {ang:+.0f}$\\degree$", fontsize=10)
        ax.set_xlim(-zoom, zoom)
        ax.set_ylim(-zoom, zoom)
        ax.set_xlabel(r"$\mu$as")
    axs[0].set_ylabel(r"$\mu$as")
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, p0))
    fig.colorbar(sm, ax=axs, fraction=0.02, pad=0.01, label="fractional polarisation $|m|$")
    save(fig, "fig_x_pol.pdf")


def fig_wormhole() -> None:
    """Gate 21 (EXOTIC, NOT Sgr A*): the GPU renders of the Dneg wormhole at
    three distances, with the fp64 CPU twin's results drawn over them — the
    edge of the other universe (exact: sin alpha = rho/r(l)) and the Einstein
    ring of the point directly behind the wormhole."""
    wh = load("wormhole.json")
    fr = json.loads((REN / "wormhole_renders.json").read_text())
    cams = {int(round(c["l_c"])): c for c in wh["cameras"]}
    thalf = math.tan(0.5 * fr["fov"])
    n = fr["size"]
    fig, axs = plt.subplots(1, len(fr["l_c"]), figsize=(7.2, 2.65))
    for ax, lc in zip(axs, fr["l_c"]):
        img = plt.imread(REN / f"wormhole_l{lc}.png")
        ax.imshow(img, extent=[0, n, n, 0])
        c = cams[lc]
        t = np.linspace(0, 2 * np.pi, 400)
        for ang, ls, col in ((c["edge_deg"], "--", OKABE["orange"]),
                             (c["einstein_ring_deg"], ":", OKABE["sky"])):
            rp = math.tan(math.radians(ang)) / thalf * n / 2
            ax.plot(n / 2 + rp * np.cos(t), n / 2 + rp * np.sin(t), ls=ls, color=col,
                    lw=0.9)
        ax.set_title(rf"$\ell_c = {lc}\rho$:  edge {c['edge_deg']:.2f}$\degree$", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)
    axs[0].plot([], [], "--", color=OKABE["orange"], label="other universe: edge (CPU)")
    axs[0].plot([], [], ":", color=OKABE["sky"], label="Einstein ring (CPU)")
    fig.legend(loc="lower center", ncol=2, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, 0.06))
    fig.suptitle("EXOTIC SPACETIME, not Sgr A*: Dneg wormhole (James et al. 2015)",
                 fontsize=9, y=0.99)
    save(fig, "fig_x_wormhole.pdf")


def fig_forecast() -> None:
    """Gate 22: (left) the no-hair reach sigma(eps_q) versus monitoring time —
    Fisher forecasts of the thesis model for two continuations, against the
    thesis' 1/sqrt(N) extrapolation; (right) the significance of S2's
    Lense–Thirring precession (a* = 0.94) for three astrometric precisions."""
    fc = load("forecasts.json")
    P = fc["assumptions"]["period_yr"]
    s0 = fc["ten_orbits"]["sigma_eps_g17"]
    span = fc["assumptions"]["g17"]["span_yr"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for key, lab, col in (("existing_quality", "existing quality (Fisher)", OKABE["blue"]),
                          ("gravity", "GRAVITY-era (Fisher)", OKABE["vermillion"])):
        c = fc["eps_vs_years"][key]
        a1.plot([e["T_yr"] for e in c], [e["sigma_eps"] for e in c], "-", color=col, lw=1.4,
                label=lab)
    T = np.linspace(0, 10 * P, 200)
    a1.plot(T, s0 / np.sqrt(1 + T / span), "--", color=OKABE["black"], lw=1.0,
            label=r"thesis: $1/\sqrt{N}$")
    a1.plot([10 * P], [fc["ten_orbits"]["sigma_eps_thesis_sqrtN"]], "o", ms=4,
            color=OKABE["black"])
    a1.plot([10 * P], [fc["ten_orbits"]["sigma_eps_fisher"]], "o", ms=4, color=OKABE["blue"])
    tp = json.loads((ROOT / "validation" / "phase5" / "s2_fit.json").read_text())["t_peri_yr"]
    t0 = fc["assumptions"]["t0_future"]
    for k in range(1, 12):
        tk = tp + k * P - t0                      # pericentres after 2017.0
        if 0 < tk <= 10 * P:
            a1.axvline(tk, color="0.85", lw=0.6, zorder=0)
    a1.set_yscale("log")
    a1.set_xlabel("years of monitoring after 2017.0")
    a1.set_ylabel(r"$\sigma(\epsilon_q)$")
    a1.legend(fontsize=7, frameon=False, loc="lower left")
    a1.set_title("no-hair reach of S2 (grey: pericentres)", fontsize=9)

    styles = {65.0: "-", 30.0: "--", 10.0: ":"}
    for e in fc["lense_thirring"]:
        c = e["curve"]
        marg = e.get("ext_marginalised", False)
        a2.plot([x["T_yr"] for x in c], [x["significance"] for x in c],
                styles[e["sig_ast_uas"]], color=OKABE["purple"] if marg else OKABE["green"],
                lw=1.0 if marg else 1.4,
                label=None if marg else rf"{e['sig_ast_uas']:.0f}$\,\mu$as")
    a2.plot([], [], "-", color=OKABE["purple"], lw=1.0, label="ext. mass free")
    for thr in (3, 5):
        a2.axhline(thr, color="0.6", lw=0.8)
        a2.text(2, thr * 1.06, rf"{thr}$\sigma$", fontsize=7, color="0.4")
    a2.set_yscale("log")
    a2.set_ylim(0.05, 30)
    a2.set_xlabel("years of monitoring after 2017.0")
    a2.set_ylabel(r"$a_* / \sigma(a_*)$  ($a_* = 0.94$)")
    a2.legend(fontsize=7, frameon=False, loc="lower right", title="astrometry",
              title_fontsize=7)
    a2.set_title("frame dragging (Lense–Thirring) in S2", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_x_forecast.pdf")


def fig_degeneracy() -> None:
    """Gate 22: (left) 1-sigma joint constraints on the extended mass enclosed
    within S2's apocentre and eps_q for GRAVITY-era monitoring; (right) the
    eps_q bias an unmodelled extended mass produces, in units of sigma(eps_q)."""
    fc = load("forecasts.json")
    k = fc["assumptions"]["enclosed_per_total"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    t = np.linspace(0, 2 * np.pi, 300)
    cols = {8.0: OKABE["orange"], 16.0: OKABE["vermillion"], 32.0: OKABE["purple"]}
    for d in fc["degeneracy"]:
        if d["T_yr"] not in cols:
            continue
        C = np.array(d["cov2"])                    # (eps_q, f_ext) covariance
        S = np.diag([1.0, 100.0 * k])              # → (eps_q, f_enc in %)
        C = S @ C @ S
        w, V = np.linalg.eigh(C)
        xy = V @ (np.sqrt(w)[:, None] * np.array([np.cos(t), np.sin(t)]))
        a1.plot(xy[1], xy[0], color=cols[d["T_yr"]], lw=1.4,
                label=f"{d['T_yr']:.0f} yr (corr {d['corr_eps_fext']:+.2f})")
    for x in (-0.1, 0.1):
        a1.axvline(x, color="0.55", lw=0.9, ls="--",
                   label="GRAVITY 1$\\sigma$ bound" if x > 0 else None)
    a1.set_xlim(-0.13, 0.13)
    a1.set_xlabel("extended mass within apocentre [% of $M$]")
    a1.set_ylabel(r"$\epsilon_q$")
    a1.legend(fontsize=7, frameon=False, loc="upper right")
    a1.set_title(r"1$\sigma$ joint constraints", fontsize=9)

    styles = {5e-4: ":", 1e-3: "-", 2e-3: "--"}
    for fe, ls in styles.items():
        rows = [b for b in fc["bias"] if abs(b["f_enc"] - fe) < 1e-12]
        a2.plot([b["T_yr"] for b in rows], [b["deps_over_sigma"] for b in rows], ls,
                marker="o", ms=3, color=OKABE["blue"], lw=1.3,
                label=f"{100 * fe:.2f} % within apocentre")
    for thr in (1, 3, 5):
        a2.axhline(thr, color="0.75", lw=0.7)
    a2.set_xlabel("years of GRAVITY-era monitoring after 2017.0")
    a2.set_ylabel(r"false $\epsilon_q$ signal  [$\sigma$]")
    a2.legend(fontsize=7, frameon=False, loc="upper left", title="unmodelled extended mass",
              title_fontsize=7)
    a2.set_title("bias of the no-hair test", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_x_degeneracy.pdf")


def fig_fall() -> None:
    """Gate 24: the in-faller's view looking ahead (top row, outside the
    horizon) and on through it (bottom row), GPU renders of the fp64
    worldline; labels: radius and proper time (from infall.json)."""
    inf = load("infall.json")
    prof = inf["profile"]
    rs = np.array([p["r"] for p in prof])[::-1]
    ts = np.array([p["tau"] for p in prof])[::-1]
    fr = json.loads((REN / "fall_renders.json").read_text())
    rp = inf["trip"]["r_plus"]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.25))
    for ax, r in zip(axs.flat, fr["r"]):
        ax.imshow(plt.imread(REN / f"fall_r{r:g}.png"))
        tau = float(np.interp(r, rs, ts))
        where = "outside" if r > rp else "inside"
        ax.set_title(rf"$r = {r:g}\,M$ ({where}), $\tau = {tau:.2f}\,M$", fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.4, h_pad=1.2)
    save(fig, "fig_x_fall.pdf")


def fig_fall_readouts() -> None:
    """Gate 24: what the in-faller's instruments read along the trip."""
    inf = load("infall.json")
    prof = inf["profile"]
    tr = inf["trip"]
    r = np.array([p["r"] for p in prof])
    tau = np.array([p["tau"] for p in prof])
    tbl = np.array([np.nan if p["t_bl"] is None else p["t_bl"] for p in prof])
    fb = np.array([p["freq_behind"] for p in prof])
    fs = np.array([p["freq_side"] for p in prof])
    st = np.array([p["stretch_2m_ms2"] for p in prof])
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax in (a1, a2, a3):
        ax.axvline(tr["r_plus"], color="0.6", lw=0.8, ls="--")
        ax.axvline(tr["r_ergo"], color="0.8", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel(r"$r$ [$M$]")
        ax.invert_xaxis()
    a1.plot(r, tau, color=OKABE["blue"], lw=1.4, label=r"your proper time $\tau$")
    a1.plot(r, tbl, color=OKABE["vermillion"], lw=1.4, label=r"far-away time $t$")
    a1.set_ylim(0, 2.2 * tau.max())
    a1.set_ylabel(r"time since the start [$M$]")
    a1.legend(fontsize=6.5, frameon=False, loc="upper left")
    a2.plot(r, fb, color=OKABE["blue"], lw=1.4, label="sky behind you")
    a2.plot(r, fs, color=OKABE["green"], lw=1.4, label="sky at your side")
    a2.axhline(1.0, color="0.7", lw=0.7)
    a2.set_ylim(0.0, 1.1)
    a2.set_ylabel(r"$\nu_{\rm seen}/\nu_\infty$")
    a2.legend(fontsize=6.5, frameon=False, loc="lower left")
    a3.plot(r, st, color=OKABE["purple"], lw=1.4)
    a3.set_yscale("log")
    a3.set_ylabel(r"tidal stretch, 2 m body [m s$^{-2}$]")
    fig.tight_layout(pad=0.4)
    save(fig, "fig_x_fall_readouts.pdf")


def main() -> None:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    fig_contrast()
    fig_images_spectral()
    fig_observers()
    fig_flare_curves()
    fig_echo()
    fig_flare_images()
    if (EXT / "riaf.json").exists():
        fig_riaf()
        fig_pol()
    for nm in ("band_visible", "band_230", "band_kband", "band_230_eht"):
        copy(REN / f"{nm}.png", f"x_{nm}.png")
    if (REN / "wormhole_renders.json").exists():
        fig_wormhole()
    if (EXT / "forecasts.json").exists():
        fig_forecast()
        fig_degeneracy()
    if (EXT / "infall.json").exists():
        fig_fall_readouts()
        if (REN / "fall_renders.json").exists():
            fig_fall()


if __name__ == "__main__":
    main()
