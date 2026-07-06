"""Generate the explanatory matplotlib schematics for the thesis
(paper/THESIS_PLAN.md §2.2b) into paper/figures/.

Same toolchain and conventions as make_figures.py: matplotlib only, no TikZ,
no hand-drawn or AI-generated content; deterministic (fixed seeds) so every
diagram is reproducible by `make figures`. Where a diagram shows physics
(light bending, anatomy radii, rosette, frame dragging) it is computed from
the same closed-form equations the C++ core implements, not sketched.

Run: py -3.11 scripts/make_diagrams.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from figstyle import OKABE, apply_style  # noqa: E402

apply_style()

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def save(figr, name: str) -> None:
    figr.tight_layout()
    figr.savefig(FIG / name)
    plt.close(figr)
    print(f"  wrote {name}")


# ---------------------------------------------------------------------------
# helper: Schwarzschild light bending u'' = -u + 3u² (M = 1), integrated in
# the orbit angle φ with RK4 — the classic exact null-geodesic shape equation.
# ---------------------------------------------------------------------------
def _bend_ray(b: float, r0: float = 60.0, steps: int = 20000):
    u = 1.0 / r0
    # inward ray: du/dφ from the null condition
    up = np.sqrt(max(1.0 / (b * b) - u * u * (1.0 - 2.0 * u), 0.0))
    phi, h = 0.0, 2e-3
    xs, ys = [], []

    def f(state):
        uu, vv = state
        return np.array([vv, -uu + 3.0 * uu * uu])

    s = np.array([u, up])
    for _ in range(steps):
        k1 = f(s); k2 = f(s + 0.5 * h * k1)
        k3 = f(s + 0.5 * h * k2); k4 = f(s + h * k3)
        s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        phi += h
        if s[0] <= 0.0 or s[0] > 0.5:  # escaped to infinity / inside horizon
            break
        r = 1.0 / s[0]
        xs.append(r * np.cos(phi))
        ys.append(r * np.sin(phi))
        if r > 1.4 * r0:
            break
    return np.array(xs), np.array(ys)


def dg_shadow_formation() -> None:
    figr, ax = plt.subplots(figsize=(6.4, 4.2))
    bc = np.sqrt(27.0)
    for b, c in [(9.0, OKABE["green"]), (7.0, OKABE["green"]),
                 (bc + 0.02, OKABE["orange"]),
                 (5.0, OKABE["vermillion"]), (4.0, OKABE["vermillion"])]:
        x, y = _bend_ray(b)
        # rotate so the ray comes in horizontally from the right at height b
        ang = -np.arcsin(b / 60.0)
        xr = x * np.cos(ang) - y * np.sin(ang)
        yr = x * np.sin(ang) + y * np.cos(ang)
        ax.plot(xr, yr, lw=1.1, color=c)
        ax.plot(xr, -yr, lw=1.1, color=c)  # mirror family
    ax.add_patch(plt.Circle((0, 0), 2.0, color="k", zorder=6))
    ax.add_patch(plt.Circle((0, 0), 3.0, fill=False, ls="--", lw=0.8,
                            color="0.5", zorder=5))
    ax.annotate("event horizon $r=2M$", (0, -3.3), ha="center", va="top",
                fontsize=8)
    ax.annotate("photon sphere $r=3M$", (2.3, 2.6), fontsize=8, color="0.4")
    ax.annotate("escapes: bent but seen", (-25, 13.5), fontsize=9,
                color=OKABE["green"])
    ax.annotate(r"critical ray $b\simeq\sqrt{27}M$: orbits", (-25, 7.0),
                fontsize=9, color=OKABE["orange"])
    ax.annotate("captured: the shadow", (-25, -1.0), fontsize=9,
                color=OKABE["vermillion"])
    ax.set_xlim(-26, 26)
    ax.set_ylim(-16, 16)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [$M$]")
    ax.set_ylabel(r"$y$ [$M$]")
    save(figr, "dg_shadow_formation.pdf")


def dg_kerr_anatomy() -> None:
    figr, axes = plt.subplots(1, 2, figsize=(7.0, 3.8))
    th = np.linspace(0, 2 * np.pi, 400)
    for ax, a in zip(axes, [0.0, 0.94]):
        rp = 1.0 + np.sqrt(1.0 - a * a)
        rE = 1.0 + np.sqrt(1.0 - a * a * np.cos(th) ** 2)
        r_ph_p = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(-a)))
        r_ph_r = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(+a)))
        z1 = 1 + (1 - a * a) ** (1 / 3) * ((1 + a) ** (1 / 3) +
                                           (1 - a) ** (1 / 3))
        z2 = np.sqrt(3 * a * a + z1 * z1)
        risco = 3 + z2 - np.sqrt(max((3 - z1) * (3 + z1 + 2 * z2), 0.0))
        # polar cross-section: x = r sinθ, z = r cosθ
        ax.fill(rp * np.sin(th), rp * np.cos(th), color="k", zorder=5)
        ax.plot(rE * np.sin(th), rE * np.cos(th), color=OKABE["orange"],
                lw=1.2, label=r"ergosphere $r_E(\theta)$")
        for rr, c, lab in [(r_ph_p, OKABE["blue"], "photon orbit (prograde)"),
                           (r_ph_r, OKABE["sky"], "photon orbit (retrograde)"),
                           (risco, OKABE["green"], "ISCO")]:
            ax.plot(rr * np.sin(th), rr * np.cos(th), ls="--", lw=1.0,
                    color=c, label=lab)
        ax.plot([-8, 8], [0, 0], color="0.85", lw=4, zorder=0)
        ax.annotate("disk", (7.0, 0.35), fontsize=8, color="0.55")
        ax.set_title(rf"$a_*={a:g}$", fontsize=10)
        ax.set_xlim(-8.5, 8.5)
        ax.set_ylim(-7.5, 7.5)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ [$M$]")
    axes[0].set_ylabel(r"$z$ (spin axis) [$M$]")
    axes[1].legend(fontsize=6.5, loc="upper right")
    save(figr, "dg_kerr_anatomy.pdf")


def dg_frame_drag() -> None:
    figr, ax = plt.subplots(figsize=(4.8, 4.4))
    a = 0.94
    rp = 1.0 + np.sqrt(1.0 - a * a)
    # ZAMO angular velocity ω = 2Mar/A in the equatorial plane
    for r in np.linspace(rp + 0.15, 7.5, 9):
        A = (r * r + a * a) ** 2 - a * a * (r * r - 2 * r + a * a)
        w = 2.0 * a * r / A
        dth = 3.5 * w  # arrow arc length encodes the dragging rate
        for t0 in np.linspace(0, 2 * np.pi, 14, endpoint=False):
            ax.annotate("",
                        xy=(r * np.cos(t0 + dth), r * np.sin(t0 + dth)),
                        xytext=(r * np.cos(t0), r * np.sin(t0)),
                        arrowprops=dict(arrowstyle="->", lw=0.9,
                                        color=plt.cm.plasma(min(w * 6, 0.95))))
    ax.add_patch(plt.Circle((0, 0), rp, color="k", zorder=6))
    ax.add_patch(plt.Circle((0, 0), 2.0, fill=False, color=OKABE["orange"],
                            lw=1.1, zorder=5))
    ax.annotate("ergosphere", (0, 2.2), ha="center", fontsize=8,
                color=OKABE["orange"])
    ax.annotate("frame dragging: spacetime itself circulates\n(arrow length "
                r"$\propto\ \omega=-g_{t\phi}/g_{\phi\phi}$)",
                (0.02, 0.02), xycoords="axes fraction", fontsize=8,
                color="0.35")
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [$M$]")
    ax.set_ylabel(r"$y$ [$M$]")
    ax.set_title(r"$a_*=0.94$, equatorial plane", fontsize=10)
    save(figr, "dg_frame_drag.pdf")


def dg_kepler_ellipse() -> None:
    figr, ax = plt.subplots(figsize=(6.0, 3.8))
    a, e = 1.0, 0.6
    b = a * np.sqrt(1 - e * e)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(a * np.cos(th) - a * e, b * np.sin(th), lw=1.3,
            color=OKABE["blue"])
    ax.plot([0], [0], "*", ms=14, color="k")
    ax.plot([-2 * a * e], [0], "+", ms=9, color="0.5")
    ax.annotate("black hole\n(occupied focus)", (0, -0.09), ha="center",
                va="top", fontsize=8)
    ax.annotate("empty focus", (-2 * a * e, 0.06), ha="center", fontsize=8,
                color="0.5")
    ax.plot([a * (1 - e)], [0], "o", ms=5, color=OKABE["vermillion"])
    ax.annotate("pericentre\n(closest)", (a * (1 - e) + 0.04, 0.05),
                fontsize=8, ha="left", color=OKABE["vermillion"])
    ax.plot([-a * (1 + e)], [0], "o", ms=5, color=OKABE["green"])
    ax.annotate("apocentre\n(farthest)", (-a * (1 + e), 0.05), fontsize=8,
                ha="right", color=OKABE["green"])
    # equal-time star positions (Kepler's second law made visible)
    Manom = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    E = Manom.copy()
    for _ in range(50):
        E = E - (E - e * np.sin(E) - Manom) / (1 - e * np.cos(E))
    ax.plot(a * np.cos(E) - a * e, b * np.sin(E), "o", ms=3,
            color=OKABE["orange"])
    ax.annotate("dots: equal time steps —\nthe star races through pericentre",
                (0.02, 0.06), xycoords="axes fraction", fontsize=8,
                color="0.35")
    ax.annotate("", xy=(-2 * a * e - 0.06, 0.32), xytext=(0.06, 0.32),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.4"))
    ax.annotate(r"$2ae$ (eccentricity $e$ = offset/size)", (-a * e, 0.36),
                ha="center", fontsize=8, color="0.4")
    ax.set_aspect("equal")
    ax.axis("off")
    save(figr, "dg_kepler_ellipse.pdf")


def dg_rosette() -> None:
    figr, ax = plt.subplots(figsize=(4.6, 4.4))
    e, p = 0.6, 1.0
    phi = np.linspace(0, 8 * np.pi, 4000)
    k = 0.88  # pericentre advances by 2π(1/k − 1) per radial period
    r = p / (1 + e * np.cos(k * phi))
    ax.plot(r * np.cos(phi), r * np.sin(phi), lw=0.9, color=OKABE["blue"])
    ax.plot([0], [0], "k*", ms=12)
    for n in range(4):  # successive pericentres
        ph_p = 2 * np.pi * n / k
        rp = p / (1 + e)
        ax.plot(rp * np.cos(ph_p), rp * np.sin(ph_p), "o", ms=5,
                color=OKABE["vermillion"], zorder=5)
    ax.annotate("pericentre direction turns\na little every orbit:\n"
                "the ellipse traces a rosette",
                (0.02, 0.02), xycoords="axes fraction", fontsize=8,
                color="0.35")
    ax.set_aspect("equal")
    ax.axis("off")
    save(figr, "dg_rosette.pdf")


def dg_three_redshifts() -> None:
    figr, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    x = np.linspace(0, 6 * np.pi, 500)
    cases = [
        ("classical Doppler", "source moves away:\nwaves stretched",
         1.0 + 0.25 * x / x.max(), OKABE["blue"]),
        ("transverse Doppler", "fast-moving clock ticks slow:\nredshift "
         r"$\approx v^2/2c^2$", np.full_like(x, 1.18), OKABE["green"]),
        ("gravitational", "climbing out of gravity's well\ncosts energy: "
         r"redshift $\approx GM/rc^2$", 1.0 + 0.35 * x / x.max(),
         OKABE["vermillion"]),
    ]
    for ax, (title, sub, stretch, c) in zip(axes, cases):
        ax.plot(x, np.sin(np.cumsum(1.0 / stretch) * (x[1] - x[0]) * 6),
                lw=1.0, color=c)
        ax.set_title(title, fontsize=9)
        ax.annotate(sub, (0.02, 0.02), xycoords="axes fraction", fontsize=7,
                    color="0.35")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(-1.7, 1.7)
    save(figr, "dg_three_redshifts.pdf")


def dg_astrometry_rv() -> None:
    figr, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    # left: sky-plane view (what a camera sees)
    e, a = 0.88, 1.0
    b = a * np.sqrt(1 - e * e)
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(a * np.cos(th) - a * e, 0.55 * b * np.sin(th), lw=1.1,
            color=OKABE["blue"])
    ax.plot([0], [0], "k*", ms=11)
    rng = np.random.default_rng(42)
    tm = np.sort(rng.uniform(0, 2 * np.pi, 22))
    ax.errorbar(a * np.cos(tm) - a * e + rng.normal(0, .01, tm.size),
                0.55 * b * np.sin(tm) + rng.normal(0, .01, tm.size),
                xerr=0.02, yerr=0.02, fmt="o", ms=2.5, lw=0.5,
                color=OKABE["orange"])
    ax.set_title("astrometry: position on the sky", fontsize=9)
    ax.annotate("milli-arcsecond dots,\nyear by year", (0.03, 0.05),
                xycoords="axes fraction", fontsize=8, color="0.35")
    ax.set_aspect("equal")
    ax.axis("off")
    # right: the radial-velocity channel (a spectral line shifting)
    lam = np.linspace(2.15, 2.185, 500)
    for v, c, lab in [(0.0, "0.6", r"at rest: 2.1661 $\mu$m"),
                      (+4000, OKABE["vermillion"], "receding: redshifted"),
                      (-4000, OKABE["blue"], "approaching: blueshifted")]:
        lam0 = 2.16612 * (1 + v / 299792.458)
        ax2.plot(lam, 1 - 0.75 * np.exp(-0.5 * ((lam - lam0) / 0.0012) ** 2),
                 lw=1.1, color=c, label=lab)
    ax2.set_title(r"spectroscopy: the Br$\gamma$ line shifts", fontsize=9)
    ax2.set_xlabel(r"wavelength [$\mu$m]")
    ax2.set_yticks([])
    ax2.legend(fontsize=6.5, loc="lower right")
    save(figr, "dg_astrometry_rv.pdf")


def dg_scales() -> None:
    figr, ax = plt.subplots(figsize=(6.8, 2.6))
    marks = [
        (np.log10(1.27e10), "Sgr A* horizon\n$\\sim$0.08 au",
         OKABE["vermillion"]),
        (np.log10(1.496e11), "1 au\n(Earth–Sun)", OKABE["blue"]),
        (np.log10(1.79e13), "S2 pericentre\n$\\sim$120 au", OKABE["green"]),
        (np.log10(3.086e16), "1 parsec", OKABE["purple"]),
        (np.log10(2.55e20), "us $\\to$ Sgr A*\n8277 pc", OKABE["orange"]),
    ]
    ax.hlines(0, 9.5, 21, color="0.4", lw=1.2)
    for i, (x, lab, c) in enumerate(marks):
        ax.plot([x], [0], "o", ms=6, color=c)
        ax.annotate(lab, (x, 0.25 if i % 2 == 0 else -0.25), ha="center",
                    va="bottom" if i % 2 == 0 else "top", fontsize=8, color=c)
    for ex in range(10, 21):
        ax.plot([ex], [0], "|", ms=8, color="0.6")
        ax.annotate(rf"$10^{{{ex}}}$", (ex, -0.62), ha="center", fontsize=6.5,
                    color="0.55")
    ax.annotate("metres (logarithmic scale: each tick is 10$\\times$ the "
                "last)", (15.2, -0.85), ha="center", fontsize=8, color="0.4")
    ax.set_ylim(-1.05, 0.9)
    ax.set_xlim(9.5, 21)
    ax.axis("off")
    save(figr, "dg_scales.pdf")


def dg_chi2() -> None:
    figr, ax = plt.subplots(figsize=(5.6, 3.4))
    rng = np.random.default_rng(7)
    t = np.linspace(0, 10, 16)
    y = 2.0 + 1.4 * np.sin(0.8 * t) + rng.normal(0, 0.35, t.size)
    ax.errorbar(t, y, yerr=0.35, fmt="o", ms=4, lw=0.8, color="0.3",
                capsize=2, label="data with error bars")
    tt = np.linspace(0, 10, 300)
    chi_g = np.sum(((y - (2.0 + 1.4 * np.sin(0.8 * t))) / 0.35) ** 2)
    chi_b = np.sum(((y - (2.0 + 1.4 * np.sin(0.95 * t))) / 0.35) ** 2)
    ax.plot(tt, 2.0 + 1.4 * np.sin(0.8 * tt), color=OKABE["green"], lw=1.3,
            label=rf"good model: $\chi^2={chi_g:.0f}$ for {t.size} points")
    ax.plot(tt, 2.0 + 1.4 * np.sin(0.95 * tt), color=OKABE["vermillion"],
            lw=1.1, ls="--", label=rf"wrong model: $\chi^2={chi_b:.0f}$")
    ax.annotate(r"$\chi^2$ counts mismatch in units of the error bar:"
                "\n" r"$\chi^2/N \approx 1$ means `fits within the errors'",
                (0.02, 0.02), xycoords="axes fraction", fontsize=8,
                color="0.35")
    ax.set_xlabel("time")
    ax.set_ylabel("measured quantity")
    ax.legend(fontsize=7, loc="upper right")
    save(figr, "dg_chi2.pdf")


def dg_mcmc_walkers() -> None:
    figr, ax = plt.subplots(figsize=(4.8, 4.0))
    rng = np.random.default_rng(11)
    rho = 0.8
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    xg, yg = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
    P = np.exp(-0.5 * (xg**2 - 2 * rho * xg * yg + yg**2) / (1 - rho**2))
    ax.contour(xg, yg, P, levels=[0.05, 0.32, 0.7], colors="0.7",
               linewidths=0.8)

    def logp(v):
        z = np.linalg.solve(L, v)
        return -0.5 * z @ z

    for c in [OKABE["blue"], OKABE["vermillion"], OKABE["green"]]:
        pos = rng.uniform(-3.5, 3.5, 2)
        path = [pos.copy()]
        for _ in range(90):
            prop = pos + rng.normal(0, 0.55, 2)
            if np.log(rng.uniform()) < logp(prop) - logp(pos):
                pos = prop
            path.append(pos.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], lw=0.6, color=c, alpha=0.85)
        ax.plot(path[0, 0], path[0, 1], "s", ms=5, color=c)
        ax.plot(path[-25:, 0], path[-25:, 1], ".", ms=3, color=c)
    ax.annotate("walkers start anywhere (squares),\nwander, and settle where "
                "the data\nsays the parameters are likely —\nthe dots ARE the "
                "uncertainty map", (0.02, 0.02), xycoords="axes fraction",
                fontsize=8, color="0.35")
    ax.set_xlabel("parameter 1")
    ax.set_ylabel("parameter 2")
    ax.set_xticks([])
    ax.set_yticks([])
    save(figr, "dg_mcmc_walkers.pdf")


def dg_quadrupole() -> None:
    figr, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))
    th = np.linspace(0, 2 * np.pi, 300)
    for ax, (title, ry, sub) in zip(axes, [
            ("Kerr: `no hair'", 0.82,
             "shape fixed by $M$ and spin alone:\n$Q=-Ma^2$, nothing to "
             "choose"),
            (r"non-Kerr: $\epsilon_q \neq 0$", 0.62,
             "extra squashing $\\epsilon_q M^3$ would\nshift stellar orbits "
             "measurably")]):
        ax.fill(np.cos(th), ry * np.sin(th), color="0.25")
        for s in (1.35, 1.75, 2.2):
            ax.plot(s * np.cos(th), (ry + 0.75 * (s - 1)) * np.sin(th),
                    lw=0.7, color=OKABE["blue"], alpha=0.6)
        ax.annotate("", xy=(0, 1.45), xytext=(0, 0.7),
                    arrowprops=dict(arrowstyle="->", lw=1.4,
                                    color=OKABE["orange"]))
        ax.annotate("spin", (0.08, 1.2), fontsize=8, color=OKABE["orange"])
        ax.set_title(title, fontsize=10)
        ax.annotate(sub, (0.02, 0.02), xycoords="axes fraction",
                    fontsize=7.5, color="0.35")
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.axis("off")
    save(figr, "dg_quadrupole.pdf")


def dg_timeline() -> None:
    figr, ax = plt.subplots(figsize=(6.8, 2.8))
    events = [
        (1992, "NACO/Keck imaging\nof the S-stars begins", OKABE["blue"], 1),
        (2002, "S2 pericentre #1", OKABE["green"], -1),
        (2008, "16-yr orbit closed:\n$M \\sim 4\\times10^6 M_\\odot$",
         OKABE["blue"], 1),
        (2018.4, "S2 pericentre #2:\nGRAVITY redshift",
         OKABE["vermillion"], -1),
        (2020, "Schwarzschild\nprecession detected", OKABE["vermillion"], 1),
        (2022, "EHT image of Sgr A*;\nGRAVITY mass/distance",
         OKABE["purple"], -1),
        (2026, "this simulator", "0.2", 1),
    ]
    ax.hlines(0, 1990, 2028, color="0.4", lw=1.2)
    for yr, lab, c, side in events:
        ax.plot([yr], [0], "o", ms=6, color=c)
        ax.annotate(lab, (yr, 0.3 * side), ha="center",
                    va="bottom" if side > 0 else "top", fontsize=7.5,
                    color=c)
    for yr in range(1990, 2029, 5):
        ax.annotate(str(yr), (yr, -0.78), ha="center", fontsize=7,
                    color="0.5")
    ax.set_xlim(1989, 2029)
    ax.set_ylim(-1.05, 1.0)
    ax.axis("off")
    save(figr, "dg_timeline.pdf")


DIAGRAMS = [
    dg_shadow_formation,
    dg_kerr_anatomy,
    dg_frame_drag,
    dg_kepler_ellipse,
    dg_rosette,
    dg_three_redshifts,
    dg_astrometry_rv,
    dg_scales,
    dg_chi2,
    dg_mcmc_walkers,
    dg_quadrupole,
    dg_timeline,
]


def main() -> None:
    print("generating thesis diagrams ...")
    for fn in DIAGRAMS:
        fn()
    print(f"done ({len(DIAGRAMS)} diagrams).")


if __name__ == "__main__":
    main()
