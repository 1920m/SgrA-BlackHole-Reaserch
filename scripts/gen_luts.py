"""Generate the visual preset LUTs and the blackbody disk
colormap (§7) as 256x1 RGBA PNGs under engine/assets/luts/.

The disk_colormap encodes the physical Planck/CIE blackbody locus from 500 K
to 60 000 K (log-spaced) — matching NovikovThorneDisk::blackbody_rgb so GPU
and CPU shading agree. The four artistic preset LUTs remap *display* colour
only; the physics (flux, redshift) is untouched by preset switching.

Run: py -3.11 scripts/gen_luts.py
"""
from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image

OUT = pathlib.Path(__file__).resolve().parent.parent / "engine" / "assets" / "luts"


def planck_xyz(T: float) -> tuple[float, float, float]:
    # Multi-lobe Gaussian CIE 1931 fits (Wyman, Sloan & Shirley 2013) — the
    # same fits as physics/src/disk_model.cpp.
    def g(x, alpha, mu, s1, s2):
        t = (x - mu) / (s1 if x < mu else s2)
        return alpha * np.exp(-0.5 * t * t)

    X = Y = Z = 0.0
    for wl in np.arange(380.0, 781.0, 5.0):
        lm = wl * 1e-9
        B = 1.0 / (lm**5 * (np.exp(1.4388e-2 / (lm * T)) - 1.0))
        X += B * (g(wl, 1.056, 599.8, 37.9, 31.0) + g(wl, 0.362, 442.0, 16.0, 26.7)
                  + g(wl, -0.065, 501.1, 20.4, 26.2))
        Y += B * (g(wl, 0.821, 568.8, 46.9, 40.5) + g(wl, 0.286, 530.9, 16.3, 31.1))
        Z += B * (g(wl, 1.217, 437.0, 11.8, 36.0) + g(wl, 0.681, 459.0, 26.0, 13.8))
    return X, Y, Z


_norm = None


def planck_rgb(T: float) -> np.ndarray:
    global _norm
    if _norm is None:
        _norm = 1.0 / planck_xyz(6500.0)[1]
    X, Y, Z = (v * _norm for v in planck_xyz(T))
    R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    B = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    return np.clip([R, G, B], 0.0, None)


def save_lut(name: str, rgb: np.ndarray) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    srgb = np.clip(rgb, 0.0, 1.0) ** (1.0 / 2.2)
    px = np.zeros((1, 256, 4), dtype=np.uint8)
    px[0, :, :3] = np.round(srgb * 255).astype(np.uint8)
    px[0, :, 3] = 255
    Image.fromarray(px, "RGBA").save(OUT / f"{name}.png")
    print(f"wrote {OUT / (name + '.png')}")


def ramp(stops: list[tuple[float, tuple[float, float, float]]]) -> np.ndarray:
    """Piecewise-linear colour ramp over u ∈ [0,1] in linear RGB."""
    u = np.linspace(0.0, 1.0, 256)
    out = np.zeros((256, 3))
    for i in range(len(stops) - 1):
        u0, c0 = stops[i]
        u1, c1 = stops[i + 1]
        m = (u >= u0) & (u <= u1)
        t = (u[m] - u0) / max(u1 - u0, 1e-9)
        out[m] = np.outer(1 - t, c0) + np.outer(t, c1)
    return out


def main() -> None:
    # Physical blackbody locus, log T in [500, 60000] K, radiance-weighted.
    T = np.exp(np.linspace(np.log(500.0), np.log(60000.0), 256))
    bb = np.array([planck_rgb(t) for t in T])
    save_lut("disk_colormap", bb / bb.max())

    # §9 presets (display ramps in linear RGB; u = normalised intensity).
    # Densely stopped so colour keeps moving over the WHOLE ramp: with the
    # live renderer's 24-stop log window the outer disk sits at u ≈ 0.1–0.4,
    # and a ramp that is flat there quantizes the whole outer disk to one
    # colour (and hides the turbulence filaments).
    save_lut("firestorm", ramp([
        (0.00, (0.010, 0.001, 0.001)), (0.10, (0.060, 0.005, 0.003)),
        (0.30, (0.28, 0.03, 0.01)), (0.50, (0.62, 0.10, 0.02)),
        (0.68, (0.92, 0.30, 0.03)), (0.85, (1.0, 0.60, 0.10)),
        (1.00, (1.0, 0.93, 0.72)),
    ]))
    save_lut("gargantua", ramp([
        (0.00, (0.008, 0.012, 0.030)), (0.12, (0.04, 0.07, 0.14)),
        (0.30, (0.14, 0.22, 0.38)), (0.50, (0.38, 0.45, 0.60)),
        (0.68, (0.75, 0.70, 0.60)), (0.85, (0.98, 0.90, 0.70)),
        (1.00, (1.0, 1.0, 1.0)),
    ]))
    save_lut("arctic", ramp([
        (0.00, (0.002, 0.010, 0.020)), (0.12, (0.02, 0.07, 0.12)),
        (0.30, (0.06, 0.22, 0.33)), (0.50, (0.16, 0.45, 0.60)),
        (0.70, (0.45, 0.78, 0.90)), (0.88, (0.80, 0.97, 1.00)),
        (1.00, (0.97, 1.0, 1.0)),
    ]))
    save_lut("nebula", ramp([
        (0.00, (0.012, 0.001, 0.025)), (0.12, (0.06, 0.01, 0.11)),
        (0.30, (0.18, 0.03, 0.28)), (0.50, (0.45, 0.09, 0.45)),
        (0.68, (0.78, 0.22, 0.52)), (0.85, (1.0, 0.52, 0.52)),
        (1.00, (1.0, 0.86, 0.66)),
    ]))


if __name__ == "__main__":
    main()
