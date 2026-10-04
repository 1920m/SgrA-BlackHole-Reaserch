"""Planck spectrum -> CIE 1931 XYZ -> linear sRGB, the Python twin of
physics/src/disk_model.cpp::planck_to_linear_srgb (same Wyman, Sloan &
Shirley 2013 multi-lobe CMF fits, same 5 nm integration, Y(6500 K) = 1).

Used by scripts/make_figures.py (fig_cie_locus). It lived in the retired
8-bit LUT generator gen_luts.py until the live renderer switched to a float
table built from the C++ function at start-up (Gate 13).
"""
from __future__ import annotations

import numpy as np


def planck_xyz(T: float) -> tuple[float, float, float]:
    """Unnormalised CIE XYZ of a blackbody at T kelvin (380-780 nm)."""
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


_norm: float | None = None


def planck_rgb(T: float) -> np.ndarray:
    """Linear sRGB (D65) of a blackbody at T, normalised so Y(6500 K) = 1."""
    global _norm
    if _norm is None:
        _norm = 1.0 / planck_xyz(6500.0)[1]
    X, Y, Z = (v * _norm for v in planck_xyz(T))
    R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    B = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    return np.clip([R, G, B], 0.0, None)
