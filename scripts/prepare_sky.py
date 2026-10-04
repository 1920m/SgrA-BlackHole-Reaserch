"""Gate 15 — pack the NASA SVS Deep Star Maps 2020 galactic sky for the engine.

Input  : data/sky/starmap_2020_8k_gal.exr   (download, ~160 MB; NOT in git)
         https://svs.gsfc.nasa.gov/4851/  — plate carrée, galactic
         coordinates, centred on l = 0 with longitude increasing to the LEFT
         (u = 0.5 − l/2π, v = 0.5 − b/π; verified: the LMC at l = 280.5°,
         b = −32.9° is 20× the median sky brightness at that pixel).
Output : engine/assets/sky/starmap_gal_8k.sky (NOT in git)
         header  b"SGRSKY01", uint32 width, height, levels, format(0)
         body    every mip level, finest first, as VK_FORMAT_B10G11R11_UFLOAT
                 texels (R bits 0-10, G 11-21, B 22-31) — 4 bytes/texel keeps
                 the 8K map + mips at ~180 MB, inside a 2 GB GPU.

Radiance stays LINEAR (the EXR is linear half-float); mips are 2×2 box
averages. Credit (required by NASA SVS): NASA/Goddard Space Flight Center
Scientific Visualization Studio; Gaia DR2: ESA/Gaia/DPAC.

Run: py -3.11 scripts/prepare_sky.py
"""
from __future__ import annotations

import os
import pathlib
import struct
import sys

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402  (env var must be set before import)
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "sky" / "starmap_2020_8k_gal.exr"
DST = ROOT / "engine" / "assets" / "sky" / "starmap_gal_8k.sky"
URL = ("https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/"
       "starmap_2020_8k_gal.exr")


def pack_b10g11r11(rgb: np.ndarray) -> np.ndarray:
    """Linear float RGB → packed unsigned 11/11/10-bit floats.

    The 11- and 10-bit formats share half-float's 5-bit exponent and bias,
    so each channel is its float16 bit pattern with the low mantissa bits
    rounded off (sign bit is zero after the clamp)."""
    h = np.clip(rgb, 0.0, 65000.0).astype(np.float16).view(np.uint16)
    h = h.astype(np.uint32)
    r = (h[..., 0] + 0x8) >> 4
    g = (h[..., 1] + 0x8) >> 4
    b = (h[..., 2] + 0x10) >> 5
    return (r & 0x7FF) | ((g & 0x7FF) << 11) | ((b & 0x3FF) << 22)


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}\ndownload it (~160 MB) from\n  {URL}")
        return 1
    img = cv2.imread(str(SRC), cv2.IMREAD_UNCHANGED)
    if img is None:
        print("OpenCV could not read the EXR (needs OPENCV_IO_ENABLE_OPENEXR)")
        return 1
    rgb = np.ascontiguousarray(img[..., :3][..., ::-1], dtype=np.float32)
    h, w = rgb.shape[:2]
    levels = [rgb]
    while levels[-1].shape[0] > 1 or levels[-1].shape[1] > 1:
        a = levels[-1]
        hh, ww = max(1, a.shape[0] // 2), max(1, a.shape[1] // 2)
        a = a[: hh * 2 if a.shape[0] > 1 else 1, : ww * 2 if a.shape[1] > 1 else 1]
        fy = 2 if a.shape[0] > 1 else 1
        fx = 2 if a.shape[1] > 1 else 1
        levels.append(a.reshape(hh, fy, ww, fx, 3).mean(axis=(1, 3)))
    DST.parent.mkdir(parents=True, exist_ok=True)
    with open(DST, "wb") as f:
        f.write(b"SGRSKY01")
        f.write(struct.pack("<4I", w, h, len(levels), 0))
        for lv in levels:
            f.write(np.ascontiguousarray(pack_b10g11r11(lv)).astype("<u4").tobytes())
    print(f"wrote {DST} ({w}x{h}, {len(levels)} levels, "
          f"{DST.stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
