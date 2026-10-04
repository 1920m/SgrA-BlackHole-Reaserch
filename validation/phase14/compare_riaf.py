"""Gate 19 — the live 230 GHz band vs its fp64 CPU twin.

riaf_tool (science/riaf) writes the fp64 image of the hot accretion flow
(main case: a* = 0.94, i = 30 deg, n0 fitted to 2.4 Jy) on a grid of
n x n pixels spanning +-12 M of impact parameter, camera at 1000 M. The live
GPU path traces the same flow (same profiles, field, emissivity, Kirchhoff
absorption; fp32) with an identical pixel grid, and writes the
Rayleigh-Jeans brightness temperature in grey (--band 230 --gray).

Checks:
  1. total flux GPU/CPU within 3 %
  2. image correlation > 0.98
  3. flux-weighted centroid within 0.5 pixel

Run:  py -3.11 validation/phase14/compare_riaf.py
Exit code 0 = pass. Results: validation/extension/g19_riaf_gpu.json
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXT = ROOT / "validation" / "extension"
OUT = ROOT / "validation" / "phase14" / "strips"
ENGINE = ROOT / "build" / "engine-rel" / "engine" / "sgr_engine.exe"

K_B, C_CGS = 1.380649e-16, 2.99792458e10


def run(cmd: list[str]) -> None:
    env = dict(os.environ)
    env.setdefault("VK_LAYER_PATH", r"C:\vcpkg\installed\x64-windows\bin")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    if res.returncode != 0:
        sys.stderr.write(res.stdout[-2000:] + res.stderr[-2000:])
        raise SystemExit(f"command failed ({res.returncode}): {cmd[0]}")


def moments(img: np.ndarray, half: float) -> tuple[float, float, float]:
    n = img.shape[0]
    b = half * (2.0 * (np.arange(n) + 0.5) / n - 1.0)
    bx, by = np.meshgrid(b, -b)
    f = img.sum()
    return f, (img * bx).sum() / f, (img * by).sum() / f


def main() -> int:
    J = json.loads((EXT / "riaf.json").read_text())
    m = J["main"]
    n, half, nu = m["n_pixels"], m["half_width_M"], J["nu_obs_hz"]
    cpu_I = np.fromfile(EXT / "riaf_main.f32", dtype=np.float32).reshape(n, n).astype(float)
    cpu = cpu_I * C_CGS ** 2 / (2.0 * K_B * nu * nu) / 1e10      # T_b / 1e10 K
    ev = math.log2(0.8 / cpu.max())
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "g19_riaf_gpu.png"
    fov = 2.0 * math.atan(half / 1000.0)
    run([str(ENGINE), "--shader", "disk", "--offscreen", "--validation",
         "--width", str(n), "--height", str(n), "--preset", "0",
         "--astar", "0.94", "--rcam", "1000", "--incl", str(m["incl_deg"]),
         "--fov", f"{fov:.12f}", "--band", "230", "--gray", "--display", "linear",
         "--exposure", f"{ev:.9f}", "--spp", "1", "--screenshot", str(png)])
    g8 = np.asarray(Image.open(png).convert("L"), dtype=np.float64)
    gpu = (g8 / 255.0) ** 2.2 / 2.0 ** ev
    fc, cxc, cyc = moments(cpu, half)
    fg, cxg, cyg = moments(gpu, half)
    px = 2.0 * half / n
    corr = float(np.corrcoef(cpu.ravel(), gpu.ravel())[0, 1])
    dc = math.hypot(cxg - cxc, cyg - cyc) / px
    res = {"flux_ratio": fg / fc, "correlation": corr, "centroid_diff_px": dc,
           "pixel_M": px, "n_pixels": n}
    ok = abs(fg / fc - 1.0) < 0.03 and corr > 0.98 and dc < 0.5
    res["pass"] = bool(ok)
    (EXT / "g19_riaf_gpu.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"flux GPU/CPU {fg / fc:.4f}  correlation {corr:.5f}  centroid {dc:.3f} px  "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
