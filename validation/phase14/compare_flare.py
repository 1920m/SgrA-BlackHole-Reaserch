"""Gate 18 — the live renderer's hot spot vs its fp64 CPU twin.

flare_tool (science/flares) writes fp64 Kerr-Schild images of the GRAVITY
2018 hot spot (R = 7 M, i = 160 deg, a* = 0.94, 256x256 pixels spanning
+-14 M of impact parameter) at four observing times over one orbit. The live
GPU path renders the SAME emitter (constants from science/flares/Hotspot.cpp,
slow light from the ray's own Kerr-Schild time) with an identical pixel grid:
camera at r = 1000 M, fov = 2 atan(14/1000), pixel-centre rays (one sample),
no disk, black sky, grey linear display (--flare-gray).

Checks, per snapshot:
  1. total flux GPU/CPU within 3 %
  2. flux-weighted centroid within 0.5 pixel (0.11 M = 0.56 uas)
  3. image correlation > 0.98 (same direct + lensed images)

Run:  py -3.11 validation/phase14/compare_flare.py
Exit code 0 = pass. Results: validation/extension/g18_flare_gpu.json
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

N, HALF, RCAM, INCL = 256, 14.0, 1000.0, 160.0
CASE = "g18_R7_i160"


def run(cmd: list[str]) -> None:
    env = dict(os.environ)
    env.setdefault("VK_LAYER_PATH", r"C:\vcpkg\installed\x64-windows\bin")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        sys.stderr.write(res.stdout[-2000:] + res.stderr[-2000:])
        raise SystemExit(f"command failed ({res.returncode}): {cmd[0]}")


def moments(img: np.ndarray) -> tuple[float, float, float]:
    b = HALF * (2.0 * (np.arange(N) + 0.5) / N - 1.0)
    bx, by = np.meshgrid(b, -b)          # row 0 = top (+b_y)
    f = img.sum()
    return f, (img * bx).sum() / f, (img * by).sum() / f


def main() -> int:
    J = json.loads((EXT / "flares.json").read_text())
    case = next(c for c in J["cases"] if c["name"] == CASE)
    P = case["period_M"]
    OUT.mkdir(parents=True, exist_ok=True)
    fov = 2.0 * math.atan(HALF / RCAM)
    px = 2.0 * HALF / N
    report = {"case": CASE, "pixel_M": px, "snapshots": []}
    ok = True
    for k in (0, 64, 128, 192):
        cpu = np.fromfile(EXT / f"flare_{CASE}_k{k}.f32", dtype=np.float32).reshape(N, N)
        cpu = cpu.astype(np.float64)
        # expose so the CPU peak sits at 0.8 of full scale (no clipping)
        ev = math.log2(0.8 / cpu.max())
        png = OUT / f"g18_flare_gpu_k{k}.png"
        run([str(ENGINE), "--shader", "disk", "--offscreen", "--validation",
             "--width", str(N), "--height", str(N), "--preset", "0",
             "--rcam", str(RCAM), "--incl", str(INCL), "--fov", f"{fov:.12f}",
             "--no-disk", "--blacksky", "--flare", "--flare-gray",
             "--flare-t", f"{P * k / 256.0:.9f}", "--display", "linear",
             "--exposure", f"{ev:.9f}", "--spp", "1", "--screenshot", str(png)])
        g8 = np.asarray(Image.open(png).convert("L"), dtype=np.float64)
        gpu = (g8 / 255.0) ** 2.2 / 2.0 ** ev
        fc, cxc, cyc = moments(cpu)
        fg, cxg, cyg = moments(gpu)
        corr = float(np.corrcoef(cpu.ravel(), gpu.ravel())[0, 1])
        dc = math.hypot(cxg - cxc, cyg - cyc)
        snap = {"k": k, "t_obs_M": P * k / 256.0, "flux_ratio": fg / fc,
                "centroid_cpu_M": [cxc, cyc], "centroid_gpu_M": [cxg, cyg],
                "centroid_diff_px": dc / px, "correlation": corr}
        report["snapshots"].append(snap)
        good = abs(fg / fc - 1.0) < 0.03 and dc / px < 0.5 and corr > 0.98
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} k={k:3d}  flux GPU/CPU {fg / fc:.4f}  "
              f"centroid diff {dc / px:.3f} px  corr {corr:.5f}")
    report["pass"] = bool(ok)
    (EXT / "g18_flare_gpu.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
