"""Gate 24 — the live renderer's in-fall camera vs the fp64 CPU reference.

For each radius in validation/extension/infall.json (written by infall_tool:
the edges of the dark region on the horizontal and vertical image axes of
the in-faller's view, bisected on the fp64 Kerr-Schild tracer in the INGOING
chart with the renderer's own camera frame, sgr_physics/infall.hpp), the GPU
renders the same frame (--flight-r) as one-row and one-column strips against
a uniform, unshifted sky with no disk, so rays that end on the past horizon
are exact zeros. Every GPU edge must lie within 1.5 pixels of a CPU edge,
and the two must find the same number of edges — inside the horizon too.

Continuity through the horizon: full frames at r₊ + 0.03, r₊ + 0.01,
r₊ − 0.01 and r₊ − 0.03 M; the fraction of pixels whose dark/sky state
changes from one frame to the next must not jump at the crossing (no seam
or flash): the change across r₊ is at most twice the larger change of the
neighbouring pairs.

Run:  py -3.11 validation/phase14/compare_infall.py
Exit code 0 = pass. Results: validation/extension/g24_infall_gpu.json
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
OUT = ROOT / "validation" / "extension"
STRIPS = ROOT / "validation" / "phase14" / "strips"
ENGINE = ROOT / "build" / "engine-rel" / "engine" / "sgr_engine.exe"
REF = OUT / "infall.json"
N = 2001


def run(cmd: list[str]) -> None:
    env = dict(os.environ)
    env.setdefault("VK_LAYER_PATH", r"C:\vcpkg\installed\x64-windows\bin")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        sys.stderr.write(res.stdout[-2000:] + res.stderr[-2000:])
        raise SystemExit(f"command failed ({res.returncode}): {cmd[0]}")
    if "validation errors: 0" not in res.stdout + res.stderr:
        raise SystemExit("Vulkan validation errors reported:\n" + res.stdout[-1500:])


def gpu_edges(r: float, T: float, vertical: bool) -> list[float]:
    STRIPS.mkdir(parents=True, exist_ok=True)
    w, h = (3, N) if vertical else (N, 3)
    fov = 2.0 * math.atan(T * (w / h if vertical else 1.0))   # vertical extent: tan(fov/2)·h/w
    path = STRIPS / f"infall_r{r:g}_{'v' if vertical else 'h'}.png"
    run([str(ENGINE), "--shader", "disk", "--offscreen", "--validation",
         "--width", str(w), "--height", str(h), "--spp", "1",
         "--flight-r", f"{r:.9g}", "--fov", f"{fov:.12g}",
         "--uniform-sky", "--no-sky-shift", "--no-disk",
         "--display", "linear", "--exposure", "0", "--bloom", "0",
         "--screenshot", str(path)])
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32).sum(axis=2)
    line = img[:, 1] if vertical else img[1]
    dark = line == 0
    # image rows run top → bottom (= +ty → −ty); columns left → right (−tx → +tx)
    t = lambda px: T * (2.0 * (px + 0.5) / N - 1.0)
    edges = [t(i + 0.5) for i in range(N - 1) if dark[i] != dark[i + 1]]
    return sorted(-e for e in edges) if vertical else edges


def dark_mask(r: float, w: int = 320, h: int = 180, fov: float = 1.6) -> np.ndarray:
    STRIPS.mkdir(parents=True, exist_ok=True)
    path = STRIPS / f"infall_frame_r{r:.5f}.png"
    run([str(ENGINE), "--shader", "disk", "--offscreen", "--validation",
         "--width", str(w), "--height", str(h), "--spp", "1",
         "--flight-r", f"{r:.9g}", "--fov", f"{fov}",
         "--uniform-sky", "--no-sky-shift", "--no-disk",
         "--display", "linear", "--exposure", "0", "--bloom", "0",
         "--screenshot", str(path)])
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int32).sum(axis=2) == 0


def continuity(rp: float) -> dict:
    radii = [rp + 0.03, rp + 0.01, rp - 0.01, rp - 0.03]
    masks = [dark_mask(r) for r in radii]
    ch = [float(np.mean(masks[i] != masks[i + 1])) for i in range(3)]
    ok = ch[1] <= 2.0 * max(ch[0], ch[2]) + 1e-3
    print(f"[infall] continuity through r+ = {rp:.4f}: changed pixels {ch[0]:.4f} (outside), "
          f"{ch[1]:.4f} (across), {ch[2]:.4f} (inside) -> {'PASS' if ok else 'FAIL'}")
    return {"radii": radii, "changed_fraction": ch, "dark_fraction": [float(m.mean()) for m in masks],
            "pass": bool(ok)}


def main() -> int:
    ref = json.loads(REF.read_text())
    ok = True
    rows = []
    worst = 0.0
    for s in ref["strips"]:
        r, T = s["r"], s["strip_T"]
        px = 2.0 * T / N
        res = {"r": r}
        for key, vertical in (("edges_h", False), ("edges_v", True)):
            cpu = sorted(s[key])
            gpu = gpu_edges(r, T, vertical)
            same = len(cpu) == len(gpu)
            devs = [abs(g - c) / px for g, c in zip(gpu, cpu)] if same else []
            m = max(devs) if devs else (0.0 if same else math.inf)
            worst = max(worst, m)
            res[key] = {"cpu": cpu, "gpu": gpu, "max_dev_px": m, "same_count": same}
            ok &= same and m < 1.5
            print(f"[infall] r = {r:4.2f} {'vertical  ' if vertical else 'horizontal'}: "
                  f"{len(cpu)} CPU / {len(gpu)} GPU edges, max |dev| {m:.2f} px")
        rows.append(res)
    cont = continuity(ref["trip"]["r_plus"])
    ok &= cont["pass"]
    out = {"strip_pixels": N, "cases": rows, "max_dev_px": worst, "continuity": cont,
           "pass": bool(ok)}
    (OUT / "g24_infall_gpu.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"compare_infall: {'PASS' if ok else 'FAIL'} (max {worst:.2f} px)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
