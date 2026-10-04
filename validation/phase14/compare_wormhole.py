"""Gate 21 — EXOTIC SPACETIME, NOT SGR A*: the live renderer's Dneg wormhole
vs the fp64 CPU twin (science/wormhole, wormhole_tool → wormhole.json).

For each camera distance l_c in wormhole.json the GPU live path renders a
one-row strip (and a one-column strip) through the image centre with the
validation mask (--uniform-sky: rays traced back to our universe = 1, to the
other universe = 0). The edge of the other universe's disk on each strip is
compared with the CPU edge, which equals the exact sin(alpha) = rho/r(l_c).

Checks:
  1. the CPU edge equals the analytic edge to 1e-6 deg (wormhole_tool)
  2. all four GPU edges (left, right, top, bottom) lie within 1 pixel of it
     (fp32 planar RK4 in the shader, pixel-centre sampling)

Run:  py -3.11 validation/phase14/compare_wormhole.py
Exit code 0 = pass. Results: validation/extension/wormhole_gpu.json
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
REF = OUT / "wormhole.json"

N = 2001                # strip length; pixel N//2 is the image centre
EDGE_AT = 0.45          # place the CPU edge at this fraction of the half-strip


def run(cmd: list[str]) -> None:
    env = dict(os.environ)
    env.setdefault("VK_LAYER_PATH", r"C:\vcpkg\installed\x64-windows\bin")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        sys.stderr.write(res.stdout[-2000:] + res.stderr[-2000:])
        raise SystemExit(f"command failed ({res.returncode}): {cmd[0]}")
    if "validation errors: 0" not in res.stdout + res.stderr:
        raise SystemExit("Vulkan validation errors reported:\n" + res.stdout[-1500:])


def render(l_c: float, thalf: float, vertical: bool) -> np.ndarray:
    """The mask along the centre line, as tan offsets of the pixel centres
    and a 0/1 'other universe' flag."""
    STRIPS.mkdir(parents=True, exist_ok=True)
    w, h = (3, N) if vertical else (N, 3)
    # the shader's vertical extent is tan(fov/2) * h/w
    fov = 2.0 * math.atan(thalf * (w / h if vertical else 1.0))
    path = STRIPS / f"wormhole_l{l_c:g}_{'v' if vertical else 'h'}.png"
    run([str(ENGINE), "--shader", "disk", "--offscreen", "--validation",
         "--width", str(w), "--height", str(h), "--spp", "1",
         "--wormhole", "--wh-l", f"{l_c:.9g}", "--fov", f"{fov:.12g}",
         "--uniform-sky", "--display", "linear", "--exposure", "0", "--bloom", "0",
         "--screenshot", str(path)])
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32).sum(axis=2)
    line = img[:, 1] if vertical else img[1]
    return line < 3 * 128


def edges(other: np.ndarray, thalf: float) -> tuple[float, float]:
    """Signed tan offsets of the other universe's disk edges: midway between
    the last our-universe pixel and the first other-universe pixel."""
    idx = np.where(other)[0]
    if idx.size == 0:
        return math.nan, math.nan
    t = lambda px: thalf * (2.0 * (px + 0.5) / N - 1.0)
    return t(idx[0] - 0.5), t(idx[-1] + 0.5)


def main() -> int:
    ref = json.loads(REF.read_text())
    ok = True
    rows = []
    for cam in ref["cameras"]:
        l_c, a_cpu = cam["l_c"], cam["edge_deg"]
        cpu_ok = abs(a_cpu - cam["edge_analytic_deg"]) < 1e-6
        ok &= cpu_ok
        t_e = math.tan(math.radians(a_cpu))
        thalf = t_e / EDGE_AT
        px_tan = 2.0 * thalf / N          # one pixel in tan units
        res = {"l_c": l_c, "cpu_edge_deg": a_cpu, "cpu_matches_analytic": cpu_ok,
               "thalf": thalf, "pixel_deg_at_edge":
                   math.degrees(px_tan / (1.0 + t_e * t_e))}
        devs = []
        for vertical, names in ((False, ("left", "right")), (True, ("top", "bottom"))):
            lo, hi = edges(render(l_c, thalf, vertical), thalf)
            # vertical strips run top → bottom (image y down = −ty)
            for name, t in zip(names, (lo, hi)):
                res[f"gpu_{name}_deg"] = math.degrees(math.atan(abs(float(t))))
                devs.append(float((abs(t) - t_e) / px_tan))
        res["max_dev_px"] = max(abs(d) for d in devs)
        res["devs_px"] = devs
        res["pass"] = bool(cpu_ok and res["max_dev_px"] < 1.0)
        ok &= res["pass"]
        rows.append(res)
        print(f"[wormhole] l_c = {l_c:5.1f}: CPU edge {a_cpu:.6f} deg; GPU "
              f"L/R/T/B {res['gpu_left_deg']:.4f}/{res['gpu_right_deg']:.4f}/"
              f"{res['gpu_top_deg']:.4f}/{res['gpu_bottom_deg']:.4f} deg; "
              f"max |dev| {res['max_dev_px']:.2f} px "
              f"({'PASS' if res['pass'] else 'FAIL'})")
    out = {"strip_pixels": N, "edge_fraction": EDGE_AT, "cases": rows,
           "max_dev_px": max(r["max_dev_px"] for r in rows), "pass": ok}
    (OUT / "wormhole_gpu.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"compare_wormhole: {'PASS' if ok else 'FAIL'} -> validation/extension/wormhole_gpu.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
