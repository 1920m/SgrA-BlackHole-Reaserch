"""Gate 17 — the live renderer's moving observers vs the fp64 CPU reference.

For each observer configuration in validation/extension/observers.json
(written by observer_tool: the shadow edges each observer sees, bisected on
the fp64 Kerr-Schild tracer with the SAME observer tetrads the engine uses,
sgr_physics/observers.hpp), the GPU live path renders a one-row strip
through the image centre against a uniform sky (--uniform-sky) with no disk,
so captured rays are exact zeros. The shadow's left/right edges on that row
are compared with the CPU edges.

Checks:
  1. at a* = 0 the CPU edges equal Synge's shadow + special-relativistic
     aberration (observer_tool's analytic columns) to 1e-8
  2. every GPU edge lies within 1.5 pixels of the CPU edge (fp32 live
     integration, pixel-centre sampling)

Run:  py -3.11 validation/phase14/compare_observers.py
Exit code 0 = pass. Results: validation/extension/g17_observers_gpu.json
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
REF = OUT / "observers.json"

W, H = 2001, 3          # row 1 is the image's horizontal centre line
THALF = 1.4             # tan(fov/2): covers every edge in the reference set
FOV = 2.0 * math.atan(THALF)


def run(cmd: list[str]) -> None:
    env = dict(os.environ)
    env.setdefault("VK_LAYER_PATH", r"C:\vcpkg\installed\x64-windows\bin")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        sys.stderr.write(res.stdout[-2000:] + res.stderr[-2000:])
        raise SystemExit(f"command failed ({res.returncode}): {cmd[0]}")


def gpu_edges(path: pathlib.Path) -> tuple[float, float]:
    """Signed tan offsets of the shadow's edges on the centre row: midway
    between the last sky pixel and the first captured pixel."""
    row = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)[1].sum(axis=1)
    dark = np.where(row == 0)[0]
    if dark.size == 0:
        return math.nan, math.nan
    tx = lambda px: THALF * (2.0 * (px + 0.5) / W - 1.0)
    return tx(dark[0] - 0.5), tx(dark[-1] + 0.5)


def main() -> int:
    ref = json.loads(REF.read_text())
    STRIPS.mkdir(parents=True, exist_ok=True)
    px = 2.0 * THALF / W
    ok = True
    report = {"pixel_tan": px, "cases": []}

    for s in ref["schwarzschild_check"]:
        for side in ("left", "right"):
            d = abs(s[f"x_{side}"] - s[f"analytic_x_{side}"])
            if d > 1e-8:
                ok = False
                print(f"FAIL analytic {s['observer']} {side}: {d:.2e}")
    report["analytic_max_diff"] = max(
        abs(s[f"x_{side}"] - s[f"analytic_x_{side}"])
        for s in ref["schwarzschild_check"] for side in ("left", "right"))

    for s in ref["shadow_edges"]:
        if not s.get("valid"):
            continue
        name = f"g17_strip_{s['observer']}_a{s['a']:g}_r{s['r']:g}_i{s['incl_deg']:g}.png"
        png = STRIPS / name
        run([str(ENGINE), "--shader", "disk", "--offscreen", "--validation",
             "--width", str(W), "--height", str(H), "--preset", "0",
             "--astar", str(s["a"]), "--rcam", str(s["r"]),
             "--incl", str(s["incl_deg"]), "--fov", f"{FOV:.9f}",
             "--observer", s["observer"], "--rout", "0.5",
             "--uniform-sky", "--display", "linear", "--exposure", "0",
             "--screenshot", str(png)])
        gl, gr = gpu_edges(png)
        dl = (gl - s["x_left"]) / px
        dr = (gr - s["x_right"]) / px
        case = {"observer": s["observer"], "a": s["a"], "r": s["r"],
                "incl_deg": s["incl_deg"], "speed": s["speed"],
                "cpu": [s["x_left"], s["x_right"]], "gpu": [gl, gr],
                "diff_px": [dl, dr]}
        report["cases"].append(case)
        good = abs(dl) <= 1.5 and abs(dr) <= 1.5
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {s['observer']:6s} a={s['a']:<4g} "
              f"r={s['r']:<4g} i={s['incl_deg']:<4g} v={s['speed']:.3f}  "
              f"left {s['x_left']:+.4f}/{gl:+.4f} ({dl:+.2f} px)  "
              f"right {s['x_right']:+.4f}/{gr:+.4f} ({dr:+.2f} px)")

    report["max_abs_diff_px"] = max(max(abs(c["diff_px"][0]), abs(c["diff_px"][1]))
                                    for c in report["cases"])

    # Record of the defect this comparison found: with the pre-G17 budget of
    # 1600 steps, near-critical rays at the weakly unstable prograde photon
    # orbit (a* = 0.94, edge-on) ran out of steps and counted as captured.
    s = next(c for c in ref["shadow_edges"]
             if c["observer"] == "zamo" and c["a"] == 0.94 and c["r"] == 10.0)
    png = STRIPS / "g17_strip_zamo_legacy1600.png"
    run([str(ENGINE), "--shader", "disk", "--offscreen", "--validation",
         "--width", str(W), "--height", str(H), "--preset", "0",
         "--astar", "0.94", "--rcam", "10", "--incl", "90", "--fov", f"{FOV:.9f}",
         "--observer", "zamo", "--rout", "0.5", "--uniform-sky",
         "--display", "linear", "--exposure", "0", "--max-steps", "1600",
         "--screenshot", str(png)])
    gl, gr = gpu_edges(png)
    import re
    hdr = (ROOT / "engine" / "src" / "app.hpp").read_text(encoding="utf-8")
    report["current_budget"] = int(re.search(r"int max_steps = (\d+);", hdr).group(1))
    report["legacy_budget"] = {"max_steps": 1600, "case": "zamo a=0.94 r=10 i=90",
                               "diff_px": [(gl - s["x_left"]) / px,
                                           (gr - s["x_right"]) / px]}
    print(f"legacy 1600-step budget: left edge {report['legacy_budget']['diff_px'][0]:+.2f} px, "
          f"right {report['legacy_budget']['diff_px'][1]:+.2f} px")
    report["pass"] = bool(ok)
    (OUT / "g17_observers_gpu.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"max |GPU - CPU| = {report['max_abs_diff_px']:.2f} px; "
          f"analytic check {report['analytic_max_diff']:.1e}; "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
