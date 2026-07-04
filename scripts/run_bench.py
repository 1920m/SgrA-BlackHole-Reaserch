"""Phase 11: assemble validation/phase11/bench.json.

Runs the Google-Benchmark CPU geodesic baseline and the live GPU renderer,
and records the numbers make_numbers.py turns into \\Bench* macros.

Usage:  py -3.11 scripts/run_bench.py
Needs:  build/core-rel/physics/geodesic_bench.exe   (-DSGR_BUILD_BENCH=ON)
        build/engine-rel/engine/sgr_engine.exe
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "validation" / "phase11" / "bench.json"

BENCH_EXE = ROOT / "build/core-rel/physics/geodesic_bench.exe"
ENGINE_EXE = ROOT / "build/engine-rel/engine/sgr_engine.exe"


def run_cpu_bench() -> dict[str, float]:
    print("[bench] CPU geodesic baseline ...")
    res = subprocess.run(
        [str(BENCH_EXE), "--benchmark_format=json",
         "--benchmark_min_time=1.0s"],
        capture_output=True, text=True, check=True, cwd=ROOT)
    data = json.loads(res.stdout)
    out: dict[str, float] = {}
    key_map = {
        "BM_NullGeodesicStep": "cpu_null_steps_per_s",
        "BM_ChristoffelEval": "cpu_christoffel_per_s",
        "BM_TimelikeGeodesicStep": "cpu_timelike_steps_per_s",
    }
    for b in data["benchmarks"]:
        k = key_map.get(b["name"])
        if k:
            out[k] = round(b["items_per_second"])
            print(f"  {b['name']}: {b['items_per_second']:.3e} items/s")
    return out


def run_gpu_fps(width: int, height: int) -> float:
    print(f"[bench] GPU live path {width}x{height} ...")
    res = subprocess.run(
        [str(ENGINE_EXE), "--shader", "disk", "--hud", "--width", str(width),
         "--height", str(height), "--frames", "500", "--no-validation"],
        capture_output=True, text=True, cwd=ROOT)
    fps = [float(m) for m in re.findall(r"\[app\] ([0-9.]+) fps", res.stdout)]
    if not fps:
        print(f"  WARNING: no fps samples ({res.stdout!r})")
        return 0.0
    # Skip the warm-up sample, average the rest.
    steady = fps[1:] if len(fps) > 1 else fps
    avg = sum(steady) / len(steady)
    print(f"  samples {fps} -> {avg:.1f} fps")
    return round(avg, 1)


def main() -> None:
    for exe in (BENCH_EXE, ENGINE_EXE):
        if not exe.exists():
            sys.exit(f"missing {exe} — build it first")
    bench: dict[str, float] = {}
    bench.update(run_cpu_bench())
    # Keys must stay letters-only: make_numbers.py turns them into LaTeX macro
    # names, and digits are not legal in LaTeX command names (hd = 1280x720,
    # fullhd = 1920x1080).
    bench["gpu_fps_hd"] = run_gpu_fps(1280, 720)
    bench["gpu_fps_fullhd"] = run_gpu_fps(1920, 1080)
    # HD live path: rays per second = pixels * fps (one geodesic per pixel),
    # in millions so the LaTeX macro reads naturally.
    if bench["gpu_fps_hd"]:
        bench["gpu_megarays_per_s_hd"] = round(
            1280 * 720 * bench["gpu_fps_hd"] / 1e6, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bench, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
