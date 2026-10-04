"""Gate 13 — live GPU disk colour vs the CPU reference shading.

Renders the PHYSICAL preset twice at identical camera/disk parameters:
  * GPU: engine/shaders/disk.comp.glsl (fp32 live path) via sgr_engine --offscreen
  * CPU: validation/render_disk_cpu (fp64 Carter tracer + NovikovThorneDisk::shade)
and compares the disk pixels. Both write clamp(c·2^EV)^(1/2.2) to 8 bit, so
after undoing the gamma the two images are directly comparable radiances.

Checks (all on disk pixels, CPU hit mask eroded to avoid fp32/fp64 edge
differences, clipped and near-black pixels excluded):
  1. median |ln(Y_gpu / Y_cpu)|              < 0.05   (luminance)
  2. median |chromaticity difference|        < 0.01   (colour)
  3. approaching/receding luminance ratio agrees to < 5 %  (Doppler beaming)
  4. GPU pure black where the CPU disk is lit: < 0.1 % of lit pixels
     (the old 8-bit LUT floor blacked out whole regions; isolated single
     pixels are fp32 live-integrator crossing misses — G16's Kerr-Schild
     kernel should bring the count to zero)

Run:  py -3.11 validation/phase14/compare_live_colour.py
Exit code 0 = pass. Results: validation/phase14/g13_colour.json
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "validation" / "phase14"
ENGINE = ROOT / "build" / "engine-rel" / "engine" / "sgr_engine.exe"
CPU = ROOT / "build" / "engine-rel" / "validation" / "render_disk_cpu.exe"

# PHYSICAL preset (engine/src/presets/DiskPreset.hpp, index 0) + live fov.
A_STAR, INCL, R_CAM, FOV, R_OUT, T_PEAK = 0.94, 80.0, 60.0, 0.70, 18.0, 6300.0
EV = -1.5          # keeps the beamed inner disk below display clipping
W, H = 480, 270


def run(cmd: list[str]) -> None:
    env = dict(os.environ)
    env.setdefault("VK_LAYER_PATH", r"C:\vcpkg\installed\x64-windows\bin")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        sys.stderr.write(res.stdout[-2000:] + res.stderr[-2000:])
        raise SystemExit(f"command failed ({res.returncode}): {cmd[0]}")


def linear(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return (b / 255.0) ** 2.2, b


def erode(m: np.ndarray, k: int) -> np.ndarray:
    out = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            out &= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gpu_png, cpu_png = OUT / "g13_gpu.png", OUT / "g13_cpu.png"
    # --display linear: the science transform clamp(c·2^EV)^(1/2.2), the same
    # as the CPU tool (no bloom/tone curve). --tpeak pins the temperature the
    # CPU side uses (presets otherwise derive it from the accretion rate).
    run([str(ENGINE), "--shader", "disk", "--offscreen", "--validation",
         "--width", str(W), "--height", str(H), "--preset", "0",
         "--display", "linear", "--tpeak", str(T_PEAK),
         "--exposure", str(EV), "--blacksky", "--screenshot", str(gpu_png)])
    run([str(CPU), str(cpu_png), str(W), str(H), str(A_STAR), str(INCL),
         str(R_CAM), str(FOV), str(R_OUT), str(T_PEAK), str(EV)])

    g, g8 = linear(gpu_png)
    c, c8 = linear(cpu_png)
    hit = np.asarray(Image.open(OUT / "g13_cpu_mask.png")) > 127
    # Both paths trace the RECEIVED photon (live Kerr-Schild kernel, Gate 16;
    # CPU Carter tracer since the handedness correction), so the images are
    # compared directly — including which side is which.
    core = erode(hit, 2)
    lum_w = np.array([0.2126, 0.7152, 0.0722])
    yg, yc = g @ lum_w, c @ lum_w
    usable = core & (g8.max(-1) < 250) & (c8.max(-1) < 250) & \
        (g8.max(-1) > 12) & (c8.max(-1) > 12)
    n = int(usable.sum())
    if n < 500:
        print(f"too few comparable disk pixels ({n})")
        return 1

    dlny = np.abs(np.log(yg[usable] / yc[usable]))
    chroma = lambda im: im / np.maximum(im.sum(-1, keepdims=True), 1e-12)
    dchroma = np.abs(chroma(g)[usable] - chroma(c)[usable]).max(-1)

    # Approaching side: the disk rotates prograde (+φ), which with this
    # camera tetrad (x = +φ̂) moves toward the observer on the image's left
    # or right depending on orientation — take the brighter half on the CPU
    # image as "approaching" and use the same split for the GPU.
    xs = np.arange(W)[None, :].repeat(H, 0)
    left, right = usable & (xs < W // 2), usable & (xs >= W // 2)
    ratio = lambda y: y[right].mean() / y[left].mean()
    r_cpu, r_gpu = ratio(yc), ratio(yg)
    beam_err = abs(r_gpu / r_cpu - 1.0)

    lit_cpu = core & (c8.max(-1) > 12)
    black_gpu = lit_cpu & (g8.max(-1) == 0)
    # Geometry: GPU disk-hit footprint (anything lit) vs the CPU hit mask —
    # disagreement confined to the 1-px fp32/fp64 boundary.
    gpu_hit = g8.max(-1) > 0
    mask_mismatch = float(np.mean(gpu_hit != hit))

    res = {
        "params": dict(a_star=A_STAR, incl_deg=INCL, r_cam=R_CAM, fov=FOV,
                       r_out=R_OUT, t_peak_k=T_PEAK, exposure_ev=EV,
                       width=W, height=H),
        "n_pixels": n,
        "median_abs_dlnY": float(np.median(dlny)),
        "p95_abs_dlnY": float(np.percentile(dlny, 95)),
        "median_abs_dchroma": float(np.median(dchroma)),
        "beaming_ratio_cpu": float(r_cpu),
        "beaming_ratio_gpu": float(r_gpu),
        "beaming_ratio_rel_err": float(beam_err),
        "gpu_black_where_cpu_lit": int(black_gpu.sum()),
        "cpu_lit_pixels": int(lit_cpu.sum()),
        "disk_mask_mismatch_frac": mask_mismatch,
    }
    checks = {
        "luminance": bool(res["median_abs_dlnY"] < 0.05),
        "colour": bool(res["median_abs_dchroma"] < 0.01),
        "beaming": bool(beam_err < 0.05),
        "no_black_floor": bool(black_gpu.sum() < 1e-3 * lit_cpu.sum()),
        "geometry": bool(mask_mismatch < 5e-3),
    }
    res["checks"] = checks
    res["pass"] = all(checks.values())
    (OUT / "g13_colour.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
