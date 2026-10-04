"""Engine renders used as figures in the extension paper (display images —
the quantitative content of every figure is validated elsewhere: Gate 13
colour, Gate 17 observers, Gate 18 flares). Each render is one documented
sgr_engine command, so every image is reproducible.

Run:  py -3.11 scripts/render_ext_gallery.py [--quick]
Output: validation/extension/renders/*.png
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE = ROOT / "build" / "engine-rel" / "engine" / "sgr_engine.exe"
OUT = ROOT / "validation" / "extension" / "renders"

QUICK = "--quick" in sys.argv
SPP = "4" if QUICK else "32"
W, H = ("640", "360") if QUICK else ("1280", "720")

# name -> extra arguments (all: --shader disk --offscreen --validation)
RENDERS: dict[str, list[str]] = {
    # Gate 14/15: the five Sgr A* presets (same spacetime, a* = 0.94)
    "preset0_physical": ["--preset", "0"],
    "preset1_earth": ["--preset", "1"],
    "preset2_close": ["--preset", "2"],
    "preset3_lowmdot": ["--preset", "3"],
    "preset4_highmdot": ["--preset", "4"],
    # Gate 13: the bolometric teaching view of the PHYSICAL preset
    "preset0_bolometric": ["--preset", "0", "--bolometric"],
    # Gate 17: whose eyes — the same place, four observers
    "obs_zamo": ["--preset", "2", "--observer", "zamo"],
    "obs_infall": ["--preset", "2", "--observer", "infall"],
    "obs_orbit": ["--preset", "2", "--observer", "orbit"],
    # Gate 17: light that visited the ergoregion (no disk, edge-on)
    "ergo_crescent": ["--preset", "0", "--rcam", "30", "--incl", "88",
                      "--fov", "0.42", "--no-disk", "--ergo"],
    # Gate 19: one camera, three bands (EHT viewing geometry)
    "band_visible": ["--preset", "1"],
    "band_230": ["--preset", "1", "--band", "230"],
    "band_kband": ["--preset", "1", "--band", "kband", "--flare", "--flare-amp", "6",
                   "--flare-t", "40"],
}


def run(name: str, args: list[str], size: tuple[str, str] = (W, H)) -> None:
    env = dict(os.environ)
    env.setdefault("VK_LAYER_PATH", r"C:\vcpkg\installed\x64-windows\bin")
    cmd = [str(ENGINE), "--shader", "disk", "--offscreen", "--validation",
           "--width", size[0], "--height", size[1], "--spp", SPP, *args,
           "--screenshot", str(OUT / f"{name}.png")]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    ok = res.returncode == 0
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {' '.join(args)}")
    if not ok:
        sys.stderr.write(res.stdout[-1500:] + res.stderr[-1500:])
        raise SystemExit(1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, args in RENDERS.items():
        run(name, args)
    # Gate 19: the 230 GHz view from far away, framed like an EHT image
    run("band_230_eht", ["--preset", "0", "--rcam", "1000", "--incl", "30",
                         "--fov", "0.024", "--band", "230"], size=("640", "640"))
    # Gate 18: four frames of the GRAVITY 2018 hot spot (quarter orbits)
    fl = json.loads((ROOT / "validation" / "extension" / "flares.json").read_text())
    P = next(c for c in fl["cases"] if c["name"] == "g18_R7_i160")["period_M"]
    run("flare", ["--preset", "0", "--rcam", "45", "--incl", "160", "--fov", "0.55",
                  "--no-disk", "--flare", "--flare-amp", "6", "--flare-t", "0",
                  "--movie", "4", "--movie-dt", f"{P / 4:.9f}"], size=("480", "480"))
    # Gate 21 — EXOTIC SPACETIME, NOT Sgr A*: the Dneg wormhole approached
    # along l (camera at incl 30 deg, cam_phi 0: the Galactic Centre behind
    # the throat). The framing is recorded for make_figures_ext.py, which
    # overlays the CPU twin's edge and Einstein ring on these images.
    wh = {"fov": 1.2, "size": 720, "l_c": [15, 6, 2]}
    for lc in wh["l_c"]:
        run(f"wormhole_l{lc}", ["--wormhole", "--wh-l", str(lc), "--incl", "30",
                                "--fov", str(wh["fov"])],
            size=(str(wh["size"]), str(wh["size"])))
    (OUT / "wormhole_renders.json").write_text(json.dumps(wh, indent=2) + "\n")
    # Gate 24: the fall into Sgr A* (rain from 30 M at 60°, a* = 0.94),
    # looking ahead, Milky Way only (the thin disk is hypothetical)
    fall = {"r": [20, 6, 3, 1.5, 1.1, 0.8], "fov": 1.6, "size": [640, 360]}
    for r in fall["r"]:
        run(f"fall_r{r:g}", ["--flight-r", str(r), "--fov", str(fall["fov"]), "--no-disk"],
            size=(str(fall["size"][0]), str(fall["size"][1])))
    (OUT / "fall_renders.json").write_text(json.dumps(fall, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
