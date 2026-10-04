"""Gate 18 movie: one orbit of the GRAVITY 2018 hot spot, rendered by the live
engine (slow light, Doppler beaming, lensed echo) and assembled into a GIF.

The spot follows the best-fitting GRAVITY 2018 orbit (R = 7 M; i = 160 deg,
i.e. seen from below the orbital plane, so the motion is clockwise on the sky
as observed) around our Sgr A* (a* = 0.94). The spot colour is FALSE colour:
2.2 um synchrotron light has no visible hue. The background is the lensed
Milky Way panorama seen from Earth (labelled caveat, Gate 15).

Run:  py -3.11 scripts/make_flare_movie.py [frames] [size]
Output: validation/extension/g18_flare_movie.gif (frames in a temp dir)
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE = ROOT / "build" / "engine-rel" / "engine" / "sgr_engine.exe"
OUT = ROOT / "validation" / "extension" / "g18_flare_movie.gif"


def main() -> int:
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 480
    J = json.loads((ROOT / "validation" / "extension" / "flares.json").read_text())
    P = next(c for c in J["cases"] if c["name"] == "g18_R7_i160")["period_M"]
    env = dict(os.environ)
    env.setdefault("VK_LAYER_PATH", r"C:\vcpkg\installed\x64-windows\bin")
    with tempfile.TemporaryDirectory() as tmp:
        shot = pathlib.Path(tmp) / "frame.png"
        cmd = [str(ENGINE), "--shader", "disk", "--offscreen", "--validation",
               "--width", str(size), "--height", str(size), "--preset", "0",
               "--rcam", "45", "--incl", "160", "--fov", "0.55", "--no-disk",
               "--flare", "--flare-amp", "6", "--flare-t", "0",
               "--movie", str(frames), "--movie-dt", f"{P / frames:.9f}",
               "--spp", "8", "--screenshot", str(shot)]
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if res.returncode != 0:
            sys.stderr.write(res.stdout[-2000:] + res.stderr[-2000:])
            return 1
        imgs = [Image.open(pathlib.Path(tmp) / f"frame_{k:04d}.png").convert("RGB")
                for k in range(frames)]
        pal = [im.quantize(colors=255, method=Image.Quantize.MEDIANCUT) for im in imgs]
        # 64 frames over one 43-minute orbit: ~40 s of Sgr A* time per frame
        pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=70, loop=0,
                    optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {frames} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
