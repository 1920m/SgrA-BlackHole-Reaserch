# SGR A* Scientific Simulator — Startup & User Guide

A ground-up, physically rigorous simulation of **Sagittarius A\***, the
supermassive black hole at the centre of the Milky Way. It integrates exact
Kerr geodesics (no post-Newtonian shortcuts, no faked visuals), reproduces
the published GRAVITY and Event Horizon Telescope observations, fits the
real orbit of the star S2, and tests the no-hair theorem with a
free-quadrupole metric. Everything you see rendered and every number in the
research paper is regenerated from this code.

**Headline results** (all reproducible with one command, see §6):

| Result | This code | Published |
|---|---|---|
| Black-hole mass from S2 | 4.270 ± 0.195 ×10⁶ M☉ | 4.297 ± 0.012 (GRAVITY 2022) |
| Distance to Sgr A* | 8272 ± 182 pc | 8277 ± 9 pc (GRAVITY 2022) |
| Photon-ring diameter | 49.68 μas | 51.8 ± 2.3 μas (EHT 2022) |
| S2 redshift excess at pericentre | 205 km/s @ 2018.379 | ≈200 km/s (GRAVITY 2018) |
| Schwarzschild precession | 11.98 arcmin/orbit | confirmed (GRAVITY 2020) |
| No-hair quadrupole deviation | ε_q = 0 ± 1.33×10⁵ (95%) | — (consistent with Kerr) |

The full write-up is the research paper: `paper/main.pdf`
(build it with `make paper`, see §6).

---

## 1. Requirements

- **Windows 10/11**, a Vulkan 1.2-capable GPU (developed on a GeForce GTX 1050)
- **Visual Studio 2022 Build Tools** (MSVC x64) — the wrapper
  `scripts\vcenv.cmd` sets the environment up (its "vswhere.exe is not
  recognized" warning is harmless)
- **vcpkg** at `C:\vcpkg` (manifest mode; dependencies install automatically
  on first configure)
- **CMake 3.25+ and Ninja** (Ninja is picked up from vcpkg's downloads by
  `vcenv.cmd`)
- **Python 3.11** with `matplotlib` and `numpy` — figures and number macros
- **TinyTeX** at `%APPDATA%\TinyTeX` (pdflatex + bibtex + `mnras` class) —
  only needed for the paper
- **GNU make** (installed via `winget install ezwinports.make`) — only
  needed for the paper/reproduce targets

## 2. Building

All commands from the repository root. Three CMake presets matter:

```bat
:: physics core + validation tools (fast)
scripts\vcenv.cmd cmake --preset core-rel
scripts\vcenv.cmd cmake --build --preset core-rel

:: + science modules (Ceres orbit fitter) — needed for the S2 fits
scripts\vcenv.cmd cmake --preset sci-rel
scripts\vcenv.cmd cmake --build --preset sci-rel

:: + the Vulkan engine (the interactive app)
scripts\vcenv.cmd cmake --preset engine-rel
scripts\vcenv.cmd cmake --build --preset engine-rel
```

The first configure of each preset lets vcpkg build the dependency tree —
that can take a long while once; afterwards builds are seconds.

Run the physics test suite (all gates):

```bat
scripts\vcenv.cmd ctest --preset sci-rel
```

## 3. Starting the app

**Easiest: double-click `LAUNCH.bat` in the repository root.** It sets the
Vulkan layer path, builds the engine automatically if it has never been
built, and starts the app. Arguments pass through, e.g.
`LAUNCH.bat --validation` or `LAUNCH.bat --preset 3`.

Manually:

```bat
build\engine-rel\engine\sgr_engine.exe
```

With no arguments the app **boots to the cinematic main menu** — an
Interstellar-style backdrop (a 4K frame of our own Kerr disk, baked by the
CPU reference tracer) with an engraved "SGR A*" title and a square-bullet
menu. Navigate with **↑/↓ + Enter** or the mouse.

To enable the Vulkan validation layers (debugging):

```bat
set VK_LAYER_PATH=C:\vcpkg\installed\x64-windows\bin
build\engine-rel\engine\sgr_engine.exe --validation
```

### Menu items (the modes)

| Item | What it does |
|---|---|
| **ENTER SIMULATION** | Live GPU Kerr renderer with the current preset. A controls window exposes spin a\*, inclination, camera radius, field of view, disk outer radius, exposure and disk temperature — all live. |
| **SCIENCE MODE** | Live renderer + the Science window: the fitted S2 orbit plotted over the real Gillessen+2017 astrometry, with the fit summary (M, R₀, χ²/dof). |
| **NO-HAIR TEST** | Panel with the Hartle–Thorne quadrupole slider ε_q: shows the predicted extra S2 pericentre precession live, next to the actual 95% confidence interval fitted from the S2 data (Phase 5). |
| **PHOTON RING SCOPE** | Panel with the measured photon sub-ring flux ratios (n=0/1/2), plus a button that reframes the live camera (r=60M, narrow fov) so the n=1 ring is visible hugging the shadow edge. |
| **SETTINGS** | Opens the same controls window as in simulation — changes apply when you enter the simulation. |
| **GALLERY** | Thumbnails of the milestone renders (Schwarzschild/Kerr gate images, disk render, sub-rings, the four presets). |
| **ABOUT** | Sgr A* fact panel (mass, distance, spin, shadow size, S2 period with sources) beside the rendered black hole. |
| **EXIT** | Quit. |

### In-simulation keys

- **1–5** — switch visual preset (see below)
- **Esc** — back to the menu (or quit, if the app was started without the menu)

### Visual presets (physics never changes — only palette/geometry)

| Key | Preset | Look |
|---|---|---|
| 1 | PHYSICAL | True blackbody colours from the Planck spectrum (default) |
| 2 | FIRESTORM | Deep crimson → orange → yellow-white, a\*=0.94 |
| 3 | GARGANTUA | Ice blue → pale gold, a\*=0.60 — the Interstellar look |
| 4 | ARCTIC | Pale cyan → electric white, a\*=0.97, 12000 K disk |
| 5 | NEBULA | Violet → magenta → rose-gold, a\*=0.35, wide disk |

### Useful command-line flags

```
--menu                boot to menu (default when run with no arguments)
--shader disk --hud   skip the menu, go straight to the live renderer
--science             open the Science window at startup
--preset N            start with preset N (0–4)
--panel NAME          open a menu panel at boot (about|nohair|scope|gallery|settings)
--astar X --incl DEG  override physics parameters
--width W --height H  window size          --scale S   render-resolution knob
--frames N            exit after N frames  --screenshot out.png
--validation          Vulkan validation layers ON (exit code = error count)
```

## 4. Command-line science tools

| Tool (in `build\sci-rel\...`) | Purpose |
|---|---|
| `validation\render_reference.exe schwarzschild\|kerr94 out.png [size]` | CPU reference render + shadow metrology JSON |
| `science\subring_tool.exe <outdir> [a*] [incl] [size]` | Disk render, Doppler asymmetry, photon sub-ring isolation |
| `science\orbit_fitter\fit_s2_tool.exe data\gillessen_2017 <outdir> kerr\|nohair` | The S2 orbit fit (Kerr or free-quadrupole) |
| `validation\sweep_tool.exe calibrate\|sweep <outdir>` | Sgr A* ring-size calibration / 20×20 spin–inclination sweep |
| `validation\render_hero.exe out.png [w h ssaa ...]` | Bake the 4K menu hero plate with the CPU tracer |

## 5. Repository map

```
physics/     Kerr metric, Carter geodesics (Mino-time 2nd order), disk model,
             CPU ray tracer + the Catch2 test suite (the ground truth)
science/     Ceres S2 orbit fitter, spectral/redshift model, sub-ring extractor
engine/      Vulkan 1.2 app: compute shaders, presets, ImGui HUD, menu
validation/  reference renders, CPU-vs-GPU comparison, calibration artefacts
data/        Gillessen et al. (2017) S2 astrometry + radial velocities (VizieR)
paper/       the research paper (MNRAS style) + auto-generated figures/numbers
scripts/     build wrapper, figure/number generators, benchmarks, LUT baker
writeup/     per-phase evidence screenshots and the build plan
```

## 6. The paper and full reproduction

```bash
# from Git Bash, with make and TinyTeX on PATH:
make figures     # regenerate all 10 paper figures from simulation output
make paper       # figures + compile paper/main.pdf (pdflatex + bibtex)
make reproduce   # re-run the ENTIRE science chain: tests → renders →
                 # sub-rings → S2 fits → calibration + sweep → figures
```

Every number printed in `paper/main.pdf` is a LaTeX macro generated by
`scripts/make_numbers.py` from the JSON artefacts `make reproduce` writes —
nothing is typed by hand, so the paper cannot silently drift from the code.

## 7. Development history (the gates)

The project was built in 12 gated phases; each gate is a git tag
(`git tag` lists them): physics foundation → Schwarzschild tracer → Kerr
tracer → disk + sub-rings → S2 fitter → no-hair probe → Vulkan shell →
GPU geodesics → full renderer → cinematic menu → Sgr A* calibration →
research paper (`v1.0-paper`). The gate criteria and their measured
results are in Appendix A of the paper and `writeup/screenshots/`.
