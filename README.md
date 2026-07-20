<div align="center">

# 🌌 SGR A* — Scientific Simulator

### *Welcome to the centre of the galaxy.*

**A first-principles, physically rigorous simulation of Sagittarius A\*, the
supermassive black hole at the heart of the Milky Way — from exact Kerr
geodesics to a reproducible research paper.**

![tests](https://img.shields.io/badge/physics_tests-32%2F32_passing-brightgreen)
![vulkan](https://img.shields.io/badge/Vulkan-1.2_·_0_validation_errors-red)
![paper](https://img.shields.io/badge/paper-MNRAS_style_·_11_pages-blue)
![reproducible](https://img.shields.io/badge/every_number-reproducible-purple)

<img src="writeup/screenshots/phase_9/menu_boot.png" width="850" alt="The cinematic main menu — our own Kerr disk, ray-traced by the CPU reference tracer"/>

*The main menu. The backdrop is not artwork — it is 33 million exact Kerr
geodesics, traced by this code.*

<br>

**[▶ Quick start](#-quick-start)** ·
**[🕳 What is this?](#-what-is-this)** ·
**[🎛 The modes](#-the-modes)** ·
**[🔬 The science](#-the-science)** ·
**[📄 The paper](#-the-research-paper)**

</div>

---

## 🕳 What is this?

This project simulates **Sagittarius A\*** from the ground up — no
post-Newtonian shortcuts, no faked visuals. It integrates the exact
Carter-separated geodesic equations in Kerr spacetime, renders the accretion
disk with the Novikov–Thorne model, **fits the real orbit of the star S2**
(145 astrometric + 44 radial-velocity measurements from Gillessen et al.
2017), and probes the **no-hair theorem** with a free-quadrupole
Hartle–Thorne metric. A validated GPU port of the same equations drives a
live, interactive renderer.

Everything it claims, it can prove: every number in the research paper is
regenerated from source by one command.

### Headline results

| Result | This code | Published |
|---|---|---|
| Black-hole mass from S2 | **4.270 ± 0.195 ×10⁶ M☉** | 4.297 ± 0.012 (GRAVITY 2022) |
| Distance to Sgr A* | **8272 ± 182 pc** | 8277 ± 9 pc (GRAVITY 2022) |
| Photon-ring diameter | **49.68 μas** | 51.8 ± 2.3 μas (EHT 2022) |
| S2 redshift excess at pericentre | **205 km/s @ 2018.379** | ≈200 km/s (GRAVITY 2018) |
| Schwarzschild precession | **11.98 arcmin/orbit** | confirmed (GRAVITY 2020) |
| No-hair quadrupole deviation | **ε_q = 0 ± 1.33×10⁵ (95%)** | — consistent with Kerr |

---

## ▶ Quick start

**Double-click `LAUNCH.bat`.** That's it — it prepares the Vulkan layer
path, builds the engine automatically the first time (one-off; vcpkg
compiles the dependencies), and boots into the cinematic menu. Arguments
pass through: `LAUNCH.bat --validation`, `LAUNCH.bat --preset 3`, …

<details>
<summary><b>Manual build & requirements</b></summary>

Requirements: Windows 10/11, a Vulkan 1.2 GPU, VS 2022 Build Tools,
vcpkg at `C:\vcpkg`, CMake 3.25+ (Ninja comes via vcpkg), Python 3.11 +
matplotlib. TinyTeX and GNU make only for the paper.

```bat
:: physics core + validation tools
scripts\vcenv.cmd cmake --preset core-rel
scripts\vcenv.cmd cmake --build --preset core-rel

:: + science modules (Ceres S2 orbit fitter)
scripts\vcenv.cmd cmake --preset sci-rel
scripts\vcenv.cmd cmake --build --preset sci-rel

:: + the Vulkan engine (the interactive app)
scripts\vcenv.cmd cmake --preset engine-rel
scripts\vcenv.cmd cmake --build --preset engine-rel

:: run the physics test suite (all 32 tests)
scripts\vcenv.cmd ctest --preset sci-rel

:: start the app
build\engine-rel\engine\sgr_engine.exe
```

To enable Vulkan validation layers: `set VK_LAYER_PATH=C:\vcpkg\installed\x64-windows\bin`
and add `--validation` (exit code = error count; the gates require zero).

</details>

---

## 🎛 The modes

Navigate the menu with **↑/↓ + Enter** or the mouse. In the simulation:

- **drag** to orbit the black hole in 3D — full 360° azimuth and inclination,
  above *and* below the disk plane
- **scroll** to zoom (camera radius 15–400 M)
- keys **1–5** switch presets, **Esc** returns to the menu
- the **turbulence** slider adds a filament layer sheared by the real
  Keplerian differential rotation — display-only, mean-1 on the
  Page–Thorne flux, so the g⁴ Doppler physics is untouched (set it to 0,
  or use the PHYSICAL preset, for the exact analytic disk)
- the **rotation [× real]** slider animates the filaments at multiples of
  real Sgr A* wall-clock time. At 1.0 (the default) the innermost orbit
  takes ~8.5 minutes — a 4-million-solar-mass black hole genuinely looks
  almost frozen. Push it to a few hundred to watch the differential shear
  wind the filaments into trailing spirals
- **auto resolution** (on by default) adjusts the internal render scale to
  hold ~13 fps at any window size; the **render scale** slider overrides it
  manually (0.25×–2×; above 1× = supersampled anti-aliasing)

| Menu item | What it does |
|---|---|
| **ENTER SIMULATION** | Live GPU Kerr renderer. Sliders for spin a\*, inclination, camera radius, field of view, disk size, exposure, disk temperature — all live. |
| **SCIENCE MODE** | Live renderer + the fitted S2 orbit plotted over the real Gillessen+2017 astrometry, with the fit summary (M, R₀, χ²/dof). |
| **NO-HAIR TEST** | The Hartle–Thorne quadrupole slider ε_q: predicted extra S2 precession live, beside the 95% confidence interval fitted from the data. |
| **PHOTON RING SCOPE** | Measured photon sub-ring flux ratios (n=0/1/2) + a live narrow-field camera that isolates the n=1 ring at the shadow edge. |
| **SETTINGS** | The controls window, from the menu. |
| **GALLERY** | Thumbnails of the milestone renders from every validation gate. |
| **ABOUT** | Sgr A* fact panel — mass, distance, spin, shadow size, S2 period, with sources — beside the rendered black hole. |
| **EXIT** | Quit. |

<div align="center">
<img src="writeup/screenshots/phase_9/menu_panel_nohair.png" width="410" alt="No-hair test panel"/>
<img src="writeup/screenshots/phase_9/menu_panel_gallery.png" width="410" alt="Gallery panel"/>
</div>

### Visual presets — the physics never changes, only palette and framing

| Key | Preset | Look |
|---|---|---|
| 1 | **PHYSICAL** | True blackbody colours from the Planck spectrum (default) |
| 2 | **FIRESTORM** | Deep crimson → orange → yellow-white, a\*=0.94 |
| 3 | **GARGANTUA** | Ice blue → pale gold — the *Interstellar* look, a\*=0.60 |
| 4 | **ARCTIC** | Pale cyan → electric white, a\*=0.97, 12 000 K disk |
| 5 | **NEBULA** | Violet → magenta → rose-gold, a\*=0.35, wide disk |
| 6 | **CINEMATIC SGR A\*** | Volumetric orbiting dust banks over the exact geodesics — warm *Interstellar* white-gold; the `turbulence` slider sets cloud density |

<div align="center">
<img src="writeup/screenshots/phase_8/preset_gargantua.png" width="410" alt="GARGANTUA preset"/>
<img src="writeup/screenshots/phase_8/preset_arctic.png" width="410" alt="ARCTIC preset"/>
</div>

<details>
<summary><b>Command-line flags</b></summary>

```
--menu                boot to menu (default when run with no arguments)
--shader disk --hud   skip the menu, go straight to the live renderer
--science             open the Science window at startup
--preset N            start with preset N (0–5)
--panel NAME          open a menu panel at boot (about|nohair|scope|gallery|settings)
--astar X --incl DEG  override physics parameters
--width W --height H  window size          --scale S   render-resolution knob
--frames N            exit after N frames  --screenshot out.png
--no-mouse            disable orbit/zoom input (repeatable screenshots)
--rot X               disk rotation speed in multiples of real time
--rout R              disk outer radius [M]   --turb X   turbulence/cloud density 0–1
--validation          Vulkan validation layers ON (exit code = error count)
```

</details>

---

## 🔬 The science

The physics core is a CPU reference implementation held to numerical
standards set by analytic theory — conserved quantities to 10⁻⁶, the
Schwarzschild shadow to 5×10⁻¹⁶ of Synge's formula, the Kerr shadow on
Bardeen's critical curve — and the GPU is validated against it pixel by
pixel. The command-line instruments (in `build\sci-rel\...`):

| Tool | Purpose |
|---|---|
| `validation\render_reference.exe` | CPU reference renders + shadow metrology JSON |
| `science\subring_tool.exe` | Disk render, Doppler beaming, photon sub-ring isolation |
| `science\orbit_fitter\fit_s2_tool.exe` | The S2 orbit fit — `kerr` or `nohair` (free ε_q) mode |
| `validation\sweep_tool.exe` | Sgr A* ring calibration / 20×20 spin–inclination sweep |
| `validation\render_hero.exe` | Bake the 4K menu hero plate with the CPU tracer |

---

## 📄 The research paper

**`paper/main.pdf`** — an 11-page, MNRAS-style research paper covering the
full theory, the numerical methods, and all results, with three appendices
(verification gates, reproduction pipeline, number provenance).

```bash
make figures     # regenerate all 10 paper figures from simulation output
make paper       # figures + compile paper/main.pdf (pdflatex + bibtex)
make reproduce   # re-run the ENTIRE science chain: tests → renders →
                 # sub-rings → S2 fits → calibration + sweep → figures
```

Every number in the paper is a LaTeX macro generated from the simulation's
JSON artefacts — nothing is typed by hand, so the paper cannot silently
drift from the code. The final verification run regenerated **all 55
numbers bit-identically**.

---

## 🗺 Repository map

```
physics/     Kerr metric, Carter geodesics (Mino-time 2nd order), disk model,
             CPU ray tracer + the Catch2 test suite (the ground truth)
science/     Ceres S2 orbit fitter, redshift model, photon sub-ring extractor
engine/      Vulkan 1.2 app: compute shaders, presets, ImGui HUD, menu
validation/  reference renders, CPU-vs-GPU comparison, calibration artefacts
data/        Gillessen et al. (2017) S2 astrometry + radial velocities (VizieR)
paper/       the research paper (main.pdf) + auto-generated figures/numbers
scripts/     build wrapper, figure/number generators, benchmarks, LUT baker
writeup/     per-phase evidence screenshots and the build plan
```

## 🏁 Development history

Built in 12 gated phases — no phase started while a previous gate was red,
and each passing gate is a git tag, from `v0.0-physics-foundation` to
`v1.0-paper`. The gate criteria and measured results are in Appendix A of
the paper and under `writeup/screenshots/`.

<div align="center">

*Every rendered image is defensible against published observations.*
*Every number in the paper is reproducible from the code.*

</div>
