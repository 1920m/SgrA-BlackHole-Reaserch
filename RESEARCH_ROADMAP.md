# Research Roadmap — Sgr A* Scientific Simulator

> Companion to the README. This document describes the *scientific* questions the simulator is
> designed to answer, the observational targets each module must reproduce, and the specific
> measurements that will appear in the research paper. Read alongside the README.

---

## Module 1 — Null geodesic integrator validation

**Question:** Does the integrator correctly reproduce the known analytic properties of the Kerr spacetime?

**Method:** Carter-separated equations, RK45 with adaptive step. Compare to closed-form expressions.

**Targets:**
- r_ISCO(a*=0) = 6M; r_ISCO(a*=0.94) ≈ 1.84M — tabulate vs BPT72 Table III
- r_ph(prograde, a*=0.94) ≈ 1.08M; retrograde ≈ 3.91M — asymmetry must be present
- E, L_z, Q conserved to 1e-6 over ≥1,000 integration steps
- Shadow boundary (critical curve) at each inclination matches Bardeen (1973) analytic result

**Paper figure:** Shadow size vs a* for 0 ≤ a* ≤ 0.998 at i=30°. Compare to EHT 2022 Fig. 3.

---

## Module 2 — Kerr black hole shadow and photon ring

**Question:** What does Sgr A* look like at the EHT resolution, and can we reproduce the EHT 2022 measurements?

**Method:** Backward ray tracing in Kerr spacetime, Novikov-Thorne disk, observer at 8,277 pc.

**Targets:**
- Shadow diameter: reproduce 48.7 ± 7 μas (EHT 2022 Papers I–VIII)
- Photon ring diameter: 51.8 ± 2.3 μas
- Shadow is non-circular: axis ratio D_x/D_y differs from 1 at a*=0.94, i=30°
- n=1 sub-ring: brighter by factor ~e^(2π/√3) ≈ 33 relative to n=0 per unit solid angle (Johnson+2020)
- n=2 sub-ring: width ~e^(2π/√3) narrower than n=1

**Paper figures:**
1. Full Sgr A* render at EHT parameters with shadow diameter annotated
2. Sub-ring isolation: n=0, n=1, n=2 in separate panels
3. Shadow diameter vs inclination at fixed a*=0.94 (spin–inclination degeneracy)

---

## Module 3 — Disk brightness asymmetry

**Question:** Can we reproduce the relativistic Doppler beaming asymmetry seen in the accretion disk?

**Method:** g⁴ = (ν_obs/ν_emit)⁴ beaming, Novikov-Thorne F(r), Ω(r) from Kerr equatorial orbit.

**Targets:**
- Approaching side of disk (left for standard inclination) is brighter by factor g⁴ ≈ 2–4 at inner disk
- Brightness asymmetry ratio must match Luminet (1979) Fig. 3 to within 10%
- Disk inner edge at r_ISCO; confirm this moves inward with increasing a*

**Paper figure:** Disk brightness profile along the major axis; label g⁴ factor at r_ISCO.

---

## Module 4 — S2 orbit fitting

**Question:** Can we recover the published Sgr A* mass and distance from the Gillessen 2017 S2 dataset?

**Method:** Full GR timelike geodesic integration. Ceres nonlinear least squares minimising χ² over 145 astrometric + 44 radial velocity measurements.

**Free parameters:** M, d, a*, i, Ω, ω, t_peri (and optionally ε_q in no-hair mode).

**Targets (from Gillessen 2017 and GRAVITY 2022):**
```
M       = 4.297 ± 0.012 × 10⁶ M☉    (recover within 2σ)
d       = 8,277 ± 9 pc                (recover within 2σ)
Period  = 16.0518 ± 0.0022 yr         (check residuals)
e       = 0.8839 ± 0.0019              (check residuals)
χ²/dof  < 2.0
```

**Critical checks:**
- Include Rømer delay (line-of-sight travel time correction)
- Include transverse Doppler + gravitational redshift in v_r model
- Post-Newtonian orbit integration is NOT sufficient near pericenter — use full geodesic

**Paper table:** Fit parameters and 1σ uncertainties vs published values. Report χ²/dof, reduced χ², and residual plots (RA residual, Dec residual, v_r residual vs epoch).

---

## Module 5 — Gravitational redshift reproduction

**Question:** Can we reproduce the GRAVITY 2018 detection of gravitational redshift in the S2 Brγ line?

**Method:** At each point in the S2 orbit, compute:
```
z_total = z_Doppler (first-order) + z_transverse (second-order) + z_grav (gravitational)
λ_obs = λ_Brγ (1 + z_total)
```

At pericenter the gravitational + transverse contribution is ~200 km/s equivalent — significantly above measurement uncertainty (~10 km/s in GRAVITY data).

**Targets:**
- Reproduce the GRAVITY 2018 Fig. 1: z_grav + z_transverse vs epoch, 2017–2018 period
- The "GR excess" above the Keplerian prediction must peak at pericenter passage (May 2018)
- Excess velocity ≈ +200 km/s at pericenter compared to pure Newtonian orbit

**Paper figure:** Radial velocity curve for S2 2017–2019 with Newtonian, GR, and observed curves overlaid.

---

## Module 6 — GR orbital precession

**Question:** Can we reproduce the GRAVITY 2020 detection of Schwarzschild precession in S2?

**Method:** Integrate multiple S2 orbits (≥3). Track argument of pericenter ω; fit linear drift dω/dt.

**Targets:**
- GR predicts ω precesses by ~12.1 arcmin per orbit prograde
- GRAVITY 2020 measured 12' ± 3' (1σ)
- Our simulation must reproduce this to within the measurement uncertainty

**Note on frame-drag (Lense-Thirring):** The Lense-Thirring precession from Kerr spin at S2's orbit radius (~1,400 R_S) is ~0.2 arcmin/orbit — below current detection threshold but above zero. Report it with a note that future monitoring will detect it. This is a research highlight.

**Paper figure:** Argument of pericenter vs orbital cycle number; slope compared to GRAVITY 2020.

---

## Module 7 — No-hair theorem test

**Question:** What constraint does the S2 orbit place on the no-hair parameter ε_q?

**Method:** Re-run orbit fitter with Hartle-Thorne metric (free ε_q). Report marginalised posterior on ε_q. Forecast improvement with 10 more S2 orbits (extrapolate σ_ε_q ∝ 1/√N).

**Physical interpretation:** If ε_q ≠ 0, Sgr A* has a quadrupole moment inconsistent with Kerr. Current S2 data provides a weak constraint (large CI); ngEHT photon ring measurements will dramatically tighten it.

**Targets:**
- ε_q consistent with 0 (i.e., 0 ∈ 95% CI) — this is the expected result
- CI width quantified: report |ε_q|_95% < X (expect X ~ 10–100 with current S2 data)
- Compare forecasted constraint from photon ring shape vs orbit fitting

**Paper section:** "No-Hair Constraints from S2 Astrometry — Current Bounds and Forecasts"

---

## Module 8 — S62 (contested — handle carefully)

**Question:** If S62 is real, what additional constraint does it provide on M, d, and ε_q?

**Method:** If including S62, use only published astrometric measurements from Peißker et al. (2020). Run orbit fitter. Compare resulting M, d to S2-only fit.

**Mandatory labelling:** Every result involving S62 must carry the disclaimer:
> *S62 identification is contested. The orbit reported by Peißker et al. (2020) may be a misidentification of S29 (see Ali et al. 2023 for the counter-claim). These results are presented as illustrative only and should not be cited without independent verification.*

**Paper structure:** S62 results go in an appendix, not the main text, explicitly labelled "Contested."

---

## Module 9 — Spin–inclination degeneracy sweep

**Question:** For what combinations of a* and inclination does our simulation predict a shadow consistent with EHT 2022 constraints?

**Method:** 2D grid sweep over a* ∈ [0, 0.998] × i ∈ [0°, 90°] at 20×20 resolution. For each point, render and measure shadow diameter and axis ratio.

**Targets:**
- Reproduce EHT 2022 Fig. 7 (spin–inclination degeneracy contours)
- Identify the degeneracy locus where shadow size = 48.7 μas
- Annotate with independent spin constraints from accretion models (a*≈0.94)

**Paper figure:** 2D colour map of shadow diameter in μas as function of (a*, i). Overplot EHT 1σ, 2σ contours.

---

## Module 10 — Performance characterisation

**Question:** What is the render time per frame and how does it scale with resolution?

**Method:** Google Benchmark (CPU geodesic throughput) + Tracy + Nsight (GPU pipeline).

**Targets:**
- Report rays/second for CPU reference tracer (expected ~10k–100k rays/s)
- Report rays/second for GPU tracer (expected ~10M–1B rays/s depending on hardware)
- Speedup factor CPU→GPU documented and explained in paper

**Paper table:** Benchmark results. GPU hardware and driver version reported.

---

## Research paper outline

The paper produced in Phase 12 covers all ten modules. A suggested title:

> **"A physically rigorous real-time simulation of Sagittarius A*: Kerr geodesics, S-star orbit fitting, and no-hair constraints from a GPU-accelerated scientific instrument"**

Or shorter:
> **"Sgr A* from First Principles: Kerr Geodesics, Photon Ring Substructure, and No-Hair Tests via S2 Orbit Fitting"**

The paper demonstrates three things:
1. The simulation correctly reproduces all published measurements of Sgr A*.
2. It independently recovers M and d from the Gillessen 2017 dataset via full GR orbit fitting.
3. It places a quantified (even if weak) bound on the no-hair parameter ε_q, consistent with Kerr.

These three together constitute a genuine scientific contribution, not just a visualisation exercise.

---

## Priority order if time is limited

If development time is constrained, build in this priority order:

1. **Must have (paper is incomplete without these):**
   - Gate 0: physics validation
   - Gate 2: Kerr CPU renderer with lopsided shadow
   - Gate 4: S2 orbit fitter (this is the centrepiece science result)
   - Gravitational redshift reproduction (Module 5)

2. **High value (significantly strengthen the paper):**
   - Gate 3: Disk beaming asymmetry (Module 3)
   - GR precession reproduction (Module 6)
   - Gate 7+8: GPU renderer (Modules 2, 3 in real-time)

3. **Optional / appendix material:**
   - No-hair ε_q constraint (Module 7) — include if time allows
   - S62 contested orbit (Module 8) — appendix only
   - Sub-ring scope (Module 2b) — beautiful but not required for paper argument

4. **Presentation (do last):**
   - Cinematic menu (Phase 9) — after all science is validated
   - Visual presets — purely aesthetic
