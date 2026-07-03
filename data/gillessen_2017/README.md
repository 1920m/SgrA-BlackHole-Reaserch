# Gillessen et al. (2017) S2 orbital data

**Source:** Gillessen, S., et al. 2017, *An Update on Monitoring Stellar Orbits
in the Galactic Center*, ApJ 837, 30 — machine-readable Table 5, obtained from
the VizieR catalogue **J/ApJ/837/30** (CDS), DOI: 10.26093/cds/vizier.18370030.

Retrieved with `scripts/fetch_gillessen_data.py` (re-run it to regenerate).

| File | Contents | Points |
|---|---|---|
| `s2_astrometry.csv` | epoch [Julian yr], RA/Dec offset of S2 relative to Sgr A* [mas], 1σ uncertainties [mas] | 145 |
| `s2_radial_velocity.csv` | epoch [Julian yr], heliocentric radial velocity [km/s] (LSR-corrected by the authors), 1σ uncertainty | 44 |

Conventions (paper §2): positions are offsets from the radio position of
Sgr A*; the astrometric reference frame is tied to Sgr A* with priors
(α, δ) = (0,0) ± 0.2 mas at epoch 2009.0 and (v_α, v_δ) = (0,0) ± 0.1 mas/yr;
radial velocities are relative to the LSR with v_z(Sgr A*) = 0 ± 5 km/s.
Positive RV = recession. These priors enter the orbit fit as additional χ²
terms (see `science/orbit_fitter/`).
