# Habibi et al. (2017) — S-star spectroscopic parameters

`s_star_parameters.csv` holds the measured effective temperature, surface
gravity and mass of the S-stars, used by the interactive instrument to
compute tidal-disruption ("Roche") radii.

**Source.** M. Habibi, S. Gillessen, F. Martins, F. Eisenhauer, P. M. Plewa,
O. Pfuhl, E. George, J. Dexter, I. Waisberg, T. Ott, S. von Fellenberg,
M. Bauböck, A. Jimenez-Rosales and R. Genzel, *"Twelve years of spectroscopic
monitoring in the Galactic Center: the closest look at S-stars near the black
hole"*, The Astrophysical Journal (2017),
DOI [10.3847/1538-4357/aa876f](https://doi.org/10.3847/1538-4357/aa876f),
Table 2. Preprint: [arXiv:1708.06353](https://arxiv.org/abs/1708.06353).

The programme obtained H- and K-band spectra of eight stars in the central
arcsecond over twelve years, deriving `Teff = 21 000–28 500 K`,
`log g = 4.1–4.2` and masses of 8–14 M☉, implying spectral types B0–B3V.

## What is measured and what is derived

| quantity | status |
|---|---|
| `teff_k`, `logg_cgs`, `mass_msun` | **measured**, transcribed from Table 2 with the paper's 1σ errors |
| stellar radius | **derived** in code as `R = √(GM★/g)` from the two columns above — not tabulated by the paper and not invented here (S2 → 5.4 R☉, consistent with its B0–B3V type) |
| tidal radius `r_t` | **derived** as `η·R★·(M_BH/M★)^⅓` using the black-hole mass from our own cluster fit |

## Coverage

The paper's sample is S1, S2, S4, S6, S7, S8, S9, S12. Four of those (S1, S2,
S8, S12) are among the five stars our cluster fit uses. **S13 is not in the
sample**: it carries the sample median with `assumed=1`, and the instrument
labels it "assumed" wherever its tidal radius is shown. No S13-specific
spectroscopic claim is made anywhere in this project.

## Scope

These values feed the *display-layer* tidal-disruption demonstration only.
No fitted orbit, figure, or number in the paper depends on them.
