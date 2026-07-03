#pragma once
// Physical constants (SI) and Sgr A* calibration targets.
// Geometric-unit convention throughout the physics library: G = c = 1 and
// lengths measured in units of the gravitational radius r_g = GM/c^2 (so M = 1).
// Conversions to SI/astronomical units live here and only here.

#include <string>

namespace sgr::constants {

// ---- SI fundamental constants -------------------------------------------
inline constexpr double c_si        = 299'792'458.0;         // m s^-1 (exact)
inline constexpr double G_si        = 6.674'30e-11;          // m^3 kg^-1 s^-2 (CODATA 2018)
inline constexpr double GM_sun_si   = 1.327'124'400'18e20;   // m^3 s^-2 (IAU 2015 nominal)
inline constexpr double M_sun_kg    = GM_sun_si / G_si;      // ≈ 1.98841e30 kg

// ---- Astronomical unit conversions --------------------------------------
inline constexpr double au_m        = 1.495'978'707e11;      // m (IAU 2012, exact)
inline constexpr double pc_m        = 3.085'677'581'491'3673e16; // m (exact via au/arcsec)
inline constexpr double yr_s        = 3.155'76e7;            // Julian year, s (exact)
inline constexpr double rad_to_uas  = 2.062'648'062'470'9636e11; // microarcsec per radian
inline constexpr double rad_to_arcmin = 3437.746'770'784'939;    // arcminutes per radian

// ---- Sgr A* calibration targets (GRAVITY 2022; EHT 2022) -----------------
inline constexpr double sgrA_mass_msun     = 4.297e6;   // ± 0.012e6 (GRAVITY 2022)
inline constexpr double sgrA_mass_sigma    = 0.012e6;
inline constexpr double sgrA_distance_pc   = 8277.0;    // ± 9 pc (GRAVITY 2022)
inline constexpr double sgrA_distance_sigma= 9.0;
inline constexpr double sgrA_spin_default  = 0.94;      // best-bet a* (EHT + accretion models)
inline constexpr double sgrA_incl_deg      = 30.0;      // observer inclination from spin axis
inline constexpr double sgrA_shadow_uas    = 48.7;      // ± 7 μas (EHT 2022)
inline constexpr double sgrA_ring_uas      = 51.8;      // ± 2.3 μas (EHT 2022)

// ---- S2 published orbital benchmarks (Gillessen et al. 2017) -------------
inline constexpr double s2_period_yr       = 16.0518;   // ± 0.0022 yr
inline constexpr double s2_eccentricity    = 0.8839;    // ± 0.0019
inline constexpr double s2_semimajor_mas   = 125.5;     // ± 0.4 mas (angular semi-major axis)

// ---- Derived geometric scales for Sgr A* ---------------------------------
// Gravitational radius r_g = GM/c^2 in metres.
inline constexpr double sgrA_GM_si   = sgrA_mass_msun * GM_sun_si;            // m^3 s^-2
inline constexpr double sgrA_rg_m    = sgrA_GM_si / (c_si * c_si);            // ≈ 6.345e9 m
inline constexpr double sgrA_rs_m    = 2.0 * sgrA_rg_m;                       // Schwarzschild radius
inline constexpr double sgrA_dist_m  = sgrA_distance_pc * pc_m;               // ≈ 2.554e20 m
// Angular size of one gravitational radius at the Sgr A* distance, in μas.
inline constexpr double sgrA_uas_per_rg = sgrA_rg_m / sgrA_dist_m * rad_to_uas; // ≈ 5.125 μas
// Geometric time unit GM/c^3 in seconds (one "M" of coordinate time).
inline constexpr double sgrA_tg_s    = sgrA_GM_si / (c_si * c_si * c_si);     // ≈ 21.2 s

// r_g expressed in AU (useful for S2 orbit scales: pericenter ≈ 119.5 AU ≈ 2820 r_g).
inline constexpr double sgrA_rg_au   = sgrA_rg_m / au_m;

// Human-readable calibration summary (implemented in constants.cpp).
std::string describe();

} // namespace sgr::constants
