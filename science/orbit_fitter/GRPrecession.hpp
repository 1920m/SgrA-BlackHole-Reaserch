#pragma once
// Analytic pericenter-advance formulas (cross-checks for the numerical
// geodesic precession; used in the paper's calculation appendix).
//
// All inputs in geometric units (lengths in M); returns radians per orbit.
//   p = a(1−e²) — semi-latus rectum
//
//   Schwarzschild:   Δω_S  = 6πM/p                     (GRAVITY 2020: 12.1'/orbit for S2)
//   Lense-Thirring:  Δω_LT = −8π a* (M/p)^{3/2}        (prograde equatorial; ≈ −0.2' for S2)
//   Quadrupole:      Δω_Q  = −3π q̃ (M/p)²              (equatorial; q̃ = Q/M³,
//                     from the Newtonian δΦ = −QP₂/r³ perturbation at θ=π/2;
//                     validated numerically against the HT integrator in Phase 5)

namespace sgr::precession {

inline double schwarzschild(double a_orbit_M, double e) {
    const double p = a_orbit_M * (1.0 - e * e);
    return 6.0 * 3.14159265358979323846 / p;
}

inline double lense_thirring(double a_orbit_M, double e, double a_star) {
    const double p = a_orbit_M * (1.0 - e * e);
    return -8.0 * 3.14159265358979323846 * a_star * std::pow(1.0 / p, 1.5);
}

inline double quadrupole(double a_orbit_M, double e, double q_tilde) {
    const double p = a_orbit_M * (1.0 - e * e);
    return -3.0 * 3.14159265358979323846 * q_tilde / (p * p);
}

} // namespace sgr::precession
