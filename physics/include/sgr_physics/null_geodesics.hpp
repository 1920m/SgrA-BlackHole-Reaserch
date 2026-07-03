#pragma once
// Carter-separated null geodesics in Kerr spacetime (Carter 1968).
//
// Formulation. Carter's first-order equations in Mino time τ (dλ = Σ dτ):
//
//     (dr/dτ)²  = R(r)  = P(r)² − Δ[(L_z − aE)² + Q],   P = E(r²+a²) − a L_z
//     (dθ/dτ)²  = Θ(θ)  = Q + cos²θ (a²E² − L_z²/sin²θ)
//      dφ/dτ    = −(aE − L_z/sin²θ) + a P/Δ
//      dt/dτ    = −a(aE sin²θ − L_z) + (r²+a²) P/Δ
//
// We integrate the *differentiated* (second-order) radial and polar equations
//
//     d²r/dτ² = R'(r)/2,      d²θ/dτ² = Θ'(θ)/2
//
// which are exactly equivalent along the geodesic and — unlike the ±√R form —
// have no sign bookkeeping: the first-order sqrt form stalls at turning points
// (r evolves only through its own equation, so dr/dτ = 0 is a fixed point of
// the truncated system), whereas the second-order dynamics carries the ray
// smoothly through R = 0 and Θ = 0. The constraint residuals
//
//     |p_r² − R(r)|,   |p_θ² − Θ(θ)|
//
// are then genuine (not enforced) invariants: together with the reconstruction
// of E, L_z, Q from the 4-momentum they form the integrator health check
// (E, L_z, Q conserved to 1e-6).
//
// θ = 0, π coordinate singularities are regularised with a 1e-8 clamp.

#include <array>
#include <cmath>

#include "sgr_physics/integrator.hpp"
#include "sgr_physics/metric.hpp"

namespace sgr {

struct NullRayConstants {
    double M = 1.0;   // black-hole mass
    double a = 0.0;   // spin a = J/M
    double E = 1.0;   // conserved energy  −k_t
    double Lz = 0.0;  // conserved axial angular momentum  k_φ
    double Q = 0.0;   // Carter constant
};

// State in Mino time: y = (t, r, θ, φ, p_r, p_θ) with p_r = dr/dτ, p_θ = dθ/dτ.
using NullState = StateVec<6>;

constexpr double kThetaClamp = 1e-8; // pole regularisation (coordinate, not physical)

// Radial potential R(r) and its derivative.
inline double radial_potential(const NullRayConstants& c, double r) {
    const double P = c.E * (r * r + c.a * c.a) - c.a * c.Lz;
    const double delta = r * r - 2.0 * c.M * r + c.a * c.a;
    const double K = (c.Lz - c.a * c.E) * (c.Lz - c.a * c.E) + c.Q;
    return P * P - delta * K;
}
inline double radial_potential_deriv(const NullRayConstants& c, double r) {
    const double P = c.E * (r * r + c.a * c.a) - c.a * c.Lz;
    const double K = (c.Lz - c.a * c.E) * (c.Lz - c.a * c.E) + c.Q;
    return 4.0 * c.E * P * r - 2.0 * (r - c.M) * K;
}

// Polar potential Θ(θ) and its derivative.
inline double polar_potential(const NullRayConstants& c, double theta) {
    const double ct = std::cos(theta), st = std::sin(theta);
    return c.Q + ct * ct * (c.a * c.a * c.E * c.E - c.Lz * c.Lz / (st * st));
}
inline double polar_potential_deriv(const NullRayConstants& c, double theta) {
    const double ct = std::cos(theta), st = std::sin(theta);
    // d/dθ [cos²θ a²E²] = −a²E² sin 2θ ;  d/dθ [−L_z² cot²θ] = +2 L_z² cosθ/sin³θ
    return -c.a * c.a * c.E * c.E * 2.0 * st * ct +
           2.0 * c.Lz * c.Lz * ct / (st * st * st);
}

// RHS of the 6-dimensional Mino-time system.
inline void null_geodesic_rhs(const NullRayConstants& c, const NullState& y,
                              NullState& dy) {
    const double r = y[1];
    double theta = y[2];
    // Regularise poles: clamp θ away from 0, π.
    if (theta < kThetaClamp) theta = kThetaClamp;
    if (theta > 3.14159265358979323846 - kThetaClamp)
        theta = 3.14159265358979323846 - kThetaClamp;

    const double st = std::sin(theta), ct = std::cos(theta);
    const double st2 = st * st;
    const double delta = r * r - 2.0 * c.M * r + c.a * c.a;
    const double P = c.E * (r * r + c.a * c.a) - c.a * c.Lz;

    dy[0] = -c.a * (c.a * c.E * st2 - c.Lz) + (r * r + c.a * c.a) * P / delta; // dt/dτ
    dy[1] = y[4];                                                             // dr/dτ
    dy[2] = y[5];                                                             // dθ/dτ
    dy[3] = -(c.a * c.E - c.Lz / st2) + c.a * P / delta;                      // dφ/dτ
    dy[4] = 0.5 * radial_potential_deriv(c, r);                               // dp_r/dτ
    dy[5] = 0.5 * polar_potential_deriv(c, theta);                            // dp_θ/dτ
    (void)ct;
}

// Build (constants, state) from a position and a null 4-momentum k^μ in
// Boyer-Lindquist components. No sign ambiguity: p_r = Σ k^r, p_θ = Σ k^θ.
struct NullRayInit {
    NullRayConstants constants;
    NullState state;
};
NullRayInit make_null_ray(const KerrMetric& metric, double t, double r,
                          double theta, double phi, const std::array<double, 4>& k);

// Diagnostics for the §13 health check: reconstruct conserved quantities from
// the instantaneous state and compare with the exact input constants.
struct NullConservedErrors {
    double dE_rel;      // |E_rec − E| / |E|
    double dLz_rel;     // |L_z,rec − L_z| / max(|L_z|, E·M)
    double dQ_rel;      // |Q_rec − Q| / max(|Q|, (E·M)²)
    double null_norm;   // |g_μν k^μ k^ν| / E²   (should be 0 for a photon)
    double r_residual;  // |p_r² − R(r)| / normalisation
    double theta_residual;
};
NullConservedErrors null_conserved_errors(const KerrMetric& metric,
                                          const NullRayConstants& c,
                                          const NullState& y);

// Reconstruct the Boyer-Lindquist 4-momentum k^μ = dx^μ/dλ from a Mino state.
std::array<double, 4> null_four_momentum(const NullRayConstants& c, const NullState& y);

} // namespace sgr
