#pragma once
// Observers in the Kerr spacetime (Gate 17): the zero-angular-momentum
// observer (ZAMO) frame and Lorentz-boosted tetrads of moving observers.
//
// One implementation serves the live renderer (engine/src/app.cpp builds the
// camera tetrad from it every frame) and the CPU validation
// (validation/observer_tool.cpp), so the numbers in the extension paper and
// the images on screen come from the same code.
//
// ZAMO at Boyer-Lindquist (r, θ), M = 1 units of the metric passed in:
//   α = √(−g_tt + g_tφ²/g_φφ)   lapse
//   ω = −g_tφ / g_φφ             frame-dragging angular velocity
//   e_t̂ = (1/α)(∂_t + ω ∂_φ),  e_r̂ = ∂_r/√g_rr,  e_θ̂ = ∂_θ/√g_θθ,
//   e_φ̂ = ∂_φ/√g_φφ.
// An observer moving with 3-velocity v (components along r̂, θ̂, φ̂) relative
// to the ZAMO has the boosted tetrad
//   e0' = γ (e_t̂ + v·ê),  e_j' = e_ĵ + (γ−1)(v_j/v²)(v·ê) + γ v_j e_t̂,
// which is orthonormal by construction; e0' is the observer's 4-velocity,
// so 1/e0'^t = dτ/dt is its clock rate against far-away (BL) time.
//
// Velocities of the standard observers relative to the ZAMO:
//   Orbiting  v_φ̂ = (Ω − ω)√g_φφ / α, Ω = 1/(r^{3/2} + a) (prograde
//             Keplerian; a geodesic only in the equatorial plane)
//   Infall    v_r̂ = −√(2Mr(r²+a²)/A): free fall from rest at infinity with
//             E = 1, L_z = Q = 0 (then γ = 1/α, and θ̇ = 0 exactly)
//   Hovering  v_φ̂ = −ω√g_φφ / α: at rest relative to the distant stars
//             (u ∝ ∂_t); needs v < 1, i.e. r outside the ergosphere
//   Moving    from coordinate rates (ṙ, θ̇, φ̇) = dx^i/dt:
//             v_r̂ = √g_rr ṙ/α, v_θ̂ = √g_θθ θ̇/α, v_φ̂ = √g_φφ (φ̇ − ω)/α

#include <array>

#include "sgr_physics/kerr_schild.hpp"
#include "sgr_physics/metric.hpp"

namespace sgr {

enum class ObserverKind { Zamo, Orbiting, Infall, Hovering, Moving };

struct ObserverFrame {
    bool valid = true;            // false: this observer needs v ≥ c here;
                                  // the frame returned is then the ZAMO's
    double alpha = 1.0;           // ZAMO lapse
    double omega = 0.0;           // frame-dragging angular velocity [1/M]
    std::array<double, 3> v{};    // velocity relative to the ZAMO [c]
    double speed = 0.0;
    double gamma = 1.0;
    // Orthonormal tetrad, Boyer-Lindquist contravariant components:
    // e[0] = 4-velocity, e[1..3] = boosted (r̂, θ̂, φ̂) legs.
    std::array<Vec4, 4> e{};
    double dtau_dt() const { return 1.0 / e[0][0]; }
};

// ZAMO-relative 3-velocity of a standard observer at BL (r, θ). For
// ObserverKind::Moving, coord_rates = (dr/dt, dθ/dt, dφ/dt).
std::array<double, 3> observer_velocity(const KerrMetric& m, ObserverKind kind,
                                        double r, double theta,
                                        const std::array<double, 3>& coord_rates = {});

// Tetrad of the observer moving with ZAMO-relative velocity v at BL (r, θ).
ObserverFrame observer_frame(const KerrMetric& m, double r, double theta,
                             const std::array<double, 3>& v);
inline ObserverFrame observer_frame(const KerrMetric& m, ObserverKind kind,
                                    double r, double theta,
                                    const std::array<double, 3>& coord_rates = {}) {
    return observer_frame(m, r, theta,
                          observer_velocity(m, kind, r, theta, coord_rates));
}

// Backward-traced ray of the photon RECEIVED by this observer at BL
// (r, θ, φ = 0) from local direction n = (n_1, n_2, n_3) along e[1..3]
// (unit): −k_rec = −e0 + n^i e_i, in Kerr-Schild components, lowered.
// With the ZAMO frame this is exactly ks_camera_ray.
KSRay ks_observer_ray(const KerrSchild& ks, double r, double theta,
                      const ObserverFrame& f, const std::array<double, 3>& n);

} // namespace sgr
