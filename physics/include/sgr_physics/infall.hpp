#pragma once
// Gate 24 — falling into Sgr A*: the in-faller's worldline, camera frame and
// tidal field, in the INGOING Kerr–Schild chart (s = +1), which is regular
// across the future horizon r₊ (Boyer–Lindquist is singular there).
//
// Worldline. A free fall is a timelike geodesic of the same Hamiltonian the
// ray tracer uses, H = ½ g^μν p_μ p_ν, with H = −½ instead of 0: the affine
// parameter is then proper time τ (KerrSchild::rhs, analytic gradient). It is
// integrated in fp64 (Dormand–Prince, tolerance 1e-12) from r0 until just
// before the inner (Cauchy) horizon r₋, where mass inflation makes the
// classical interior unreliable — the trip stops there by design.
//
// Starts. RAIN: dropped from rest at infinity (E = 1, L_z = 0, Q = 0); its
// polar angle stays constant. REST: released from rest relative to the local
// ZAMO at r0 (E = α(r0), L_z = 0, u_r = u_θ = 0) — for a = 0, the classic
// radial fall from rest.
//
// Frame. e0 = u; spatial legs by Gram–Schmidt (with the full metric) of the
// Kerr–Schild coordinate directions ∂_r, ∂_θ, ∂_φ projected orthogonal to u.
// Those are smooth through r₊ in this chart, so the camera frame is too. The
// legs mirror the exterior ZAMO triad (outward, θ̂, φ̂), so the renderer's
// look basis applies unchanged.
//
// Tidal field. E_ij = R_{iαjβ} u^α u^β (the relative acceleration of two
// free-falling particles separated by ξ is −E_ij ξ^j) and the magnetic part
// B_ij = ½ ε_ipq R_{pqj0}, from the Riemann tensor of the analytic metric:
// Christoffels by complex-step differentiation (exact), their derivatives by
// central differences. Checked against the closed-form Kretschmann scalar,
// K = 8(E_ij E^ij − B_ij B^ij) in vacuum, inside and outside r₊.

#include <array>
#include <vector>

#include "sgr_physics/kerr_schild.hpp"

namespace sgr {

enum class InfallStartKind { Rain, Rest };

struct InfallStart {
    InfallStartKind kind = InfallStartKind::Rain;
    double r0 = 30.0;                 // BL radius of the start [M]
    double theta0 = 1.0471975511965976;   // BL polar angle [rad] (60°)
    double stop_above_rminus = 0.05;  // stop at r = r₋ + this [M]
};

struct InfallSample {
    double tau = 0.0;     // proper time since the start [M]
    Vec4 X{};             // ingoing KS event (t, x, y, z)
    Vec4 U{};             // 4-velocity u^μ (KS components)
    double r = 0.0, theta = 0.0;   // BL radius and polar angle
    double t_bl = 0.0;    // BL (far-away) time; NaN at and inside r₊
    double E = 0.0, Lz = 0.0, Q = 0.0;   // conserved quantities from p = g·u
    double norm = 0.0;    // g_μν u^μ u^ν + 1
};

class InfallTrajectory {
public:
    InfallTrajectory(double a_star, const InfallStart& start);
    const KerrSchild& chart() const { return ks_; }
    const std::vector<InfallSample>& samples() const { return s_; }
    InfallSample at(double tau) const;     // interpolated (Hermite in X)
    // proper time at a given BL radius (first passage), by interpolation
    double tau_at_radius(double r) const;
    double tau_horizon() const { return tau_hor_; }   // at r = r₊
    double tau_end() const { return s_.back().tau; }  // at r₋ + δ
    // worst conservation errors over the trajectory: |ΔE|/E, |ΔL_z|/max(|L|,M E),
    // |ΔQ|/max(|Q|, M² E²) (Q away from |r − r₊| < 0.05 M, where its BL
    // reconstruction is ill-conditioned), |g u u + 1|
    double max_dE() const { return dE_; }
    double max_dL() const { return dL_; }
    double max_dQ() const { return dQ_; }
    double max_norm() const { return dn_; }

private:
    KerrSchild ks_;
    std::vector<InfallSample> s_;
    double tau_hor_ = 0.0, dE_ = 0.0, dL_ = 0.0, dQ_ = 0.0, dn_ = 0.0;
};

// Orthonormal frame of an observer with 4-velocity U at X (any chart):
// e[0] = U, e[1] ≈ outward, e[2] ≈ θ̂ (southward), e[3] ≈ φ̂ (prograde).
std::array<Vec4, 4> infall_tetrad(const KerrSchild& ks, const Vec4& X, const Vec4& U);

struct TidalField {
    std::array<std::array<double, 3>, 3> E{}, B{};   // frame components [1/M²]
    std::array<double, 3> eig{};   // eigenvalues of E, ascending [1/M²]
    double kretschmann = 0.0;      // R_abcd R^abcd from the same frame components
};
TidalField tidal_field(const KerrSchild& ks, const Vec4& X, const std::array<Vec4, 4>& e);

// Closed-form Kretschmann scalar of Kerr (M = 1 units): 48 M² (r⁶ − 15a²r⁴c²
// + 15a⁴r²c⁴ − a⁶c⁶)/Σ⁶, c = cos θ.
double kerr_kretschmann(double M, double a, double r, double cos_theta);

}  // namespace sgr
