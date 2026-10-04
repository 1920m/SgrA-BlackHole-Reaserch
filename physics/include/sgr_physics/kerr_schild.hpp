#pragma once
// Kerr spacetime in Cartesian Kerr-Schild coordinates (Kerr 1963) and a
// Hamiltonian null-geodesic tracer (Gate 16). CPU twin of the live renderer's
// fp32 kernel; the Carter/Mino Boyer-Lindquist path (null_geodesics.hpp)
// stays the science reference and is what this file is validated against.
//
//   g_μν = η_μν + f l_μ l_ν,      g^μν = η^μν − f l^μ l^ν,   η = diag(−1,1,1,1)
//   f    = 2 M r³ / (r⁴ + a² z²)
//   r    : r⁴ − (x² + y² + z² − a²) r² − a² z² = 0   (r ≥ 0 root)
//   l_μ  = (1,  s (r x + s a y)/(r² + a²),  s (r y − s a x)/(r² + a²),  s z/r)
//
// l is null for η and for g, so the inverse needs no matrix inversion, and
// the Hamiltonian H = ½ g^μν p_μ p_ν has a closed-form gradient:
//   dx^μ/dλ = g^μν p_ν,   dp_i/dλ = ½ ∂_i [ f (l·p)² ],   dp_t/dλ = 0,
// with l·p = l^μ p_μ = −p_t + l_x p_x + l_y p_y + l_z p_z.
//
// Chart choice s (both are exact Kerr; they differ in which horizon they
// cover regularly):
//   s = +1  INGOING  — regular across the future horizon (in-falling
//           observers, G24);
//   s = −1  OUTGOING — regular across the past horizon. Camera rays are
//           traced BACKWARD in time (−k of the received photon); rays that
//           end in the hole reach the past horizon, which this chart crosses
//           smoothly — no 1/Δ blow-up, and Cartesian coordinates have no
//           polar-axis singularity at all (the BL "zipper").
// Boyer-Lindquist relation:
//   x + i y = (r + i s a) sinθ e^{iφ_KS},   z = r cosθ,
//   t_KS = t_BL + s T(r),   φ_KS = φ_BL + s Φ(r),
//   T(r) = ∫ 2Mr/Δ dr,      Φ(r) = ∫ a/Δ dr  (both → 0 / finite at r → ∞).
//
// Camera-ray handedness. A pixel looking along local direction n RECEIVES a
// photon with k = e_t − n·e (it travels toward the camera); its source is
// found by tracing −k = −e_t + n·e. Launching +e_t + n·e instead (the photon
// the camera would EMIT toward n) is exact in static spacetimes but in Kerr
// traces the time-reversed — i.e. spin-reversed — spacetime: a left–right
// mirror image. ks_camera_ray uses the received photon.

#include <array>

#include "sgr_physics/metric.hpp"

namespace sgr {

using Vec4 = std::array<double, 4>;

class KerrSchild {
public:
    KerrSchild(double M, double a_star, int chart_sign = -1);

    double M() const { return M_; }
    double a() const { return a_; }
    int chart() const { return s_; }
    double r_plus() const { return rp_; }
    double r_minus() const { return rm_; }

    // Boyer-Lindquist radius of a Cartesian point.
    double radius(double x, double y, double z) const;

    Metric4<double> metric(const Vec4& X) const;          // g_μν
    Metric4<double> inverse_metric(const Vec4& X) const;  // g^μν
    double hamiltonian(const Vec4& X, const Vec4& P) const;
    // Hamilton's equations (analytic gradient).
    void rhs(const Vec4& X, const Vec4& P, Vec4& dX, Vec4& dP) const;

    // BL ↔ KS. φ_bl is the Boyer-Lindquist azimuth.
    double T_shift(double r) const;   // ∫ 2Mr/Δ dr
    double Phi_shift(double r) const; // ∫ a/Δ dr, → 0 as r → ∞
    Vec4 point_from_bl(double t, double r, double theta, double phi_bl) const;
    // J^μ_ν = ∂x^μ_KS / ∂x^ν_BL at a BL point.
    std::array<Vec4, 4> jacobian_bl_to_ks(double r, double theta,
                                          double phi_bl) const;
    Vec4 vector_from_bl(double r, double theta, double phi_bl,
                        const Vec4& v_bl) const;           // contravariant
    // (t_BL, r, θ, φ_BL) of a KS point.
    Vec4 bl_coords(const Vec4& X) const;
    // p^BL_ν = p^KS_μ J^μ_ν (covariant).
    Vec4 covector_to_bl(const Vec4& X, const Vec4& P) const;

    // Conserved quantities of a geodesic: E = −p_t, L_z = x p_y − y p_x,
    // Carter Q (via the BL covector; light or timelike, μ² = −2H).
    double energy(const Vec4& P) const { return -P[0]; }
    double angular_momentum(const Vec4& X, const Vec4& P) const {
        return X[1] * P[2] - X[2] * P[1];
    }
    double carter_Q(const Vec4& X, const Vec4& P) const;

private:
    double M_, a_;
    int s_;
    double rp_, rm_;
};

// Backward-traced ray of the photon RECEIVED by a ZAMO camera at BL
// (r_cam, θ_cam, φ = 0) from local direction n = (n_r, n_θ, n_φ) (unit).
// Returns (X, P) in KS components with P = −k_received lowered.
struct KSRay {
    Vec4 X, P;
};
KSRay ks_camera_ray(const KerrSchild& ks, double r_cam, double theta_cam,
                    const std::array<double, 3>& n);
// The same for a camera with any orthonormal frame e (e[0] its 4-velocity)
// at event X: n = components along e[1..3]; traces −k = −e0 + n^i e_i.
KSRay ks_frame_ray(const KerrSchild& ks, const Vec4& X, const std::array<Vec4, 4>& e,
                   const std::array<double, 3>& n);

// Adaptive (Dormand-Prince) trace with equatorial-crossing location.
enum class KSOutcome { Captured, Escaped, MaxSteps };
struct KSTraceOptions {
    double rtol = 1e-11, atol = 1e-11;
    double r_escape = 1500.0;
    int max_steps = 200000;
    int max_crossings = 4;
};
struct KSTraceResult {
    KSOutcome outcome = KSOutcome::MaxSteps;
    int n_cross = 0;
    std::array<double, 4> cross_r{};      // BL radius of each z = 0 crossing
    std::array<double, 4> cross_phi_bl{}; // BL azimuth of each crossing
    double esc_theta = 0.0, esc_phi = 0.0; // asymptotic source direction
    double max_dQ_rel = 0.0, max_dL_rel = 0.0, max_H_rel = 0.0;
    KSRay final_state{};
};
KSTraceResult ks_trace(const KerrSchild& ks, const KSRay& ray,
                       const KSTraceOptions& opt = {});

} // namespace sgr
