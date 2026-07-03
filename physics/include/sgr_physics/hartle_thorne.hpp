#pragma once
// Hartle-Thorne free-quadrupole metric ("quasi-Kerr", Glampedakis & Babak 2006,
// CQG 23, 4167; gr-qc/0510057), the simulator's no-hair test module.
//
//     g_ab = g_ab^Kerr(M, a)  +  ε h_ab(r, θ)
//
// where h_ab is the leading-order quadrupole deviation of a vacuum exterior
// (Hartle & Thorne 1968), diagonal in BL-type coordinates:
//
//     h_tt = (1 − 2M/r)   (1 − 3cos²θ) F₁(r)
//     h_rr = (1 − 2M/r)⁻¹ (1 − 3cos²θ) F₁(r)
//     h_θθ = −r²          (1 − 3cos²θ) F₂(r)
//     h_φφ = −r² sin²θ    (1 − 3cos²θ) F₂(r)
//
// (covariant components, obtained from GB06 eqs. (16)-(18) by lowering with the
// background to the order of validity). F₁, F₂ are GB06 eq. (A.4):
//
//   F₁ = −5(r−M)(2M²+6Mr−3r²) / (8Mr(r−2M)) − 15r(r−2M) ln(r/(r−2M)) / (16M²)
//   F₂ =  5(2M²−3Mr−3r²) / (8Mr)            + 15(r²−2M²) ln(r/(r−2M)) / (16M²)
//
// Quadrupole conventions. GB06 define Q = −J²/M − ε M³; this project defines
// Q_total = −M a² + ε_q M³. Hence ε(GB06) = −ε_q, and this class takes ε_q with
// this project's sign:  g_ab = g_ab^Kerr − ε_q h_ab.
// At ε_q = 0 this reproduces Kerr *exactly* (machine precision) by
// construction.
//
// Numerical note: the two terms of F₁, F₂ are individually O(r) but cancel to
// O(M³/r³) — catastrophic cancellation for r ≳ 20M (S2 lives at ~3000M, where
// naive evaluation would carry ~100% relative error). For r > kSeriesSwitch we
// therefore evaluate the exact large-r series (generated symbolically, ~1e-16
// relative), and the closed form below the switch. Both branches are verified
// against each other in the overlap region by the Phase-0 tests.
//
// The quasi-Kerr metric is NOT Carter-separable: geodesics must be integrated
// through the Christoffel path (timelike_geodesics.hpp). E and L_z remain
// conserved (stationary, axisymmetric); Q does not.

#include <complex>

#include "sgr_physics/metric.hpp"

namespace sgr {

class HartleThorneMetric final : public IMetric {
public:
    // M: mass; a_star: dimensionless spin a/M; eps_q: quadrupole
    // deviation, Q_total = −M a² + ε_q M³.
    HartleThorneMetric(double M, double a_star, double eps_q)
        : kerr_(M, a_star), eps_q_(eps_q) {}

    Metric4<double> components(double r, double theta) const override;
    Metric4<std::complex<double>> components(std::complex<double> r,
                                             std::complex<double> theta) const override;

    double mass() const override { return kerr_.mass(); }
    double spin() const override { return kerr_.spin(); }
    double eps_q() const { return eps_q_; }
    const KerrMetric& kerr() const { return kerr_; }

    // Quadrupole radial profiles (exposed for tests). Templated on scalar for
    // the complex-step path; M is taken from the metric.
    double F1(double r) const;
    double F2(double r) const;

private:
    template <typename T>
    Metric4<T> components_impl(T r, T theta) const;

    KerrMetric kerr_;
    double eps_q_;
};

} // namespace sgr
