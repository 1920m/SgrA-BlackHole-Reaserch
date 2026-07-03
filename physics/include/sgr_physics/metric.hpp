#pragma once
// Kerr metric in Boyer-Lindquist coordinates, key radii, and the generic
// metric interface used by the Christoffel-based geodesic integrator.
//
// Conventions: G = c = 1, signature (−,+,+,+), coordinates x^μ = (t, r, θ, φ).
// All lengths in units of M unless a mass is passed explicitly.
//
// Metric components are implemented as a template on the scalar type so the
// same code path can be evaluated with std::complex<double> for complex-step
// differentiation: ∂g/∂r = Im[g(r + ih)]/h with h ~ 1e-100 is exact to machine
// precision (no subtractive cancellation), which gives Christoffel symbols
// accurate to ~1e-15 for *any* metric without hand-derived derivatives.

#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace sgr {

// Symmetric 4x4 metric tensor storage, generic scalar.
template <typename T>
struct Metric4 {
    // g[μ][ν], symmetric; only Boyer-Lindquist-type structure is used in this
    // project: nonzero tt, tφ, rr, θθ, φφ (t–φ block plus diagonal r, θ).
    std::array<std::array<T, 4>, 4> g{};

    T& operator()(int mu, int nu) { return g[mu][nu]; }
    const T& operator()(int mu, int nu) const { return g[mu][nu]; }
};

// Inverse for BL-block metrics (t–φ block + diagonal rr, θθ).
template <typename T>
Metric4<T> invert_bl_block(const Metric4<T>& m) {
    Metric4<T> inv{};
    const T det_tphi = m(0, 0) * m(3, 3) - m(0, 3) * m(0, 3);
    inv(0, 0) = m(3, 3) / det_tphi;
    inv(3, 3) = m(0, 0) / det_tphi;
    inv(0, 3) = -m(0, 3) / det_tphi;
    inv(3, 0) = inv(0, 3);
    inv(1, 1) = T(1) / m(1, 1);
    inv(2, 2) = T(1) / m(2, 2);
    return inv;
}

// ---------------------------------------------------------------------------
// Kerr metric components (templated core, shared by double and complex paths)
// ---------------------------------------------------------------------------
template <typename T>
Metric4<T> kerr_components(T r, T theta, double M, double a) {
    using std::cos;
    using std::sin;
    const T ct = cos(theta), st = sin(theta);
    const T r2 = r * r;
    const T a2 = T(a * a);
    const T sigma = r2 + a2 * ct * ct;
    const T delta = r2 - T(2.0 * M) * r + a2;
    const T st2 = st * st;
    // A = (r²+a²)² − a²Δ sin²θ
    const T A = (r2 + a2) * (r2 + a2) - a2 * delta * st2;

    Metric4<T> m{};
    m(0, 0) = -(T(1) - T(2.0 * M) * r / sigma);
    m(0, 3) = -T(2.0 * M * a) * r * st2 / sigma;
    m(3, 0) = m(0, 3);
    m(1, 1) = sigma / delta;
    m(2, 2) = sigma;
    m(3, 3) = A * st2 / sigma;
    return m;
}

// ---------------------------------------------------------------------------
// Abstract metric interface (double and complex evaluation for autodiff)
// ---------------------------------------------------------------------------
class IMetric {
public:
    virtual ~IMetric() = default;
    virtual Metric4<double> components(double r, double theta) const = 0;
    virtual Metric4<std::complex<double>> components(std::complex<double> r,
                                                     std::complex<double> theta) const = 0;
    virtual double mass() const = 0;
    virtual double spin() const = 0; // a = J/M (dimensionful, units of M)
};

// Christoffel symbols Γ^λ_{μν} at (r, θ) by complex-step differentiation.
// Only ∂_r and ∂_θ are nonzero (stationary + axisymmetric).
// Returns gamma[λ][μ][ν].
using Christoffel = std::array<std::array<std::array<double, 4>, 4>, 4>;
Christoffel christoffel(const IMetric& metric, double r, double theta);

// ---------------------------------------------------------------------------
// Kerr metric + key radii
// ---------------------------------------------------------------------------
class KerrMetric final : public IMetric {
public:
    KerrMetric(double M, double a_star) : M_(M), a_(a_star * M) {
        if (std::abs(a_star) > 1.0)
            throw std::invalid_argument("KerrMetric: |a*| > 1 is a naked singularity");
    }

    Metric4<double> components(double r, double theta) const override {
        return kerr_components<double>(r, theta, M_, a_);
    }
    Metric4<std::complex<double>> components(std::complex<double> r,
                                             std::complex<double> theta) const override {
        return kerr_components<std::complex<double>>(r, theta, M_, a_);
    }
    double mass() const override { return M_; }
    double spin() const override { return a_; }
    double spin_star() const { return a_ / M_; }

    // -- Key radii (Bardeen, Press & Teukolsky 1972) -----------------------
    // Outer horizon r₊ = M + √(M²−a²)
    double r_horizon() const { return M_ + std::sqrt(M_ * M_ - a_ * a_); }
    // Ergosphere radius at polar angle θ: r_E = M + √(M²−a²cos²θ)
    double r_ergosphere(double theta) const {
        const double ct = std::cos(theta);
        return M_ + std::sqrt(M_ * M_ - a_ * a_ * ct * ct);
    }
    // Equatorial circular photon orbit radius.
    // Prograde: r_ph = 2M{1 + cos[(2/3) arccos(−a/M)]}; retrograde with +a/M.
    double r_photon(bool prograde) const {
        const double x = (prograde ? -1.0 : +1.0) * a_ / M_;
        return 2.0 * M_ * (1.0 + std::cos((2.0 / 3.0) * std::acos(x)));
    }
    // ISCO via the exact BPT72 Z₁, Z₂ formula.
    double r_isco(bool prograde) const {
        const double q = a_ / M_;
        const double z1 = 1.0 + std::cbrt(1.0 - q * q) *
                                    (std::cbrt(1.0 + q) + std::cbrt(1.0 - q));
        const double z2 = std::sqrt(3.0 * q * q + z1 * z1);
        const double sgn = prograde ? -1.0 : +1.0;
        return M_ * (3.0 + z2 + sgn * std::sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2)));
    }

    // -- Equatorial circular timelike orbits (BPT72 eqs. 2.12–2.16) --------
    // Angular velocity Ω = ±√M / (r^{3/2} ± a√M); upper sign prograde.
    double omega_circular(double r, bool prograde = true) const {
        const double sM = std::sqrt(M_);
        const double s = prograde ? 1.0 : -1.0;
        return s * sM / (std::pow(r, 1.5) + s * a_ * sM);
    }
    // Specific energy E and angular momentum L of a circular orbit.
    double energy_circular(double r, bool prograde = true) const {
        const double s = prograde ? 1.0 : -1.0;
        const double v = std::sqrt(M_ / (r * r * r));
        const double num = 1.0 - 2.0 * M_ / r + s * a_ * v;
        const double den = std::sqrt(1.0 - 3.0 * M_ / r + s * 2.0 * a_ * v);
        return num / den;
    }
    // L = ±√(Mr)·(1 + a²/r² ∓ 2a√M/r^{3/2}) / √(1 − 3M/r ± 2a√M/r^{3/2})
    double angmom_circular(double r, bool prograde = true) const {
        const double s = prograde ? 1.0 : -1.0;
        const double v = std::sqrt(M_ / (r * r * r)); // √M / r^{3/2}
        const double num = s * std::sqrt(M_ * r) *
                           (1.0 + a_ * a_ / (r * r) - s * 2.0 * a_ * v);
        const double den = std::sqrt(1.0 - 3.0 * M_ / r + s * 2.0 * a_ * v);
        return num / den;
    }

private:
    double M_; // mass
    double a_; // spin parameter a = J/M, |a| ≤ M
};

} // namespace sgr
