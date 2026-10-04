#include "sgr_physics/disk_model.hpp"

#include <algorithm>
#include <cmath>

#include "sgr_physics/constants.hpp"
#include "sgr_physics/metric.hpp"

namespace sgr {

namespace {
constexpr double kPi = 3.14159265358979323846;

// CIE 1931 colour matching functions, multi-lobe Gaussian fits
// (Wyman, Sloan & Shirley 2013). Wavelength in nanometres.
inline double gauss_piece(double x, double alpha, double mu, double s1, double s2) {
    const double t = (x - mu) * (x < mu ? 1.0 / s1 : 1.0 / s2);
    return alpha * std::exp(-0.5 * t * t);
}
inline double cie_x(double wl) {
    return gauss_piece(wl, 1.056, 599.8, 37.9, 31.0) +
           gauss_piece(wl, 0.362, 442.0, 16.0, 26.7) +
           gauss_piece(wl, -0.065, 501.1, 20.4, 26.2);
}
inline double cie_y(double wl) {
    return gauss_piece(wl, 0.821, 568.8, 46.9, 40.5) +
           gauss_piece(wl, 0.286, 530.9, 16.3, 31.1);
}
inline double cie_z(double wl) {
    return gauss_piece(wl, 1.217, 437.0, 11.8, 36.0) +
           gauss_piece(wl, 0.681, 459.0, 26.0, 13.8);
}

} // namespace

namespace {
std::array<double, 3> planck_xyz_raw(double T) {
    // Integrate B_λ(T)·CMF over 380–780 nm. B_λ ∝ 1/(λ⁵ (e^{c₂/λT} − 1)),
    // c₂ = hc/k = 1.4388e-2 m·K. Absolute scale ∝ radiance (T⁴ weighting kept).
    const double c2 = 1.4388e-2;
    double X = 0.0, Y = 0.0, Z = 0.0;
    for (double wl = 380.0; wl <= 780.0; wl += 5.0) {
        const double lm = wl * 1e-9;
        const double B = 1.0 / (lm * lm * lm * lm * lm *
                                (std::exp(c2 / (lm * T)) - 1.0));
        X += B * cie_x(wl);
        Y += B * cie_y(wl);
        Z += B * cie_z(wl);
    }
    return {X, Y, Z};
}
} // namespace

std::array<double, 3> planck_to_xyz(double T) {
    static const double norm = 1.0 / planck_xyz_raw(6500.0)[1];
    const auto xyz = planck_xyz_raw(T);
    return {xyz[0] * norm, xyz[1] * norm, xyz[2] * norm};
}

std::array<double, 3> planck_to_linear_srgb(double T) {
    // Normalise so a 6500 K blackbody has luminance Y = 1.
    static const double norm = 1.0 / planck_xyz_raw(6500.0)[1];
    const auto xyz = planck_xyz_raw(T);
    const double X = xyz[0] * norm, Y = xyz[1] * norm, Z = xyz[2] * norm;

    // XYZ → linear sRGB (D65)
    const double R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z;
    const double G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
    const double B = 0.0557 * X - 0.2040 * Y + 1.0570 * Z;
    return {std::max(0.0, R), std::max(0.0, G), std::max(0.0, B)};
}

NovikovThorneDisk::NovikovThorneDisk(double a_star, double r_in, double r_out,
                                     double T_peak)
    : a_(a_star), r_in_(r_in), r_out_(r_out) {
    x0_ = std::sqrt(r_in);
    const double ac = std::acos(std::clamp(a_star, -1.0, 1.0));
    x1_ = 2.0 * std::cos((ac - kPi) / 3.0);
    x2_ = 2.0 * std::cos((ac + kPi) / 3.0);
    x3_ = -2.0 * std::cos(ac / 3.0);

    // Find the flux maximum to set the temperature scale.
    double fmax = 0.0;
    for (int i = 0; i <= 2000; ++i) {
        const double r = r_in + (r_out - r_in) * i / 2000.0;
        fmax = std::max(fmax, flux(r));
    }
    Tscale_ = T_peak / std::pow(fmax, 0.25);

    // Blackbody LUT (log-spaced).
    lut_logT_min_ = std::log(500.0);
    lut_logT_max_ = std::log(60000.0);
    bb_lut_.resize(kLutN);
    for (int i = 0; i < kLutN; ++i) {
        const double lt = lut_logT_min_ +
                          (lut_logT_max_ - lut_logT_min_) * i / (kLutN - 1);
        const auto rgb = planck_to_linear_srgb(std::exp(lt));
        bb_lut_[i] = {static_cast<float>(rgb[0]), static_cast<float>(rgb[1]),
                      static_cast<float>(rgb[2])};
    }
}

double NovikovThorneDisk::flux(double r) const {
    if (r <= r_in_ * (1.0 + 1e-12) || r > r_out_) return 0.0;
    const double x = std::sqrt(r);
    const double denom_poly = x * x * x - 3.0 * x + 2.0 * a_;
    if (denom_poly <= 0.0) return 0.0; // inside the photon region — no orbits

    auto term = [&](double xi, double xj, double xk) {
        // At a* = 0 the root x₂ = 0 and its coefficient 3(x₂−a)²/x₂ → 0;
        // guard the 0/0 (the omitted term is O(a) ≤ 1e-10 at the threshold).
        if (std::abs(xi) < 1e-10) return 0.0;
        return 3.0 * (xi - a_) * (xi - a_) /
               (xi * (xi - xj) * (xi - xk)) *
               std::log((x - xi) / (x0_ - xi));
    };
    const double bracket = x - x0_ - 1.5 * a_ * std::log(x / x0_) -
                           term(x1_, x2_, x3_) - term(x2_, x1_, x3_) -
                           term(x3_, x1_, x2_);
    const double f = bracket / (x * x * denom_poly);
    // Page-Thorne F(r) ∝ f(x)/x² ... full radial dependence: F ∝ f(x)/x²
    // with f as assembled above already carrying 1/(x²(x³−3x+2a)); the extra
    // 1/x² makes F ∝ r^{-3} × (finite-ISCO factor) in the Newtonian limit.
    return std::max(0.0, f / (x * x));
}

double NovikovThorneDisk::fourth_root_flux(double r) const {
    return std::pow(flux(r), 0.25);
}

std::array<float, 3> NovikovThorneDisk::blackbody_rgb(double T) const {
    const double lt = std::log(std::clamp(T, 501.0, 59999.0));
    const double u = (lt - lut_logT_min_) / (lut_logT_max_ - lut_logT_min_) *
                     (kLutN - 1);
    const int i0 = std::clamp(static_cast<int>(u), 0, kLutN - 2);
    const float w = static_cast<float>(u - i0);
    std::array<float, 3> out;
    for (int c = 0; c < 3; ++c)
        out[c] = bb_lut_[i0][c] * (1.0f - w) + bb_lut_[i0 + 1][c] * w;
    return out;
}

namespace {
// 3 Ṁ c⁶ / (8π G² M²) in W m⁻² (the Page-Thorne prefactor with G, c restored).
double nt_prefactor_si(double mass_msun, double mdot_msun_yr) {
    using namespace constants;
    const double M = mass_msun * M_sun_kg;
    const double mdot = mdot_msun_yr * M_sun_kg / yr_s;
    const double c3 = c_si * c_si * c_si;
    return 3.0 * mdot * c3 * c3 / (8.0 * kPi * G_si * G_si * M * M);
}
} // namespace

double nt_flux_si(double mass_msun, double mdot_msun_yr, double a_star,
                  double r_out, double r) {
    const double r_in = KerrMetric(1.0, a_star).r_isco(true);
    const NovikovThorneDisk disk(a_star, r_in, r_out);
    return nt_prefactor_si(mass_msun, mdot_msun_yr) * disk.flux(r);
}

double nt_peak_temperature_k(double mass_msun, double mdot_msun_yr,
                             double a_star, double r_out) {
    const double r_in = KerrMetric(1.0, a_star).r_isco(true);
    const NovikovThorneDisk disk(a_star, r_in, r_out);
    // Same 2000-step scan as the constructor's T_scale, refined by golden-
    // section search around the best sample (the maximum is smooth).
    double best_r = r_in, best_f = 0.0;
    for (int i = 1; i <= 2000; ++i) {
        const double r = r_in + (r_out - r_in) * i / 2000.0;
        const double f = disk.flux(r);
        if (f > best_f) { best_f = f; best_r = r; }
    }
    const double dr = (r_out - r_in) / 2000.0;
    double lo = std::max(r_in, best_r - dr), hi = std::min(r_out, best_r + dr);
    const double gr = 0.5 * (std::sqrt(5.0) - 1.0);
    for (int it = 0; it < 80; ++it) {
        const double m1 = hi - gr * (hi - lo), m2 = lo + gr * (hi - lo);
        if (disk.flux(m1) < disk.flux(m2)) lo = m1; else hi = m2;
    }
    best_f = std::max(best_f, disk.flux(0.5 * (lo + hi)));
    const double F = nt_prefactor_si(mass_msun, mdot_msun_yr) * best_f;
    return std::pow(F / constants::sigma_sb_si, 0.25);
}

std::array<float, 3> NovikovThorneDisk::shade(double r, double g) const {
    const double T = temperature(r);
    if (T < 505.0) return {0.f, 0.f, 0.f};
    // Observed blackbody at g·T: colour and the full g⁴ beaming in one step.
    return blackbody_rgb(g * T);
}

} // namespace sgr
