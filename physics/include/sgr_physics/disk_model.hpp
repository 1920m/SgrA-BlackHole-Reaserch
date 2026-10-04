#pragma once
// Novikov-Thorne thin-disk emission model (Novikov & Thorne 1973), using the
// closed-form time-averaged flux of Page & Thorne (1974), eq. (15n):
//
//   F(r) = (3 Ṁ / 8π M²) · f(x) / x²  with x = √(r/M),
//
//   f(x) = 1/(x²(x³ − 3x + 2a*)) · [ x − x₀ − (3/2) a* ln(x/x₀)
//          − Σ_{i=1..3} (3(x_i − a*)²)/(x_i (x_i − x_j)(x_i − x_k)) ln((x−x_i)/(x₀−x_i)) ]
//
// where x₀ = √(r_isco/M) and x₁, x₂, x₃ are the roots of x³ − 3x + 2a* = 0:
//   x₁ = 2cos((arccos a* − π)/3), x₂ = 2cos((arccos a* + π)/3),
//   x₃ = −2cos(arccos a* /3).
//
// The overall Ṁ normalisation is arbitrary for imaging; the model exposes a
// *relative* flux and an effective temperature T_eff(r) = T_scale · F^{1/4},
// with T_scale chosen so that max_r T_eff = T_peak (a display parameter).
//
// Shading: a blackbody redshifted by g is exactly a blackbody at g·T
// (I_ν/ν³ invariance), so observed radiance ∝ (g·T_eff)⁴ — the g⁴ Doppler
// beaming falls out identically, and the colour comes from
// the Planck spectrum at g·T_eff through the CIE 1931 observer.

#include <array>
#include <vector>

namespace sgr {

class NovikovThorneDisk {
public:
    // a_star: dimensionless spin; r_in/r_out in units of M (r_in usually ISCO);
    // T_peak: peak effective temperature in kelvin (display scale).
    NovikovThorneDisk(double a_star, double r_in, double r_out,
                      double T_peak = 6300.0);

    // Relative Page-Thorne flux (arbitrary normalisation, 0 at/below r_in).
    double flux(double r) const;
    // Effective temperature in kelvin.
    double temperature(double r) const { return Tscale_ * fourth_root_flux(r); }
    // Linear-sRGB radiance of a blackbody at temperature T (kelvin), absolute
    // scale normalised so T = 6500 K has luminance ~1. LUT-accelerated.
    std::array<float, 3> blackbody_rgb(double T) const;

    // Shade a disk intersection: radius r, redshift factor g.
    // Returns linear sRGB with the full g⁴ beaming folded in.
    std::array<float, 3> shade(double r, double g) const;

    // Bolometric observed intensity, normalised to (T/6500K)⁴ units:
    // I_obs = g⁴ I_emit ≡ (g·T_eff/6500K)⁴. This is the channel used for the
    // quantitative beaming-asymmetry and sub-ring flux measurements (the RGB
    // shade above is the correct in-band spectral image, which does not scale
    // as g⁴ — only the bolometric intensity does).
    double bolometric(double r, double g) const {
        const double t = g * temperature(r) / 6500.0;
        return t * t * t * t;
    }

    double r_in() const { return r_in_; }
    double r_out() const { return r_out_; }

private:
    double fourth_root_flux(double r) const;

    double a_;
    double r_in_, r_out_;
    double x0_, x1_, x2_, x3_;
    double Tscale_ = 1.0;
    // blackbody LUT: log-spaced in T over [500 K, 60000 K]
    static constexpr int kLutN = 512;
    std::vector<std::array<float, 3>> bb_lut_;
    double lut_logT_min_ = 0.0, lut_logT_max_ = 0.0;
};

// Planck spectrum → linear sRGB via CIE 1931 colour matching (numerical
// integration; used to build the LUT and by the LUT generator tool).
std::array<double, 3> planck_to_linear_srgb(double T_kelvin);
// The same integration as CIE 1931 XYZ (normalised to Y(6500 K) = 1, no
// gamut clipping) — for chromaticities.
std::array<double, 3> planck_to_xyz(double T_kelvin);

// Physical (SI) Novikov-Thorne normalisation. Restoring G and c in the
// Page-Thorne flux above gives
//   F(r) = 3 Ṁ c⁶ / (8π G² M²) · NovikovThorneDisk::flux(r)   [W m⁻²]
// (large-r limit: the Shakura-Sunyaev 3GMṀ/(8πR³)(1 − √(R_in/R))), and the
// local effective temperature follows from σ T⁴ = F. r is in units of M;
// the inner edge is the prograde ISCO of spin a_star.
double nt_flux_si(double mass_msun, double mdot_msun_yr, double a_star,
                  double r_out, double r);
// max over r ∈ (r_isco, r_out] of T_eff, in kelvin. For Sgr A* this is a
// HYPOTHETICAL: the thin, radiatively efficient disk the measured accretion
// rate would power — the real flow is a hot RIAF.
double nt_peak_temperature_k(double mass_msun, double mdot_msun_yr,
                             double a_star, double r_out);

} // namespace sgr
