#pragma once
// Hot accretion flow of Sgr A* at 230 GHz (Gate 19): a semi-analytic
// radiatively inefficient accretion flow (RIAF), thermal synchrotron
// radiative transfer along Kerr-Schild rays, normalised to the measured
// flux density.
//
// Flow (Broderick & Loeb 2006, MNRAS 367, 905, eq. 4 and Table 1; after
// Yuan, Quataert & Narayan 2003), with ρ the cylindrical radius and z the
// height (BL r sinθ, r cosθ), M = 1:
//   n_e = n0 (ρ/M)^-1.1 exp(-z²/2ρ²)
//   T_e = T0 (r/M)^-0.84,        T0 = 1.5e11 K  (their a* = 0.998 model)
//   B   : toroidal, B²/8π = β⁻¹ n_e m_p c² r_S/(12 r), β = 10
//         (Broderick et al. 2011, ApJ 735, 110, eq. 4)
//   u   : Keplerian circulation Ω = 1/(ρ^{3/2} + a) outside the ISCO;
//         inside it, ballistic plunge with the ISCO's E and L.
// Only thermal electrons are modelled (Broderick et al. attribute roughly
// 30 % of the 1.3 mm flux to a non-thermal population; omitted, stated).
//
// Emissivity: the thermal-synchrotron fit of Leung, Gammie & Noble (2011),
// as implemented in grmonty (Dolence et al. 2009), cgs:
//   j_ν = √2 π e² n_e ν_s / (3 c K₂(1/Θ_e)) · (X^{1/2} + 2^{11/12} X^{1/6})² e^{-X^{1/3}}
//   X = ν/ν_s,  ν_s = (2/9) ν_c Θ_e² sin θ_B,  ν_c = eB/(2π m_e c),  Θ_e = kT_e/m_e c²
// θ_B: angle between the photon and the field in the fluid frame.
// Absorption from Kirchhoff's law, α_ν = j_ν / B_ν(T_e).
//
// Transfer. Along a backward ray normalised so the camera measures unit
// frequency, the fluid sees ε = P·u (ν_f = ν_obs ε) and the proper length
// per affine step is ε dλ. For a thermal source the formal solution is
//   I_obs/ν_obs³ = Σ_k [B_{ν_f}(T_e)/ν_f³] (1 − e^{−Δτ_k}) e^{−τ_k},
//   Δτ_k = α_{ν_f} ε Δλ r_g,
// composited front-to-back from the camera (I_ν/ν³ invariant).

#include <array>
#include <complex>
#include <vector>

#include "sgr_physics/kerr_schild.hpp"
#include "sgr_physics/metric.hpp"

namespace sgr {

namespace cgs {
inline constexpr double e = 4.80320471e-10;     // esu
inline constexpr double m_e = 9.1093837015e-28; // g
inline constexpr double m_p = 1.67262192369e-24;
inline constexpr double c = 2.99792458e10;      // cm/s
inline constexpr double h = 6.62607015e-27;     // erg s
inline constexpr double k_B = 1.380649e-16;     // erg/K
} // namespace cgs

// Magnetic-field geometry (Gate 20 compares the three; the RIAF default is
// toroidal). The direction is projected orthogonal to the flow and
// normalised: radial ∝ ∂_r, vertical ∝ ∂_z = cosθ ∂_r − (sinθ/r) ∂_θ.
enum class FieldGeometry { Toroidal, Radial, Vertical };

struct RiafParams {
    double a_star = 0.94;
    double n0 = 1.0e7;        // cm^-3 (fitted to the flux density)
    double T0 = 1.5e11;       // K at r = M
    double beta = 10.0;       // gas-to-magnetic pressure (ions)
    double p_n = -1.1, p_T = -0.84;
    double rg_cm = 0.0;       // GM/c² of the black hole [cm]
    FieldGeometry field = FieldGeometry::Toroidal;
    bool static_flow = false; // tests only: fluid at rest relative to the stars
};

// Plasma state at a point, with 4-vectors in BL components.
struct RiafFluid {
    double n_e = 0.0, T_e = 0.0, B = 0.0; // cm^-3, K, G
    Vec4 u{};                             // 4-velocity (BL)
    Vec4 b{};                             // unit field direction (BL), b·u = 0
    bool valid = false;
};

class RiafModel {
public:
    explicit RiafModel(const RiafParams& p);
    const RiafParams& params() const { return p_; }
    // State at BL (r, θ); r_isco splits Keplerian circulation from plunge.
    RiafFluid fluid(double r, double theta) const;
    // Thermal synchrotron emissivity [erg s^-1 cm^-3 Hz^-1 sr^-1].
    static double j_nu(double nu, double n_e, double T_e, double B, double sin_theta);
    static double planck(double nu, double T);   // B_ν(T), cgs
    double r_isco() const { return r_isco_; }

private:
    RiafParams p_;
    KerrMetric m_;
    double r_isco_, E_isco_, L_isco_;
};

// Front-to-back accumulator of the thermal formal solution (above).
struct TransferAccumulator {
    double inv = 0.0;   // Σ (B/ν³)(1 − e^{−Δτ}) e^{−τ}
    double tau = 0.0;   // optical depth from the camera
    void add(double nu_f, double T_e, double alpha, double dl_cm);
    double intensity(double nu_obs) const { return inv * nu_obs * nu_obs * nu_obs; }
};

struct RiafCamera {
    double r_cam = 1000.0;
    double incl_deg = 30.0;
    double half_width_M = 12.0;
    int n = 160;
    double nu_obs = 230e9;   // Hz
    double r_max = 60.0;     // outer edge of the emitting region [M]
};

struct RiafImage {
    int n = 0;
    double half_width_M = 0.0;
    std::vector<double> I;   // I_ν [erg s^-1 cm^-2 Hz^-1 sr^-1], row 0 = top
    // Gate 20 (when requested): linear polarisation, sky convention — EVPA
    // measured from image-up toward image-left (north through east with
    // north up, east left), P = Q + iU = |P| e^{2iχ}.
    std::vector<double> Q, U;
    double max_dH = 0.0;
};
// pol_fraction > 0: also transport the polarisation (Gate 20) with a
// constant intrinsic linear polarisation fraction of the emission.
RiafImage riaf_image(const RiafModel& model, const RiafCamera& cam,
                     double pol_fraction = 0.0);

// ---- Gate 20: polarisation transport -------------------------------------
// Walker & Penrose (1970) constant of a null geodesic k with a parallel-
// transported vector f orthogonal to it (BL contravariant components; e.g.
// Connors, Piran & Stark 1980; Himwich et al. 2020):
//   κ = (A − iB)(r − i a cosθ),
//   A = (k^t f^r − k^r f^t) + a sin²θ (k^r f^φ − k^φ f^r),
//   B = [(r² + a²)(k^φ f^θ − k^θ f^φ) − a (k^t f^θ − k^θ f^t)] sinθ.
// κ is unchanged by f → f + c k, so it fixes the polarisation direction
// anywhere along the ray: at the camera the EVPA is the angle χ for which
// κ(cos χ e_1 + sin χ e_2) is a real multiple of the emission point's κ.
std::complex<double> walker_penrose(double a, double r, double theta, const Vec4& k,
                                    const Vec4& f);
// ε^{μνρσ} a_ν b_ρ c_σ for covariant inputs at BL (r, θ) (√−g = Σ sinθ);
// the synchrotron polarisation of the emission is f ∝ ε(u, k, b).
Vec4 levi_civita_cross(double a, double r, double theta, const Vec4& a_low,
                       const Vec4& b_low, const Vec4& c_low);

// Total flux density [Jy] of an image at distance D [cm] (pixel solid angle
// (Δb r_g / D)², Δb the pixel size in M).
double riaf_flux_jy(const RiafImage& img, double rg_cm, double dist_cm);

// Azimuthal polarisation modes (Palumbo, Wong & Prather 2020; EHT 2024):
// β_m = Σ P e^{−imφ} / Σ I, φ the pixel's position angle measured like the
// EVPA (from image-up toward image-left). ∠β2 = 0: radial EVPA pattern,
// ±180°: toroidal; rotation-invariant, conjugated by a mirror image.
std::complex<double> beta_mode(const std::vector<double>& I, const std::vector<double>& Q,
                               const std::vector<double>& U, int n, double half_width,
                               int m);

} // namespace sgr
