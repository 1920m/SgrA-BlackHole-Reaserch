#pragma once
// Brγ spectral-line model for S2 (GRAVITY 2018, A&A 615 L15).
//
// The observed line centroid of a photospheric line emitted at rest wavelength
// λ_rest follows λ_obs(t) = λ_rest (1 + z_tot(t)) with
//   1 + z_tot = γ (1 − β·n̂) / α        (S2Model's exact ZAMO-frame redshift)
// which decomposes to first PN order as z_tot ≈ β_los + β²/2 + M/r c⁻²-terms:
// the "GR excess" over a purely Newtonian (β_los-only) orbit is
//   Δv_GR(t) = c (z_tot − β_los) ≈ c(z_transverse + z_gravitational),
// peaking at ≈ +200 km/s at the 2018.38 pericenter passage — the GRAVITY 2018
// detection (their redshift parameter f = 0.90 ± 0.17, f = 1 ⇔ GR).

#include <array>
#include <string>
#include <vector>

#include "orbit_fitter/SkyProjection.hpp"

namespace sgr {

class SpectralLineFitter {
public:
    static constexpr double kBrGammaUm = 2.16612; // µm, rest wavelength

    explicit SpectralLineFitter(const S2Model& model) : model_(&model) {}

    struct Sample {
        double t_yr;
        double lambda_obs_um;   // predicted Brγ centroid
        double v_total_kms;     // c·z_tot
        double v_newton_kms;    // c·β_los (Keplerian Doppler only)
        double v_excess_kms;    // GR excess: v_total − v_newton
        double v_trans_kms;     // c·(γ−1)
        double v_grav_kms;      // c·(1/α − 1)
    };

    // Predict the line curve over [t0, t1] with n samples.
    std::vector<Sample> predict(double t0_yr, double t1_yr, int n) const;

    // Peak GR excess and its epoch within [t0, t1] (fine-sampled).
    std::array<double, 2> peak_excess(double t0_yr, double t1_yr) const;

private:
    const S2Model* model_;
};

} // namespace sgr
