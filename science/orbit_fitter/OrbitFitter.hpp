#pragma once
// Ceres-based nonlinear least squares fit of the S2 orbit.
//
// χ² = Σ_i [(RA_i − RA_model)²/σ² + (Dec_i − Dec_model)²/σ²]
//    + Σ_j [(vr_j − vr_model)²/σ²]
//    + coordinate-system priors (Gillessen 2017 §2.3 / Plewa 2015):
//      (x0, y0) = (0,0) ± 0.2 mas at 2009.0, (vx, vy) = (0,0) ± 0.1 mas/yr,
//      vz = 0 ± 5 km/s.
//
// The model is a full GR geodesic (S2Model); Jacobians via central numeric
// differentiation (the model is an ODE integration — no autodiff through it).

#include <string>
#include <vector>

#include "orbit_fitter/SkyProjection.hpp"

namespace sgr {

struct AstrometryPoint {
    double epoch_yr, ra_mas, dec_mas, sig_ra_mas, sig_dec_mas;
};
struct RadialVelocityPoint {
    double epoch_yr, vr_kms, sig_vr_kms;
};

struct S2Dataset {
    std::vector<AstrometryPoint> astrometry;
    std::vector<RadialVelocityPoint> rv;

    static S2Dataset load(const std::string& astrometry_csv,
                          const std::string& rv_csv);
    int n_data() const {
        return 2 * static_cast<int>(astrometry.size()) +
               static_cast<int>(rv.size());
    }
};

struct FitResult {
    S2Parameters params;
    std::array<double, S2Parameters::kNumParams> sigma{}; // 1σ from covariance
    double chi2 = 0.0;
    int n_residuals = 0;   // data residuals (excl. priors)
    int dof = 0;
    double chi2_red = 0.0;
    bool converged = false;
    std::string report;

    // Derived quantities with propagated uncertainty (linearised):
    double period_yr = 0.0, sigma_period_yr = 0.0;
};

class OrbitFitter {
public:
    // a_star/eps_q select the metric for the model orbit.
    OrbitFitter(const S2Dataset& data, double a_star = 0.0, double eps_q = 0.0)
        : data_(&data), a_star_(a_star), eps_q_(eps_q) {}

    // Run the fit from the given starting point. hold_eps_q applies to the
    // Phase-5 variant (14th parameter); here the 13-parameter fit.
    FitResult fit(const S2Parameters& start) const;

    // Phase 5: 14-parameter fit with the Hartle-Thorne quadrupole deviation
    // ε_q free. Returns the usual result plus ε_q best fit and 1σ.
    FitResult fit_nohair(const S2Parameters& start, double eps_q_start,
                         double* eps_q_best, double* eps_q_sigma) const;

    // χ² at a parameter point (diagnostics).
    double chi2(const S2Parameters& p) const;

    // Parameter covariance at the best fit from an explicit central-difference
    // Jacobian: C = (JᵀJ)⁻¹ for σ-normalised residuals. Fills `sigma` (1σ) and
    // returns the full correlation matrix (row-major kNumParams²). More
    // transparent than the Ceres SVD covariance, which truncates the strongly
    // correlated M–R0 direction to zero variance.
    std::vector<double> covariance(const S2Parameters& best,
                                   std::array<double, S2Parameters::kNumParams>& sigma) const;

private:
    const S2Dataset* data_;
    double a_star_, eps_q_;
};

} // namespace sgr
