#include "spectral_fitter/SpectralLineFitter.hpp"

#include <cmath>

#include "sgr_physics/constants.hpp"

namespace sgr {

std::vector<SpectralLineFitter::Sample> SpectralLineFitter::predict(
    double t0_yr, double t1_yr, int n) const {
    std::vector<Sample> out;
    out.reserve(n);
    const double c_kms = constants::c_si / 1000.0;
    for (int i = 0; i < n; ++i) {
        const double t = t0_yr + (t1_yr - t0_yr) * i / (n - 1);
        const auto z = model_->redshift_terms(t); // {z_tot, β_los, z_trans, z_grav}
        Sample s;
        s.t_yr = t;
        s.v_total_kms = z[0] * c_kms;
        s.v_newton_kms = z[1] * c_kms;
        s.v_excess_kms = (z[0] - z[1]) * c_kms;
        s.v_trans_kms = z[2] * c_kms;
        s.v_grav_kms = z[3] * c_kms;
        s.lambda_obs_um = kBrGammaUm * (1.0 + z[0]);
        out.push_back(s);
    }
    return out;
}

std::array<double, 2> SpectralLineFitter::peak_excess(double t0_yr,
                                                      double t1_yr) const {
    double best_v = -1e30, best_t = t0_yr;
    for (int i = 0; i <= 4000; ++i) {
        const double t = t0_yr + (t1_yr - t0_yr) * i / 4000.0;
        const auto z = model_->redshift_terms(t);
        const double v = (z[0] - z[1]) * constants::c_si / 1000.0;
        if (v > best_v) {
            best_v = v;
            best_t = t;
        }
    }
    return {best_t, best_v};
}

} // namespace sgr
