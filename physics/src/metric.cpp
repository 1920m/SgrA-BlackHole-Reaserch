#include "sgr_physics/metric.hpp"

namespace sgr {

// Christoffel symbols by complex-step differentiation of the metric:
// ∂g/∂x = Im[g(x + ih)]/h with h = 1e-100 is exact to machine precision for
// the analytic (rational/trig/log) component functions used in this project —
// no subtractive cancellation, unlike finite differences. Cross-validated
// against the analytic Schwarzschild symbols in test_schwarzschild_limit.
Christoffel christoffel(const IMetric& metric, double r, double theta) {
    constexpr double h = 1e-100;
    using C = std::complex<double>;

    const Metric4<double> g = metric.components(r, theta);
    const Metric4<C> g_dr = metric.components(C(r, h), C(theta, 0.0));
    const Metric4<C> g_dth = metric.components(C(r, 0.0), C(theta, h));

    // dg[σ][μ][ν] = ∂_σ g_{μν}; only σ = 1 (r) and σ = 2 (θ) are nonzero
    // (stationarity and axisymmetry).
    double dg[4][4][4] = {};
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            dg[1][mu][nu] = g_dr(mu, nu).imag() / h;
            dg[2][mu][nu] = g_dth(mu, nu).imag() / h;
        }
    }

    const Metric4<double> ginv = invert_bl_block(g);

    Christoffel gam{};
    for (int lam = 0; lam < 4; ++lam) {
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = mu; nu < 4; ++nu) {
                double s = 0.0;
                for (int sig = 0; sig < 4; ++sig) {
                    s += ginv(lam, sig) *
                         (dg[mu][sig][nu] + dg[nu][sig][mu] - dg[sig][mu][nu]);
                }
                gam[lam][mu][nu] = 0.5 * s;
                gam[lam][nu][mu] = gam[lam][mu][nu];
            }
        }
    }
    return gam;
}

} // namespace sgr
