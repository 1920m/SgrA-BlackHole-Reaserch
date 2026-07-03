#include "sgr_physics/null_geodesics.hpp"

#include <algorithm>

namespace sgr {

NullRayInit make_null_ray(const KerrMetric& metric, double t, double r,
                          double theta, double phi,
                          const std::array<double, 4>& k) {
    const double M = metric.mass();
    const double a = metric.spin();
    const Metric4<double> g = metric.components(r, theta);

    NullRayConstants c;
    c.M = M;
    c.a = a;
    // E = −k_t = −(g_tt k^t + g_tφ k^φ);  L_z = k_φ = g_tφ k^t + g_φφ k^φ
    c.E = -(g(0, 0) * k[0] + g(0, 3) * k[3]);
    c.Lz = g(0, 3) * k[0] + g(3, 3) * k[3];

    const double ct = std::cos(theta), st = std::sin(theta);
    const double sigma = r * r + a * a * ct * ct;
    const double p_theta = sigma * k[2]; // = g_θθ k^θ = Σ k^θ
    // Null Carter constant: Q = p_θ² + cos²θ (L_z²/sin²θ − a²E²)
    c.Q = p_theta * p_theta +
          ct * ct * (c.Lz * c.Lz / (st * st) - a * a * c.E * c.E);

    NullState y{};
    y[0] = t;
    y[1] = r;
    y[2] = theta;
    y[3] = phi;
    y[4] = sigma * k[1]; // p_r = Σ dr/dλ = dr/dτ (Mino)
    y[5] = p_theta;      // p_θ = Σ dθ/dλ = dθ/dτ (Mino)
    return {c, y};
}

std::array<double, 4> null_four_momentum(const NullRayConstants& c,
                                         const NullState& y) {
    const double r = y[1], theta = y[2];
    const double st = std::sin(theta);
    const double ct = std::cos(theta);
    const double sigma = r * r + c.a * c.a * ct * ct;
    const double delta = r * r - 2.0 * c.M * r + c.a * c.a;
    const double P = c.E * (r * r + c.a * c.a) - c.a * c.Lz;

    // k^μ = dx^μ/dλ = (dx^μ/dτ)/Σ
    std::array<double, 4> k{};
    k[0] = (-c.a * (c.a * c.E * st * st - c.Lz) + (r * r + c.a * c.a) * P / delta) / sigma;
    k[1] = y[4] / sigma;
    k[2] = y[5] / sigma;
    k[3] = (-(c.a * c.E - c.Lz / (st * st)) + c.a * P / delta) / sigma;
    return k;
}

NullConservedErrors null_conserved_errors(const KerrMetric& metric,
                                          const NullRayConstants& c,
                                          const NullState& y) {
    const double r = y[1], theta = y[2];
    const double st = std::sin(theta), ct = std::cos(theta);
    const Metric4<double> g = metric.components(r, theta);
    const std::array<double, 4> k = null_four_momentum(c, y);

    const double E_rec = -(g(0, 0) * k[0] + g(0, 3) * k[3]);
    const double Lz_rec = g(0, 3) * k[0] + g(3, 3) * k[3];
    const double sigma = r * r + c.a * c.a * ct * ct;
    const double p_theta = sigma * k[2];
    const double Q_rec = p_theta * p_theta +
                         ct * ct * (Lz_rec * Lz_rec / (st * st) -
                                    c.a * c.a * E_rec * E_rec);

    double norm = 0.0;
    for (int mu = 0; mu < 4; ++mu)
        for (int nu = 0; nu < 4; ++nu) norm += g(mu, nu) * k[mu] * k[nu];

    const double eM = std::abs(c.E) * metric.mass();

    NullConservedErrors err{};
    err.dE_rel = std::abs(E_rec - c.E) / std::abs(c.E);
    err.dLz_rel = std::abs(Lz_rec - c.Lz) / std::max(std::abs(c.Lz), eM);
    err.dQ_rel = std::abs(Q_rec - c.Q) / std::max(std::abs(c.Q), eM * eM);
    err.null_norm = std::abs(norm) / (c.E * c.E);

    // Constraint residuals of the second-order Mino formulation.
    const double R = radial_potential(c, r);
    const double Th = polar_potential(c, theta);
    const double P = c.E * (r * r + c.a * c.a) - c.a * c.Lz;
    const double scaleR = std::max(P * P, eM * eM * eM * eM);
    err.r_residual = std::abs(y[4] * y[4] - R) / scaleR;
    err.theta_residual =
        std::abs(y[5] * y[5] - Th) / std::max(std::abs(c.Q), eM * eM);
    return err;
}

} // namespace sgr
