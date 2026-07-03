#include "sgr_physics/timelike_geodesics.hpp"

#include <cmath>
#include <stdexcept>

namespace sgr {

void TimelikeGeodesic::rhs(const TimelikeState& y, TimelikeState& dy) const {
    const double r = y[1];
    double theta = y[2];
    constexpr double pi = 3.14159265358979323846;
    if (theta < 1e-8) theta = 1e-8;
    if (theta > pi - 1e-8) theta = pi - 1e-8;

    const Christoffel gam = christoffel(*metric_, r, theta);

    const double u[4] = {y[4], y[5], y[6], y[7]};
    dy[0] = u[0];
    dy[1] = u[1];
    dy[2] = u[2];
    dy[3] = u[3];
    for (int lam = 0; lam < 4; ++lam) {
        double acc = 0.0;
        for (int mu = 0; mu < 4; ++mu)
            for (int nu = 0; nu < 4; ++nu)
                acc += gam[lam][mu][nu] * u[mu] * u[nu];
        dy[4 + lam] = -acc;
    }
}

double TimelikeGeodesic::solve_ut(const Metric4<double>& g, double ur,
                                  double utheta, double uphi) {
    // g_tt (u^t)² + 2(g_tφ u^φ) u^t + [g_rr u^r² + g_θθ u^θ² + g_φφ u^φ² + 1] = 0
    const double A = g(0, 0);
    const double B = 2.0 * g(0, 3) * uphi;
    const double C = g(1, 1) * ur * ur + g(2, 2) * utheta * utheta +
                     g(3, 3) * uphi * uphi + 1.0;
    const double disc = B * B - 4.0 * A * C;
    if (disc < 0.0)
        throw std::domain_error("solve_ut: spacelike initial data (disc < 0)");
    // g_tt < 0 outside the ergosphere; the future-directed root has u^t > 0.
    const double root1 = (-B + std::sqrt(disc)) / (2.0 * A);
    const double root2 = (-B - std::sqrt(disc)) / (2.0 * A);
    return std::max(root1, root2) > 0.0 ? std::max(root1, root2)
                                        : std::min(root1, root2);
}

double TimelikeGeodesic::energy(const TimelikeState& y) const {
    const Metric4<double> g = metric_->components(y[1], y[2]);
    return -(g(0, 0) * y[4] + g(0, 3) * y[7]);
}

double TimelikeGeodesic::angmom(const TimelikeState& y) const {
    const Metric4<double> g = metric_->components(y[1], y[2]);
    return g(0, 3) * y[4] + g(3, 3) * y[7];
}

double TimelikeGeodesic::norm(const TimelikeState& y) const {
    const Metric4<double> g = metric_->components(y[1], y[2]);
    const double u[4] = {y[4], y[5], y[6], y[7]};
    double n = 0.0;
    for (int mu = 0; mu < 4; ++mu)
        for (int nu = 0; nu < 4; ++nu) n += g(mu, nu) * u[mu] * u[nu];
    return n;
}

double TimelikeGeodesic::carter_q(const TimelikeState& y, double a) const {
    // Q = p_θ² + cos²θ [a²(μ² − E²) + L_z²/sin²θ],  μ = 1 (timelike)
    const double theta = y[2];
    const double st = std::sin(theta), ct = std::cos(theta);
    const Metric4<double> g = metric_->components(y[1], theta);
    const double p_theta = g(2, 2) * y[6];
    const double E = energy(y);
    const double Lz = angmom(y);
    return p_theta * p_theta +
           ct * ct * (a * a * (1.0 - E * E) + Lz * Lz / (st * st));
}

void TimelikeStepper::step(double& tau, TimelikeState& y, double& h) const {
    auto rhs_fn = [this](double, const TimelikeState& s, TimelikeState& d) {
        geo->rhs(s, d);
    };
    DormandPrince54<8, decltype(rhs_fn)> rk(rhs_fn, atol, rtol);
    for (int attempt = 0; attempt < 60; ++attempt) {
        const StepResult<8> res = rk.try_step(tau, y, h);
        h = res.h_next;
        if (res.accepted) return;
    }
    throw std::runtime_error("TimelikeStepper: step size collapsed (60 rejections)");
}

} // namespace sgr
