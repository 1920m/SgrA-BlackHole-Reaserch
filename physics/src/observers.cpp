#include "sgr_physics/observers.hpp"

#include <algorithm>
#include <cmath>

namespace sgr {

std::array<double, 3> observer_velocity(const KerrMetric& m, ObserverKind kind,
                                        double r, double th,
                                        const std::array<double, 3>& rates) {
    const Metric4<double> g = m.components(r, th);
    const double gpp = g(3, 3), gtp = g(0, 3);
    const double omega = -gtp / gpp;
    const double alpha = std::sqrt(-(g(0, 0) - gtp * gtp / gpp));
    const double M = m.mass(), a = m.spin();
    std::array<double, 3> v{};
    switch (kind) {
        case ObserverKind::Orbiting: {
            const double Om = m.omega_circular(r, true);
            v[2] = (Om - omega) * std::sqrt(gpp) / alpha;
            break;
        }
        case ObserverKind::Infall: {
            const double st = std::sin(th);
            const double delta = r * r - 2.0 * M * r + a * a;
            const double A = (r * r + a * a) * (r * r + a * a) - a * a * delta * st * st;
            v[0] = -std::sqrt(2.0 * M * r * (r * r + a * a) / A);
            break;
        }
        case ObserverKind::Hovering:
            v[2] = -omega * std::sqrt(gpp) / alpha;
            break;
        case ObserverKind::Moving:
            v[0] = std::sqrt(g(1, 1)) * rates[0] / alpha;
            v[1] = std::sqrt(g(2, 2)) * rates[1] / alpha;
            v[2] = std::sqrt(gpp) * (rates[2] - omega) / alpha;
            break;
        case ObserverKind::Zamo:
        default:
            break;
    }
    return v;
}

ObserverFrame observer_frame(const KerrMetric& m, double r, double th,
                             const std::array<double, 3>& v_in) {
    ObserverFrame f;
    const Metric4<double> g = m.components(r, th);
    const double gpp = g(3, 3), gtp = g(0, 3);
    f.omega = -gtp / gpp;
    f.alpha = std::sqrt(-(g(0, 0) - gtp * gtp / gpp));
    const Vec4 et{1.0 / f.alpha, 0.0, 0.0, f.omega / f.alpha};
    const Vec4 ei[3] = {{0.0, 1.0 / std::sqrt(g(1, 1)), 0.0, 0.0},
                        {0.0, 0.0, 1.0 / std::sqrt(g(2, 2)), 0.0},
                        {0.0, 0.0, 0.0, 1.0 / std::sqrt(gpp)}};

    std::array<double, 3> v = v_in;
    double v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    f.valid = v2 < 1.0;
    if (!f.valid) {
        v = {0.0, 0.0, 0.0};
        v2 = 0.0;
    }
    f.v = v;
    f.speed = std::sqrt(v2);
    f.gamma = 1.0 / std::sqrt(1.0 - v2);

    Vec4 ve{};
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 4; ++k) ve[k] += v[i] * ei[i][k];
    for (int k = 0; k < 4; ++k) f.e[0][k] = f.gamma * (et[k] + ve[k]);
    for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 4; ++k)
            f.e[j + 1][k] = ei[j][k] +
                            (v2 > 0.0 ? (f.gamma - 1.0) * v[j] / v2 * ve[k] : 0.0) +
                            f.gamma * v[j] * et[k];
    return f;
}

KSRay ks_observer_ray(const KerrSchild& ks, double r, double th,
                      const ObserverFrame& f, const std::array<double, 3>& n) {
    Vec4 v_bl{};
    for (int k = 0; k < 4; ++k)
        v_bl[k] = -f.e[0][k] + n[0] * f.e[1][k] + n[1] * f.e[2][k] + n[2] * f.e[3][k];
    KSRay ray;
    ray.X = ks.point_from_bl(0.0, r, th, 0.0);
    const Vec4 v = ks.vector_from_bl(r, th, 0.0, v_bl);
    const Metric4<double> gk = ks.metric(ray.X);
    for (int mu = 0; mu < 4; ++mu) {
        ray.P[mu] = 0.0;
        for (int k = 0; k < 4; ++k) ray.P[mu] += gk(mu, k) * v[k];
    }
    return ray;
}

} // namespace sgr
