#include "sgr_physics/kerr_schild.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "sgr_physics/integrator.hpp"

namespace sgr {

namespace {
constexpr double kPi = 3.14159265358979323846;

// Everything rhs() needs at a point, computed once.
struct KSLocal {
    double r, f;
    double l[4];      // covariant l_μ (l_t = 1)
    double r_i[3];    // ∂r/∂x^i
    double f_i[3];    // ∂f/∂x^i
    double l_ij[3][3]; // ∂_i l_j (spatial)
};

KSLocal local(double M, double a, int s, const Vec4& X) {
    const double x = X[1], y = X[2], z = X[3];
    KSLocal L{};
    const double rho2 = x * x + y * y + z * z - a * a;
    const double disc = std::sqrt(rho2 * rho2 + 4.0 * a * a * z * z);
    // r² = ½(ρ² + √(ρ⁴ + 4a²z²)), cancellation-free when ρ² < 0
    const double r2 = rho2 >= 0.0 ? 0.5 * (rho2 + disc)
                                  : (disc > 0.0 ? 2.0 * a * a * z * z / (disc - rho2)
                                                : 0.0);
    const double r = std::sqrt(std::max(r2, 0.0));
    L.r = r;
    const double W = r2 * r2 + a * a * z * z;
    const double D = r2 + a * a;
    L.f = 2.0 * M * r2 * r / W;
    const double Nx = r * x + s * a * y, Ny = r * y - s * a * x;
    L.l[0] = 1.0;
    L.l[1] = s * Nx / D;
    L.l[2] = s * Ny / D;
    L.l[3] = s * z / r;

    const double xs[3] = {x, y, z};
    for (int i = 0; i < 3; ++i)
        L.r_i[i] = r * (r2 * xs[i] + (i == 2 ? a * a * z : 0.0)) / W;
    for (int i = 0; i < 3; ++i) {
        const double dW = 4.0 * r2 * r * L.r_i[i] + (i == 2 ? 2.0 * a * a * z : 0.0);
        L.f_i[i] = 2.0 * M * (3.0 * r2 * L.r_i[i] * W - r2 * r * dW) / (W * W);
        const double dD = 2.0 * r * L.r_i[i];
        const double dNx = L.r_i[i] * x + (i == 0 ? r : 0.0) + (i == 1 ? s * a : 0.0);
        const double dNy = L.r_i[i] * y + (i == 1 ? r : 0.0) - (i == 0 ? s * a : 0.0);
        L.l_ij[i][0] = s * (dNx * D - Nx * dD) / (D * D);
        L.l_ij[i][1] = s * (dNy * D - Ny * dD) / (D * D);
        L.l_ij[i][2] = s * ((i == 2 ? r : 0.0) - z * L.r_i[i]) / r2;
    }
    return L;
}
} // namespace

KerrSchild::KerrSchild(double M, double a_star, int chart_sign)
    : M_(M), a_(a_star * M), s_(chart_sign >= 0 ? 1 : -1) {
    if (std::abs(a_star) >= 1.0)
        throw std::invalid_argument("KerrSchild: |a*| must be < 1");
    const double d = std::sqrt(M_ * M_ - a_ * a_);
    rp_ = M_ + d;
    rm_ = M_ - d;
}

double KerrSchild::radius(double x, double y, double z) const {
    return local(M_, a_, s_, Vec4{0.0, x, y, z}).r;
}

Metric4<double> KerrSchild::metric(const Vec4& X) const {
    const KSLocal L = local(M_, a_, s_, X);
    Metric4<double> g{};
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 4; ++n)
            g(m, n) = (m == n ? (m == 0 ? -1.0 : 1.0) : 0.0) + L.f * L.l[m] * L.l[n];
    return g;
}

Metric4<double> KerrSchild::inverse_metric(const Vec4& X) const {
    const KSLocal L = local(M_, a_, s_, X);
    const double lu[4] = {-L.l[0], L.l[1], L.l[2], L.l[3]};
    Metric4<double> g{};
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 4; ++n)
            g(m, n) = (m == n ? (m == 0 ? -1.0 : 1.0) : 0.0) - L.f * lu[m] * lu[n];
    return g;
}

double KerrSchild::hamiltonian(const Vec4& X, const Vec4& P) const {
    const KSLocal L = local(M_, a_, s_, X);
    const double lp = -P[0] + L.l[1] * P[1] + L.l[2] * P[2] + L.l[3] * P[3];
    const double eta = -P[0] * P[0] + P[1] * P[1] + P[2] * P[2] + P[3] * P[3];
    return 0.5 * (eta - L.f * lp * lp);
}

void KerrSchild::rhs(const Vec4& X, const Vec4& P, Vec4& dX, Vec4& dP) const {
    const KSLocal L = local(M_, a_, s_, X);
    const double lp = -P[0] + L.l[1] * P[1] + L.l[2] * P[2] + L.l[3] * P[3];
    dX[0] = -P[0] + L.f * lp;               // η^tt p_t − f l^t (l·p), l^t = −1
    for (int i = 1; i < 4; ++i) dX[i] = P[i] - L.f * L.l[i] * lp;
    dP[0] = 0.0;                            // stationary: E conserved exactly
    for (int i = 0; i < 3; ++i) {
        double dlp = 0.0;
        for (int j = 0; j < 3; ++j) dlp += P[j + 1] * L.l_ij[i][j];
        dP[i + 1] = 0.5 * L.f_i[i] * lp * lp + L.f * lp * dlp;
    }
}

double KerrSchild::T_shift(double r) const {
    const double d = rp_ - rm_;
    return 2.0 * M_ / d *
           (rp_ * std::log(std::abs(r - rp_)) - rm_ * std::log(std::abs(r - rm_)));
}

double KerrSchild::Phi_shift(double r) const {
    if (a_ == 0.0) return 0.0;
    return a_ / (rp_ - rm_) * std::log(std::abs((r - rp_) / (r - rm_)));
}

Vec4 KerrSchild::point_from_bl(double t, double r, double th, double ph) const {
    const double pk = ph + s_ * Phi_shift(r);
    const double st = std::sin(th), ct = std::cos(th);
    const double cp = std::cos(pk), sp = std::sin(pk);
    return {t + s_ * T_shift(r), (r * cp - s_ * a_ * sp) * st,
            (r * sp + s_ * a_ * cp) * st, r * ct};
}

std::array<Vec4, 4> KerrSchild::jacobian_bl_to_ks(double r, double th,
                                                  double ph) const {
    const double pk = ph + s_ * Phi_shift(r);
    const double st = std::sin(th), ct = std::cos(th);
    const double cp = std::cos(pk), sp = std::sin(pk);
    const double delta = r * r - 2.0 * M_ * r + a_ * a_;
    const double dpk_dr = s_ * a_ / delta;
    std::array<Vec4, 4> J{};
    J[0] = {1.0, s_ * 2.0 * M_ * r / delta, 0.0, 0.0};
    const double xr = r * cp - s_ * a_ * sp, yr = r * sp + s_ * a_ * cp;
    const double dx_dpk = (-r * sp - s_ * a_ * cp) * st;
    const double dy_dpk = (r * cp - s_ * a_ * sp) * st;
    J[1] = {0.0, cp * st + dx_dpk * dpk_dr, xr * ct, dx_dpk};
    J[2] = {0.0, sp * st + dy_dpk * dpk_dr, yr * ct, dy_dpk};
    J[3] = {0.0, ct, -r * st, 0.0};
    return J;
}

Vec4 KerrSchild::vector_from_bl(double r, double th, double ph,
                                const Vec4& v) const {
    const auto J = jacobian_bl_to_ks(r, th, ph);
    Vec4 out{};
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 4; ++n) out[m] += J[m][n] * v[n];
    return out;
}

Vec4 KerrSchild::bl_coords(const Vec4& X) const {
    const double r = radius(X[1], X[2], X[3]);
    const double th = std::acos(std::clamp(X[3] / r, -1.0, 1.0));
    const double pk = std::atan2(X[2], X[1]) - std::atan2(s_ * a_, r);
    return {X[0] - s_ * T_shift(r), r, th, pk - s_ * Phi_shift(r)};
}

Vec4 KerrSchild::covector_to_bl(const Vec4& X, const Vec4& P) const {
    const Vec4 b = bl_coords(X);
    const auto J = jacobian_bl_to_ks(b[1], b[2], b[3]);
    Vec4 out{};
    for (int n = 0; n < 4; ++n)
        for (int m = 0; m < 4; ++m) out[n] += P[m] * J[m][n];
    return out;
}

double KerrSchild::carter_Q(const Vec4& X, const Vec4& P) const {
    const Vec4 b = bl_coords(X);
    const Vec4 p = covector_to_bl(X, P);
    const double ct = std::cos(b[2]), st = std::sin(b[2]);
    const double E = -p[0], Lz = p[3];
    // Carter's constant for light (μ² = 0) and for unit timelike geodesics
    // (μ² = 1, G24 in-fall): μ² = −g^μν p_μ p_ν = −2H.
    const double mu2 = -2.0 * hamiltonian(X, P);
    return p[2] * p[2] + ct * ct * (Lz * Lz / (st * st) + a_ * a_ * (mu2 - E * E));
}

KSRay ks_camera_ray(const KerrSchild& ks, double r_cam, double th_cam,
                    const std::array<double, 3>& n) {
    const KerrMetric bl(ks.M(), ks.a() / ks.M());
    const Metric4<double> g = bl.components(r_cam, th_cam);
    const double omega = -g(0, 3) / g(3, 3);
    const double alpha = std::sqrt(-(g(0, 0) - g(0, 3) * g(0, 3) / g(3, 3)));
    // −k_received = −e_t + n·e in BL components (see header).
    const Vec4 v_bl = {-1.0 / alpha, n[0] / std::sqrt(g(1, 1)),
                       n[1] / std::sqrt(g(2, 2)),
                       -omega / alpha + n[2] / std::sqrt(g(3, 3))};
    KSRay ray;
    ray.X = ks.point_from_bl(0.0, r_cam, th_cam, 0.0);
    const Vec4 v = ks.vector_from_bl(r_cam, th_cam, 0.0, v_bl);
    const Metric4<double> gk = ks.metric(ray.X);
    for (int m = 0; m < 4; ++m) {
        ray.P[m] = 0.0;
        for (int k = 0; k < 4; ++k) ray.P[m] += gk(m, k) * v[k];
    }
    return ray;
}

KSRay ks_frame_ray(const KerrSchild& ks, const Vec4& X, const std::array<Vec4, 4>& e,
                   const std::array<double, 3>& n) {
    Vec4 v{};
    for (int m = 0; m < 4; ++m)
        v[m] = -e[0][m] + n[0] * e[1][m] + n[1] * e[2][m] + n[2] * e[3][m];
    const Metric4<double> g = ks.metric(X);
    KSRay ray;
    ray.X = X;
    for (int m = 0; m < 4; ++m) {
        ray.P[m] = 0.0;
        for (int k = 0; k < 4; ++k) ray.P[m] += g(m, k) * v[k];
    }
    return ray;
}

KSTraceResult ks_trace(const KerrSchild& ks, const KSRay& ray,
                       const KSTraceOptions& opt) {
    using State = StateVec<8>;
    auto f = [&ks](double, const State& y, State& dy) {
        const Vec4 X = {y[0], y[1], y[2], y[3]}, P = {y[4], y[5], y[6], y[7]};
        Vec4 dX, dP;
        ks.rhs(X, P, dX, dP);
        for (int i = 0; i < 4; ++i) { dy[i] = dX[i]; dy[i + 4] = dP[i]; }
    };
    DormandPrince54<8, decltype(f)> rk(f, opt.atol, opt.rtol);
    DormandPrince54<8, decltype(f)> rkb(f, opt.atol, opt.rtol);

    State y{};
    for (int i = 0; i < 4; ++i) { y[i] = ray.X[i]; y[i + 4] = ray.P[i]; }
    auto X_of = [](const State& s) { return Vec4{s[0], s[1], s[2], s[3]}; };
    auto P_of = [](const State& s) { return Vec4{s[4], s[5], s[6], s[7]}; };

    const double L0 = ks.angular_momentum(ray.X, ray.P);
    const double Q0 = ks.carter_Q(ray.X, ray.P);
    const double E0 = ks.energy(ray.P);
    const double scaleL = std::max(std::abs(L0), std::abs(E0) * ks.M());
    const double scaleQ = std::max(std::abs(Q0), E0 * E0 * ks.M() * ks.M());

    KSTraceResult out;
    double lam = 0.0;
    double h = 1e-3 * ks.radius(ray.X[1], ray.X[2], ray.X[3]);
    for (int step = 0; step < opt.max_steps; ++step) {
        const State prev = y;
        const double lam_prev = lam;
        const auto res = rk.try_step(lam, y, h);
        h = res.h_next;
        if (!res.accepted) continue;

        const Vec4 X = X_of(y), P = P_of(y);
        const double r = ks.radius(X[1], X[2], X[3]);
        out.max_dL_rel = std::max(out.max_dL_rel,
                                  std::abs(ks.angular_momentum(X, P) - L0) / scaleL);
        if (r > ks.r_plus() * 1.05)   // Q's BL reconstruction is ill-conditioned at Δ → 0
            out.max_dQ_rel = std::max(out.max_dQ_rel,
                                      std::abs(ks.carter_Q(X, P) - Q0) / scaleQ);
        out.max_H_rel = std::max(out.max_H_rel,
                                 std::abs(ks.hamiltonian(X, P)) / (E0 * E0));

        // equatorial crossing z = 0 (bisected on the step fraction)
        if (prev[3] * y[3] < 0.0 && out.n_cross < opt.max_crossings) {
            double lo = 0.0, hi = res.h_used;
            State cr = y;
            for (int it = 0; it < 60; ++it) {
                const double mid = 0.5 * (lo + hi);
                State trial = prev;
                double t0 = lam_prev;
                rkb.try_step(t0, trial, mid); // accepted at these tolerances
                if (trial[3] * prev[3] > 0.0) lo = mid; else hi = mid;
                cr = trial;
            }
            const Vec4 b = ks.bl_coords(X_of(cr));
            out.cross_r[out.n_cross] = b[1];
            out.cross_phi_bl[out.n_cross] = b[3];
            ++out.n_cross;
        }
        // Capture = the backward ray reaches the PAST horizon. The outgoing
        // chart crosses it regularly (r < r₊). The ingoing chart (G24
        // in-fall cameras) crosses the FUTURE horizon instead: backward rays
        // from inside leave outward through it, and a backward ray heading
        // inward from outside is bound for the past horizon, reached in this
        // chart only as t → −∞. Same rule as the live shader.
        const double r_prev = ks.radius(prev[1], prev[2], prev[3]);
        if (ks.chart() < 0 ? r < ks.r_plus() : (r < ks.r_plus() + 0.02 && r < r_prev)) {
            out.outcome = KSOutcome::Captured;
            break;
        }
        if (r > opt.r_escape) {
            Vec4 dX, dP;
            ks.rhs(X, P, dX, dP);
            if (X[1] * dX[1] + X[2] * dX[2] + X[3] * dX[3] > 0.0) {
                // Far out and receding: the backward ray now runs straight
                // toward its source, so its direction of motion IS the sky
                // direction the light came from (no r_escape parallax).
                const double n =
                    std::sqrt(dX[1] * dX[1] + dX[2] * dX[2] + dX[3] * dX[3]);
                out.esc_theta = std::acos(std::clamp(dX[3] / n, -1.0, 1.0));
                out.esc_phi = std::atan2(dX[2], dX[1]);
                out.outcome = KSOutcome::Escaped;
                break;
            }
        }
    }
    out.final_state = {X_of(y), P_of(y)};
    return out;
}

} // namespace sgr
