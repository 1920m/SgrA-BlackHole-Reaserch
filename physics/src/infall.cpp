#include "sgr_physics/infall.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

#include "sgr_physics/integrator.hpp"
#include "sgr_physics/metric.hpp"

namespace sgr {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
using State = StateVec<8>;

double dot(const Metric4<double>& g, const Vec4& u, const Vec4& v) {
    double s = 0.0;
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 4; ++n) s += g(m, n) * u[m] * v[n];
    return s;
}

Vec4 lower(const Metric4<double>& g, const Vec4& u) {
    Vec4 p{};
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 4; ++n) p[m] += g(m, n) * u[n];
    return p;
}

Vec4 raise(const Metric4<double>& gi, const Vec4& p) {
    Vec4 u{};
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 4; ++n) u[m] += gi(m, n) * p[n];
    return u;
}

// Kerr–Schild metric g_μν at a (possibly complex) Cartesian point — the same
// formulas as KerrSchild::metric, templated for complex-step differentiation.
template <typename T>
void ks_metric_T(double M, double a, int s, T x, T y, T z, T g[4][4]) {
    using std::sqrt;
    const T rho2 = x * x + y * y + z * z;
    const T w = rho2 - a * a;
    const T r2 = 0.5 * (w + sqrt(w * w + 4.0 * a * a * z * z));
    const T r = sqrt(r2);
    const T f = 2.0 * M * r * r2 / (r2 * r2 + a * a * z * z);
    const T l[4] = {T(1.0), double(s) * (r * x + double(s) * a * y) / (r2 + a * a),
                    double(s) * (r * y - double(s) * a * x) / (r2 + a * a),
                    double(s) * z / r};
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 4; ++n)
            g[m][n] = (m == n ? T(m == 0 ? -1.0 : 1.0) : T(0.0)) + f * l[m] * l[n];
}

using Gamma = std::array<std::array<std::array<double, 4>, 4>, 4>;

// Γ^λ_μν at a real point: ∂_i g by complex step (exact to rounding), ∂_t g = 0.
Gamma christoffel_ks(const KerrSchild& ks, const Vec4& X) {
    const double M = ks.M(), a = ks.a();
    const int s = ks.chart();
    double dg[4][4][4] = {};   // dg[k][m][n] = ∂_k g_mn
    const double h = 1e-20;
    for (int k = 1; k < 4; ++k) {
        std::complex<double> xc[3] = {X[1], X[2], X[3]};
        xc[k - 1] += std::complex<double>(0.0, h);
        std::complex<double> g[4][4];
        ks_metric_T<std::complex<double>>(M, a, s, xc[0], xc[1], xc[2], g);
        for (int m = 0; m < 4; ++m)
            for (int n = 0; n < 4; ++n) dg[k][m][n] = g[m][n].imag() / h;
    }
    const Metric4<double> gi = ks.inverse_metric(X);
    Gamma G{};
    for (int l = 0; l < 4; ++l)
        for (int m = 0; m < 4; ++m)
            for (int n = 0; n < 4; ++n) {
                double v = 0.0;
                for (int q = 0; q < 4; ++q)
                    v += gi(l, q) * (dg[m][q][n] + dg[n][q][m] - dg[q][m][n]);
                G[l][m][n] = 0.5 * v;
            }
    return G;
}

// Jacobi eigenvalues of a symmetric 3×3 matrix, ascending.
std::array<double, 3> eig3(std::array<std::array<double, 3>, 3> A) {
    for (int sweep = 0; sweep < 50; ++sweep) {
        double off = std::abs(A[0][1]) + std::abs(A[0][2]) + std::abs(A[1][2]);
        if (off < 1e-30) break;
        for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q) {
                if (std::abs(A[p][q]) < 1e-300) continue;
                const double th = 0.5 * (A[q][q] - A[p][p]) / A[p][q];
                const double t = (th >= 0 ? 1.0 : -1.0) / (std::abs(th) + std::sqrt(th * th + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), sn = t * c;
                for (int k = 0; k < 3; ++k) {   // A ← Jᵀ A J
                    const double akp = A[k][p], akq = A[k][q];
                    A[k][p] = c * akp - sn * akq;
                    A[k][q] = sn * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const double apk = A[p][k], aqk = A[q][k];
                    A[p][k] = c * apk - sn * aqk;
                    A[q][k] = sn * apk + c * aqk;
                }
            }
    }
    std::array<double, 3> e = {A[0][0], A[1][1], A[2][2]};
    std::sort(e.begin(), e.end());
    return e;
}

}  // namespace

InfallTrajectory::InfallTrajectory(double a_star, const InfallStart& st)
    : ks_(1.0, a_star, +1) {
    const double r0 = st.r0, th0 = st.theta0;
    if (r0 <= ks_.r_plus() * 1.01)
        throw std::invalid_argument("InfallTrajectory: start outside the horizon");
    // ---- initial 4-velocity in Boyer–Lindquist (outside r₊), then to KS --
    const KerrMetric bl(1.0, a_star);
    const Metric4<double> g = bl.components(r0, th0);
    const Metric4<double> gi = invert_bl_block(g);
    Vec4 ucov{};   // (u_t, u_r, u_θ, u_φ)
    if (st.kind == InfallStartKind::Rain) {
        ucov[0] = -1.0;   // E = 1, L_z = 0, p_θ = 0 (Q = 0)
        ucov[1] = -std::sqrt((-1.0 - gi(0, 0)) / gi(1, 1));   // falling inward
    } else {
        ucov[0] = -std::sqrt(-1.0 / gi(0, 0));   // E = α: at rest w.r.t. the ZAMO
    }
    const Vec4 ubl = raise(gi, ucov);
    Vec4 X = ks_.point_from_bl(0.0, r0, th0, 0.0);
    Vec4 U = ks_.vector_from_bl(r0, th0, 0.0, ubl);
    Vec4 P = lower(ks_.metric(X), U);

    auto f = [this](double, const State& y, State& dy) {
        const Vec4 Xs = {y[0], y[1], y[2], y[3]}, Ps = {y[4], y[5], y[6], y[7]};
        Vec4 dX, dP;
        ks_.rhs(Xs, Ps, dX, dP);
        for (int i = 0; i < 4; ++i) {
            dy[i] = dX[i];
            dy[i + 4] = dP[i];
        }
    };
    DormandPrince54<8, decltype(f)> rk(f, 1e-12, 1e-12);
    DormandPrince54<8, decltype(f)> rkb(f, 1e-12, 1e-12);

    const double E0 = -P[0], L0 = ks_.angular_momentum(X, P), Q0 = ks_.carter_Q(X, P);
    const double scaleL = std::max(std::abs(L0), E0), scaleQ = std::max(std::abs(Q0), E0 * E0);
    const double rp = ks_.r_plus(), rm = ks_.r_minus();
    const double r_stop = rm + st.stop_above_rminus;

    auto sample = [&](double tau, const State& y) {
        InfallSample sm;
        sm.tau = tau;
        sm.X = {y[0], y[1], y[2], y[3]};
        const Vec4 Py = {y[4], y[5], y[6], y[7]};
        sm.U = raise(ks_.inverse_metric(sm.X), Py);
        sm.r = ks_.radius(sm.X[1], sm.X[2], sm.X[3]);
        sm.theta = std::acos(std::clamp(sm.X[3] / sm.r, -1.0, 1.0));
        sm.t_bl = sm.r > rp * (1.0 + 1e-9) ? ks_.bl_coords(sm.X)[0] : kNaN;
        sm.E = -Py[0];
        sm.Lz = ks_.angular_momentum(sm.X, Py);
        const bool q_ok = std::abs(sm.r - rp) > 0.05 && std::abs(sm.r - rm) > 0.05;
        sm.Q = q_ok ? ks_.carter_Q(sm.X, Py) : kNaN;
        sm.norm = 2.0 * ks_.hamiltonian(sm.X, Py) + 1.0;
        dE_ = std::max(dE_, std::abs(sm.E - E0) / E0);
        dL_ = std::max(dL_, std::abs(sm.Lz - L0) / scaleL);
        if (q_ok) dQ_ = std::max(dQ_, std::abs(sm.Q - Q0) / scaleQ);
        dn_ = std::max(dn_, std::abs(sm.norm));
        s_.push_back(sm);
    };

    State y{};
    for (int i = 0; i < 4; ++i) {
        y[i] = X[i];
        y[i + 4] = P[i];
    }
    double tau = 0.0, h = 1e-3 * r0;
    sample(tau, y);
    tau_hor_ = kNaN;
    for (int step = 0; step < 5000000; ++step) {
        const State prev = y;
        const double tau_prev = tau;
        const double r_prev = s_.back().r;
        const auto res = rk.try_step(tau, y, std::min(h, 0.02 * r_prev));
        h = res.h_next;
        if (!res.accepted) continue;
        const double r = ks_.radius(y[1], y[2], y[3]);
        // bisect a crossing of radius rc inside the last step
        auto land_on = [&](double rc) {
            double lo = 0.0, hi = res.h_used;
            State st_mid = y;
            double t_mid = tau;
            for (int it = 0; it < 80; ++it) {
                const double mid = 0.5 * (lo + hi);
                State trial = prev;
                double t0 = tau_prev;
                rkb.try_step(t0, trial, mid);   // accepted at this size
                if (ks_.radius(trial[1], trial[2], trial[3]) > rc) lo = mid; else hi = mid;
                st_mid = trial;
                t_mid = t0;
            }
            return std::pair{t_mid, st_mid};
        };
        if (std::isnan(tau_hor_) && r_prev > rp && r <= rp) {
            const auto [th, sh] = land_on(rp);
            tau_hor_ = th;
            (void)sh;
        }
        if (r <= r_stop) {
            const auto [te, se] = land_on(r_stop);
            sample(te, se);
            return;
        }
        sample(tau, y);
    }
    throw std::runtime_error("InfallTrajectory: step budget exhausted");
}

InfallSample InfallTrajectory::at(double tau) const {
    if (tau <= s_.front().tau) return s_.front();
    if (tau >= s_.back().tau) return s_.back();
    const auto it = std::upper_bound(s_.begin(), s_.end(), tau,
                                     [](double t, const InfallSample& s) { return t < s.tau; });
    const InfallSample& b = *it;
    const InfallSample& a = *(it - 1);
    const double dt = b.tau - a.tau, u = (tau - a.tau) / dt;
    const double h00 = (1 + 2 * u) * (1 - u) * (1 - u), h10 = u * (1 - u) * (1 - u);
    const double h01 = u * u * (3 - 2 * u), h11 = u * u * (u - 1);
    InfallSample o;
    o.tau = tau;
    for (int m = 0; m < 4; ++m) {
        o.X[m] = h00 * a.X[m] + h10 * dt * a.U[m] + h01 * b.X[m] + h11 * dt * b.U[m];
        o.U[m] = (1 - u) * a.U[m] + u * b.U[m];
    }
    const double n = std::sqrt(-dot(ks_.metric(o.X), o.U, o.U));
    for (double& c : o.U) c /= n;
    o.r = ks_.radius(o.X[1], o.X[2], o.X[3]);
    o.theta = std::acos(std::clamp(o.X[3] / o.r, -1.0, 1.0));
    o.t_bl = o.r > ks_.r_plus() * (1.0 + 1e-9) ? ks_.bl_coords(o.X)[0] : kNaN;
    o.E = (1 - u) * a.E + u * b.E;
    o.Lz = (1 - u) * a.Lz + u * b.Lz;
    o.Q = (1 - u) * a.Q + u * b.Q;
    o.norm = (1 - u) * a.norm + u * b.norm;
    return o;
}

double InfallTrajectory::tau_at_radius(double r) const {
    // bracket on the stored steps, then bisect on the Hermite interpolant
    for (size_t k = 1; k < s_.size(); ++k)
        if (s_[k - 1].r >= r && s_[k].r < r) {
            double lo = s_[k - 1].tau, hi = s_[k].tau;
            for (int it = 0; it < 64; ++it) {
                const double mid = 0.5 * (lo + hi);
                if (at(mid).r >= r) lo = mid; else hi = mid;
            }
            return 0.5 * (lo + hi);
        }
    return kNaN;
}

std::array<Vec4, 4> infall_tetrad(const KerrSchild& ks, const Vec4& X, const Vec4& U) {
    const Metric4<double> g = ks.metric(X);
    const double a = ks.a(), s = ks.chart();
    const double r = ks.radius(X[1], X[2], X[3]);
    const double ct = std::clamp(X[3] / r, -1.0, 1.0), st = std::sqrt(1.0 - ct * ct);
    const double ph = std::atan2(X[2], X[1]) - std::atan2(s * a, r);   // KS azimuth
    const double cp = std::cos(ph), sp = std::sin(ph);
    // Kerr–Schild coordinate directions at fixed t: x + iy = (r + i s a) sinθ e^{iφ}
    const Vec4 vr = {0.0, st * cp, st * sp, ct};
    const Vec4 vt = {0.0, (r * cp - s * a * sp) * ct, (r * sp + s * a * cp) * ct, -r * st};
    const Vec4 vp = {0.0, -(r * sp + s * a * cp) * st, (r * cp - s * a * sp) * st, 0.0};
    auto proj = [&](Vec4 v) {   // remove the component along U (U·U = −1)
        const double k = dot(g, U, v);
        for (int m = 0; m < 4; ++m) v[m] += k * U[m];
        return v;
    };
    auto normed = [&](Vec4 v) {
        const double n = std::sqrt(dot(g, v, v));
        for (double& c : v) c /= n;
        return v;
    };
    std::array<Vec4, 4> e;
    e[0] = U;
    e[1] = normed(proj(vr));
    Vec4 w = proj(vt);
    const double k21 = dot(g, e[1], w);
    for (int m = 0; m < 4; ++m) w[m] -= k21 * e[1][m];
    e[2] = normed(w);
    w = proj(vp);
    const double k31 = dot(g, e[1], w), k32 = dot(g, e[2], w);
    for (int m = 0; m < 4; ++m) w[m] -= k31 * e[1][m] + k32 * e[2][m];
    e[3] = normed(w);
    return e;
}

TidalField tidal_field(const KerrSchild& ks, const Vec4& X, const std::array<Vec4, 4>& e) {
    const double r = ks.radius(X[1], X[2], X[3]);
    const double hd = 1e-4 * std::max(r, 0.5);
    // ∂_k Γ by central differences of the (exact) complex-step Christoffels
    Gamma dG[4] = {};
    for (int k = 1; k < 4; ++k) {
        Vec4 Xp = X, Xm = X;
        Xp[k] += hd;
        Xm[k] -= hd;
        const Gamma Gp = christoffel_ks(ks, Xp), Gm = christoffel_ks(ks, Xm);
        for (int l = 0; l < 4; ++l)
            for (int m = 0; m < 4; ++m)
                for (int n = 0; n < 4; ++n) dG[k][l][m][n] = (Gp[l][m][n] - Gm[l][m][n]) / (2 * hd);
    }
    const Gamma G = christoffel_ks(ks, X);
    // R^ρ_σμν = ∂_μ Γ^ρ_νσ − ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ − Γ^ρ_νλ Γ^λ_μσ
    double R[4][4][4][4];
    for (int p = 0; p < 4; ++p)
        for (int sg = 0; sg < 4; ++sg)
            for (int m = 0; m < 4; ++m)
                for (int n = 0; n < 4; ++n) {
                    double v = dG[m][p][n][sg] - dG[n][p][m][sg];
                    for (int l = 0; l < 4; ++l) v += G[p][m][l] * G[l][n][sg] - G[p][n][l] * G[l][m][sg];
                    R[p][sg][m][n] = v;
                }
    // lower the first index, then project on the frame one index at a time
    const Metric4<double> g = ks.metric(X);
    double A[4][4][4][4], Bt[4][4][4][4];
    for (int a = 0; a < 4; ++a)
        for (int sg = 0; sg < 4; ++sg)
            for (int m = 0; m < 4; ++m)
                for (int n = 0; n < 4; ++n) {
                    double v = 0.0;
                    for (int p = 0; p < 4; ++p) v += g(a, p) * R[p][sg][m][n];
                    A[a][sg][m][n] = v;
                }
    auto contract = [&](int slot) {   // A ← A with index `slot` projected on e
        for (int i0 = 0; i0 < 4; ++i0)
            for (int i1 = 0; i1 < 4; ++i1)
                for (int i2 = 0; i2 < 4; ++i2)
                    for (int i3 = 0; i3 < 4; ++i3) {
                        int idx[4] = {i0, i1, i2, i3};
                        double v = 0.0;
                        for (int c = 0; c < 4; ++c) {
                            idx[slot] = c;
                            v += A[idx[0]][idx[1]][idx[2]][idx[3]] * e[slot == 0 ? i0 : slot == 1 ? i1 : slot == 2 ? i2 : i3][c];
                        }
                        Bt[i0][i1][i2][i3] = v;
                    }
        std::copy(&Bt[0][0][0][0], &Bt[0][0][0][0] + 256, &A[0][0][0][0]);
    };
    for (int slot = 0; slot < 4; ++slot) contract(slot);
    TidalField t;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) t.E[i][j] = A[i + 1][0][j + 1][0];
    // B_ij = ½ ε_ipq R_{pq j 0}
    const int eps[3][3][3] = {{{0, 0, 0}, {0, 0, 1}, {0, -1, 0}},
                              {{0, 0, -1}, {0, 0, 0}, {1, 0, 0}},
                              {{0, 1, 0}, {-1, 0, 0}, {0, 0, 0}}};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double v = 0.0;
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) v += eps[i][p][q] * A[p + 1][q + 1][j + 1][0];
            t.B[i][j] = 0.5 * v;
        }
    double K = 0.0;
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b)
            for (int c = 0; c < 4; ++c)
                for (int d = 0; d < 4; ++d) {
                    const int n0 = (a == 0) + (b == 0) + (c == 0) + (d == 0);
                    K += (n0 % 2 ? -1.0 : 1.0) * A[a][b][c][d] * A[a][b][c][d];
                }
    t.kretschmann = K;
    t.eig = eig3(t.E);
    return t;
}

double kerr_kretschmann(double M, double a, double r, double c) {
    const double r2 = r * r, a2 = a * a, c2 = c * c;
    const double S = r2 + a2 * c2;
    return 48.0 * M * M *
           (r2 * r2 * r2 - 15.0 * a2 * r2 * r2 * c2 + 15.0 * a2 * a2 * r2 * c2 * c2 -
            a2 * a2 * a2 * c2 * c2 * c2) /
           std::pow(S, 6);
}

}  // namespace sgr
