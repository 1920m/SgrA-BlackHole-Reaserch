#include "Hotspot.hpp"

#include <algorithm>
#include <cmath>
#include <execution>
#include <numeric>

#include "sgr_physics/integrator.hpp"
#include "sgr_physics/metric.hpp"

namespace sgr {

namespace {
constexpr double kPi = 3.14159265358979323846;

void rotate_z(double c, double s, double& x, double& y) {
    const double xn = c * x - s * y;
    y = s * x + c * y;
    x = xn;
}
} // namespace

Hotspot::Hotspot(const HotspotParams& p)
    : p_(p), ks_(1.0, p.a_star, -1) {
    const KerrMetric m(1.0, p.a_star);
    Om_ = m.omega_circular(p.R, true);
    const Metric4<double> g = m.components(p.R, 0.5 * kPi);
    ut_ = 1.0 / std::sqrt(-(g(0, 0) + 2.0 * Om_ * g(0, 3) + Om_ * Om_ * g(3, 3)));
    // Comoving metric h = g + u⊗u at the reference centre (KS azimuth 0);
    // distance2() rotates displacements into that frame — rotations about z
    // are isometries of Kerr-Schild Cartesian coordinates.
    const double s = ks_.chart(), a = ks_.a();
    const Vec4 C{0.0, p.R, s * a, 0.0};
    const Metric4<double> gk = ks_.metric(C);
    const Vec4 u{ut_, -Om_ * C[2] * ut_, Om_ * C[1] * ut_, 0.0};
    double ul[4] = {0.0, 0.0, 0.0, 0.0};
    for (int m2 = 0; m2 < 4; ++m2)
        for (int n = 0; n < 4; ++n) ul[m2] += gk(m2, n) * u[n];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) h_[i][j] = gk(i + 1, j + 1) + ul[i + 1] * ul[j + 1];
}

double Hotspot::period() const { return 2.0 * kPi / std::abs(Om_); }

std::array<double, 3> Hotspot::centre(double t) const {
    const double ph = p_.phase0 + Om_ * t;
    const double s = ks_.chart(), a = ks_.a();
    const double c = std::cos(ph), sn = std::sin(ph);
    return {p_.R * c - s * a * sn, p_.R * sn + s * a * c, 0.0};
}

double Hotspot::distance2(const Vec4& X) const {
    const auto c = centre(X[0]);
    double d[3] = {X[1] - c[0], X[2] - c[1], X[3] - c[2]};
    const double ph = p_.phase0 + Om_ * X[0];
    rotate_z(std::cos(ph), -std::sin(ph), d[0], d[1]);
    double d2 = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) d2 += h_[i][j] * d[i] * d[j];
    return d2;
}

double Hotspot::emissivity(const Vec4& X) const {
    return std::exp(-0.5 * distance2(X) / (p_.sigma * p_.sigma));
}

double Hotspot::redshift(const Vec4& X, const Vec4& P) const {
    const double Lz = X[1] * P[2] - X[2] * P[1];
    return 1.0 / (ut_ * (P[0] + Om_ * Lz));
}

// --------------------------------------------------------------------------
FlareRayTable build_flare_table(const HotspotParams& geo, const FlareCamera& cam) {
    FlareRayTable T;
    T.cam = cam;
    T.geometry = geo;
    const int n = cam.n;
    const KerrSchild ks(1.0, geo.a_star, -1);
    const double th_cam = cam.incl_deg * kPi / 180.0;
    const double R = geo.R, sg = geo.sigma;
    const double roi = 6.0 * sg;

    struct RowOut {
        std::vector<FlareRayTable::Sample> s;
        std::vector<uint32_t> count;
        std::vector<double> Pt, Lz;
        std::vector<uint8_t> cap;
        double dH = 0.0, dL = 0.0;
    };
    std::vector<RowOut> rows(n);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    T.t_cam = ks.point_from_bl(0.0, cam.r_cam, th_cam, 0.0)[0];

    std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int j) {
        RowOut& out = rows[j];
        using State = StateVec<8>;
        auto f = [&ks](double, const State& y, State& dy) {
            const Vec4 X{y[0], y[1], y[2], y[3]}, P{y[4], y[5], y[6], y[7]};
            Vec4 dX, dP;
            ks.rhs(X, P, dX, dP);
            for (int i = 0; i < 4; ++i) { dy[i] = dX[i]; dy[i + 4] = dP[i]; }
        };
        for (int i = 0; i < n; ++i) {
            const double bx = cam.half_width_M * (2.0 * (i + 0.5) / n - 1.0);
            const double by = cam.half_width_M * (1.0 - 2.0 * (j + 0.5) / n);
            const double tx = bx / cam.r_cam, ty = by / cam.r_cam;
            const double inv = 1.0 / std::sqrt(1.0 + tx * tx + ty * ty);
            const KSRay ray = ks_camera_ray(ks, cam.r_cam, th_cam,
                                            {-inv, -ty * inv, tx * inv});
            DormandPrince54<8, decltype(f)> rk(f, 1e-11, 1e-11);
            State y{};
            for (int k = 0; k < 4; ++k) { y[k] = ray.X[k]; y[k + 4] = ray.P[k]; }
            const double E0 = ray.P[0];
            const double L0 = ks.angular_momentum(ray.X, ray.P);
            const double scaleL = std::max(std::abs(L0), E0);
            double lam = 0.0, h = 1.0;
            int passage = -1;
            bool was_in = false, cap = false;
            uint32_t cnt = 0;
            for (int step = 0; step < 400000; ++step) {
                const Vec4 X{y[0], y[1], y[2], y[3]}, P{y[4], y[5], y[6], y[7]};
                const double r = ks.radius(X[1], X[2], X[3]);
                // step cap: never jump into or across the emitting region,
                // and sample it at ≤ s/5 spacing inside
                Vec4 dX, dP;
                ks.rhs(X, P, dX, dP);
                const double speed = std::sqrt(dX[1] * dX[1] + dX[2] * dX[2] + dX[3] * dX[3]);
                const double dist = std::max({std::abs(r - R) - roi, std::abs(X[3]) - roi, 0.0});
                const double Lmax = std::max(0.2 * sg, 0.5 * dist);
                const double hcap = Lmax / std::max(speed, 1e-30);
                const auto res = rk.try_step(lam, y, std::min(h, hcap));
                h = res.h_next;
                if (!res.accepted) continue;
                const Vec4 Xn{y[0], y[1], y[2], y[3]}, Pn{y[4], y[5], y[6], y[7]};
                const double rn = ks.radius(Xn[1], Xn[2], Xn[3]);
                out.dH = std::max(out.dH, std::abs(ks.hamiltonian(Xn, Pn)) / (E0 * E0));
                out.dL = std::max(out.dL,
                                  std::abs(ks.angular_momentum(Xn, Pn) - L0) / scaleL);
                const bool in = std::abs(rn - R) <= roi && std::abs(Xn[3]) <= roi;
                if (in) {
                    if (!was_in) ++passage;
                    FlareRayTable::Sample s;
                    s.dt = static_cast<float>(Xn[0] - T.t_cam);
                    s.x = static_cast<float>(Xn[1]);
                    s.y = static_cast<float>(Xn[2]);
                    s.z = static_cast<float>(Xn[3]);
                    s.dlam = static_cast<float>(res.h_used);
                    s.order = static_cast<uint8_t>(std::min(passage, 255));
                    out.s.push_back(s);
                    ++cnt;
                }
                was_in = in;
                if (rn < ks.r_plus()) { cap = true; break; }
                if (rn > 1.2 * cam.r_cam) {
                    ks.rhs(Xn, Pn, dX, dP);
                    if (Xn[1] * dX[1] + Xn[2] * dX[2] + Xn[3] * dX[3] > 0.0) break;
                }
            }
            out.count.push_back(cnt);
            out.Pt.push_back(ray.P[0]);
            out.Lz.push_back(L0);
            out.cap.push_back(cap ? 1 : 0);
        }
    });

    T.first.assign(static_cast<size_t>(n) * n + 1, 0);
    size_t total = 0;
    for (const auto& r : rows) total += r.s.size();
    T.samples.reserve(total);
    uint32_t pos = 0;
    for (int j = 0; j < n; ++j) {
        const RowOut& r = rows[j];
        T.samples.insert(T.samples.end(), r.s.begin(), r.s.end());
        for (int i = 0; i < n; ++i) {
            T.first[static_cast<size_t>(j) * n + i] = pos;
            pos += r.count[i];
            T.P_t.push_back(r.Pt[i]);
            T.Lz.push_back(r.Lz[i]);
            T.captured.push_back(r.cap[i]);
        }
        T.max_dH = std::max(T.max_dH, r.dH);
        T.max_dL = std::max(T.max_dL, r.dL);
    }
    T.first[static_cast<size_t>(n) * n] = pos;
    return T;
}

std::vector<double> flare_image(const FlareRayTable& T, const Hotspot& hs,
                                double t_obs, double t_flash, double tau,
                                unsigned order_mask) {
    const int n = T.cam.n;
    std::vector<double> img(static_cast<size_t>(n) * n, 0.0);
    const double expo = 2.0 + hs.params().alpha;
    const double inv2s2 = 0.5 / (hs.params().sigma * hs.params().sigma);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int j) {
        for (int i = 0; i < n; ++i) {
            const size_t p = static_cast<size_t>(j) * n + i;
            const uint32_t a = T.first[p], b = T.first[p + 1];
            if (a == b) continue;
            const double g = 1.0 / (hs.ut() * (T.P_t[p] + hs.omega() * T.Lz[p]));
            if (!(g > 0.0)) continue;
            const double w = std::pow(g, expo);
            double acc = 0.0;
            for (uint32_t k = a; k < b; ++k) {
                const auto& s = T.samples[k];
                if (!(order_mask & (1u << std::min<int>(s.order, 31)))) continue;
                const double te = T.t_cam + t_obs + s.dt;
                double e = std::exp(-hs.distance2({te, s.x, s.y, s.z}) * inv2s2);
                if (tau > 0.0) {
                    const double q = (te - t_flash) / tau;
                    e *= std::exp(-0.5 * q * q);
                }
                acc += e * s.dlam;
            }
            img[p] = w * acc;
        }
    });
    return img;
}

std::vector<std::vector<double>> flare_flash_curves(const FlareRayTable& T,
                                                    const Hotspot& hs,
                                                    double t_flash, double tau,
                                                    const std::vector<double>& t_obs,
                                                    int max_order) {
    const int n = T.cam.n;
    const int NT = static_cast<int>(t_obs.size());
    const int NO = max_order + 1;
    const double t0 = t_obs.front();
    const double dt = (NT > 1) ? (t_obs[1] - t_obs[0]) : 1.0;
    const double sg = hs.params().sigma;
    const double inv2s2 = 0.5 / (sg * sg);
    const double expo = 2.0 + hs.params().alpha;
    // the centre moves < 1 per unit time: a sample farther than this from
    // the centre at t_flash cannot be lit during the ±5τ flash window
    const double reach = 6.0 * sg + 5.0 * tau;
    std::vector<std::vector<double>> rows(n, std::vector<double>(static_cast<size_t>(NO) * NT, 0.0));
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int j) {
        auto& acc = rows[j];
        for (int i = 0; i < n; ++i) {
            const size_t p = static_cast<size_t>(j) * n + i;
            const uint32_t a = T.first[p], b = T.first[p + 1];
            if (a == b) continue;
            const double g = 1.0 / (hs.ut() * (T.P_t[p] + hs.omega() * T.Lz[p]));
            if (!(g > 0.0)) continue;
            const double w = std::pow(g, expo);
            for (uint32_t k = a; k < b; ++k) {
                const auto& s = T.samples[k];
                if (s.order > max_order) continue;
                if (hs.distance2({t_flash, s.x, s.y, s.z}) > reach * reach * 4.0) continue;
                const double t_arr = t_flash - T.t_cam - s.dt;   // t_obs with t_e = t_flash
                const int lo = std::max(0, static_cast<int>(std::ceil((t_arr - 5.0 * tau - t0) / dt)));
                const int hi = std::min(NT - 1, static_cast<int>(std::floor((t_arr + 5.0 * tau - t0) / dt)));
                for (int q = lo; q <= hi; ++q) {
                    const double te = T.t_cam + t_obs[q] + s.dt;
                    const double u = (te - t_flash) / tau;
                    const double e = std::exp(-hs.distance2({te, s.x, s.y, s.z}) * inv2s2 - 0.5 * u * u);
                    acc[static_cast<size_t>(s.order) * NT + q] += w * e * s.dlam;
                }
            }
        }
    });
    std::vector<std::vector<double>> out(NO, std::vector<double>(NT, 0.0));
    for (const auto& r : rows)
        for (int o = 0; o < NO; ++o)
            for (int q = 0; q < NT; ++q) out[o][q] += r[static_cast<size_t>(o) * NT + q];
    return out;
}

FlareMoments flare_moments(const FlareRayTable& T, const std::vector<double>& img) {
    FlareMoments m;
    const int n = T.cam.n;
    double sx = 0.0, sy = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const double v = img[static_cast<size_t>(j) * n + i];
            const double bx = T.cam.half_width_M * (2.0 * (i + 0.5) / n - 1.0);
            const double by = T.cam.half_width_M * (1.0 - 2.0 * (j + 0.5) / n);
            m.flux += v;
            sx += v * bx;
            sy += v * by;
        }
    if (m.flux > 0.0) {
        m.cx = sx / m.flux;
        m.cy = sy / m.flux;
    }
    return m;
}

} // namespace sgr
