// Gate 18: the orbiting hot spot — orbit, comoving geometry, slow-light
// timing against the Carter t-equation, and the equatorial mirror symmetry
// of the flare light curve and centroid track.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "Hotspot.hpp"
#include "sgr_physics/integrator.hpp"
#include "sgr_physics/kerr_schild.hpp"
#include "sgr_physics/metric.hpp"
#include "sgr_physics/null_geodesics.hpp"

using namespace sgr;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {
constexpr double kPi = 3.14159265358979323846;
}

TEST_CASE("Hot spot rides the prograde circular geodesic", "[gate18][flares]") {
    for (double a : {0.0, 0.94})
        for (double R : {6.0, 7.0, 9.0}) {
            HotspotParams p;
            p.a_star = a;
            p.R = R;
            p.phase0 = 0.3;
            const Hotspot hs(p);
            CHECK_THAT(hs.period(), WithinRel(2.0 * kPi * (std::pow(R, 1.5) + a), 1e-13));
            const KerrSchild ks(1.0, a, -1);
            for (double t : {0.0, 17.3, 250.0}) {
                const auto c = hs.centre(t);
                const auto c2 = hs.centre(t + hs.period());
                for (int i = 0; i < 3; ++i) CHECK_THAT(c2[i], WithinAbs(c[i], 1e-9));
                CHECK_THAT(ks.radius(c[0], c[1], c[2]), WithinRel(R, 1e-13));
                CHECK_THAT(hs.distance2({t, c[0], c[1], c[2]}), WithinAbs(0.0, 1e-24));
                CHECK_THAT(hs.emissivity({t, c[0], c[1], c[2]}), WithinAbs(1.0, 1e-15));
            }
        }
}

TEST_CASE("Comoving distance co-rotates with the spot", "[gate18][flares]") {
    HotspotParams p;
    p.a_star = 0.94;
    p.R = 7.0;
    const Hotspot hs(p);
    const auto c0 = hs.centre(0.0);
    const double d[3] = {0.31, -0.22, 0.17};
    const double D0 = hs.distance2({0.0, c0[0] + d[0], c0[1] + d[1], c0[2] + d[2]});
    CHECK(D0 > 0.0);
    for (double t : {5.0, 40.0, 123.4}) {
        // rotate the displaced point with the spot: same comoving distance
        const double ph = hs.omega() * t;
        const double x = c0[0] + d[0], y = c0[1] + d[1];
        const double xr = std::cos(ph) * x - std::sin(ph) * y;
        const double yr = std::sin(ph) * x + std::cos(ph) * y;
        CHECK_THAT(hs.distance2({t, xr, yr, c0[2] + d[2]}), WithinRel(D0, 1e-12));
    }
}

TEST_CASE("Slow light: Kerr-Schild ray time equals the Carter t-equation",
          "[gate18][flares]") {
    // A ray from a far ZAMO camera to the equatorial plane near the spot
    // orbit. KS time at the crossing, t_KS = t_BL + s T(r), must agree with
    // the independent Carter/Mino integration of dt/dτ.
    const double a = 0.94, r_cam = 1000.0, th = 160.0 * kPi / 180.0;
    const KerrSchild ks(1.0, a, -1);
    const KerrMetric m(1.0, a);
    for (double bx : {-7.0, 3.0, 8.5}) {
        const double tx = bx / r_cam, ty = 2.0 / r_cam;
        const double inv = 1.0 / std::sqrt(1.0 + tx * tx + ty * ty);
        const std::array<double, 3> n{-inv, -ty * inv, tx * inv};
        // Kerr-Schild path
        const KSRay ray = ks_camera_ray(ks, r_cam, th, n);
        using S8 = StateVec<8>;
        auto f = [&ks](double, const S8& y, S8& dy) {
            const Vec4 X{y[0], y[1], y[2], y[3]}, P{y[4], y[5], y[6], y[7]};
            Vec4 dX, dP;
            ks.rhs(X, P, dX, dP);
            for (int i = 0; i < 4; ++i) { dy[i] = dX[i]; dy[i + 4] = dP[i]; }
        };
        DormandPrince54<8, decltype(f)> rk(f, 1e-12, 1e-12), rkb(f, 1e-12, 1e-12);
        S8 y{};
        for (int k = 0; k < 4; ++k) { y[k] = ray.X[k]; y[k + 4] = ray.P[k]; }
        double lam = 0.0, h = 1.0, t_ks = 0.0, r_ks = 0.0;
        for (int step = 0; step < 200000; ++step) {
            const S8 prev = y;
            const double lp = lam;
            const auto res = rk.try_step(lam, y, h);
            h = res.h_next;
            if (!res.accepted) continue;
            if (prev[3] * y[3] < 0.0) {
                double lo = 0.0, hi = res.h_used;
                S8 cr = y;
                for (int it = 0; it < 60; ++it) {
                    const double mid = 0.5 * (lo + hi);
                    S8 tr = prev;
                    double l0 = lp;
                    rkb.try_step(l0, tr, mid);
                    if (tr[3] * prev[3] > 0.0) lo = mid; else hi = mid;
                    cr = tr;
                }
                t_ks = cr[0];
                r_ks = ks.radius(cr[1], cr[2], cr[3]);
                break;
            }
        }
        REQUIRE(r_ks > 0.0);
        // Carter path from the same received photon (backward tangent, BL)
        const auto g = m.components(r_cam, th);
        const double om = -g(0, 3) / g(3, 3);
        const double al = std::sqrt(-(g(0, 0) - g(0, 3) * g(0, 3) / g(3, 3)));
        const std::array<double, 4> k_bl{-1.0 / al, n[0] / std::sqrt(g(1, 1)),
                                         n[1] / std::sqrt(g(2, 2)),
                                         -om / al + n[2] / std::sqrt(g(3, 3))};
        NullRayInit init = make_null_ray(m, 0.0, r_cam, th, 0.0, k_bl);
        const NullRayConstants c = init.constants;
        auto fc = [&c](double, const NullState& s, NullState& ds) { null_geodesic_rhs(c, s, ds); };
        DormandPrince54<6, decltype(fc)> rc(fc, 1e-12, 1e-12), rcb(fc, 1e-12, 1e-12);
        NullState z = init.state;
        double tau = 0.0, hc = 1e-4, t_bl = 0.0, r_bl = 0.0;
        for (int step = 0; step < 400000; ++step) {
            const NullState prev = z;
            const double tp = tau;
            const auto res = rc.try_step(tau, z, hc);
            hc = res.h_next;
            if (!res.accepted) continue;
            // constraint projection, exactly as RaytraceCpu::trace: the
            // second-order Mino system otherwise drifts O(rtol r^4) on the
            // long r ~ 1000 M leg
            {
                const double Rp = radial_potential(c, z[1]);
                if (Rp >= 0.0) z[4] = std::copysign(std::sqrt(Rp), z[4]);
                const double Tp = polar_potential(c, z[2]);
                if (Tp >= 0.0) z[5] = std::copysign(std::sqrt(Tp), z[5]);
            }
            if (std::cos(prev[2]) * std::cos(z[2]) < 0.0) {
                double lo = 0.0, hi = res.h_used;
                NullState cr = z;
                for (int it = 0; it < 60; ++it) {
                    const double mid = 0.5 * (lo + hi);
                    NullState tr = prev;
                    double t0 = tp;
                    rcb.try_step(t0, tr, mid);
                    if (std::cos(tr[2]) * std::cos(prev[2]) > 0.0) lo = mid; else hi = mid;
                    cr = tr;
                }
                t_bl = cr[0];
                r_bl = cr[1];
                break;
            }
        }
        REQUIRE(r_bl > 0.0);
        CHECK_THAT(r_ks, WithinRel(r_bl, 1e-9));
        const double s = ks.chart();
        // camera event: t_BL = 0 ↔ t_KS = s T(r_cam)
        CHECK_THAT(t_ks, WithinAbs(t_bl + s * ks.T_shift(r_bl), 1e-7));
    }
}

TEST_CASE("Flare light curves are mirror-symmetric about the equator",
          "[gate18][flares]") {
    // Kerr is symmetric under z → −z; the equatorial orbit maps to itself.
    // A camera at θ and one at π − θ see vertically mirrored images: the
    // same flux and x-centroid at every time, opposite y-centroid.
    HotspotParams p;
    p.a_star = 0.94;
    p.R = 7.0;
    FlareCamera ca, cb;
    ca.n = cb.n = 40;
    ca.incl_deg = 40.0;
    cb.incl_deg = 140.0;
    const FlareRayTable A = build_flare_table(p, ca);
    const FlareRayTable B = build_flare_table(p, cb);
    CHECK(A.max_dH < 1e-8);       // §4 integrator health
    CHECK(A.max_dL < 1e-6);
    const Hotspot hs(p);
    for (int k = 0; k < 6; ++k) {
        const double t = hs.period() * k / 6.0;
        const FlareMoments ma = flare_moments(A, flare_image(A, hs, t));
        const FlareMoments mb = flare_moments(B, flare_image(B, hs, t));
        REQUIRE(ma.flux > 0.0);
        CHECK_THAT(mb.flux, WithinRel(ma.flux, 1e-6));
        CHECK_THAT(mb.cx, WithinAbs(ma.cx, 1e-6));
        CHECK_THAT(mb.cy, WithinAbs(-ma.cy, 1e-6));
    }
}
