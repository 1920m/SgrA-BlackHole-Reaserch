// Gate 16: Cartesian Kerr-Schild Hamiltonian tracer (CPU twin of the live
// kernel) — exact Kerr, conservation (to 1e-6), agreement with the
// Carter/Mino science path, and the physical image handedness.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "sgr_physics/integrator.hpp"
#include "sgr_physics/kerr_schild.hpp"
#include "sgr_physics/metric.hpp"
#include "sgr_physics/null_geodesics.hpp"
#include "sgr_physics/raytrace_cpu.hpp"

using namespace sgr;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {
constexpr double kPi = 3.14159265358979323846;

std::array<double, 3> look(double tx, double ty) { // pinhole, looks along −r̂
    const double inv = 1.0 / std::sqrt(1.0 + tx * tx + ty * ty);
    return {-inv, -ty * inv, tx * inv};
}

// Backward tangent −k_received = −e_t + n·e of a ZAMO camera, BL components.
Vec4 received_backward_bl(const KerrMetric& m, double r, double th,
                          const std::array<double, 3>& n) {
    const auto g = m.components(r, th);
    const double om = -g(0, 3) / g(3, 3);
    const double al = std::sqrt(-(g(0, 0) - g(0, 3) * g(0, 3) / g(3, 3)));
    return {-1.0 / al, n[0] / std::sqrt(g(1, 1)), n[1] / std::sqrt(g(2, 2)),
            -om / al + n[2] / std::sqrt(g(3, 3))};
}

// First equatorial crossing radius on the Carter/Mino path (reference).
double carter_first_crossing(const KerrMetric& m, double r, double th,
                             const Vec4& k_bl) {
    NullRayInit init = make_null_ray(m, 0.0, r, th, 0.0,
                                     {k_bl[0], k_bl[1], k_bl[2], k_bl[3]});
    const NullRayConstants c = init.constants;
    auto f = [&c](double, const NullState& y, NullState& dy) {
        null_geodesic_rhs(c, y, dy);
    };
    DormandPrince54<6, decltype(f)> rk(f, 1e-12, 1e-12), rkb(f, 1e-12, 1e-12);
    NullState y = init.state;
    double tau = 0.0, h = 1e-4;
    for (int step = 0; step < 400000; ++step) {
        const NullState prev = y;
        const double tau_prev = tau;
        const auto res = rk.try_step(tau, y, h);
        h = res.h_next;
        if (!res.accepted) continue;
        if (std::cos(prev[2]) * std::cos(y[2]) < 0.0) {
            double lo = 0.0, hi = res.h_used;
            NullState cr = y;
            for (int it = 0; it < 60; ++it) {
                const double mid = 0.5 * (lo + hi);
                NullState trial = prev;
                double t0 = tau_prev;
                rkb.try_step(t0, trial, mid);
                if (std::cos(trial[2]) * std::cos(prev[2]) > 0.0) lo = mid; else hi = mid;
                cr = trial;
            }
            return cr[1];
        }
        if (y[1] < m.r_horizon() * 1.0001 || y[1] > 2000.0) break;
    }
    return -1.0;
}

// Redshift g of the disk at the crossing, from the backward KS covector:
// g = (p·u_cam)/(p·u_disk) with p·u_cam = 1 by construction.
double disk_g(const KerrSchild& ks, double r_x, const Vec4& X, const Vec4& P) {
    const KerrMetric m(ks.M(), ks.a() / ks.M());
    const double Om = m.omega_circular(r_x, true);
    const auto g = m.components(r_x, kPi / 2);
    const double ut =
        1.0 / std::sqrt(-(g(0, 0) + 2.0 * Om * g(0, 3) + Om * Om * g(3, 3)));
    return 1.0 / (ut * (P[0] + Om * ks.angular_momentum(X, P)));
}

// Smallest |tx| on one side of the look axis whose ray escapes (bisection).
double shadow_edge(const KerrSchild& ks, double r_cam, double th, double sign) {
    double lo = 0.0, hi = 0.02; // tan(angle); hi escapes at r_cam = 1000
    KSTraceOptions o;
    o.rtol = o.atol = 1e-10;
    o.max_crossings = 0;
    for (int it = 0; it < 40; ++it) {
        const double mid = 0.5 * (lo + hi);
        const auto res = ks_trace(ks, ks_camera_ray(ks, r_cam, th, look(sign * mid, 0.0)), o);
        if (res.outcome == KSOutcome::Escaped) hi = mid; else lo = mid;
    }
    return 0.5 * (lo + hi);
}
} // namespace

TEST_CASE("Kerr-Schild metric is exactly Kerr (transforms to Boyer-Lindquist)",
          "[kerr-schild][gate16]") {
    for (int s : {+1, -1}) {
        for (double a : {0.0, 0.5, 0.94}) {
            const KerrSchild ks(1.0, a, s);
            const KerrMetric bl(1.0, a);
            for (double r : {1.6, 2.5, 6.0, 40.0}) {
                if (r <= ks.r_plus() * 1.01) continue;
                for (double th : {0.3, 1.1, kPi / 2, 2.6}) {
                    for (double ph : {0.0, 1.9, -2.7}) {
                        const auto J = ks.jacobian_bl_to_ks(r, th, ph);
                        const auto gk = ks.metric(ks.point_from_bl(0.0, r, th, ph));
                        const auto gb = bl.components(r, th);
                        double scale = 0.0, err = 0.0;
                        for (int al = 0; al < 4; ++al)
                            for (int be = 0; be < 4; ++be) {
                                double v = 0.0;
                                for (int m = 0; m < 4; ++m)
                                    for (int n = 0; n < 4; ++n)
                                        v += J[m][al] * J[n][be] * gk(m, n);
                                err = std::max(err, std::abs(v - gb(al, be)));
                                scale = std::max(scale, std::abs(gb(al, be)));
                            }
                        INFO("s=" << s << " a=" << a << " r=" << r << " th=" << th
                                  << " ph=" << ph << " err=" << err);
                        CHECK(err / scale < 1e-12);
                    }
                }
            }
        }
    }
}

TEST_CASE("Kerr-Schild inverse metric and analytic Hamiltonian gradient",
          "[kerr-schild][gate16]") {
    const KerrSchild ks(1.0, 0.94, -1);
    const Vec4 X = {0.3, 2.1, -1.7, 0.9}; // well outside r₊, off-axis
    const Vec4 P = {-0.8, 0.37, -0.52, 0.21};
    const auto g = ks.metric(X), gi = ks.inverse_metric(X);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            double v = 0.0;
            for (int k = 0; k < 4; ++k) v += g(i, k) * gi(k, j);
            CHECK_THAT(v, WithinAbs(i == j ? 1.0 : 0.0, 1e-13));
        }
    Vec4 dX, dP;
    ks.rhs(X, P, dX, dP);
    const double eps = 1e-6;
    for (int i = 1; i < 4; ++i) {   // dp_i/dλ = −∂H/∂x^i (central difference)
        Vec4 Xp = X, Xm = X;
        Xp[i] += eps;
        Xm[i] -= eps;
        const double num = -(ks.hamiltonian(Xp, P) - ks.hamiltonian(Xm, P)) / (2 * eps);
        CHECK_THAT(dP[i], WithinAbs(num, 1e-8));
    }
    for (int m = 0; m < 4; ++m) {   // dx^μ/dλ = ∂H/∂p_μ
        Vec4 Pp = P, Pm = P;
        Pp[m] += eps;
        Pm[m] -= eps;
        const double num = (ks.hamiltonian(X, Pp) - ks.hamiltonian(X, Pm)) / (2 * eps);
        CHECK_THAT(dX[m], WithinAbs(num, 1e-8));
    }
    CHECK(dP[0] == 0.0);
}

TEST_CASE("Kerr-Schild a*=0: critical impact parameter is sqrt(27) M",
          "[kerr-schild][schwarzschild][gate16]") {
    const KerrSchild ks(1.0, 0.0, -1);
    const double r_cam = 1000.0, th = kPi / 2;
    const double t = shadow_edge(ks, r_cam, th, +1.0);
    const KSRay ray = ks_camera_ray(ks, r_cam, th, look(t, 0.0));
    const double b = std::abs(ks.angular_momentum(ray.X, ray.P) / ks.energy(ray.P));
    INFO("b_c = " << b);
    CHECK_THAT(b, WithinRel(std::sqrt(27.0), 1e-6));
}

// (ASCII test name: ctest passes it back as a filter and mangles non-ASCII.)
TEST_CASE("Kerr-Schild rays conserve E, L_z, Q and stay null",
          "[kerr-schild][conservation][gate16]") {
    const KerrSchild ks(1.0, 0.94, -1);
    KSTraceOptions o;
    o.rtol = o.atol = 1e-12;
    for (double tx : {-0.32, -0.12, 0.05, 0.11, 0.26})
        for (double ty : {-0.2, 0.0, 0.15}) {
            const auto res = ks_trace(ks, ks_camera_ray(ks, 60.0, 80.0 * kPi / 180.0,
                                                        look(tx, ty)), o);
            INFO("tx=" << tx << " ty=" << ty << " dL=" << res.max_dL_rel
                       << " dQ=" << res.max_dQ_rel << " H=" << res.max_H_rel);
            CHECK(res.max_dL_rel < 1e-6);
            CHECK(res.max_dQ_rel < 1e-6);
            CHECK(res.max_H_rel < 1e-8);
        }
}

TEST_CASE("Kerr-Schild crossings agree with the Carter/Mino path",
          "[kerr-schild][carter][gate16]") {
    const double a = 0.94, r_cam = 60.0, th = 80.0 * kPi / 180.0;
    const KerrSchild ks(1.0, a, -1);
    const KerrMetric bl(1.0, a);
    KSTraceOptions o;
    o.rtol = o.atol = 1e-12;
    int compared = 0;
    for (double tx : {-0.25, -0.1, 0.08, 0.2})
        for (double ty : {-0.15, -0.05, 0.1}) {
            const auto n = look(tx, ty);
            const auto res = ks_trace(ks, ks_camera_ray(ks, r_cam, th, n), o);
            const double r_carter =
                carter_first_crossing(bl, r_cam, th, received_backward_bl(bl, r_cam, th, n));
            if (res.n_cross == 0 || r_carter < 0.0) continue;
            INFO("tx=" << tx << " ty=" << ty << " KS r=" << res.cross_r[0]
                       << " Carter r=" << r_carter);
            CHECK_THAT(res.cross_r[0], WithinRel(r_carter, 1e-7));
            ++compared;
        }
    CHECK(compared >= 8);
}

TEST_CASE("Image handedness: receding side redshifted, prograde side flattened",
          "[kerr-schild][handedness][gate16]") {
    // Camera above the plane (θ = 80°), image right = +φ̂, up = north (spin).
    // A prograde (counter-clockwise from north) disk then RECEDES on the
    // right of the image and APPROACHES on the left.
    const double r_cam = 1000.0, th = 80.0 * kPi / 180.0;
    for (double a : {0.0, 0.94}) {
        const KerrSchild ks(1.0, a, -1);
        KSTraceOptions o;
        o.rtol = o.atol = 1e-11;
        auto g_at = [&](double tx, double& phi_hit) {
            const auto res = ks_trace(ks, ks_camera_ray(ks, r_cam, th, look(tx, 0.0)), o);
            REQUIRE(res.n_cross >= 1);
            phi_hit = res.cross_phi_bl[0];
            // re-trace to the crossing state for g (tight bisection already
            // gave r; evaluate g with conserved E, L_z — exact along the ray)
            const KSRay ray = ks_camera_ray(ks, r_cam, th, look(tx, 0.0));
            return disk_g(ks, res.cross_r[0], ray.X, ray.P);
        };
        double ph_r = 0.0, ph_l = 0.0;
        const double g_right = g_at(+0.020, ph_r); // ~20 M right of the hole
        const double g_left = g_at(-0.020, ph_l);
        INFO("a=" << a << " g_right=" << g_right << " (phi " << ph_r << ") g_left="
                  << g_left << " (phi " << ph_l << ")");
        CHECK(std::sin(ph_r) > 0.0);  // right of image = +y side of the hole
        CHECK(g_right < 1.0);         // receding: redshift
        CHECK(g_left > g_right);      // approaching side is the bright one
    }
    // Kerr shadow: prograde photons (they arrive from the image LEFT) orbit
    // closer, so the left edge sits nearer the look axis.
    const KerrSchild ks(1.0, 0.94, -1);
    const double left = shadow_edge(ks, 1000.0, kPi / 2, -1.0);
    const double right = shadow_edge(ks, 1000.0, kPi / 2, +1.0);
    INFO("left edge " << left << " right edge " << right);
    CHECK(left < right);
}

TEST_CASE("Carter (BL) and Kerr-Schild cameras render the same image",
          "[kerr-schild][handedness][gate16]") {
    // Both science paths now trace the RECEIVED photon backward; two
    // independent coordinate systems and integrators must agree pixel by
    // pixel — disk-hit radius and redshift — including which side is which.
    const double a = 0.94, r_cam = 1000.0, th = 80.0;
    const KerrMetric metric(1.0, a);
    CameraConfig cam;
    cam.r_cam = r_cam;
    cam.incl_deg = th;
    cam.fov_rad = 0.06;
    DiskConfig dc;
    dc.enabled = true;
    dc.r_in = metric.r_isco(true);
    dc.r_out = 40.0;
    const RaytraceCpu cpu(metric, cam, dc);
    const KerrSchild ks(1.0, a, -1);
    KSTraceOptions o;
    o.rtol = o.atol = 1e-11;
    const double t_half = std::tan(0.5 * cam.fov_rad);
    for (double x : {-0.6, -0.35, 0.35, 0.6}) {
        const TraceResult c = cpu.trace(x, 0.0);
        REQUIRE(c.outcome == RayOutcome::DiskHit);
        const KSRay ray = ks_camera_ray(ks, r_cam, th * kPi / 180.0,
                                        look(x * t_half, 0.0));
        const auto res = ks_trace(ks, ray, o);
        REQUIRE(res.n_cross >= 1);
        const double g_ks = disk_g(ks, res.cross_r[0], ray.X, ray.P);
        INFO("x=" << x << " CPU r=" << c.disk.r << " g=" << c.disk.g
                  << " | KS r=" << res.cross_r[0] << " g=" << g_ks);
        CHECK_THAT(c.disk.r, WithinRel(res.cross_r[0], 1e-6));
        CHECK_THAT(c.disk.g, WithinRel(g_ks, 1e-6));
    }
}

TEST_CASE("The legacy emitted-photon camera renders the exact mirror image",
          "[handedness][gate16]") {
    // Phases 1–12 launched k = e_t + n·e (the photon the camera would EMIT
    // toward n). In Kerr that traces the time-reversed (spin-reversed)
    // spacetime, so its pixel x shows what physically appears at −x — the
    // relation documented in the thesis appendix "Image handedness".
    // Scalar results (shadow/ring size, sub-ring ratios, |beaming|) are
    // invariant under the mirror; image left/right is not.
    const double a = 0.94;
    const KerrMetric metric(1.0, a);
    CameraConfig cam;
    cam.r_cam = 1000.0;
    cam.incl_deg = 80.0;
    cam.fov_rad = 0.06;
    DiskConfig dc;
    dc.enabled = true;
    dc.r_in = metric.r_isco(true);
    dc.r_out = 40.0;
    const RaytraceCpu now(metric, cam, dc);
    CameraConfig old_cam = cam;
    old_cam.legacy_emitted_photon = true;
    const RaytraceCpu old(metric, old_cam, dc);
    for (double x : {-0.6, -0.35, 0.35, 0.6}) {
        const TraceResult o = old.trace(x, 0.0), m = now.trace(-x, 0.0);
        REQUIRE(o.outcome == RayOutcome::DiskHit);
        REQUIRE(m.outcome == RayOutcome::DiskHit);
        INFO("x=" << x << " legacy r=" << o.disk.r << " g=" << o.disk.g
                  << " | physical(-x) r=" << m.disk.r << " g=" << m.disk.g);
        CHECK_THAT(o.disk.r, WithinRel(m.disk.r, 1e-9));
        CHECK_THAT(o.disk.g, WithinRel(m.disk.g, 1e-9));
        CHECK_THAT(o.disk.phi, WithinAbs(-m.disk.phi, 1e-8)); // azimuth mirrored
    }
}
