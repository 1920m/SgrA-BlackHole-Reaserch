// Gate 2: the Kerr shadow at a* = 0.94 must be lopsided (prograde side
// flattened) and its boundary must lie on the Bardeen (1973) analytic critical
// curve: ξ(r*) = [r*²(3M−r*) − a²(r*+M)]/[a(r*−M)], η(r*) = r*³[4a²M −
// r*(r*−3M)²]/[a²(r*−M)²] for some photon-shell radius r* ∈ [r_ph⁺, r_ph⁻].
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "sgr_physics/raytrace_cpu.hpp"

using namespace sgr;

namespace {

double bardeen_xi(double r, double M, double a) {
    return (r * r * (3.0 * M - r) - a * a * (r + M)) / (a * (r - M));
}
double bardeen_eta(double r, double M, double a) {
    return r * r * r * (4.0 * a * a * M - r * (r - 3.0 * M) * (r - 3.0 * M)) /
           (a * a * (r - M) * (r - M));
}

} // namespace

TEST_CASE("Kerr a*=0.94 shadow boundary lies on the Bardeen critical curve",
          "[raytrace][shadow][kerr][gate2]") {
    const double a = 0.94;
    const KerrMetric metric(1.0, a);
    CameraConfig cam;
    cam.r_cam = 1000.0;
    cam.incl_deg = 90.0; // edge-on: full critical curve visible
    cam.fov_rad = 0.032;
    const RaytraceCpu tracer(metric, cam, DiskConfig{});

    const double r_in = metric.r_photon(true);   // prograde shell edge
    const double r_out = metric.r_photon(false); // retrograde shell edge

    double alpha_east = 0.0, alpha_west = 0.0;
    int checked = 0;
    const int n_dirs = 16;
    for (int k = 0; k < n_dirs; ++k) {
        const double psi = 2.0 * 3.14159265358979323846 * k / n_dirs;
        const double s = tracer.shadow_boundary_angle(psi, 0.010);
        REQUIRE(s > 0.0);
        if (k == 0) alpha_east = s;
        if (k == n_dirs / 2) alpha_west = s;

        // Conserved quantities of the boundary ray:
        const double t_half = std::tan(0.5 * cam.fov_rad);
        const auto init = tracer.camera_ray(std::tan(s) * std::cos(psi) / t_half,
                                            std::tan(s) * std::sin(psi) / t_half);
        const double xi = init.constants.Lz / init.constants.E;
        const double eta = init.constants.Q / (init.constants.E * init.constants.E);

        // Solve ξ(r*) = ξ_measured on the photon shell (ξ is monotonic in r*),
        // then the ray is on the critical curve iff η(r*) matches.
        double lo = r_in, hi = r_out;
        // ξ decreases from ξ(r_in) > 0 to ξ(r_out) < 0 as r* grows.
        if (xi > bardeen_xi(r_in, 1.0, a) || xi < bardeen_xi(r_out, 1.0, a)) {
            INFO("psi=" << psi << " xi=" << xi << " outside shell range — skip");
            continue; // boundary ray at the very cusp; η-check degenerate
        }
        for (int it = 0; it < 80; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (bardeen_xi(mid, 1.0, a) > xi) lo = mid; else hi = mid;
        }
        const double r_star = 0.5 * (lo + hi);
        const double eta_analytic = bardeen_eta(r_star, 1.0, a);

        INFO("psi=" << psi << " s=" << s << " xi=" << xi << " eta=" << eta
                    << " r*=" << r_star << " eta_analytic=" << eta_analytic);
        // Normalise by the curve scale (η ranges up to ~27):
        CHECK(std::abs(eta - eta_analytic) / 27.0 < 2e-4);
        ++checked;
    }
    CHECK(checked >= 12); // the η-check ran for most directions

    // Lopsidedness: the two equatorial extents from the look axis must be
    // asymmetric by far more than any numerical wiggle.
    const double ratio = std::max(alpha_east, alpha_west) /
                         std::min(alpha_east, alpha_west);
    INFO("east extent=" << alpha_east << " west extent=" << alpha_west
                        << " ratio=" << ratio);
    CHECK(ratio > 1.5);
    // Handedness (the camera fixes it: image up = spin axis, image right =
    // +φ̂). Prograde photons (L_z > 0) reach the camera from the image LEFT
    // and skim the hole closest, so the WEST edge sits nearer the axis. The
    // Phases 1–12 camera put it on the right — the mirror image (thesis
    // appendix "Image handedness"); this check now pins it.
    CHECK(alpha_west < alpha_east);
}

TEST_CASE("Kerr shadow width shrinks and displaces with spin",
          "[raytrace][shadow][kerr]") {
    // Analytic expectation at i = 90°: horizontal extent spans ξ ∈ [ξ(r⁻), ξ(r⁺)],
    // width |Δα| = ξ(r_pro) − ξ(r_retro) ≈ 9.54 M for a* = 0.94 vs 2√27 ≈ 10.39 M
    // for Schwarzschild.
    const double a = 0.94;
    const KerrMetric metric(1.0, a);
    CameraConfig cam;
    cam.r_cam = 1000.0;
    cam.incl_deg = 90.0;
    cam.fov_rad = 0.032;
    const RaytraceCpu tracer(metric, cam, DiskConfig{});

    const double s_east = tracer.shadow_boundary_angle(0.0, 0.010);
    const double s_west = tracer.shadow_boundary_angle(3.14159265358979323846, 0.010);

    // Convert observer angles to asymptotic impact parameters: b = ξ measured
    // from the boundary-ray constants directly.
    const double t_half = std::tan(0.5 * cam.fov_rad);
    const auto ray_e = tracer.camera_ray(std::tan(s_east) / t_half, 0.0);
    const auto ray_w = tracer.camera_ray(-std::tan(s_west) / t_half, 0.0);
    const double xi_e = ray_e.constants.Lz / ray_e.constants.E;
    const double xi_w = ray_w.constants.Lz / ray_w.constants.E;

    // Prograde (ξ > 0) boundary photons arrive from the image left (west),
    // retrograde ones from the right.
    CHECK(xi_w > 0.0);
    CHECK(xi_e < 0.0);
    const double width = std::abs(xi_e - xi_w);
    const double width_analytic =
        bardeen_xi(metric.r_photon(true), 1.0, a) -
        bardeen_xi(metric.r_photon(false), 1.0, a);
    INFO("xi_east=" << xi_e << " xi_west=" << xi_w << " width=" << width
                    << " analytic=" << width_analytic);
    CHECK_THAT(width, Catch::Matchers::WithinRel(width_analytic, 1e-3));
    // Narrower than Schwarzschild:
    CHECK(width < 2.0 * std::sqrt(27.0) - 0.5);
}
