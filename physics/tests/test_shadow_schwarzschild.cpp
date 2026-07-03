// Gate 1: the Schwarzschild shadow must be a perfect circle whose angular size
// matches the analytic Synge (1966) formula sin α = √27 M √(1−2M/r_obs)/r_obs.
// The boundary is located by bisection in camera-angle space (no pixel grid),
// so this checks the ray tracer's geometry to integrator precision.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "sgr_physics/raytrace_cpu.hpp"

using namespace sgr;

TEST_CASE("Schwarzschild shadow is circular with the analytic angular size",
          "[raytrace][shadow][gate1]") {
    const KerrMetric metric(1.0, 0.0);
    CameraConfig cam;
    cam.r_cam = 1000.0;
    cam.incl_deg = 90.0;
    cam.fov_rad = 0.032;
    DiskConfig disk; // disabled
    const RaytraceCpu tracer(metric, cam, disk);

    const double analytic = schwarzschild_shadow_angle(1.0, cam.r_cam);

    double min_r = 1e9, max_r = 0.0, sum = 0.0;
    const int n_dirs = 16;
    for (int k = 0; k < n_dirs; ++k) {
        const double psi = 2.0 * 3.14159265358979323846 * k / n_dirs;
        const double alpha = tracer.shadow_boundary_angle(psi, 0.009);
        INFO("psi = " << psi << " alpha = " << alpha
                      << " (analytic " << analytic << ")");
        REQUIRE(alpha > 0.0);
        REQUIRE(alpha < 0.009);
        min_r = std::min(min_r, alpha);
        max_r = std::max(max_r, alpha);
        sum += alpha;
    }
    const double mean = sum / n_dirs;

    // Perfect circle: direction-to-direction spread ≪ the radius itself.
    CHECK((max_r - min_r) / mean < 1e-4);
    // Angular size matches the analytic value.
    CHECK_THAT(mean, Catch::Matchers::WithinRel(analytic, 1e-3));
}

// Hidden diagnostic (run with: sgr_physics_tests "[.probe]"): classify rays in
// the boundary layer around the critical angle at several tolerances.
TEST_CASE("Shadow boundary layer probe", "[.probe]") {
    const KerrMetric metric(1.0, 0.0);
    CameraConfig cam;
    cam.r_cam = 1000.0;
    cam.fov_rad = 0.032;
    const double analytic = schwarzschild_shadow_angle(1.0, cam.r_cam);
    const double t_half = std::tan(0.5 * cam.fov_rad);

    for (double tol : {1e-9, 1e-11, 1e-13}) {
        TraceOptions opts;
        opts.atol = opts.rtol = tol;
        const RaytraceCpu tracer(metric, cam, DiskConfig{}, opts);
        for (double f : {0.995, 0.998, 0.9985, 0.999, 0.9995, 0.9999,
                         1.0001, 1.0005, 1.001, 1.005}) {
            const auto res = tracer.trace(std::tan(f * analytic) / t_half, 0.0);
            const char* what = res.outcome == RayOutcome::Captured ? "CAPTURED"
                               : res.outcome == RayOutcome::Escaped ? "escaped"
                                                                    : "MAXSTEPS";
            WARN("tol=" << tol << " f=" << f << " -> " << what
                        << " (crossings=" << res.total_equator_crossings << ")");
        }
    }
}

TEST_CASE("Rays inside/outside the critical impact parameter behave correctly",
          "[raytrace][shadow]") {
    // Direct physics check on equatorial rays: b < √27 M captured, b > √27 M
    // escapes, independent of the camera machinery.
    const KerrMetric metric(1.0, 0.0);
    CameraConfig cam;
    cam.r_cam = 1000.0;
    cam.fov_rad = 0.032;
    const RaytraceCpu tracer(metric, cam, DiskConfig{});

    const double analytic = schwarzschild_shadow_angle(1.0, cam.r_cam);
    const double t_half = std::tan(0.5 * cam.fov_rad);

    // 4% inside the critical angle → captured
    auto res_in = tracer.trace(std::tan(0.96 * analytic) / t_half, 0.0);
    CHECK(res_in.outcome == RayOutcome::Captured);
    // 4% outside → escapes
    auto res_out = tracer.trace(std::tan(1.04 * analytic) / t_half, 0.0);
    CHECK(res_out.outcome == RayOutcome::Escaped);
}
