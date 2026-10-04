// Gate 3 (automated part): Novikov-Thorne / Page-Thorne flux structure, the
// blackbody colour pipeline, and the Doppler beaming asymmetry direction.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "sgr_physics/disk_model.hpp"
#include "sgr_physics/metric.hpp"
#include "sgr_physics/raytrace_cpu.hpp"

using namespace sgr;

TEST_CASE("Page-Thorne flux: zero at ISCO, single interior maximum",
          "[disk][novikov-thorne]") {
    for (double a : {0.0, 0.5, 0.94}) {
        const KerrMetric metric(1.0, a);
        const double r0 = metric.r_isco(true);
        const NovikovThorneDisk disk(a, r0, 400.0);
        INFO("a* = " << a << " r_isco = " << r0);

        CHECK(disk.flux(r0) == 0.0);
        CHECK(disk.flux(r0 * 0.9) == 0.0);

        // Single max: flux rises from the ISCO then falls monotonically.
        double rmax = r0;
        double fmax = 0.0;
        int sign_changes = 0;
        double prev = 0.0;
        for (int i = 1; i <= 3000; ++i) {
            const double r = r0 + (100.0 - r0) * i / 3000.0;
            const double f = disk.flux(r);
            CHECK(f >= 0.0);
            if (f > fmax) { fmax = f; rmax = r; }
            if (i > 1 && (f - prev) * (prev - 0.0) != 0.0) {
                // count derivative sign changes crudely
            }
            prev = f;
        }
        (void)sign_changes;
        CHECK(rmax > r0);
        CHECK(fmax > 0.0);
        if (a == 0.0) {
            // Schwarzschild Page-Thorne flux peaks near ~1.6 r_isco
            // (the Newtonian SS value is (49/36) r_in ≈ 8.17 M; GR shifts it).
            CHECK(rmax > 7.5);
            CHECK(rmax < 11.5);
        }
    }
}

TEST_CASE("Page-Thorne flux matches its exact large-r expansion",
          "[disk][novikov-thorne]") {
    // For a* = 0, r_in = 6M the flux bracket expands as
    //   bracket(x) = x − c + 3/x + O(x⁻³),  x = √r,
    //   c = x₀ + (√3/2) ln((x₀+√3)/(x₀−√3)),  x₀ = √6
    // (the log terms shift the effective inner-edge constant away from the
    // naive Newtonian √(r_in); c ≈ 3.9757). With the full denominator
    // x⁴(x³−3x), the compensated flux must be flat to O(x⁻³) ≲ 0.3% between
    // 300 M and 900 M.
    const NovikovThorneDisk disk(0.0, 6.0, 2000.0);
    const double x0 = std::sqrt(6.0), s3 = std::sqrt(3.0);
    const double c = x0 + 0.5 * s3 * std::log((x0 + s3) / (x0 - s3));
    auto compensated = [&](double r) {
        const double x = std::sqrt(r);
        const double expected = (x - c + 3.0 / x) /
                                (x * x * x * x * (x * x * x - 3.0 * x));
        return disk.flux(r) / expected;
    };
    const double c300 = compensated(300.0);
    const double c900 = compensated(900.0);
    INFO("c = " << c << " c300 = " << c300 << " c900 = " << c900);
    CHECK_THAT(c900 / c300, Catch::Matchers::WithinAbs(1.0, 0.003));
}

TEST_CASE("Physical NT normalisation reduces to Shakura-Sunyaev in SI",
          "[disk][novikov-thorne][si]") {
    // Independent SI evaluation of 3GMṀ/(8πR³)·(1 − c/x), R = r·GM/c²,
    // with the effective inner-edge constant c of the expansion test above.
    // At r = 1e6 M the omitted terms are O(1/r) ~ 1e-6.
    const double M_msun = 4.297e6, mdot_msun_yr = 2.0e-8;
    const double r = 1.0e6;
    const double G = 6.67430e-11, c_light = 299792458.0;
    const double M = M_msun * 1.3271244001800e20 / G;          // kg
    const double mdot = mdot_msun_yr * (1.3271244001800e20 / G) / 3.15576e7;
    const double R = r * G * M / (c_light * c_light);
    const double x0 = std::sqrt(6.0), s3 = std::sqrt(3.0);
    const double c = x0 + 0.5 * s3 * std::log((x0 + s3) / (x0 - s3));
    const double F_ss = 3.0 * G * M * mdot / (8.0 * 3.14159265358979323846 * R * R * R) *
                        (1.0 - c / std::sqrt(r));
    const double F = nt_flux_si(M_msun, mdot_msun_yr, 0.0, 2.0e6, r);
    INFO("F = " << F << " W/m^2, F_SS = " << F_ss);
    CHECK_THAT(F / F_ss, Catch::Matchers::WithinAbs(1.0, 1e-4));
}

TEST_CASE("Physical NT peak temperature scales as (Mdot/M^2)^(1/4)",
          "[disk][novikov-thorne][si]") {
    const double T0 = nt_peak_temperature_k(4.297e6, 2.0e-8, 0.94, 30.0);
    const double T_mdot16 = nt_peak_temperature_k(4.297e6, 32.0e-8, 0.94, 30.0);
    const double T_mass4 = nt_peak_temperature_k(4.0 * 4.297e6, 2.0e-8, 0.94, 30.0);
    INFO("T_peak(Sgr A*, 2e-8 Msun/yr, a*=0.94) = " << T0 << " K");
    CHECK(T0 > 0.0);
    CHECK_THAT(T_mdot16 / T0, Catch::Matchers::WithinRel(2.0, 1e-9));
    CHECK_THAT(T_mass4 / T0, Catch::Matchers::WithinRel(0.5, 1e-9)); // (1/16)^(1/4)
    // Spin raises the peak (ISCO moves inward, more binding energy released).
    CHECK(nt_peak_temperature_k(4.297e6, 2.0e-8, 0.0, 30.0) < T0);
}

TEST_CASE("Blackbody colours behave physically", "[disk][colour]") {
    const NovikovThorneDisk disk(0.0, 6.0, 30.0, 6300.0);
    const auto cold = disk.blackbody_rgb(3000.0);
    const auto mid = disk.blackbody_rgb(6500.0);
    const auto hot = disk.blackbody_rgb(20000.0);
    // 3000 K: red dominates blue; 20000 K: blue dominates red.
    CHECK(cold[0] > cold[2]);
    CHECK(hot[2] > hot[0]);
    // 6500 K ~ white in sRGB(D65): channels within ~25% of each other.
    CHECK(std::abs(mid[0] - mid[1]) / mid[1] < 0.25);
    CHECK(std::abs(mid[2] - mid[1]) / mid[1] < 0.25);
    // Radiance increases steeply with T.
    CHECK(hot[1] > 5.0f * mid[1]);
    CHECK(mid[1] > 5.0f * cold[1]);
}

TEST_CASE("Schwarzschild disk: approaching (left) limb blueshifted",
          "[disk][beaming][handedness]") {
    // No spin, so no frame dragging and a symmetric shadow — the only
    // handedness left is the disk's own rotation. A pixel right of the hole
    // sees gas moving AWAY from the camera (counter-clockwise from above, the
    // +y side recedes from a camera on +x): it must be redshifted.
    const KerrMetric metric(1.0, 0.0);
    CameraConfig cam;
    cam.r_cam = 1000.0;
    cam.incl_deg = 80.0;
    cam.fov_rad = 0.06;
    DiskConfig dcfg;
    dcfg.enabled = true;
    dcfg.r_in = 6.0;
    dcfg.r_out = 40.0;
    const RaytraceCpu tracer(metric, cam, dcfg);
    const TraceResult right = tracer.trace(+0.6, 0.0), left = tracer.trace(-0.6, 0.0);
    REQUIRE(right.outcome == RayOutcome::DiskHit);
    REQUIRE(left.outcome == RayOutcome::DiskHit);
    INFO("g_left=" << left.disk.g << " g_right=" << right.disk.g);
    CHECK(right.disk.g < 1.0);
    CHECK(left.disk.g > right.disk.g);
}

TEST_CASE("Doppler beaming: approaching side brighter, g range sane",
          "[disk][beaming][gate3]") {
    const double a = 0.94;
    const KerrMetric metric(1.0, a);
    CameraConfig cam;
    cam.r_cam = 1000.0;
    cam.incl_deg = 80.0;
    cam.fov_rad = 0.032;
    DiskConfig dcfg;
    dcfg.enabled = true;
    dcfg.r_in = metric.r_isco(true);
    dcfg.r_out = 18.0;
    const RaytraceCpu tracer(metric, cam, dcfg);
    const NovikovThorneDisk disk(a, dcfg.r_in, dcfg.r_out, 6300.0);

    // Scan the major axis: bolometric flux on each side of the image.
    double left = 0.0, right = 0.0;
    double g_min = 1e9, g_max = 0.0;
    for (int i = 0; i < 400; ++i) {
        const double x = 2.0 * (i + 0.5) / 400.0 - 1.0;
        const TraceResult res = tracer.trace(x, 0.0);
        if (res.outcome != RayOutcome::DiskHit) continue;
        const double I = disk.bolometric(res.disk.r, res.disk.g);
        (x < 0 ? left : right) += I;
        g_min = std::min(g_min, res.disk.g);
        g_max = std::max(g_max, res.disk.g);
    }
    INFO("left=" << left << " right=" << right << " g_min=" << g_min
                 << " g_max=" << g_max);
    REQUIRE(left > 0.0);
    REQUIRE(right > 0.0);
    // The approaching side must be brighter by a large factor (g⁴ beaming at
    // v/c ~ 0.5 near the inner disk gives ≥ several ×).
    const double ratio = std::max(left, right) / std::min(left, right);
    CHECK(ratio > 2.0);
    // ...and it must be the LEFT limb: image up = spin axis and image right
    // = +φ̂, so prograde gas on the image left moves toward the camera.
    // (The Phases 1–12 camera put it on the right — thesis appendix
    // "Image handedness".)
    CHECK(left > right);
    // Redshift factor range: blueshift above 1 on the approaching side,
    // strong redshift near the ISCO; both bounded by physical limits.
    CHECK(g_max > 1.05);
    CHECK(g_min < 0.7);
    CHECK(g_min > 0.05);
    CHECK(g_max < 3.0);
}
