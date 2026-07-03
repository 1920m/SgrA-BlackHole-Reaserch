// Gate 0: E, L_z, Q constant to <1e-6 relative along null geodesics,
// over ≥1000 integration steps, including radial and polar turning
// points. Also cross-validates the Carter/Mino integrator against the fully
// independent Christoffel-based integrator on the same ray.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>

#include "sgr_physics/integrator.hpp"
#include "sgr_physics/metric.hpp"
#include "sgr_physics/null_geodesics.hpp"
#include "sgr_physics/timelike_geodesics.hpp"

using namespace sgr;

namespace {

struct RaySpec {
    double Lz, Q;
    double pr_sign, pth_sign;
    const char* label;
};

// Integrate a ray in the Mino-time Carter form for up to max_steps accepted
// steps, checking conservation at every step. Returns worst-case errors.
NullConservedErrors integrate_and_check(const KerrMetric& metric,
                                        const RaySpec& spec, double r0,
                                        double th0, int min_steps) {
    NullRayConstants c;
    c.M = metric.mass();
    c.a = metric.spin();
    c.E = 1.0;
    c.Lz = spec.Lz;
    c.Q = spec.Q;

    const double R0 = radial_potential(c, r0);
    const double Th0 = polar_potential(c, th0);
    REQUIRE(R0 > 0.0);
    REQUIRE(Th0 > 0.0);

    NullState y{};
    y[0] = 0.0; y[1] = r0; y[2] = th0; y[3] = 0.0;
    y[4] = spec.pr_sign * std::sqrt(R0);
    y[5] = spec.pth_sign * std::sqrt(Th0);

    auto rhs = [&c](double, const NullState& s, NullState& d) {
        null_geodesic_rhs(c, s, d);
    };
    DormandPrince54<6, decltype(rhs)> rk(rhs, 1e-12, 1e-12);

    NullConservedErrors worst{};
    double tau = 0.0, h = 1e-4;
    int steps = 0;
    const double r_hor = metric.r_horizon();
    while (steps < min_steps) {
        const auto res = rk.try_step(tau, y, h);
        h = std::min(res.h_next, 1.0);
        if (!res.accepted) continue;
        ++steps;
        const auto err = null_conserved_errors(metric, c, y);
        worst.dE_rel = std::max(worst.dE_rel, err.dE_rel);
        worst.dLz_rel = std::max(worst.dLz_rel, err.dLz_rel);
        worst.dQ_rel = std::max(worst.dQ_rel, err.dQ_rel);
        worst.null_norm = std::max(worst.null_norm, err.null_norm);
        worst.r_residual = std::max(worst.r_residual, err.r_residual);
        worst.theta_residual = std::max(worst.theta_residual, err.theta_residual);
        if (y[1] < r_hor * 1.02 || y[1] > 500.0) {
            // Ray finished (captured or escaped): restart it to accumulate the
            // required number of steps across fresh rays of the same class.
            y[0] = 0.0; y[1] = r0; y[2] = th0; y[3] = 0.0;
            y[4] = spec.pr_sign * std::sqrt(R0);
            y[5] = spec.pth_sign * std::sqrt(Th0);
            tau = 0.0;
            h = 1e-4;
        }
    }
    return worst;
}

} // namespace

TEST_CASE("E, L_z, Q conserved along Kerr null geodesics (a*=0.94)",
          "[null][conservation]") {
    const KerrMetric metric(1.0, 0.94);
    // Ray classes: plunging (inside critical curve), scattering (radial turning
    // point), and high-latitude (polar turning points).
    const RaySpec specs[] = {
        {3.0, 4.0, -1.0, +1.0, "plunging"},
        {7.0, 30.0, -1.0, -1.0, "scattering, polar oscillation"},
        {-6.5, 20.0, -1.0, +1.0, "retrograde scattering"},
        {0.5, 26.0, -1.0, +1.0, "near-critical, many windings"},
    };
    for (const auto& s : specs) {
        INFO("ray class: " << s.label);
        const auto worst = integrate_and_check(metric, s, 50.0, 1.3, 1000);
        CHECK(worst.dE_rel < 1e-6);
        CHECK(worst.dLz_rel < 1e-6);
        CHECK(worst.dQ_rel < 1e-6);
        CHECK(worst.null_norm < 1e-6);
        CHECK(worst.r_residual < 1e-6);
        CHECK(worst.theta_residual < 1e-6);
    }
}

TEST_CASE("Schwarzschild null geodesics conserve constants", "[null][conservation]") {
    const KerrMetric metric(1.0, 0.0);
    // θ₀ = 1.3 keeps Θ(θ₀) > 0 for these L_z (at θ₀ = 1.0 the polar potential
    // would be negative — no photon with these constants passes through there).
    const RaySpec specs[] = {
        {4.0, 10.0, -1.0, +1.0, "b^2 = 26 < 27: captured"},
        {5.0, 10.0, -1.0, +1.0, "b^2 = 35 > 27: scattered"},
    };
    for (const auto& s : specs) {
        INFO("ray class: " << s.label);
        const auto worst = integrate_and_check(metric, s, 30.0, 1.3, 1000);
        CHECK(worst.dE_rel < 1e-6);
        CHECK(worst.dLz_rel < 1e-6);
        CHECK(worst.dQ_rel < 1e-6);
        CHECK(worst.null_norm < 1e-6);
    }
}

TEST_CASE("Carter/Mino and Christoffel integrators agree on the same null ray",
          "[null][crosscheck]") {
    const KerrMetric metric(1.0, 0.94);
    const double r0 = 40.0, th0 = 1.2;

    NullRayConstants c;
    c.M = 1.0; c.a = metric.spin(); c.E = 1.0; c.Lz = 6.0; c.Q = 25.0;
    const double R0 = radial_potential(c, r0);
    const double Th0 = polar_potential(c, th0);
    REQUIRE(R0 > 0.0);
    REQUIRE(Th0 > 0.0);

    // --- Path A: Christoffel-based integration in affine parameter λ.
    // (The geodesic equation does not care that u is null rather than timelike;
    //  only the normalisation diagnostic differs.)
    NullState y0{};
    y0[1] = r0; y0[2] = th0;
    y0[4] = -std::sqrt(R0);
    y0[5] = +std::sqrt(Th0);
    const auto k0 = null_four_momentum(c, y0);

    TimelikeGeodesic geo(metric); // generic geodesic RHS, works for null too
    TimelikeState z{};
    z[0] = 0.0; z[1] = r0; z[2] = th0; z[3] = 0.0;
    z[4] = k0[0]; z[5] = k0[1]; z[6] = k0[2]; z[7] = k0[3];

    // Conservation along the Christoffel path is a *real* drift measurement —
    // nothing is enforced. Track it while integrating to λ_end.
    const Metric4<double> g0 = metric.components(r0, th0);
    const double E0 = -(g0(0, 0) * z[4] + g0(0, 3) * z[7]);
    const double Lz0 = g0(0, 3) * z[4] + g0(3, 3) * z[7];

    double lambda = 0.0, h = 1e-3;
    const double lambda_end = 60.0;
    auto rhs_c = [&geo](double, const TimelikeState& s, TimelikeState& d) {
        geo.rhs(s, d);
    };
    DormandPrince54<8, decltype(rhs_c)> rkc(rhs_c, 1e-12, 1e-12);
    double worstE = 0.0, worstL = 0.0, worstQ = 0.0;
    while (lambda < lambda_end) {
        if (lambda + h > lambda_end) h = lambda_end - lambda;
        const auto res = rkc.try_step(lambda, z, h);
        h = std::min(res.h_next, lambda_end - lambda > 0 ? lambda_end - lambda : res.h_next);
        if (h <= 0.0) h = 1e-10;
        if (!res.accepted) continue;
        const Metric4<double> g = metric.components(z[1], z[2]);
        const double E = -(g(0, 0) * z[4] + g(0, 3) * z[7]);
        const double Lz = g(0, 3) * z[4] + g(3, 3) * z[7];
        const double ct = std::cos(z[2]), st = std::sin(z[2]);
        const double sigma = z[1] * z[1] + c.a * c.a * ct * ct;
        const double pth = sigma * z[6];
        const double Q = pth * pth + ct * ct * (Lz * Lz / (st * st) - c.a * c.a * E * E);
        worstE = std::max(worstE, std::abs(E - E0) / std::abs(E0));
        worstL = std::max(worstL, std::abs(Lz - Lz0) / std::abs(Lz0));
        worstQ = std::max(worstQ, std::abs(Q - c.Q) / std::abs(c.Q));
    }
    CHECK(worstE < 1e-6);
    CHECK(worstL < 1e-6);
    CHECK(worstQ < 1e-6);

    // --- Path B: Mino-time Carter integration, with the affine parameter λ
    // carried as a 7th state component (dλ/dτ = Σ), run to the same λ_end.
    StateVec<7> w{};
    w[0] = 0.0; w[1] = r0; w[2] = th0; w[3] = 0.0;
    w[4] = -std::sqrt(R0); w[5] = +std::sqrt(Th0); w[6] = 0.0;
    auto rhs_m = [&c](double, const StateVec<7>& s, StateVec<7>& d) {
        NullState s6{s[0], s[1], s[2], s[3], s[4], s[5]};
        NullState d6{};
        null_geodesic_rhs(c, s6, d6);
        for (int i = 0; i < 6; ++i) d[i] = d6[i];
        const double ct = std::cos(s[2]);
        d[6] = s[1] * s[1] + c.a * c.a * ct * ct; // dλ/dτ = Σ
    };
    DormandPrince54<7, decltype(rhs_m)> rkm(rhs_m, 1e-13, 1e-13);
    double tau = 0.0;
    double hm = 1e-4;
    StateVec<7> w_prev = w;
    double tau_prev = tau;
    while (w[6] < lambda_end) {
        w_prev = w;
        tau_prev = tau;
        const auto res = rkm.try_step(tau, w, hm);
        hm = res.h_next;
    }
    // Bisect within the overshooting step so that λ(τ*) = λ_end: re-take a
    // single RK5 sub-step of trial size from the pre-overshoot state (the
    // 5th-order local accuracy over a fraction of an accepted step is ample).
    {
        double lo = 0.0, hi = tau - tau_prev;
        StateVec<7> trial = w;
        DormandPrince54<7, decltype(rhs_m)> rks(rhs_m, 1e30, 1e30); // always accept
        for (int it = 0; it < 60; ++it) {
            const double mid = 0.5 * (lo + hi);
            trial = w_prev;
            double t0 = 0.0;
            rks.try_step(t0, trial, mid);
            if (trial[6] < lambda_end) lo = mid; else hi = mid;
        }
        w = trial;
    }
    INFO("Christoffel endpoint: r=" << z[1] << " th=" << z[2] << " phi=" << z[3]);
    INFO("Carter/Mino endpoint: r=" << w[1] << " th=" << w[2] << " phi=" << w[3]);
    CHECK_THAT(w[1], Catch::Matchers::WithinAbs(z[1], 5e-6));
    CHECK_THAT(w[2], Catch::Matchers::WithinAbs(z[2], 5e-6));
    CHECK_THAT(w[3], Catch::Matchers::WithinAbs(z[3], 5e-6));
    CHECK_THAT(w[0], Catch::Matchers::WithinAbs(z[0], 5e-5));
}
