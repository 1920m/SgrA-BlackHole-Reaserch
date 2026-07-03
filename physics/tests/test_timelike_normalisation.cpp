// Gate 0: |g_μν u^μ u^ν + 1| < 1e-8 along timelike geodesics,
// via the Christoffel-based integrator (the one the S-star orbit fitter uses).
// Also validates the physics end-to-end: the measured pericenter advance of a
// weak-field orbit must match the analytic Schwarzschild precession formula.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "sgr_physics/integrator.hpp"
#include "sgr_physics/metric.hpp"
#include "sgr_physics/timelike_geodesics.hpp"

using namespace sgr;

namespace {

// Set up a bound orbit starting at pericenter r_p in the equatorial-ish plane
// with local inclination angle iota (0 = equatorial), using the Newtonian
// pericenter speed as the initial tangential velocity in the local static
// frame (exact ellipse data are not needed — any bound orbit does).
TimelikeState pericenter_start(const IMetric& metric, double r_p, double e,
                               double iota) {
    const double M = metric.mass();
    const double v = std::sqrt(M * (1.0 + e) / r_p); // Newtonian peri speed
    const double th0 = 1.5707963267948966;
    const Metric4<double> g = metric.components(r_p, th0);
    // Local orthonormal tangential direction split between φ̂ and θ̂:
    const double uphi = v * std::cos(iota) / std::sqrt(g(3, 3));
    const double uth = v * std::sin(iota) / std::sqrt(g(2, 2));

    TimelikeState y{};
    y[1] = r_p;
    y[2] = th0;
    y[5] = 0.0;    // u^r = 0 at pericenter
    y[6] = uth;
    y[7] = uphi;
    y[4] = TimelikeGeodesic::solve_ut(g, y[5], y[6], y[7]);
    return y;
}

struct OrbitRunResult {
    double worst_norm_err = 0.0;   // max |g u u + 1|
    double worst_dE = 0.0;         // max relative E drift
    double worst_dL = 0.0;         // max relative L_z drift
    double worst_dQ = 0.0;         // max relative Carter Q drift (Kerr only)
    double delta_phi = 0.0;        // φ advance between successive pericenters
    int steps = 0;
};

// Integrate from pericenter through apocenter to the next pericenter (u^r
// crossing − → +), monitoring conserved quantities at every accepted step.
OrbitRunResult run_one_radial_period(const IMetric& metric, TimelikeState y,
                                     bool track_carter_q = false) {
    TimelikeGeodesic geo(metric);
    TimelikeStepper stepper{&geo, 1e-12, 1e-12};

    const double E0 = geo.energy(y);
    const double L0 = geo.angmom(y);
    const double Q0 = track_carter_q ? geo.carter_q(y, metric.spin()) : 0.0;

    OrbitRunResult out;
    double tau = 0.0, h = 1.0;
    bool left_pericenter = false;
    TimelikeState y_prev = y;
    double tau_prev = tau;

    for (int i = 0; i < 2'000'000; ++i) {
        y_prev = y;
        tau_prev = tau;
        stepper.step(tau, y, h);
        ++out.steps;

        out.worst_norm_err =
            std::max(out.worst_norm_err, std::abs(geo.norm(y) + 1.0));
        out.worst_dE =
            std::max(out.worst_dE, std::abs(geo.energy(y) - E0) / std::abs(E0));
        out.worst_dL =
            std::max(out.worst_dL, std::abs(geo.angmom(y) - L0) / std::abs(L0));
        if (track_carter_q && std::abs(Q0) > 1e-12) {
            out.worst_dQ = std::max(
                out.worst_dQ,
                std::abs(geo.carter_q(y, metric.spin()) - Q0) / std::abs(Q0));
        }

        if (y[5] < -1e-14) left_pericenter = true; // moving inward: past apo
        if (left_pericenter && y_prev[5] < 0.0 && y[5] >= 0.0) {
            // Crossed the next pericenter within this step: bisect on u^r.
            double lo = tau_prev, hi = tau;
            TimelikeState ylo = y_prev;
            for (int it = 0; it < 60; ++it) {
                const double mid = 0.5 * (lo + hi);
                TimelikeState trial = ylo;
                double t0 = lo;
                // one fixed sub-step to mid
                auto rhs_fn = [&geo](double, const TimelikeState& s,
                                     TimelikeState& d) { geo.rhs(s, d); };
                DormandPrince54<8, decltype(rhs_fn)> rk(rhs_fn, 1e30, 1e30);
                rk.try_step(t0, trial, mid - lo);
                if (trial[5] < 0.0) {
                    lo = mid;
                    ylo = trial;
                } else {
                    hi = mid;
                }
            }
            out.delta_phi = ylo[3] - 0.0; // φ started at 0
            return out;
        }
    }
    FAIL("run_one_radial_period: did not return to pericenter");
    return out;
}

} // namespace

TEST_CASE("4-velocity normalisation on an S2-scale Kerr orbit",
          "[timelike][normalisation]") {
    // S2 scales in units of M: pericenter ≈ 2818 M, e ≈ 0.884, slight
    // inclination to exercise all four velocity components.
    const KerrMetric metric(1.0, 0.94);
    const auto y0 = pericenter_start(metric, 2818.0, 0.884, 0.35);
    const auto res = run_one_radial_period(metric, y0, /*track_carter_q=*/true);

    INFO("steps = " << res.steps << ", delta_phi = " << res.delta_phi);
    CHECK(res.steps > 100);            // enough steps for the drift to show
    CHECK(res.worst_norm_err < 1e-8);
    CHECK(res.worst_dE < 1e-6);
    CHECK(res.worst_dL < 1e-6);
    CHECK(res.worst_dQ < 1e-6);
}

TEST_CASE("4-velocity normalisation on a strong-field Kerr orbit",
          "[timelike][normalisation]") {
    // Pericenter at 8 M around a* = 0.94: strong-field, rapid precession.
    const KerrMetric metric(1.0, 0.94);
    const auto y0 = pericenter_start(metric, 8.0, 0.5, 0.2);
    const auto res = run_one_radial_period(metric, y0, /*track_carter_q=*/true);

    INFO("steps = " << res.steps << ", delta_phi = " << res.delta_phi);
    CHECK(res.worst_norm_err < 1e-8);
    CHECK(res.worst_dE < 1e-6);
    CHECK(res.worst_dL < 1e-6);
    CHECK(res.worst_dQ < 1e-6);
}

// NOTE: ASCII-only test name (ctest Windows filter mangles Unicode).
TEST_CASE("Measured pericenter precession matches 6piM/(a(1-e^2)) in weak field",
          "[timelike][precession][physics]") {
    // Schwarzschild, equatorial orbit with p = a(1−e²) ≈ 1000 M. The leading
    // GR precession is Δω = 6πM/p per orbit; higher-order corrections are
    // O(M/p) ~ 1e-3 relative, so agreement at 1% is a genuine validation of
    // the Christoffel pipeline (this is the same formula scaled to S2 that
    // gives the GRAVITY 2020 12.1 arcmin/orbit).
    const KerrMetric metric(1.0, 0.0);
    const double e = 0.4, r_p = 700.0;
    const auto y0 = pericenter_start(metric, r_p, e, 0.0);
    const auto res = run_one_radial_period(metric, y0);

    const double precession = res.delta_phi - 2.0 * 3.14159265358979323846;
    // The orbit launched with the *Newtonian* pericenter speed is not exactly
    // the (p, e) ellipse, but its own p can be measured: p = L²/M in the
    // Newtonian limit. Use the conserved L of the actual orbit.
    TimelikeGeodesic geo(metric);
    const double L = geo.angmom(y0);
    const double p_orbit = L * L / metric.mass();
    const double analytic = 6.0 * 3.14159265358979323846 * metric.mass() / p_orbit;

    INFO("measured precession = " << precession << " rad/orbit, analytic = "
                                  << analytic);
    CHECK(precession > 0.0); // prograde advance
    CHECK_THAT(precession, Catch::Matchers::WithinRel(analytic, 0.01));
}
