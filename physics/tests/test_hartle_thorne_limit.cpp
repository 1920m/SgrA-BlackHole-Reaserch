// Gate 0: HartleThorneMetric at ε_q = 0 must reproduce Kerr to 1e-12,
// plus structural checks of the quadrupole deviation.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "sgr_physics/hartle_thorne.hpp"
#include "sgr_physics/metric.hpp"
#include "sgr_physics/timelike_geodesics.hpp"

using namespace sgr;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("HT metric at eps_q = 0 equals Kerr to 1e-12", "[hartle-thorne][limit]") {
    for (double astar : {0.0, 0.5, 0.94}) {
        const KerrMetric kerr(1.0, astar);
        const HartleThorneMetric ht(1.0, astar, 0.0);
        for (double r : {2.5, 6.0, 15.0, 100.0, 3000.0}) {
            for (double th : {0.2, 0.9, 1.5707963267948966, 2.6}) {
                const auto gk = kerr.components(r, th);
                const auto gh = ht.components(r, th);
                for (int mu = 0; mu < 4; ++mu) {
                    for (int nu = 0; nu < 4; ++nu) {
                        INFO("a*=" << astar << " r=" << r << " th=" << th
                                   << " mu=" << mu << " nu=" << nu);
                        CHECK_THAT(gh(mu, nu),
                                   WithinAbs(gk(mu, nu),
                                             1e-12 * std::max(1.0, std::abs(gk(mu, nu)))));
                    }
                }
            }
        }
    }
}

TEST_CASE("F1/F2 branch continuity at the series switch radius",
          "[hartle-thorne][numerics]") {
    // The closed form (r < 9M) and the symbolic large-r series (r > 9M) must
    // agree where they meet; a coefficient error would show up as a jump.
    const HartleThorneMetric ht(1.0, 0.0, 1.0);
    const double below = ht.F1(9.0 - 1e-9), above = ht.F1(9.0 + 1e-9);
    CHECK_THAT(below, WithinRel(above, 1e-9));
    const double below2 = ht.F2(9.0 - 1e-9), above2 = ht.F2(9.0 + 1e-9);
    CHECK_THAT(below2, WithinRel(above2, 1e-9));
    // Spot values from 50-digit mpmath evaluation of GB06 (A.4):
    CHECK_THAT(ht.F1(8.0), WithinRel(-0.00298492699681, 1e-9));
    CHECK_THAT(ht.F1(10.0), WithinRel(-0.00139134856573, 1e-9));
    CHECK_THAT(ht.F1(3000.0), WithinRel(-3.70741034901e-11, 1e-9));
    CHECK_THAT(ht.F2(8.0), WithinRel(0.00277046125977, 1e-9));
    CHECK_THAT(ht.F2(10.0), WithinRel(0.00131377699302, 1e-9));
    CHECK_THAT(ht.F2(3000.0), WithinRel(3.70679224123e-11, 1e-9));
}

TEST_CASE("Quadrupole deviation has the correct far-field structure",
          "[hartle-thorne][physics]") {
    // δg_tt = −ε_q (1−2M/r)(1−3cos²θ) F1(r) → +ε_q (1−3cos²θ) M³/r³ as r→∞
    // (F1 → −M³/r³). Angular structure ∝ P₂; radial decay r⁻³.
    const double eps_q = 0.05;
    const KerrMetric kerr(1.0, 0.5);
    const HartleThorneMetric ht(1.0, 0.5, eps_q);
    for (double r : {200.0, 500.0, 2000.0}) {
        for (double th : {0.3, 1.0, 2.0}) {
            const double dgtt = ht.components(r, th)(0, 0) - kerr.components(r, th)(0, 0);
            const double ct = std::cos(th);
            const double expected = eps_q * (1.0 - 3.0 * ct * ct) / (r * r * r);
            INFO("r=" << r << " th=" << th);
            // Tolerance: O(M/r) series corrections plus the float cancellation
            // noise of subtracting two O(1) metric components (~1e-16/|δg|).
            const double tol = 5.0 / r + 1e-15 / std::abs(expected);
            CHECK_THAT(dgtt, WithinRel(expected, tol));
        }
    }
    // Equatorial deviation is nonzero; polar angle θ with 3cos²θ = 1 kills it.
    const double th_node = std::acos(std::sqrt(1.0 / 3.0));
    const double dg_node =
        ht.components(50.0, th_node)(0, 0) - kerr.components(50.0, th_node)(0, 0);
    CHECK_THAT(dg_node, WithinAbs(0.0, 1e-14));
}

TEST_CASE("Timelike geodesics in HT metric conserve E, L_z and normalisation",
          "[hartle-thorne][geodesic]") {
    // The quasi-Kerr metric is stationary and axisymmetric, so E and L_z stay
    // conserved and the 4-velocity norm must hold; Carter Q is *not* conserved
    // (separability is broken) and is deliberately not asserted here.
    const HartleThorneMetric ht(1.0, 0.94, 0.3);
    TimelikeGeodesic geo(ht);

    // Bound inclined orbit at ~20 M.
    const double r0 = 20.0, th0 = 1.3;
    const Metric4<double> g0 = ht.components(r0, th0);
    TimelikeState y{};
    y[1] = r0;
    y[2] = th0;
    y[5] = 0.0;
    y[6] = 0.03 / std::sqrt(g0(2, 2));
    y[7] = 0.21 / std::sqrt(g0(3, 3));
    y[4] = TimelikeGeodesic::solve_ut(g0, y[5], y[6], y[7]);

    const double E0 = geo.energy(y), L0 = geo.angmom(y);
    REQUIRE(E0 < 1.0); // bound

    TimelikeStepper stepper{&geo, 1e-12, 1e-12};
    double tau = 0.0, h = 0.5;
    double worst_norm = 0.0, worst_dE = 0.0, worst_dL = 0.0;
    for (int i = 0; i < 3000; ++i) {
        stepper.step(tau, y, h);
        worst_norm = std::max(worst_norm, std::abs(geo.norm(y) + 1.0));
        worst_dE = std::max(worst_dE, std::abs(geo.energy(y) - E0) / E0);
        worst_dL = std::max(worst_dL, std::abs(geo.angmom(y) - L0) / std::abs(L0));
        REQUIRE(y[1] > 2.5); // must not plunge for this parameter choice
    }
    CHECK(worst_norm < 1e-8);
    CHECK(worst_dE < 1e-6);
    CHECK(worst_dL < 1e-6);
}
