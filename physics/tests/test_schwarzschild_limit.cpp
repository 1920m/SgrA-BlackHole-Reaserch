// Gate 0: at a* = 0 every Kerr formula must reduce exactly to Schwarzschild.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "sgr_physics/metric.hpp"

using namespace sgr;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Schwarzschild key radii at a*=0", "[schwarzschild][radii]") {
    const double M = 1.0;
    const KerrMetric g(M, 0.0);

    CHECK_THAT(g.r_horizon(), WithinAbs(2.0 * M, 1e-14));
    CHECK_THAT(g.r_ergosphere(0.3), WithinAbs(2.0 * M, 1e-14));   // ∀θ at a=0
    CHECK_THAT(g.r_ergosphere(1.2), WithinAbs(2.0 * M, 1e-14));
    CHECK_THAT(g.r_photon(true), WithinAbs(3.0 * M, 1e-13));
    CHECK_THAT(g.r_photon(false), WithinAbs(3.0 * M, 1e-13));
    CHECK_THAT(g.r_isco(true), WithinAbs(6.0 * M, 1e-13));
    CHECK_THAT(g.r_isco(false), WithinAbs(6.0 * M, 1e-13));
}

TEST_CASE("Schwarzschild metric components at a*=0", "[schwarzschild][metric]") {
    const double M = 1.0;
    const KerrMetric kerr(M, 0.0);

    for (double r : {2.5, 4.0, 10.0, 100.0}) {
        for (double th : {0.3, 1.0, 1.5707963267948966}) {
            const Metric4<double> g = kerr.components(r, th);
            const double f = 1.0 - 2.0 * M / r;
            CHECK_THAT(g(0, 0), WithinRel(-f, 1e-14));
            CHECK_THAT(g(1, 1), WithinRel(1.0 / f, 1e-14));
            CHECK_THAT(g(2, 2), WithinRel(r * r, 1e-14));
            CHECK_THAT(g(3, 3), WithinRel(r * r * std::sin(th) * std::sin(th), 1e-14));
            CHECK_THAT(g(0, 3), WithinAbs(0.0, 1e-15));
        }
    }
}

TEST_CASE("Complex-step Christoffels match analytic Schwarzschild symbols",
          "[schwarzschild][christoffel]") {
    const double M = 1.0;
    const KerrMetric kerr(M, 0.0);

    for (double r : {3.0, 6.0, 20.0}) {
        for (double th : {0.7, 1.3}) {
            const Christoffel gam = christoffel(kerr, r, th);
            const double f = 1.0 - 2.0 * M / r;
            const double st = std::sin(th), ct = std::cos(th);

            // Analytic nonzero Schwarzschild symbols (t,r,θ,φ) = (0,1,2,3):
            CHECK_THAT(gam[0][0][1], WithinRel(M / (r * r * f), 1e-12));      // Γ^t_tr
            CHECK_THAT(gam[1][0][0], WithinRel(M * f / (r * r), 1e-12));      // Γ^r_tt
            CHECK_THAT(gam[1][1][1], WithinRel(-M / (r * r * f), 1e-12));     // Γ^r_rr
            CHECK_THAT(gam[1][2][2], WithinRel(-(r - 2.0 * M), 1e-12));       // Γ^r_θθ
            CHECK_THAT(gam[1][3][3], WithinRel(-(r - 2.0 * M) * st * st, 1e-12));
            CHECK_THAT(gam[2][1][2], WithinRel(1.0 / r, 1e-12));              // Γ^θ_rθ
            CHECK_THAT(gam[2][3][3], WithinRel(-st * ct, 1e-12));             // Γ^θ_φφ
            CHECK_THAT(gam[3][1][3], WithinRel(1.0 / r, 1e-12));              // Γ^φ_rφ
            CHECK_THAT(gam[3][2][3], WithinRel(ct / st, 1e-12));              // Γ^φ_θφ

            // A few symbols that must vanish in Schwarzschild:
            CHECK_THAT(gam[0][0][0], WithinAbs(0.0, 1e-13));
            CHECK_THAT(gam[1][0][3], WithinAbs(0.0, 1e-13));
            CHECK_THAT(gam[3][0][1], WithinAbs(0.0, 1e-13));
        }
    }
}

TEST_CASE("Kerr metric determinant identity", "[kerr][metric]") {
    // det g = −Σ² sin²θ holds for all Kerr metrics in BL coordinates —
    // a strong algebraic check of every component at once.
    for (double astar : {0.0, 0.5, 0.94, 0.998}) {
        const KerrMetric kerr(1.0, astar);
        for (double r : {2.2, 5.0, 50.0}) {
            for (double th : {0.4, 1.1, 1.5707963267948966}) {
                const Metric4<double> g = kerr.components(r, th);
                const double det_tphi = g(0, 0) * g(3, 3) - g(0, 3) * g(0, 3);
                const double det = det_tphi * g(1, 1) * g(2, 2);
                const double a = astar;
                const double ct = std::cos(th), st = std::sin(th);
                const double sigma = r * r + a * a * ct * ct;
                CHECK_THAT(det, WithinRel(-sigma * sigma * st * st, 1e-12));
            }
        }
    }
}
