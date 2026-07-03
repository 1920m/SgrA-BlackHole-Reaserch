// Gate 0: key Kerr radii — prograde/retrograde asymmetry, extremal limits,
// published reference values, and self-consistency with the radial potential.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "sgr_physics/metric.hpp"
#include "sgr_physics/null_geodesics.hpp"

using namespace sgr;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Prograde/retrograde asymmetry for spinning holes", "[kerr][radii]") {
    for (double astar : {0.5, 0.94}) {
        const KerrMetric g(1.0, astar);
        INFO("a* = " << astar);
        CHECK(g.r_photon(true) < g.r_photon(false));
        CHECK(g.r_isco(true) < g.r_isco(false));
        // Prograde radii shrink below the Schwarzschild values, retrograde grow.
        CHECK(g.r_photon(true) < 3.0);
        CHECK(g.r_photon(false) > 3.0);
        CHECK(g.r_isco(true) < 6.0);
        CHECK(g.r_isco(false) > 6.0);
    }
}

TEST_CASE("Extremal Kerr limits (BPT72 exact values)", "[kerr][radii]") {
    const KerrMetric g(1.0, 1.0);
    CHECK_THAT(g.r_horizon(), WithinAbs(1.0, 1e-14));
    CHECK_THAT(g.r_photon(true), WithinAbs(1.0, 1e-7));   // acos(±1) endpoint
    CHECK_THAT(g.r_photon(false), WithinAbs(4.0, 1e-7));
    CHECK_THAT(g.r_isco(true), WithinAbs(1.0, 1e-6));
    CHECK_THAT(g.r_isco(false), WithinAbs(9.0, 1e-6));
}

TEST_CASE("Published reference radii", "[kerr][radii]") {
    // Reference values evaluated to 12 digits with mpmath (30-digit precision)
    // from the same BPT72 closed forms — guards against transcription and
    // floating-point-order errors in the C++ implementation.
    const KerrMetric g09(1.0, 0.9);
    CHECK_THAT(g09.r_isco(true), WithinAbs(2.32088304176, 1e-9));
    const KerrMetric g94(1.0, 0.94);
    CHECK_THAT(g94.r_isco(true), WithinAbs(2.0235931047, 1e-9));
    CHECK_THAT(g94.r_photon(true), WithinAbs(1.4252442687, 1e-9));
    CHECK_THAT(g94.r_photon(false), WithinAbs(3.94636607748, 1e-9));
    // Horizon at a* = 0.94: r+ = 1 + sqrt(1 - 0.94²) = 1.34117...
    CHECK_THAT(g94.r_horizon(), WithinRel(1.0 + std::sqrt(1.0 - 0.94 * 0.94), 1e-14));
}

// NOTE: test names must stay pure ASCII — ctest's Windows command-line filter
// mangles Unicode and then matches no test case at all.
TEST_CASE("ISCO moves toward horizon as a* goes to 1 (prograde)", "[kerr][radii]") {
    double prev = 7.0;
    for (double astar : {0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.94, 0.98, 0.998}) {
        const KerrMetric g(1.0, astar);
        const double ri = g.r_isco(true);
        CHECK(ri < prev);
        CHECK(ri > g.r_horizon() - 1e-12);
        prev = ri;
    }
}

TEST_CASE("Photon orbit radius solves R(r)=R'(r)=0 (self-consistency)",
          "[kerr][radii][potential]") {
    // For the circular photon orbit the radial potential and its derivative
    // vanish simultaneously with the Bardeen critical impact parameters:
    //   ξ = L_z/E = (r²(3M−r) − a²(r+M)) / (a(r−M))  ... prograde sign built in
    //   η = Q/E²  = r³(4a²M − r(r−3M)²) / (a²(r−M)²)
    for (double astar : {0.3, 0.7, 0.94}) {
        const double M = 1.0, a = astar;
        const KerrMetric g(M, astar);
        const double r_in = g.r_photon(true), r_out = g.r_photon(false);
        // Sample the equatorial endpoints (where η = 0) and interior radii
        // (spherical photon orbits with η > 0 crossing the poles).
        for (double r : {r_in, r_in + 0.25 * (r_out - r_in),
                         0.5 * (r_in + r_out), r_out}) {
            const double xi = (r * r * (3.0 * M - r) - a * a * (r + M)) /
                              (a * (r - M));
            const double eta = r * r * r *
                               (4.0 * a * a * M - r * (r - 3.0 * M) * (r - 3.0 * M)) /
                               (a * a * (r - M) * (r - M));
            NullRayConstants c;
            c.M = M; c.a = a; c.E = 1.0; c.Lz = xi; c.Q = eta;
            const double scale = std::pow(r * r + a * a + std::abs(a * xi), 2.0);
            INFO("a*=" << astar << " r=" << r << " xi=" << xi << " eta=" << eta);
            CHECK(std::abs(radial_potential(c, r)) < 1e-10 * scale);
            CHECK(std::abs(radial_potential_deriv(c, r)) < 1e-9 * scale);
        }
    }
}

TEST_CASE("Circular-orbit E and L are stationary at the ISCO", "[kerr][isco]") {
    // At r_isco, dE/dr = 0 (marginal stability) — checks the BPT72 E(r), L(r)
    // formulas and the ISCO radius against each other.
    for (double astar : {0.0, 0.5, 0.94}) {
        const KerrMetric g(1.0, astar);
        for (bool pro : {true, false}) {
            const double ri = g.r_isco(pro);
            const double h = 1e-5;
            const double dE = (g.energy_circular(ri + h, pro) -
                               g.energy_circular(ri - h, pro)) / (2.0 * h);
            const double dL = (g.angmom_circular(ri + h, pro) -
                               g.angmom_circular(ri - h, pro)) / (2.0 * h);
            INFO("a*=" << astar << " prograde=" << pro << " r_isco=" << ri);
            CHECK(std::abs(dE) < 1e-6);
            CHECK(std::abs(dL) < 1e-5);
        }
    }
}
