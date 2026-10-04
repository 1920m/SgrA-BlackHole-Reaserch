// Gate 21 (exotic spacetime, not Sgr A*): the Dneg wormhole of James et al.
// (2015) — its lensing width, the conserved quantities of its rays, and the
// exact edge of the other universe's disk on the camera's sky.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "Wormhole.hpp"

using namespace sgr;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {
constexpr double kPi = 3.14159265358979323846;
}

TEST_CASE("Dneg lensing width matches the JVFT closed form", "[gate21][wormhole]") {
    for (double M : {0.035, 0.5, 2.0}) {
        DnegParams p;
        p.M = M;
        p.a = 0.3;
        const DnegWormhole w(p);
        CHECK_THAT(w.lensing_width_numeric(), WithinRel(DnegWormhole::lensing_width_closed_form(M), 1e-10));
    }
    CHECK_THAT(DnegWormhole::lensing_width_closed_form(1.0), WithinAbs(1.42953, 5e-6));
    // radius: continuous through the mouths, ρ inside, ≈ |ℓ| far out
    const DnegWormhole w(DnegParams{});
    CHECK(w.r(0.0) == 1.0);
    CHECK_THAT(w.r(w.params().a + 1e-12), WithinAbs(1.0, 1e-9));
    CHECK(std::abs(w.drdl(500.0) - 1.0) < 1e-3);
}

TEST_CASE("Wormhole rays conserve B^2 and stay null", "[gate21][wormhole]") {
    DnegParams p;
    p.a = 0.5;
    p.M = 0.3;
    const DnegWormhole w(p);
    for (double alpha : {0.1, 0.4, 0.8, 1.3})
        for (double tilt : {0.0, 0.7}) {
            const std::array<double, 3> N{-std::cos(alpha), std::sin(alpha) * std::cos(tilt),
                                          std::sin(alpha) * std::sin(tilt)};
            const auto o = w.trace(w.camera_ray(6.0, 1.3, 0.2, N));
            CHECK(o.max_dB2 < 1e-9);
            CHECK(o.max_H < 1e-9);
        }
}

TEST_CASE("The other universe fills exactly sin(alpha) < rho/r(l_c)", "[gate21][wormhole]") {
    // A ray of impact parameter B = r_c sin α crosses the throat iff B < ρ:
    // the edge of the other universe's disk is at sin α = ρ / r(ℓ_c).
    DnegParams p;
    p.a = 0.5;
    p.M = 0.3;
    const DnegWormhole w(p);
    for (double l_c : {3.0, 8.0}) {
        auto other = [&](double alpha) {
            const std::array<double, 3> N{-std::cos(alpha), std::sin(alpha), 0.0};
            return w.trace(w.camera_ray(l_c, 0.5 * kPi, 0.0, N)).universe < 0;
        };
        REQUIRE(other(0.01));
        REQUIRE_FALSE(other(1.2));
        double lo = 0.01, hi = 1.2;
        for (int it = 0; it < 40; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (other(mid)) lo = mid; else hi = mid;
        }
        CHECK_THAT(0.5 * (lo + hi), WithinAbs(std::asin(p.rho / w.r(l_c)), 1e-8));
    }
}
