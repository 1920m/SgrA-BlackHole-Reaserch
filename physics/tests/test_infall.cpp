// Gate 24: falling into Sgr A* — the in-faller's worldline (ingoing
// Kerr–Schild, through the horizon), frame and tidal field.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "sgr_physics/infall.hpp"

using namespace sgr;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {
constexpr double kPi = 3.14159265358979323846;

double dot(const Metric4<double>& g, const Vec4& u, const Vec4& v) {
    double s = 0.0;
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 4; ++n) s += g(m, n) * u[m] * v[n];
    return s;
}
}  // namespace

TEST_CASE("Radial fall from rest in Schwarzschild follows the cycloid, through r = 2M",
          "[gate24][infall]") {
    // τ(r) = √(r0³/8M) (η + sin η),  r = (r0/2)(1 + cos η)
    for (double r0 : {6.0, 10.0, 30.0}) {
        InfallStart st;
        st.kind = InfallStartKind::Rest;
        st.r0 = r0;
        st.theta0 = 1.1;
        const InfallTrajectory tr(0.0, st);
        auto tau_exact = [&](double r) {
            const double eta = std::acos(2.0 * r / r0 - 1.0);
            return std::sqrt(r0 * r0 * r0 / 8.0) * (eta + std::sin(eta));
        };
        CHECK_THAT(tr.tau_horizon(), WithinRel(tau_exact(2.0), 1e-9));
        for (double r : {0.7 * r0, 3.0, 1.0, 0.3})   // outside and inside the horizon
            CHECK_THAT(tr.tau_at_radius(r), WithinRel(tau_exact(r), 1e-6));
        CHECK_THAT(tr.samples().back().r, WithinAbs(0.05, 1e-9));   // stopped at r₋ + δ
    }
}

TEST_CASE("Rain from infinity in Schwarzschild: tau = (2/3)(r0^1.5 - r^1.5)/sqrt(2M)",
          "[gate24][infall]") {
    InfallStart st;
    st.r0 = 25.0;
    const InfallTrajectory tr(0.0, st);
    for (double r : {20.0, 5.0, 2.0, 1.2, 0.4}) {
        const double exact = 2.0 / 3.0 * (std::pow(st.r0, 1.5) - std::pow(r, 1.5)) / std::sqrt(2.0);
        CHECK_THAT(tr.tau_at_radius(r), WithinRel(exact, 1e-6));
    }
}

TEST_CASE("Kerr in-fall conserves E, L_z, Q and the norm through the horizon",
          "[gate24][infall]") {
    for (auto kind : {InfallStartKind::Rain, InfallStartKind::Rest})
        for (double th : {0.5, 1.0471975511965976, kPi / 2}) {
            InfallStart st;
            st.kind = kind;
            st.r0 = 30.0;
            st.theta0 = th;
            const InfallTrajectory tr(0.94, st);
            INFO("theta0 = " << th << ", kind " << static_cast<int>(kind));
            CHECK(tr.max_dE() < 1e-6);
            CHECK(tr.max_dL() < 1e-6);
            CHECK(tr.max_dQ() < 1e-6);
            CHECK(tr.max_norm() < 1e-8);
            CHECK(std::isfinite(tr.tau_horizon()));
            CHECK(tr.samples().back().r < tr.chart().r_plus());   // went through
            CHECK_THAT(tr.samples().back().r,
                       WithinAbs(tr.chart().r_minus() + st.stop_above_rminus, 1e-9));
            // r decreases monotonically all the way
            bool mono = true;
            for (size_t k = 1; k < tr.samples().size(); ++k)
                mono = mono && tr.samples()[k].r < tr.samples()[k - 1].r;
            CHECK(mono);
        }
}

TEST_CASE("The in-faller's frame is orthonormal inside and outside the horizon",
          "[gate24][infall]") {
    InfallStart st;
    const InfallTrajectory tr(0.94, st);
    for (double r : {20.0, 3.0, 1.4, 1.0, 0.75}) {
        const InfallSample s = tr.at(tr.tau_at_radius(r));
        const auto e = infall_tetrad(tr.chart(), s.X, s.U);
        const auto g = tr.chart().metric(s.X);
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) {
                const double eta = a == b ? (a == 0 ? -1.0 : 1.0) : 0.0;
                CHECK_THAT(dot(g, e[a], e[b]), WithinAbs(eta, 1e-10));
            }
    }
}

TEST_CASE("Tidal field: Schwarzschild radial values and the Kerr Kretschmann scalar",
          "[gate24][infall]") {
    {
        InfallStart st;
        const InfallTrajectory tr(0.0, st);
        for (double r : {8.0, 2.5, 1.5, 0.8}) {   // r = 1.5, 0.8 inside r = 2M
            const InfallSample s = tr.at(tr.tau_at_radius(r));
            const auto e = infall_tetrad(tr.chart(), s.X, s.U);
            const TidalField t = tidal_field(tr.chart(), s.X, e);
            const double rr = s.r, k = 1.0 / (rr * rr * rr);
            INFO("r = " << rr);
            CHECK_THAT(t.E[0][0], WithinRel(-2.0 * k, 1e-6));   // radial stretch
            CHECK_THAT(t.E[1][1], WithinRel(k, 1e-6));          // transverse squeeze
            CHECK_THAT(t.E[2][2], WithinRel(k, 1e-6));
            CHECK(std::abs(t.E[0][1]) < 1e-6 * k);
            CHECK(std::abs(t.B[0][0]) < 1e-6 * k);
        }
    }
    InfallStart st;
    const InfallTrajectory tr(0.94, st);
    for (double r : {12.0, 2.0, 1.36, 1.2, 0.8}) {
        const InfallSample s = tr.at(tr.tau_at_radius(r));
        const auto e = infall_tetrad(tr.chart(), s.X, s.U);
        const TidalField t = tidal_field(tr.chart(), s.X, e);
        const double K = kerr_kretschmann(1.0, 0.94, s.r, s.X[3] / s.r);
        double EE = 0.0, BB = 0.0, trE = 0.0;
        for (int i = 0; i < 3; ++i) {
            trE += t.E[i][i];
            for (int j = 0; j < 3; ++j) {
                EE += t.E[i][j] * t.E[i][j];
                BB += t.B[i][j] * t.B[i][j];
            }
        }
        INFO("r = " << s.r);
        CHECK_THAT(t.kretschmann, WithinRel(K, 1e-6));
        CHECK_THAT(8.0 * (EE - BB), WithinRel(K, 1e-6));
        CHECK(std::abs(trE) < 1e-6 * std::sqrt(EE));   // vacuum: E is trace-free
    }
}
