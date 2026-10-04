// Gate 17: observer tetrads (ZAMO + Lorentz boosts) — orthonormality, the
// 4-velocities of the standard observers against their closed forms
// (BPT72 circular orbits, E = 1 rain, static observers), the ergosphere
// limit of hovering, and the aberrated shadow of moving observers against
// Synge's formula plus special-relativistic aberration (Schwarzschild).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "sgr_physics/kerr_schild.hpp"
#include "sgr_physics/metric.hpp"
#include "sgr_physics/observers.hpp"

using namespace sgr;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {
constexpr double kPi = 3.14159265358979323846;

double dot_bl(const Metric4<double>& g, const Vec4& u, const Vec4& w) {
    double s = 0.0;
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 4; ++n) s += g(m, n) * u[m] * w[n];
    return s;
}

// Is the backward ray from local direction n captured?
bool captured(const KerrSchild& ks, double r, double th, const ObserverFrame& f,
              const std::array<double, 3>& n) {
    KSTraceOptions opt;
    opt.rtol = opt.atol = 1e-10;
    opt.r_escape = 4.0 * r;
    return ks_trace(ks, ks_observer_ray(ks, r, th, f, n), opt).outcome ==
           KSOutcome::Captured;
}

// Angle from the look direction (−e_1, i.e. at the hole) to the shadow edge
// in the plane of e_1 and e_3 (image horizontal), on the side `side` = ±1.
double shadow_edge(const KerrSchild& ks, double r, double th,
                   const ObserverFrame& f, double side) {
    double lo = 0.0, hi = 0.5 * kPi;   // lo inside the shadow, hi outside
    for (int it = 0; it < 40; ++it) {
        const double psi = 0.5 * (lo + hi);
        const std::array<double, 3> n{-std::cos(psi), 0.0, side * std::sin(psi)};
        if (captured(ks, r, th, f, n)) lo = psi; else hi = psi;
    }
    return 0.5 * (lo + hi);
}
} // namespace

TEST_CASE("Observer tetrads are orthonormal", "[gate17][observers]") {
    for (double a : {0.0, 0.5, 0.94}) {
        const KerrMetric m(1.0, a);
        for (double r : {3.5, 6.0, 15.0, 60.0}) // r = 3 at a = 0 is the null orbit
            for (double th : {0.4, 1.2, 0.5 * kPi}) {
                const auto g = m.components(r, th);
                for (auto kind : {ObserverKind::Zamo, ObserverKind::Orbiting,
                                  ObserverKind::Infall, ObserverKind::Hovering}) {
                    const ObserverFrame f = observer_frame(m, kind, r, th);
                    if (!f.valid) continue;
                    for (int i = 0; i < 4; ++i)
                        for (int j = 0; j < 4; ++j) {
                            const double eta = (i == j) ? (i == 0 ? -1.0 : 1.0) : 0.0;
                            CHECK_THAT(dot_bl(g, f.e[i], f.e[j]), WithinAbs(eta, 1e-12));
                        }
                    CHECK_THAT(f.dtau_dt() * f.e[0][0], WithinAbs(1.0, 1e-15));
                }
            }
    }
}

TEST_CASE("Standard observers have their closed-form 4-velocities",
          "[gate17][observers]") {
    for (double a : {0.0, 0.94}) {
        const KerrMetric m(1.0, a);
        // circular equatorial orbit: u^t (BPT72), u^phi = Omega u^t
        for (double r : {6.0, 10.0, 60.0}) {
            const ObserverFrame f =
                observer_frame(m, ObserverKind::Orbiting, r, 0.5 * kPi);
            REQUIRE(f.valid);
            const double sr = std::sqrt(r);
            const double ut = (r * sr + a) /
                              (std::pow(r, 0.75) * std::sqrt(r * sr - 3.0 * sr + 2.0 * a));
            CHECK_THAT(f.e[0][0], WithinRel(ut, 1e-12));
            CHECK_THAT(f.e[0][3] / f.e[0][0], WithinRel(m.omega_circular(r), 1e-12));
            CHECK_THAT(f.e[0][1], WithinAbs(0.0, 1e-15));
            // E and L of the orbit (lowered u: u_t = -E, u_phi = L)
            const auto g = m.components(r, 0.5 * kPi);
            const double ut_low = g(0, 0) * f.e[0][0] + g(0, 3) * f.e[0][3];
            const double up_low = g(3, 0) * f.e[0][0] + g(3, 3) * f.e[0][3];
            CHECK_THAT(-ut_low, WithinRel(m.energy_circular(r), 1e-12));
            CHECK_THAT(up_low, WithinRel(m.angmom_circular(r), 1e-12));
        }
        // rain: E = 1, L_z = 0, no polar motion, at any (r, theta)
        for (double r : {2.5, 6.0, 40.0})
            for (double th : {0.3, 1.0, 0.5 * kPi}) {
                const ObserverFrame f = observer_frame(m, ObserverKind::Infall, r, th);
                REQUIRE(f.valid);
                const auto g = m.components(r, th);
                CHECK_THAT(g(0, 0) * f.e[0][0] + g(0, 3) * f.e[0][3],
                           WithinAbs(-1.0, 1e-12));
                CHECK_THAT(g(3, 0) * f.e[0][0] + g(3, 3) * f.e[0][3],
                           WithinAbs(0.0, 1e-12));
                CHECK(f.e[0][1] < 0.0);
                CHECK_THAT(f.e[0][2], WithinAbs(0.0, 1e-15));
                CHECK_THAT(f.gamma, WithinRel(1.0 / f.alpha, 1e-12));
            }
        // hovering: u ∝ ∂_t, u^t = 1/sqrt(-g_tt)
        for (double r : {3.0, 15.0}) {
            const ObserverFrame f = observer_frame(m, ObserverKind::Hovering, r, 1.0);
            REQUIRE(f.valid);
            const auto g = m.components(r, 1.0);
            CHECK_THAT(f.e[0][0], WithinRel(1.0 / std::sqrt(-g(0, 0)), 1e-12));
            CHECK_THAT(f.e[0][3], WithinAbs(0.0, 1e-14));
        }
    }
}

TEST_CASE("Hovering is impossible exactly inside the ergosphere",
          "[gate17][observers]") {
    const KerrMetric m(1.0, 0.94);
    for (double th : {0.4, 1.0, 0.5 * kPi}) {
        const double rE = m.r_ergosphere(th);
        CHECK(observer_frame(m, ObserverKind::Hovering, rE * 1.001, th).valid);
        CHECK_FALSE(observer_frame(m, ObserverKind::Hovering, rE * 0.999, th).valid);
    }
    // the rain observer is subluminal down to the horizon, reaching c there
    const double rp = m.r_horizon();
    const ObserverFrame f = observer_frame(m, ObserverKind::Infall, rp * 1.0001, 1.0);
    CHECK(f.valid);
    CHECK(f.speed > 0.99);
}

TEST_CASE("Moving observer reproduces the standard ones from coordinate rates",
          "[gate17][observers]") {
    const KerrMetric m(1.0, 0.94);
    const double r = 8.0, th = 0.5 * kPi;
    const double Om = m.omega_circular(r);
    const auto vo = observer_velocity(m, ObserverKind::Orbiting, r, th);
    const auto vm = observer_velocity(m, ObserverKind::Moving, r, th, {0.0, 0.0, Om});
    for (int i = 0; i < 3; ++i) CHECK_THAT(vm[i], WithinAbs(vo[i], 1e-15));
    const auto vh = observer_velocity(m, ObserverKind::Hovering, r, th);
    const auto vs = observer_velocity(m, ObserverKind::Moving, r, th, {0.0, 0.0, 0.0});
    for (int i = 0; i < 3; ++i) CHECK_THAT(vs[i], WithinAbs(vh[i], 1e-15));
}

TEST_CASE("ZAMO observer ray is the Gate-16 camera ray", "[gate17][observers]") {
    const KerrSchild ks(1.0, 0.94, -1);
    const KerrMetric m(1.0, 0.94);
    const ObserverFrame f = observer_frame(m, ObserverKind::Zamo, 30.0, 1.1);
    const std::array<double, 3> n{-0.8, 0.36, 0.48};
    const KSRay a = ks_camera_ray(ks, 30.0, 1.1, n);
    const KSRay b = ks_observer_ray(ks, 30.0, 1.1, f, n);
    for (int k = 0; k < 4; ++k) {
        CHECK_THAT(b.X[k], WithinAbs(a.X[k], 1e-12));
        CHECK_THAT(b.P[k], WithinAbs(a.P[k], 1e-12));
    }
}

TEST_CASE("Moving observers see the aberrated Schwarzschild shadow",
          "[gate17][observers][slow]") {
    // a = 0: static = ZAMO, shadow half-angle sin(psi) = 3√3 √(1−2/r)/r
    // (Synge 1966). A moving observer sees every direction aberrated:
    // cos χ' = (cos χ + v)/(1 + v cos χ), χ measured from the velocity.
    const KerrSchild ks(1.0, 0.0, -1);
    const KerrMetric m(1.0, 0.0);
    const double r = 10.0, th = 0.5 * kPi;
    const double psi_s = std::asin(3.0 * std::sqrt(3.0) * std::sqrt(1.0 - 2.0 / r) / r);

    const ObserverFrame fz = observer_frame(m, ObserverKind::Zamo, r, th);
    CHECK_THAT(shadow_edge(ks, r, th, fz, +1.0), WithinAbs(psi_s, 1e-6));

    // infall along −r̂ = the look direction: the shadow shrinks
    const ObserverFrame fi = observer_frame(m, ObserverKind::Infall, r, th);
    const double v = fi.speed;
    const double psi_i = std::acos((std::cos(psi_s) + v) / (1.0 + v * std::cos(psi_s)));
    CHECK_THAT(shadow_edge(ks, r, th, fi, +1.0), WithinAbs(psi_i, 1e-6));
    CHECK(psi_i < psi_s);

    // orbiting along +φ̂ = image right: the shadow shifts toward the motion.
    // A static-frame edge direction (−cos ψ, 0, ±sin ψ) makes cos χ = ±sin ψ
    // with the velocity; aberrate χ, then measure ψ' from the look axis.
    const ObserverFrame fo = observer_frame(m, ObserverKind::Orbiting, r, th);
    const double vo = fo.speed;
    for (double side : {+1.0, -1.0}) {
        const double cchi = side * std::sin(psi_s);
        const double cchi_p = (cchi + vo) / (1.0 + vo * cchi);
        // ψ' from the look axis (−r̂): the aberrated direction keeps its
        // r̂ component sign; its angle to −r̂ is acos(√(1 − cos²χ')) on the
        // right (cos χ' > 0 is to the right)
        const double psi_p = std::asin(std::abs(cchi_p));
        const double right = (cchi_p > 0.0) ? +1.0 : -1.0;
        // measured edge on that side of the image
        const double meas = shadow_edge(ks, r, th, fo, right);
        CHECK_THAT(meas, WithinAbs(psi_p, 1e-6));
    }
}
