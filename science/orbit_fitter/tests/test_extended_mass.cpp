// Gate 22: the optional extended mass of S2Model (a Plummer sphere in g_tt).
//  1. Off (frac = 0) it is bit-identical to the thesis model.
//  2. A Plummer core much larger than the orbit is a harmonic potential,
//     Φ ≈ const + ½Ω²r² with Ω² = frac·M/b³, for which first-order
//     perturbation theory gives the classical retrograde apsidal precession
//     Δω = −3π frac (a/b)³ √(1 − e²) per orbit. Measured as the change in
//     the orbit-averaged Laplace–Runge–Lenz direction between two
//     consecutive orbits, relative to the same orbit without the core (so
//     the Schwarzschild advance cancels).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <vector>

#include "orbit_fitter/SkyProjection.hpp"

using namespace sgr;

namespace {
constexpr double kPi = 3.14159265358979323846;
using V3 = std::array<double, 3>;

S2Parameters nominal() {
    S2Parameters p;
    p.M_msun = 4.30e6;
    p.R0_pc = 8300.0;
    p.a_mas = 125.5;
    p.e = 0.8839;
    p.incl_deg = 134.7;
    p.Omega_deg = 227.9;
    p.omega_deg = 66.1;
    p.t_peri_yr = 2002.33;
    return p;
}

V3 cross(const V3& a, const V3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Time-averaged Newtonian LRL vector A = v × (x × v) − x̂ (M = 1) over
// [t0, t1], trapezoid in the integrator's own samples, and the mean orbit
// normal over the same window.
void mean_lrl(const S2Model& m, double t0, double t1, V3& A, V3& n) {
    A = {0, 0, 0};
    n = {0, 0, 0};
    const auto& tr = m.trajectory();
    for (size_t k = 1; k < tr.size(); ++k) {
        const double ta = tr[k - 1].t_yr, tb = tr[k].t_yr;
        if (tb <= t0 || ta >= t1) continue;
        const double w = 0.5 * (std::min(tb, t1) - std::max(ta, t0));
        for (size_t j : {k - 1, k}) {
            const V3& x = tr[j].pos_M;
            const V3& v = tr[j].vel;
            const V3 L = cross(x, v);
            const V3 vxL = cross(v, L);
            const double r = std::sqrt(dot(x, x));
            for (int c = 0; c < 3; ++c) {
                A[c] += w * (vxL[c] - x[c] / r);
                n[c] += w * L[c];
            }
        }
    }
    const double nn = std::sqrt(dot(n, n));
    for (double& c : n) c /= nn;
}

// signed angle from a to b about the axis n
double angle(const V3& a, const V3& b, const V3& n) {
    return std::atan2(dot(cross(a, b), n), dot(a, b));
}
}  // namespace

TEST_CASE("Extended mass switched off is bit-identical to the thesis model", "[gate22][s2model]") {
    const S2Parameters p = nominal();
    S2Model plain(0.0, 0.0), off(0.0, 0.0);
    off.set_extended_mass(0.0, 300.0);
    REQUIRE(plain.integrate(p, 1992.0, 2030.0));
    REQUIRE(off.integrate(p, 1992.0, 2030.0));
    std::vector<double> ep;
    for (double t = 1992.5; t < 2030.0; t += 0.37) ep.push_back(t);
    const auto a = plain.evaluate(p, ep, ep), b = off.evaluate(p, ep, ep);
    for (size_t i = 0; i < ep.size(); ++i) {
        CHECK(a.ra_mas[i] == b.ra_mas[i]);
        CHECK(a.dec_mas[i] == b.dec_mas[i]);
        CHECK(a.vr_kms[i] == b.vr_kms[i]);
    }
}

TEST_CASE("A large Plummer core precesses S2 retrograde as a harmonic core", "[gate22][s2model]") {
    const S2Parameters p = nominal();
    const double frac = 0.01;
    const double b_mas = 10.0 * p.a_mas * (1.0 + p.e);   // core ≫ apocentre
    S2Model m0(0.0, 0.0), m1(0.0, 0.0);
    m1.set_extended_mass(frac, b_mas);
    const double a_M = p.a_mas / m0.mas_per_M(p);
    const double P = 2.0 * kPi * std::pow(a_M, 1.5) * m0.yr_per_M(p);
    const double t0 = p.t_peri_yr + 0.5 * P;   // windows from apocentre to apocentre
    REQUIRE(m0.integrate(p, p.t_peri_yr, t0 + 2.0 * P + 0.5));
    REQUIRE(m1.integrate(p, p.t_peri_yr, t0 + 2.0 * P + 0.5));
    V3 A1, A2, B1, B2, n0, n1, nn;
    mean_lrl(m0, t0, t0 + P, A1, n0);
    mean_lrl(m0, t0 + P, t0 + 2.0 * P, A2, nn);
    mean_lrl(m1, t0, t0 + P, B1, n1);
    mean_lrl(m1, t0 + P, t0 + 2.0 * P, B2, nn);
    const double gr = angle(A1, A2, n0);          // Schwarzschild advance alone
    const double both = angle(B1, B2, n1);        // plus the core
    const double expected =
        -3.0 * kPi * frac * std::pow(p.a_mas / b_mas, 3) * std::sqrt(1.0 - p.e * p.e);
    INFO("GR advance " << gr << " rad/orbit; core " << both - gr << " vs expected " << expected);
    CHECK(gr > 0.0);                               // prograde Schwarzschild advance
    CHECK_THAT(both - gr, Catch::Matchers::WithinRel(expected, 0.03));
}
