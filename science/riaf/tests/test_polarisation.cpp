// Gate 20: polarisation — the Walker–Penrose constant is conserved by
// explicit parallel transport along a Kerr null geodesic, a face-on disk with
// a toroidal field shows a radial EVPA pattern (∠β2 = 0), and an image seen
// from the other side of the equator has the conjugate β2.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>

#include "Riaf.hpp"
#include "sgr_physics/constants.hpp"
#include "sgr_physics/integrator.hpp"
#include "sgr_physics/metric.hpp"

using namespace sgr;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {
constexpr double kPi = 3.14159265358979323846;
}

TEST_CASE("Walker-Penrose constant survives parallel transport", "[gate20][polarisation]") {
    // Geodesic and parallel transport in BL with Christoffel symbols:
    //   ẍ^μ = −Γ^μ_νρ k^ν k^ρ,   ḟ^μ = −Γ^μ_νρ k^ν f^ρ.
    const double a = 0.94;
    const KerrMetric m(1.0, a);
    const double r0 = 30.0, th0 = 1.1;
    const auto g = m.components(r0, th0);
    // an inward photon with angular momentum, and f orthogonal to it
    const double om = -g(0, 3) / g(3, 3);
    const double al = std::sqrt(-(g(0, 0) - g(0, 3) * g(0, 3) / g(3, 3)));
    const Vec4 et{1.0 / al, 0.0, 0.0, om / al};
    const Vec4 er{0.0, 1.0 / std::sqrt(g(1, 1)), 0.0, 0.0};
    const Vec4 eth{0.0, 0.0, 1.0 / std::sqrt(g(2, 2)), 0.0};
    const Vec4 eph{0.0, 0.0, 0.0, 1.0 / std::sqrt(g(3, 3))};
    const double n[3] = {-0.8, 0.36, 0.48};               // unit
    Vec4 k{}, f{};
    for (int mu = 0; mu < 4; ++mu) {
        k[mu] = et[mu] + n[0] * er[mu] + n[1] * eth[mu] + n[2] * eph[mu];
        f[mu] = 0.6 * er[mu] + 0.8 * eph[mu];             // ⊥ n in the ZAMO frame
        f[mu] = f[mu] - (0.6 * n[0] + 0.8 * n[2]) * (n[0] * er[mu] + n[1] * eth[mu] + n[2] * eph[mu]);
    }
    using S = StateVec<12>;
    auto rhs = [&m](double, const S& y, S& dy) {
        const Christoffel G = christoffel(m, y[1], y[2]);
        for (int mu = 0; mu < 4; ++mu) {
            dy[mu] = y[4 + mu];
            double ak = 0.0, af = 0.0;
            for (int nu = 0; nu < 4; ++nu)
                for (int rh = 0; rh < 4; ++rh) {
                    ak += G[mu][nu][rh] * y[4 + nu] * y[4 + rh];
                    af += G[mu][nu][rh] * y[4 + nu] * y[8 + rh];
                }
            dy[4 + mu] = -ak;
            dy[8 + mu] = -af;
        }
    };
    DormandPrince54<12, decltype(rhs)> rk(rhs, 1e-12, 1e-12);
    S y{0.0, r0, th0, 0.0, k[0], k[1], k[2], k[3], f[0], f[1], f[2], f[3]};
    const std::complex<double> k0 = walker_penrose(a, r0, th0, k, f);
    REQUIRE(std::abs(k0) > 1e-3);
    double lam = 0.0, h = 0.01, worst = 0.0;
    for (int step = 0; step < 20000 && y[1] > 3.0 && y[1] < 200.0 && lam < 60.0; ++step) {
        const auto res = rk.try_step(lam, y, h);
        h = res.h_next;
        if (!res.accepted) continue;
        const Vec4 kk{y[4], y[5], y[6], y[7]}, ff{y[8], y[9], y[10], y[11]};
        worst = std::max(worst, std::abs(walker_penrose(a, y[1], y[2], kk, ff) - k0) / std::abs(k0));
    }
    CHECK(lam > 10.0);
    CHECK(worst < 1e-8);
}

TEST_CASE("Face-on disk with a toroidal field: radial EVPA pattern", "[gate20][polarisation]") {
    // a = 0, fluid at rest, toroidal field, camera almost on the axis: by
    // symmetry E ⊥ B is radial everywhere and β2 is real and positive.
    RiafParams p;
    p.a_star = 0.0;
    p.n0 = 3e6;
    p.rg_cm = constants::sgrA_rg_m * 100.0;
    p.static_flow = true;
    p.field = FieldGeometry::Toroidal;
    const RiafModel model(p);
    RiafCamera cam;
    cam.n = 48;
    cam.incl_deg = 0.5;
    const RiafImage img = riaf_image(model, cam, 1.0);
    const auto b2 = beta_mode(img.I, img.Q, img.U, cam.n, cam.half_width_M, 2);
    CHECK(std::abs(b2) > 0.1);
    CHECK_THAT(std::arg(b2) * 180.0 / kPi, WithinAbs(0.0, 2.0));
}

TEST_CASE("Polarisation of the mirrored view is the conjugate", "[gate20][polarisation]") {
    RiafParams p;
    p.a_star = 0.94;
    p.n0 = 3e6;
    p.rg_cm = constants::sgrA_rg_m * 100.0;
    const RiafModel model(p);
    RiafCamera ca, cb;
    ca.n = cb.n = 40;
    ca.incl_deg = 35.0;
    cb.incl_deg = 145.0;
    const RiafImage ia = riaf_image(model, ca, 1.0);
    const RiafImage ib = riaf_image(model, cb, 1.0);
    const auto ba = beta_mode(ia.I, ia.Q, ia.U, ca.n, ca.half_width_M, 2);
    const auto bb = beta_mode(ib.I, ib.Q, ib.U, cb.n, cb.half_width_M, 2);
    CHECK_THAT(std::abs(bb), WithinRel(std::abs(ba), 1e-6));
    CHECK_THAT(std::arg(bb), WithinAbs(-std::arg(ba), 1e-6));
}
