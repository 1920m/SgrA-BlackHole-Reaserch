// Gate 19: the hot accretion flow — fluid kinematics, the thermal transfer
// solution in its two limits, the emissivity's basic scalings, and the
// equatorial mirror symmetry of the 230 GHz image.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "Riaf.hpp"
#include "sgr_physics/constants.hpp"

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
} // namespace

TEST_CASE("RIAF fluid: unit timelike flow, unit toroidal field orthogonal to it",
          "[gate19][riaf]") {
    RiafParams p;
    p.a_star = 0.94;
    const RiafModel model(p);
    const KerrMetric m(1.0, 0.94);
    for (double r : {1.6, 1.9, 2.5, 6.0, 20.0})          // plunge and circulation
        for (double th : {0.6, 1.2, 0.5 * kPi, 2.0}) {
            const RiafFluid f = model.fluid(r, th);
            REQUIRE(f.valid);
            const auto g = m.components(r, th);
            CHECK_THAT(dot_bl(g, f.u, f.u), WithinAbs(-1.0, 1e-10));
            CHECK_THAT(dot_bl(g, f.b, f.b), WithinAbs(1.0, 1e-10));
            CHECK_THAT(dot_bl(g, f.u, f.b), WithinAbs(0.0, 1e-10));
            if (r < model.r_isco()) CHECK(f.u[1] < 0.0);   // falling in
            else CHECK_THAT(f.u[1], WithinAbs(0.0, 1e-15));
        }
}

TEST_CASE("Thermal transfer: optically thick slab radiates B_nu(T), thin one sums j",
          "[gate19][riaf]") {
    const double nu = 230e9, T = 5e10;
    const double Bn = RiafModel::planck(nu, T);
    TransferAccumulator thick;
    for (int k = 0; k < 400; ++k) thick.add(nu, T, 1.0, 0.1);   // τ = 40
    CHECK_THAT(thick.intensity(nu), WithinRel(Bn, 1e-12));
    TransferAccumulator thin;
    const double alpha = 1e-12, dl = 1e8;                        // τ = 1e-4 per step
    for (int k = 0; k < 10; ++k) thin.add(nu, T, alpha, dl);
    // I = Σ j dl (1 − O(τ)), j = α B
    CHECK_THAT(thin.intensity(nu), WithinRel(10.0 * alpha * Bn * dl, 1e-3));
}

TEST_CASE("Thermal synchrotron emissivity scalings", "[gate19][riaf]") {
    const double nu = 230e9, T = 3e10, B = 30.0;
    const double j1 = RiafModel::j_nu(nu, 1e6, T, B, 0.8);
    CHECK(j1 > 0.0);
    CHECK_THAT(RiafModel::j_nu(nu, 2e6, T, B, 0.8), WithinRel(2.0 * j1, 1e-12));
    // far above the synchrotron frequency the emission dies exponentially
    CHECK(RiafModel::j_nu(1e3 * nu, 1e6, T, B, 0.8) < 1e-3 * j1);
    // no emission along the field
    CHECK(RiafModel::j_nu(nu, 1e6, T, B, 0.0) == 0.0);
}

TEST_CASE("230 GHz image is mirror-symmetric about the equator", "[gate19][riaf]") {
    RiafParams p;
    p.a_star = 0.94;
    p.n0 = 3e6;
    p.rg_cm = constants::sgrA_rg_m * 100.0;
    const RiafModel model(p);
    RiafCamera a, b;
    a.n = b.n = 32;
    a.incl_deg = 40.0;
    b.incl_deg = 140.0;
    const RiafImage ia = riaf_image(model, a);
    const RiafImage ib = riaf_image(model, b);
    CHECK(ia.max_dH < 1e-8);
    const double D = constants::sgrA_dist_m * 100.0;
    const double Fa = riaf_flux_jy(ia, p.rg_cm, D), Fb = riaf_flux_jy(ib, p.rg_cm, D);
    REQUIRE(Fa > 0.0);
    CHECK_THAT(Fb, WithinRel(Fa, 1e-6));
    // row j of one image is row n−1−j of the other
    for (int j = 0; j < a.n; ++j)
        for (int i = 0; i < a.n; ++i)
            CHECK_THAT(ib.I[static_cast<size_t>(a.n - 1 - j) * a.n + i],
                       WithinRel(ia.I[static_cast<size_t>(j) * a.n + i], 1e-6));
}
