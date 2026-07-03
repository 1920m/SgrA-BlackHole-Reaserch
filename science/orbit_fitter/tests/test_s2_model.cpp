// S2 model sanity: Keplerian limit, Rømer delay direction, redshift scales.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "orbit_fitter/SkyProjection.hpp"
#include "sgr_physics/constants.hpp"

using namespace sgr;

namespace {
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
} // namespace

TEST_CASE("S2 model: radial period matches Kepler's third law", "[s2model]") {
    const S2Parameters p = nominal();
    S2Model model;
    REQUIRE(model.integrate(p, 1990.0, 2022.0));

    // Kepler period from the fitted elements:
    const double a_au = p.a_mas / 1000.0 * p.R0_pc;
    const double P_kepler = 2.0 * 3.14159265358979323846 *
                            std::sqrt(std::pow(a_au * constants::au_m, 3) /
                                      (p.M_msun * constants::GM_sun_si)) /
                            constants::yr_s;
    INFO("Kepler P = " << P_kepler << " yr");
    CHECK(P_kepler > 15.0);
    CHECK(P_kepler < 17.0);

    // Radial period from the trajectory: successive minima of r ≈ |pos|.
    const auto& traj = model.trajectory();
    double t_peri2 = 0.0;
    double best = 1e30;
    for (const auto& s : traj) {
        const double r = std::sqrt(s.pos_M[0] * s.pos_M[0] +
                                   s.pos_M[1] * s.pos_M[1] +
                                   s.pos_M[2] * s.pos_M[2]);
        if (s.t_yr > p.t_peri_yr + 10.0 && s.t_yr < p.t_peri_yr + 20.0 &&
            r < best) {
            best = r;
            t_peri2 = s.t_yr;
        }
    }
    const double P_meas = t_peri2 - p.t_peri_yr;
    INFO("measured radial period = " << P_meas << " yr");
    // GR radial (anomalistic) period differs from the Keplerian value by
    // O(M/p) corrections; agreement to 0.5% validates the unit pipeline.
    CHECK_THAT(P_meas, Catch::Matchers::WithinRel(P_kepler, 5e-3));
}

TEST_CASE("S2 model: pericenter speed ~ 7650 km/s and redshift structure",
          "[s2model]") {
    const S2Parameters p = nominal();
    S2Model model;
    REQUIRE(model.integrate(p, 2000.0, 2004.0));

    const TrajectorySample s = model.sample_at(p.t_peri_yr);
    const double v = std::sqrt(s.vel[0] * s.vel[0] + s.vel[1] * s.vel[1] +
                               s.vel[2] * s.vel[2]) *
                     constants::c_si / 1000.0;
    INFO("pericenter speed = " << v << " km/s");
    CHECK(v > 7000.0);
    CHECK(v < 8300.0);

    // Redshift decomposition at pericenter: transverse + gravitational parts
    // are each ~100 km/s (their sum ~200 km/s is the GRAVITY 2018 detection).
    const auto z = model.redshift_terms(p.t_peri_yr);
    const double c_kms = constants::c_si / 1000.0;
    INFO("z_trans*c = " << z[2] * c_kms << " z_grav*c = " << z[3] * c_kms);
    CHECK(z[2] * c_kms > 60.0);
    CHECK(z[2] * c_kms < 160.0);
    CHECK(z[3] * c_kms > 60.0);
    CHECK(z[3] * c_kms < 160.0);
}

TEST_CASE("S2 model: Roemer delay shifts epochs with the correct sign",
          "[s2model]") {
    const S2Parameters p = nominal();
    S2Model model;
    REQUIRE(model.integrate(p, 1995.0, 2010.0));

    // Two epochs where the star sits on opposite LOS sides: the apparent
    // (Rømer-corrected) position must lead/lag the instantaneous one.
    // Verify: evaluate() with the Rømer solve differs from naive sampling by
    // |z|/c, and the sign matches z.
    std::vector<double> epochs = {1996.0, 2002.0, 2006.0};
    const S2Observables obs = model.evaluate(p, epochs, {});
    for (size_t i = 0; i < epochs.size(); ++i) {
        const TrajectorySample naive = model.sample_at(epochs[i]);
        const double z_yr = naive.pos_M[2] * model.yr_per_M(p);
        const TrajectorySample lagged = model.sample_at(epochs[i] - z_yr);
        const double masM = model.mas_per_M(p);
        // Rømer-corrected RA must match sampling at t_obs − z/c, not t_obs:
        const double ra_roemer = lagged.pos_M[0] * masM;
        const double ra_naive = naive.pos_M[0] * masM;
        const double ra_model = obs.ra_mas[i];
        INFO("epoch " << epochs[i] << ": naive " << ra_naive << " roemer "
                      << ra_roemer << " model " << ra_model
                      << " (z_yr=" << z_yr << ")");
        CHECK(std::abs(ra_model - ra_roemer) <
              0.2 * std::abs(ra_roemer - ra_naive) + 1e-4);
    }
}
