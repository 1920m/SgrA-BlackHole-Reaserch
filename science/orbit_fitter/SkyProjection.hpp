#pragma once
// S2 orbit model: full GR timelike geodesic integration → sky observables.
//
// Parameterisation (13 parameters, matching Gillessen et al. 2017):
//   M_msun            black-hole mass [10⁰ M☉]
//   R0_pc             GC distance [pc]
//   a_mas             angular semi-major axis [mas]
//   e                 eccentricity
//   incl_deg          inclination i [deg]
//   Omega_deg         PA of the ascending node Ω, East of North [deg]
//   omega_deg         argument of pericenter ω [deg]
//   t_peri_yr         pericenter epoch [Julian yr]
//   x0_mas, y0_mas    Sgr A* position offset at epoch 2009.0 [mas]
//   vx_masyr, vy_masyr  Sgr A* proper-motion drift [mas/yr]
//   vz_kms            radial-velocity zero point [km/s]
//
// Conventions: sky frame x = ΔRA (East +), y = ΔDec (North +), z away from
// the observer (so RV > 0 = recession and the Rømer delay is t_obs = t_em +
// z/c). Ascending node = star crossing the sky plane with ż > 0. The BH spin
// axis is the coordinate z-axis (line of sight); for the Phase-4 fit a* = 0,
// where the choice is irrelevant.
//
// The osculating elements define Keplerian initial conditions at pericenter;
// the trajectory itself is a numerically integrated GR geodesic (Kerr or
// Hartle-Thorne via IMetric), so periapsis advance, gravitational redshift and
// transverse Doppler are all emergent, not bolted on.
//
// Observables include:
//   • Rømer retardation: emission epoch solved from t_em + z(t_em)/c = t_obs
//   • relativistic redshift 1+z = γ(1 − β·n̂)/α in the ZAMO frame (exact up to
//     the ~1e-3 rad photon-bending correction, ≲5 km/s at pericenter, well
//     below the ≥17 km/s measurement errors — stated in the paper)

#include <array>
#include <vector>

#include "sgr_physics/metric.hpp"

namespace sgr {

struct S2Parameters {
    double M_msun = 4.35e6;
    double R0_pc = 8300.0;
    double a_mas = 125.5;
    double e = 0.8839;
    double incl_deg = 134.6;
    double Omega_deg = 228.1;
    double omega_deg = 66.3;
    double t_peri_yr = 2002.33;
    double x0_mas = 0.0, y0_mas = 0.0;
    double vx_masyr = 0.0, vy_masyr = 0.0;
    double vz_kms = 0.0;

    static constexpr int kNumParams = 13;
    std::array<double, kNumParams> to_array() const;
    static S2Parameters from_array(const double* p);
};

// One sample of the integrated trajectory in BH-frame Cartesian coordinates
// (units of M for x,y,z; coordinate time in years relative to J2000-like
// absolute scale carried separately).
struct TrajectorySample {
    double t_yr;                    // BH coordinate time mapped to Julian yr
    std::array<double, 3> pos_M;    // x (East), y (North), z (away)
    std::array<double, 3> vel;      // dx/dt in units of c
    double redshift;                // 1+z of a LOS photon emitted here
};

struct S2Observables {
    // Astrometry [mas] at the requested observation epochs.
    std::vector<double> ra_mas, dec_mas;
    // Radial velocity [km/s] at the requested RV epochs.
    std::vector<double> vr_kms;
    bool valid = false;             // false if the orbit failed to integrate
};

class S2Model {
public:
    // metric_factory: builds the metric for given mass-normalised units
    // (a* and eps_q fixed inside). For Phase 4, a* = 0.
    explicit S2Model(double a_star = 0.0, double eps_q = 0.0)
        : a_star_(a_star), eps_q_(eps_q) {}

    // Integrate the orbit over [t_min_yr, t_max_yr] and cache the trajectory.
    // Returns false if initial conditions are unphysical (fit exploring bad
    // parameter space) — the caller should return a large residual.
    bool integrate(const S2Parameters& p, double t_min_yr, double t_max_yr);

    // Evaluate observables at given epochs (requires integrate() first).
    S2Observables evaluate(const S2Parameters& p,
                           const std::vector<double>& astrometry_epochs,
                           const std::vector<double>& rv_epochs) const;

    // Dense trajectory access (paper figures, precession measurement).
    const std::vector<TrajectorySample>& trajectory() const { return traj_; }

    // Interpolate the trajectory at emission time t_yr (Hermite).
    TrajectorySample sample_at(double t_yr) const;

    // Redshift decomposition at time t (for the spectral module):
    // returns {z_total, z_kepler_los, z_transverse, z_gravitational} where
    // z_kepler_los = β_los/(1−β_los) − ... first-order Doppler only,
    // z_transverse = γ − 1, z_grav = 1/α − 1 (each ×c gives km/s equivalent).
    std::array<double, 4> redshift_terms(double t_yr) const;

    double mas_per_M(const S2Parameters& p) const;
    double yr_per_M(const S2Parameters& p) const;

    // Validity floors used by integrate(). The defaults are the values every
    // fit has always used and must not be changed for fitting; the
    // interactive instrument lowers them to integrate hypothetical orbits
    // scaled down from the fitted ones (still ordinary bound GR orbits, just
    // far smaller than any fit would propose).
    void set_limits(double a_mas_min, double a_M_min, double r_min_M) {
        lim_a_mas_ = a_mas_min;
        lim_a_M_ = a_M_min;
        lim_r_min_ = r_min_M;
    }

    // Extension paper (Gate 22): an optional extended mass around the black
    // hole — a Plummer sphere of mass frac·M and scale radius scale_mas
    // [mas] (GRAVITY 2020 used 0.3″), entering as its Newtonian potential
    // Φ = −frac·M/√(r² + b²) in g_tt = −(1 − 2M/r + 2Φ) (its own 1PN terms
    // are O(frac·M/r) smaller still and are neglected). frac = 0, the
    // default, leaves the metric untouched: every thesis fit is unchanged.
    void set_extended_mass(double frac, double scale_mas) {
        ext_frac_ = frac;
        ext_scale_mas_ = scale_mas;
    }

private:
    double a_star_, eps_q_;
    double lim_a_mas_ = 5.0;    // minimum angular semi-major axis [mas]
    double lim_a_M_ = 100.0;    // minimum semi-major axis [M]
    double lim_r_min_ = 50.0;   // abort below this radius (plunge) [M]
    double ext_frac_ = 0.0;     // extended mass / M (0 = pure Kerr/HT)
    double ext_scale_mas_ = 300.0;   // its Plummer scale radius [mas]
    std::vector<TrajectorySample> traj_; // sorted by t_yr
    double t_peri_cache_ = 0.0;
    double yrM_ = 1.0; // years per geometric time unit (set by integrate)
};

} // namespace sgr
