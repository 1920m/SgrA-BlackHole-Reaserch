#include "orbit_fitter/SkyProjection.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

#include "sgr_physics/constants.hpp"
#include "sgr_physics/hartle_thorne.hpp"
#include "sgr_physics/timelike_geodesics.hpp"

namespace sgr {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

struct CartesianState {
    std::array<double, 3> x;  // units of M
    std::array<double, 3> v;  // dx/dt (units of c)
};

// BL/spherical → Cartesian with velocities (a=0 exact; a≠0: BL≈spherical to
// O(a²/r²) ~ 1e-7 at S2 radii, far below measurement precision).
CartesianState to_cartesian(const TimelikeState& y) {
    const double r = y[1], th = y[2], ph = y[3];
    const double st = std::sin(th), ct = std::cos(th);
    const double sp = std::sin(ph), cp = std::cos(ph);
    CartesianState c;
    c.x = {r * st * cp, r * st * sp, r * ct};
    const double ut = y[4];
    const double rdot = y[5] / ut, thdot = y[6] / ut, phdot = y[7] / ut;
    c.v = {
        rdot * st * cp + r * ct * cp * thdot - r * st * sp * phdot,
        rdot * st * sp + r * ct * sp * thdot + r * st * cp * phdot,
        rdot * ct - r * st * thdot,
    };
    return c;
}

// Exact static/ZAMO-frame redshift of a line-of-sight photon emitted at y:
// 1+z = γ(1 − β·n̂)/α with n̂ the unit vector from star toward Earth (−ẑ).
double line_of_sight_redshift(const IMetric& metric, const TimelikeState& y) {
    const double r = y[1], th = y[2];
    const Metric4<double> g = metric.components(r, th);
    const double omega_z = -g(0, 3) / g(3, 3);
    const double alpha = std::sqrt(-(g(0, 0) - g(0, 3) * g(0, 3) / g(3, 3)));

    const double ut_hat = alpha * y[4];
    const double br = std::sqrt(g(1, 1)) * y[5] / ut_hat;
    const double bth = std::sqrt(g(2, 2)) * y[6] / ut_hat;
    const double bph = std::sqrt(g(3, 3)) * (y[7] - omega_z * y[4]) / ut_hat;
    const double beta2 = br * br + bth * bth + bph * bph;
    const double gamma = 1.0 / std::sqrt(1.0 - beta2);

    const double st = std::sin(th), ct = std::cos(th);
    const double n_r = -ct, n_th = st; // Earth direction −ẑ in (r̂, θ̂, φ̂)
    const double beta_dot_n = br * n_r + bth * n_th;
    (void)bph; // n̂ has no φ̂ component
    return gamma * (1.0 - beta_dot_n) / alpha; // = 1 + z
}

// Gate 22: Kerr/Hartle–Thorne plus the Newtonian potential of a Plummer
// sphere of mass mu (units of M) and scale radius b [M], in g_tt only.
class PlummerMetric final : public IMetric {
public:
    PlummerMetric(std::unique_ptr<IMetric> base, double mu, double b)
        : base_(std::move(base)), mu_(mu), b_(b) {}
    Metric4<double> components(double r, double theta) const override {
        Metric4<double> g = base_->components(r, theta);
        g(0, 0) += 2.0 * mu_ / std::sqrt(r * r + b_ * b_);
        return g;
    }
    Metric4<std::complex<double>> components(std::complex<double> r,
                                             std::complex<double> theta) const override {
        Metric4<std::complex<double>> g = base_->components(r, theta);
        g(0, 0) += 2.0 * mu_ / std::sqrt(r * r + b_ * b_);
        return g;
    }
    double mass() const override { return base_->mass(); }
    double spin() const override { return base_->spin(); }

private:
    std::unique_ptr<IMetric> base_;
    double mu_, b_;
};

} // namespace

std::array<double, S2Parameters::kNumParams> S2Parameters::to_array() const {
    return {M_msun,    R0_pc,   a_mas,  e,        incl_deg, Omega_deg,
            omega_deg, t_peri_yr, x0_mas, y0_mas, vx_masyr, vy_masyr,
            vz_kms};
}
S2Parameters S2Parameters::from_array(const double* p) {
    S2Parameters s;
    s.M_msun = p[0];   s.R0_pc = p[1];    s.a_mas = p[2];    s.e = p[3];
    s.incl_deg = p[4]; s.Omega_deg = p[5]; s.omega_deg = p[6];
    s.t_peri_yr = p[7]; s.x0_mas = p[8];  s.y0_mas = p[9];
    s.vx_masyr = p[10]; s.vy_masyr = p[11]; s.vz_kms = p[12];
    return s;
}

double S2Model::mas_per_M(const S2Parameters& p) const {
    const double rg_m = p.M_msun * constants::GM_sun_si /
                        (constants::c_si * constants::c_si);
    const double rad = rg_m / (p.R0_pc * constants::pc_m);
    return rad * constants::rad_to_uas / 1000.0;
}
double S2Model::yr_per_M(const S2Parameters& p) const {
    const double tg_s = p.M_msun * constants::GM_sun_si /
                        (constants::c_si * constants::c_si * constants::c_si);
    return tg_s / constants::yr_s;
}

bool S2Model::integrate(const S2Parameters& p, double t_min_yr, double t_max_yr) {
    traj_.clear();
    // Sanity guards against runaway fit parameter space. The defaults (5 mas,
    // 100 M, and the 50 M floor below) sit orders of magnitude below the real
    // S-star scales and are what every fit uses — unchanged. They are
    // configurable only so the interactive instrument can integrate its
    // hypothetical *shrunken* orbits, which are legitimate GR orbits merely
    // far smaller than anything a fit explores. See set_limits().
    if (p.M_msun < 1e5 || p.R0_pc < 1000.0 || p.e <= 0.0 || p.e >= 0.999 ||
        p.a_mas < lim_a_mas_)
        return false;

    const double masM = mas_per_M(p);
    const double a_M = p.a_mas / masM;
    yrM_ = yr_per_M(p);
    if (a_M < lim_a_M_) return false;

    // --- Keplerian pericenter initial conditions ----------------------------
    const double i = p.incl_deg * kDeg, Om = p.Omega_deg * kDeg,
                 om = p.omega_deg * kDeg;
    const std::array<double, 3> e_node{std::sin(Om), std::cos(Om), 0.0};
    const std::array<double, 3> e_perp{std::cos(Om) * std::cos(i),
                                       -std::sin(Om) * std::cos(i),
                                       std::sin(i)};
    // Darwin-style GR elements: choose E, L so the Schwarzschild radial
    // potential (dr/dτ)² = E² − (1−2M/r)(1+L²/r²) vanishes exactly at
    // r_p = a(1−e) and r_a = a(1+e). This keeps (a, e) geometrically
    // meaningful (turning radii) in the relativistic orbit — launching with
    // the *Newtonian* pericenter speed instead shifts the effective period by
    // O(v_p⁴/|ε_bind|) ≈ 1.6% for S2's eccentricity, which would poison the
    // comparison with the published Keplerian elements.
    const double r_p = a_M * (1.0 - p.e);
    const double r_a = a_M * (1.0 + p.e);
    const double fp = 1.0 - 2.0 / r_p, fa = 1.0 - 2.0 / r_a;
    const double L2 = 2.0 * (1.0 / r_p - 1.0 / r_a) /
                      (fp / (r_p * r_p) - fa / (r_a * r_a));
    if (L2 <= 0.0) return false;
    const double E2 = fp * (1.0 + L2 / (r_p * r_p));
    if (E2 >= 1.0 || E2 <= 0.0) return false; // unbound or invalid
    const double u_tang = std::sqrt(L2) / r_p;  // proper tangential speed at peri

    std::array<double, 3> xhat{}, vhat{};
    for (int k = 0; k < 3; ++k) {
        xhat[k] = std::cos(om) * e_node[k] + std::sin(om) * e_perp[k];
        vhat[k] = -std::sin(om) * e_node[k] + std::cos(om) * e_perp[k];
    }
    const std::array<double, 3> x0{r_p * xhat[0], r_p * xhat[1], r_p * xhat[2]};
    // Proper-velocity (u^i) components — not coordinate velocity.
    const std::array<double, 3> v0{u_tang * vhat[0], u_tang * vhat[1],
                                   u_tang * vhat[2]};

    const double r0 = r_p;
    const double th0 = std::acos(std::clamp(x0[2] / r0, -1.0, 1.0));
    const double ph0 = std::atan2(x0[1], x0[0]);
    const double st = std::sin(th0), ct = std::cos(th0);
    const double sp = std::sin(ph0), cp = std::cos(ph0);
    const double vr = v0[0] * st * cp + v0[1] * st * sp + v0[2] * ct;
    const double vth = (v0[0] * ct * cp + v0[1] * ct * sp - v0[2] * st) / r0;
    const double vph = (-v0[0] * sp + v0[1] * cp) / (r0 * st);

    std::unique_ptr<IMetric> metric;
    if (eps_q_ == 0.0)
        metric = std::make_unique<KerrMetric>(1.0, a_star_);
    else
        metric = std::make_unique<HartleThorneMetric>(1.0, a_star_, eps_q_);
    if (ext_frac_ != 0.0)
        metric = std::make_unique<PlummerMetric>(std::move(metric), ext_frac_,
                                                 ext_scale_mas_ / masM);

    const Metric4<double> g0 = metric->components(r0, th0);
    TimelikeState y0{};
    y0[0] = 0.0; y0[1] = r0; y0[2] = th0; y0[3] = ph0;
    y0[5] = vr; y0[6] = vth; y0[7] = vph; // proper-velocity components u^i
    try {
        y0[4] = TimelikeGeodesic::solve_ut(g0, y0[5], y0[6], y0[7]);
    } catch (const std::exception&) {
        return false;
    }

    TimelikeGeodesic geo(*metric);
    TimelikeStepper stepper{&geo, 1e-11, 1e-11};

    auto push_sample = [&](const TimelikeState& y) {
        const CartesianState c = to_cartesian(y);
        TrajectorySample s;
        s.t_yr = p.t_peri_yr + y[0] * yrM_;
        s.pos_M = c.x;
        s.vel = c.v;
        s.redshift = line_of_sight_redshift(*metric, y);
        traj_.push_back(s);
    };

    const double t_fwd_M = (t_max_yr - p.t_peri_yr) / yrM_;
    const double t_bwd_M = (t_min_yr - p.t_peri_yr) / yrM_; // negative
    for (const double target : {t_fwd_M, t_bwd_M}) {
        TimelikeState y = y0;
        double tau = 0.0, h = (target > 0 ? 50.0 : -50.0);
        if (target > 0) push_sample(y);
        int guard = 0;
        while ((target > 0 ? y[0] < target : y[0] > target)) {
            try {
                stepper.step(tau, y, h);
            } catch (const std::exception&) {
                return false;
            }
            if (y[1] < lim_r_min_ || y[1] > 1e7) return false; // plunged/escaped
            push_sample(y);
            if (++guard > 400000) return false;
        }
    }
    std::sort(traj_.begin(), traj_.end(),
              [](const TrajectorySample& a, const TrajectorySample& b) {
                  return a.t_yr < b.t_yr;
              });
    t_peri_cache_ = p.t_peri_yr;
    return traj_.size() > 10;
}

TrajectorySample S2Model::sample_at(double t_yr) const {
    const auto it = std::lower_bound(
        traj_.begin(), traj_.end(), t_yr,
        [](const TrajectorySample& s, double t) { return s.t_yr < t; });
    if (it == traj_.begin()) return traj_.front();
    if (it == traj_.end()) return traj_.back();
    const TrajectorySample& b = *it;
    const TrajectorySample& a = *(it - 1);
    const double dt = b.t_yr - a.t_yr;
    if (dt <= 0.0) return a;
    const double u = (t_yr - a.t_yr) / dt;
    const double h00 = (1 + 2 * u) * (1 - u) * (1 - u);
    const double h10 = u * (1 - u) * (1 - u);
    const double h01 = u * u * (3 - 2 * u);
    const double h11 = u * u * (u - 1);

    TrajectorySample out;
    out.t_yr = t_yr;
    for (int k = 0; k < 3; ++k) {
        // Hermite tangents in M/yr: vel is dx/dt in c-units and coordinate
        // time maps to years via yrM_, so dx/dt_yr = vel / yrM_.
        const double ta = a.vel[k] / yrM_;
        const double tb = b.vel[k] / yrM_;
        out.pos_M[k] = h00 * a.pos_M[k] + h10 * dt * ta + h01 * b.pos_M[k] +
                       h11 * dt * tb;
        out.vel[k] = (1 - u) * a.vel[k] + u * b.vel[k];
    }
    out.redshift = (1 - u) * a.redshift + u * b.redshift;
    return out;
}

S2Observables S2Model::evaluate(const S2Parameters& p,
                                const std::vector<double>& astrometry_epochs,
                                const std::vector<double>& rv_epochs) const {
    S2Observables obs;
    if (traj_.size() < 2) return obs;
    const double masM = mas_per_M(p);
    const double c_kms = constants::c_si / 1000.0;

    // Rømer-corrected emission epoch: t_obs = t_em + z_M · yrM_ (light travel
    // time of the LOS offset z; the constant R0 leg is absorbed in the epoch
    // definition). Fixed point converges in 2–3 iterations (v/c ≤ 0.026).
    auto emission_time = [&](double t_obs) {
        double t_em = t_obs;
        for (int k = 0; k < 3; ++k) {
            const TrajectorySample s = sample_at(t_em);
            t_em = t_obs - s.pos_M[2] * yrM_;
        }
        return t_em;
    };

    for (const double t_obs : astrometry_epochs) {
        const TrajectorySample s = sample_at(emission_time(t_obs));
        obs.ra_mas.push_back(s.pos_M[0] * masM + p.x0_mas +
                             p.vx_masyr * (t_obs - 2009.0));
        obs.dec_mas.push_back(s.pos_M[1] * masM + p.y0_mas +
                              p.vy_masyr * (t_obs - 2009.0));
    }
    for (const double t_obs : rv_epochs) {
        const TrajectorySample s = sample_at(emission_time(t_obs));
        obs.vr_kms.push_back((s.redshift - 1.0) * c_kms + p.vz_kms);
    }
    obs.valid = true;
    return obs;
}

std::array<double, 4> S2Model::redshift_terms(double t_yr) const {
    const TrajectorySample s = sample_at(t_yr);
    const double r = std::sqrt(s.pos_M[0] * s.pos_M[0] +
                               s.pos_M[1] * s.pos_M[1] +
                               s.pos_M[2] * s.pos_M[2]);
    const double beta2 = s.vel[0] * s.vel[0] + s.vel[1] * s.vel[1] +
                         s.vel[2] * s.vel[2];
    const double beta_los = s.vel[2]; // ż > 0 = receding
    const double z_total = s.redshift - 1.0;
    const double z_trans = 1.0 / std::sqrt(1.0 - beta2) - 1.0;
    const double z_grav = 1.0 / std::sqrt(1.0 - 2.0 / r) - 1.0;
    return {z_total, beta_los, z_trans, z_grav};
}

} // namespace sgr
