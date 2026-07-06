// Thesis data tool (paper/THESIS_PLAN.md §2.1): dump CSV tables of exact Kerr
// quantities over a spin grid for the report figures. Pure formula evaluation
// plus a handful of reference geodesic integrations — additive, read-only with
// respect to the validated physics core.
//
//   physics_tables <out_dir>
//
// Outputs (CSV, one header row):
//   radii_vs_spin.csv       r₊, r_E, r_ph±, r_ISCO±, r_mb± over a* ∈ [0, 0.998]
//   efficiency_vs_spin.csv  E_ISCO± and accretion efficiency η = 1 − E_ISCO
//   energetics_r.csv        E(r), L(r), Ω(r) prograde at several spins
//   disk_profiles.csv       Page–Thorne F(r) (relative) and T/T_peak at several spins
//   photon_potential.csv    R(r)/E² for equatorial photons at several (a*, b)
//   timelike_potential.csv  effective potential V_eff(r) = E₊(r) at several L
//   trajectories_photon.csv equatorial photon paths (bending / capture / dragging)
//   trajectory_rosette.csv  equatorial timelike rosette (precession showcase)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "sgr_physics/disk_model.hpp"
#include "sgr_physics/integrator.hpp"
#include "sgr_physics/metric.hpp"
#include "sgr_physics/null_geodesics.hpp"
#include "sgr_physics/timelike_geodesics.hpp"

using namespace sgr;
namespace fs = std::filesystem;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Marginally bound circular orbit radius (BPT72 eq. 2.19):
// r_mb = 2M ∓ a + 2√(M(M ∓ a)), upper sign prograde.
double r_marginally_bound(double M, double a, bool prograde) {
    const double s = prograde ? +1.0 : -1.0;
    return 2.0 * M - s * a + 2.0 * std::sqrt(M * (M - s * a));
}

// Bardeen critical impact parameter ξ(r_ph) for equatorial photons.
double xi_critical(double M, double a, double r_ph) {
    return (r_ph * r_ph * (3.0 * M - r_ph) - a * a * (r_ph + M)) /
           (a * (r_ph - M));
}

// Equatorial photon path: E = 1, L_z = b, Q = 0, launched ingoing from r0.
// Integrates the second-order Mino form with per-step projection and appends
// "label,a_star,b,r,phi" rows until capture/escape/step budget.
void dump_photon_path(std::ofstream& csv, const std::string& label,
                      double a_star, double b, double r0) {
    const KerrMetric metric(1.0, a_star);
    NullRayConstants c;
    c.M = 1.0;
    c.a = metric.spin();
    c.E = 1.0;
    c.Lz = b;
    c.Q = 0.0;
    const double R0 = radial_potential(c, r0);
    if (R0 < 0.0) return; // not a valid launch radius for this b
    NullState y{0.0, r0, kPi / 2.0, 0.0, -std::sqrt(R0), 0.0};

    auto rhs = [&c](double, const NullState& s, NullState& d) {
        null_geodesic_rhs(c, s, d);
    };
    DormandPrince54<6, decltype(rhs)> rk(rhs, 1e-11, 1e-11);

    const double r_capture = metric.r_horizon() * 1.001 + 1e-6;
    const double r_escape = 1.5 * r0 + 30.0;
    double tau = 0.0, h = 1e-3;
    for (int step = 0; step < 60000; ++step) {
        const auto res = rk.try_step(tau, y, h);
        h = std::clamp(res.h_next, 1e-14, 5.0);
        if (!res.accepted) continue;
        const double Rp = radial_potential(c, y[1]);
        if (Rp >= 0.0) y[4] = std::copysign(std::sqrt(Rp), y[4]);
        csv << label << ',' << a_star << ',' << b << ',' << y[1] << ','
            << y[3] << '\n';
        if (y[1] < r_capture || y[1] > r_escape) break;
        if (std::abs(y[3]) > 8.0 * kPi) break; // enough winding for a plot
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <out_dir>\n", argv[0]);
        return 2;
    }
    const fs::path out = argv[1];
    fs::create_directories(out);

    const std::vector<double> profile_spins{0.0, 0.5, 0.94, 0.998};

    // ---- radii_vs_spin.csv + efficiency_vs_spin.csv -----------------------
    {
        std::ofstream radii(out / "radii_vs_spin.csv");
        radii << "a_star,r_plus,r_ergo_eq,r_ph_pro,r_ph_retro,"
                 "r_isco_pro,r_isco_retro,r_mb_pro,r_mb_retro\n";
        std::ofstream eff(out / "efficiency_vs_spin.csv");
        eff << "a_star,E_isco_pro,eta_pro,E_isco_retro,eta_retro\n";
        const int N = 200;
        for (int i = 0; i <= N; ++i) {
            const double a = 0.998 * i / N;
            const KerrMetric m(1.0, a);
            radii << a << ',' << m.r_horizon() << ','
                  << m.r_ergosphere(kPi / 2.0) << ',' << m.r_photon(true) << ','
                  << m.r_photon(false) << ',' << m.r_isco(true) << ','
                  << m.r_isco(false) << ',' << r_marginally_bound(1.0, a, true)
                  << ',' << r_marginally_bound(1.0, a, false) << '\n';
            const double Ep = m.energy_circular(m.r_isco(true), true);
            const double Er = m.energy_circular(m.r_isco(false), false);
            eff << a << ',' << Ep << ',' << 1.0 - Ep << ',' << Er << ','
                << 1.0 - Er << '\n';
        }
    }

    // ---- energetics_r.csv --------------------------------------------------
    {
        std::ofstream csv(out / "energetics_r.csv");
        csv << "a_star,r,E,L,Omega\n";
        for (double a : profile_spins) {
            const KerrMetric m(1.0, a);
            const double r0 = m.r_isco(true);
            for (int i = 0; i <= 400; ++i) {
                const double r = r0 + (20.0 - r0) * i / 400.0;
                csv << a << ',' << r << ',' << m.energy_circular(r, true) << ','
                    << m.angmom_circular(r, true) << ','
                    << m.omega_circular(r, true) << '\n';
            }
        }
    }

    // ---- disk_profiles.csv --------------------------------------------------
    {
        std::ofstream csv(out / "disk_profiles.csv");
        csv << "a_star,r,flux_rel,T_over_Tpeak\n";
        for (double a : profile_spins) {
            const KerrMetric m(1.0, a);
            const double rin = m.r_isco(true);
            NovikovThorneDisk disk(a, rin, 50.0);
            // normalise flux to its own maximum for a shape comparison
            double fmax = 0.0;
            for (int i = 0; i <= 600; ++i) {
                const double r = rin + (50.0 - rin) * i / 600.0;
                fmax = std::max(fmax, disk.flux(r));
            }
            const double tpeak = 6300.0;
            for (int i = 0; i <= 600; ++i) {
                const double r = rin + (50.0 - rin) * i / 600.0;
                csv << a << ',' << r << ',' << disk.flux(r) / fmax << ','
                    << disk.temperature(r) / tpeak << '\n';
            }
        }
    }

    // ---- photon_potential.csv ----------------------------------------------
    {
        std::ofstream csv(out / "photon_potential.csv");
        csv << "a_star,b,r,R_over_E2\n";
        // Schwarzschild: below / at / above the critical b = √27
        const double bcrit0 = std::sqrt(27.0);
        std::vector<std::pair<double, double>> sets; // (a*, b)
        for (double b : {4.8, bcrit0, 6.2}) sets.emplace_back(0.0, b);
        // Kerr 0.94: prograde and retrograde critical impact parameters
        {
            const KerrMetric m(1.0, 0.94);
            const double xp = xi_critical(1.0, m.spin(), m.r_photon(true));
            const double xr = xi_critical(1.0, m.spin(), m.r_photon(false));
            for (double b : {xp - 1.0, xp, xp + 1.0, xr}) sets.emplace_back(0.94, b);
        }
        for (const auto& [a, b] : sets) {
            const KerrMetric m(1.0, a);
            NullRayConstants c;
            c.M = 1.0; c.a = m.spin(); c.E = 1.0; c.Lz = b; c.Q = 0.0;
            const double r0 = m.r_horizon() + 1e-3;
            for (int i = 0; i <= 500; ++i) {
                const double r = r0 + (12.0 - r0) * i / 500.0;
                csv << a << ',' << b << ',' << r << ','
                    << radial_potential(c, r) << '\n';
            }
        }
    }

    // ---- timelike_potential.csv --------------------------------------------
    // V_eff(r) = E₊(r), the energy at which r is a radial turning point:
    // R(r) = A E² + B E + C with A = (r²+a²)² − Δa², B = −4MraL,
    // C = −Δr² + L²(a² − Δ)  →  E₊ = (−B + √(B²−4AC)) / 2A.
    {
        std::ofstream csv(out / "timelike_potential.csv");
        csv << "a_star,L,r,Veff\n";
        for (double a : {0.0, 0.94}) {
            const KerrMetric m(1.0, a);
            const double Lisco = m.angmom_circular(m.r_isco(true), true);
            for (double fL : {0.94, 1.0, 1.15, 1.35}) {
                const double L = fL * Lisco;
                const double aa = m.spin();
                const double r0 = m.r_horizon() + 0.05;
                for (int i = 0; i <= 600; ++i) {
                    const double r = r0 + (40.0 - r0) * i / 600.0;
                    const double delta = r * r - 2.0 * r + aa * aa;
                    const double A = (r * r + aa * aa) * (r * r + aa * aa) -
                                     delta * aa * aa;
                    const double B = -4.0 * r * aa * L;
                    const double C = -delta * r * r + L * L * (aa * aa - delta);
                    const double disc = B * B - 4.0 * A * C;
                    if (disc < 0.0) continue;
                    csv << a << ',' << L << ',' << r << ','
                        << (-B + std::sqrt(disc)) / (2.0 * A) << '\n';
                }
            }
        }
    }

    // ---- trajectories_photon.csv -------------------------------------------
    {
        std::ofstream csv(out / "trajectories_photon.csv");
        csv << "label,a_star,b,r,phi\n";
        const double bcrit0 = std::sqrt(27.0);
        // Schwarzschild: weak bending → capture, around the critical b.
        for (double b : {8.0, 6.0, 5.4, bcrit0 + 0.002, bcrit0 - 0.05, 4.0, 2.0})
            dump_photon_path(csv, "schw", 0.0, b, 30.0);
        // Kerr a* = 0.94: prograde vs retrograde asymmetry (frame dragging).
        {
            const KerrMetric m(1.0, 0.94);
            const double xp = xi_critical(1.0, m.spin(), m.r_photon(true));
            const double xr = xi_critical(1.0, m.spin(), m.r_photon(false));
            for (double b : {xp + 0.02, xp + 1.5, xp - 0.4})
                dump_photon_path(csv, "kerr_pro", 0.94, b, 30.0);
            for (double b : {xr - 0.02, xr - 1.5, xr + 0.4})
                dump_photon_path(csv, "kerr_retro", 0.94, b, 30.0);
        }
    }

    // ---- trajectory_rosette.csv --------------------------------------------
    // Schwarzschild equatorial bound orbit r ∈ [8, 25] M: E², L² from the two
    // turning-point conditions E²r⁴ = Δ(r² + L²) at r_p and r_a (Darwin
    // elements, the same construction the S2 model uses), integrated with the
    // generic timelike (Christoffel) integrator for several radial periods.
    {
        const double rp = 8.0, ra = 25.0;
        const double dp = rp * rp - 2.0 * rp; // Δ(r_p), a = 0
        const double da = ra * ra - 2.0 * ra;
        // E² rp⁴ − dp L² = dp rp²;  E² ra⁴ − da L² = da ra²  → 2×2 solve
        // (Cramer with det = dp ra⁴ − da rp⁴)
        const double rp4 = rp * rp * rp * rp, ra4 = ra * ra * ra * ra;
        const double det = dp * ra4 - da * rp4;
        const double E2 = dp * da * (ra * ra - rp * rp) / det;
        const double L2 = (rp4 * da * ra * ra - dp * rp * rp * ra4) / det;
        const double E = std::sqrt(E2), L = std::sqrt(L2);

        const KerrMetric metric(1.0, 0.0);
        const TimelikeGeodesic geo(metric);
        TimelikeState y{};
        y[0] = 0.0; y[1] = ra; y[2] = kPi / 2.0; y[3] = 0.0;
        y[5] = 0.0; y[6] = 0.0; y[7] = L / (ra * ra);
        y[4] = TimelikeGeodesic::solve_ut(metric.components(ra, kPi / 2.0),
                                          y[5], y[6], y[7]);
        TimelikeStepper stepper{&geo, 1e-12, 1e-12};

        std::ofstream csv(out / "trajectory_rosette.csv");
        csv << "tau,r,phi,E,L,norm\n";
        double tau = 0.0, h = 0.1;
        for (int step = 0; step < 400000 && y[3] < 12.0 * kPi; ++step) {
            stepper.step(tau, y, h);
            if (step % 8 == 0)
                csv << tau << ',' << y[1] << ',' << y[3] << ',' << geo.energy(y)
                    << ',' << geo.angmom(y) << ',' << geo.norm(y) << '\n';
        }
        std::printf("rosette: E=%.12f L=%.12f (target E=%.12f L=%.12f)\n",
                    geo.energy(y), geo.angmom(y), E, L);
    }

    std::printf("physics_tables: wrote 8 CSVs to %s\n", out.string().c_str());
    return 0;
}
