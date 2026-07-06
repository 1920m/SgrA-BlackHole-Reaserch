// Phase 4/5 tool: fit the S2 orbit to the Gillessen 2017 dataset.
//
//   fit_s2_tool <data_dir> <out_dir> [eps_q_mode]
//
//   eps_q_mode: "kerr" (default, 13-parameter fit, a*=0) or "nohair"
//               (Phase 5: 14th free parameter ε_q via profile scan)
//
// Outputs: s2_fit.json, residuals CSV, model trajectory CSV, spectral CSV.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "orbit_fitter/GRPrecession.hpp"
#include "orbit_fitter/OrbitFitter.hpp"
#include "spectral_fitter/SpectralLineFitter.hpp"
#include "sgr_physics/constants.hpp"

using namespace sgr;
namespace fs = std::filesystem;

namespace {

// Thesis MCMC evaluation server (paper/THESIS_PLAN.md §2.1): read one
// whitespace-separated parameter vector per stdin line — 13 values (kerr) or
// 14 with a trailing ε_q (nohair) — and print χ² per line. Persistent process
// so a Python sampler (scripts/run_mcmc.py) pays the dataset/model setup once.
int run_chi2_server(const fs::path& data_dir, const std::string& mode) {
    const S2Dataset data =
        S2Dataset::load((data_dir / "s2_astrometry.csv").string(),
                        (data_dir / "s2_radial_velocity.csv").string());
    const bool nohair = (mode == "nohair");
    const int nvals = S2Parameters::kNumParams + (nohair ? 1 : 0);
    std::string line;
    std::vector<double> v(nvals);
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        bool ok = true;
        for (int i = 0; i < nvals; ++i)
            if (!(iss >> v[i])) { ok = false; break; }
        double c2 = 1e30;
        if (ok) {
            try {
                const S2Parameters p = S2Parameters::from_array(v.data());
                const OrbitFitter f(data, /*a_star=*/0.0,
                                    /*eps_q=*/nohair ? v[13] : 0.0);
                c2 = f.chi2(p);
                if (!std::isfinite(c2)) c2 = 1e30;
            } catch (...) {
                c2 = 1e30;
            }
        }
        std::printf("%.10e\n", c2);
        std::fflush(stdout);
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <data_dir> <out_dir> [kerr|nohair]\n"
                     "       %s --chi2 <data_dir> [kerr|nohair]\n",
                     argv[0], argv[0]);
        return 2;
    }
    if (std::string(argv[1]) == "--chi2")
        return run_chi2_server(argv[2], argc > 3 ? argv[3] : "kerr");
    const fs::path data_dir = argv[1];
    const fs::path out_dir = argv[2];
    const std::string mode = argc > 3 ? argv[3] : "kerr";
    fs::create_directories(out_dir);

    const S2Dataset data =
        S2Dataset::load((data_dir / "s2_astrometry.csv").string(),
                        (data_dir / "s2_radial_velocity.csv").string());
    spdlog::info("loaded {} astrometric + {} RV points",
                 data.astrometry.size(), data.rv.size());

    // Published Gillessen 2017 values as the starting point.
    S2Parameters start;
    start.M_msun = 4.35e6;
    start.R0_pc = 8330.0;
    start.a_mas = 125.5;
    start.e = 0.8839;
    start.incl_deg = 134.7;
    start.Omega_deg = 227.9;
    start.omega_deg = 66.1;
    start.t_peri_yr = 2002.33;

    OrbitFitter fitter(data, /*a_star=*/0.0, /*eps_q=*/0.0);
    spdlog::info("chi2 at start = {:.2f}", fitter.chi2(start));

    const FitResult res = fitter.fit(start);
    spdlog::info("converged={} chi2={:.2f} dof={} chi2red={:.4f}",
                 res.converged, res.chi2, res.dof, res.chi2_red);
    const auto& p = res.params;
    spdlog::info("M = ({:.4f} +- {:.4f}) x 1e6 Msun", p.M_msun / 1e6,
                 res.sigma[0] / 1e6);
    spdlog::info("R0 = {:.1f} +- {:.1f} pc", p.R0_pc, res.sigma[1]);
    spdlog::info("a = {:.3f} +- {:.3f} mas, e = {:.5f} +- {:.5f}", p.a_mas,
                 res.sigma[2], p.e, res.sigma[3]);
    spdlog::info("i = {:.2f}, Omega = {:.2f}, omega = {:.2f} deg", p.incl_deg,
                 p.Omega_deg, p.omega_deg);
    spdlog::info("t_peri = {:.4f} +- {:.4f}", p.t_peri_yr, res.sigma[7]);
    spdlog::info("P = {:.4f} +- {:.4f} yr (Kepler-derived)", res.period_yr,
                 res.sigma_period_yr);

    // ---- model artefacts at the best fit -----------------------------------
    S2Model model(0.0, 0.0);
    model.integrate(p, 1990.0, 2020.0);

    // Residuals CSV
    {
        std::vector<double> ae, re;
        for (const auto& a : data.astrometry) ae.push_back(a.epoch_yr);
        for (const auto& v : data.rv) re.push_back(v.epoch_yr);
        const S2Observables obs = model.evaluate(p, ae, re);
        std::ofstream csv(out_dir / "s2_residuals.csv");
        csv << "type,epoch_yr,obs,model,sigma,norm_resid\n";
        for (size_t i = 0; i < data.astrometry.size(); ++i) {
            const auto& a = data.astrometry[i];
            csv << "ra," << a.epoch_yr << "," << a.ra_mas << "," << obs.ra_mas[i]
                << "," << a.sig_ra_mas << ","
                << (obs.ra_mas[i] - a.ra_mas) / a.sig_ra_mas << "\n";
            csv << "dec," << a.epoch_yr << "," << a.dec_mas << ","
                << obs.dec_mas[i] << "," << a.sig_dec_mas << ","
                << (obs.dec_mas[i] - a.dec_mas) / a.sig_dec_mas << "\n";
        }
        for (size_t j = 0; j < data.rv.size(); ++j) {
            const auto& v = data.rv[j];
            csv << "rv," << v.epoch_yr << "," << v.vr_kms << "," << obs.vr_kms[j]
                << "," << v.sig_vr_kms << ","
                << (obs.vr_kms[j] - v.vr_kms) / v.sig_vr_kms << "\n";
        }
    }

    // Dense model trajectory on the sky (figure data)
    {
        std::ofstream csv(out_dir / "s2_model_track.csv");
        csv << "epoch_yr,ra_mas,dec_mas,vr_kms\n";
        std::vector<double> ep;
        for (double t = 1992.0; t <= 2019.5; t += 0.02) ep.push_back(t);
        const S2Observables obs = model.evaluate(p, ep, ep);
        for (size_t i = 0; i < ep.size(); ++i)
            csv << ep[i] << "," << obs.ra_mas[i] << "," << obs.dec_mas[i] << ","
                << obs.vr_kms[i] << "\n";
    }

    // Spectral module: redshift decomposition through the 2018.38 pericenter
    double peak_t = 0.0, peak_v = 0.0;
    {
        SpectralLineFitter spec(model);
        const auto curve = spec.predict(2016.0, 2020.0, 1200);
        std::ofstream csv(out_dir / "s2_redshift_curve.csv");
        csv << "epoch_yr,lambda_obs_um,v_total_kms,v_newton_kms,v_excess_kms,"
               "v_trans_kms,v_grav_kms\n";
        for (const auto& s : curve)
            csv << s.t_yr << "," << s.lambda_obs_um << "," << s.v_total_kms
                << "," << s.v_newton_kms << "," << s.v_excess_kms << ","
                << s.v_trans_kms << "," << s.v_grav_kms << "\n";
        const auto pk = spec.peak_excess(2017.5, 2019.0);
        peak_t = pk[0];
        peak_v = pk[1];
        spdlog::info("GR excess peak: {:.1f} km/s at epoch {:.3f} "
                     "(GRAVITY 2018: ~200 km/s at 2018.38)",
                     peak_v, peak_t);
    }

    // Numerical pericenter precession from the trajectory vs analytic.
    const double masM = model.mas_per_M(p);
    const double aM = p.a_mas / masM;
    const double dw_s = precession::schwarzschild(aM, p.e);
    const double dw_lt = precession::lense_thirring(aM, p.e, 0.94);
    spdlog::info("analytic precession: Schwarzschild {:.3f} arcmin/orbit, "
                 "LT(a*=0.94) {:.4f} arcmin/orbit",
                 dw_s * constants::rad_to_arcmin,
                 dw_lt * constants::rad_to_arcmin);

    // ---- Phase 5: no-hair probe --------------------------------------------
    nlohmann::json nohair;
    if (mode == "nohair") {
        double eq_best = 0.0, eq_sigma = 0.0;
        spdlog::info("no-hair mode: 14-parameter fit with free eps_q ...");
        const FitResult nh = fitter.fit_nohair(p, 0.0, &eq_best, &eq_sigma);
        spdlog::info("eps_q = {:.1f} +- {:.1f} (chi2={:.2f}, chi2red={:.4f})",
                     eq_best, eq_sigma, nh.chi2, nh.chi2_red);

        // Profile-likelihood scan for a robust 95% CI (Δχ² = 3.84).
        nlohmann::json profile = nlohmann::json::array();
        std::vector<std::pair<double, double>> prof;
        for (double f : {-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0}) {
            const double eq = eq_best + f * eq_sigma;
            OrbitFitter pf(data, 0.0, eq);
            const FitResult r = pf.fit(nh.params);
            prof.push_back({eq, r.chi2});
            profile.push_back({{"eps_q", eq}, {"chi2", r.chi2}});
            spdlog::info("profile eps_q={:.1f}: chi2={:.3f}", eq, r.chi2);
        }
        // quadratic fit of χ²(ε_q) around the minimum → 95% CI half-width
        double cmin = 1e30, emin = 0.0;
        for (auto& pr : prof)
            if (pr.second < cmin) { cmin = pr.second; emin = pr.first; }
        // effective curvature from the ±2σ points
        double curv = 0.0;
        int ncurv = 0;
        for (auto& pr : prof) {
            const double de = pr.first - emin;
            if (std::abs(de) > 0.5 * eq_sigma) {
                curv += 2.0 * (pr.second - cmin) / (de * de);
                ++ncurv;
            }
        }
        curv /= std::max(1, ncurv);
        const double ci95 = std::sqrt(3.84 / std::max(curv, 1e-30));
        spdlog::info("no-hair 95% CI: eps_q = {:.1f} +- {:.1f} "
                     "(0 within CI: {})",
                     emin, ci95, std::abs(emin) < ci95 ? "YES" : "NO");
        nohair = {{"eps_q_best", eq_best},
                  {"eps_q_sigma", eq_sigma},
                  {"eps_q_profile_min", emin},
                  {"eps_q_ci95_halfwidth", ci95},
                  {"zero_within_ci95", std::abs(emin) < ci95},
                  {"chi2_nohair", nh.chi2},
                  {"chi2red_nohair", nh.chi2_red},
                  {"profile", profile},
                  {"forecast_sigma_10_more_orbits",
                   eq_sigma / std::sqrt(1.0 + 10.0 / 1.5)}};
    }

    nlohmann::json j{
        {"mode", mode},
        {"n_astrometry", data.astrometry.size()},
        {"n_rv", data.rv.size()},
        {"converged", res.converged},
        {"chi2", res.chi2},
        {"dof", res.dof},
        {"chi2_red", res.chi2_red},
        {"M_msun", p.M_msun},
        {"sigma_M_msun", res.sigma[0]},
        {"R0_pc", p.R0_pc},
        {"sigma_R0_pc", res.sigma[1]},
        {"a_mas", p.a_mas}, {"sigma_a_mas", res.sigma[2]},
        {"e", p.e}, {"sigma_e", res.sigma[3]},
        {"incl_deg", p.incl_deg}, {"sigma_incl_deg", res.sigma[4]},
        {"Omega_deg", p.Omega_deg}, {"sigma_Omega_deg", res.sigma[5]},
        {"omega_deg", p.omega_deg}, {"sigma_omega_deg", res.sigma[6]},
        {"t_peri_yr", p.t_peri_yr}, {"sigma_t_peri_yr", res.sigma[7]},
        {"x0_mas", p.x0_mas}, {"y0_mas", p.y0_mas},
        {"vx_masyr", p.vx_masyr}, {"vy_masyr", p.vy_masyr},
        {"vz_kms", p.vz_kms},
        {"period_yr", res.period_yr},
        {"sigma_period_yr", res.sigma_period_yr},
        {"gr_excess_peak_kms", peak_v},
        {"gr_excess_peak_epoch", peak_t},
        {"precession_schw_arcmin", dw_s * constants::rad_to_arcmin},
        {"precession_lt_arcmin_a094", dw_lt * constants::rad_to_arcmin},
        {"published", {{"M_msun", 4.297e6}, {"sigma_M_msun", 0.012e6},
                       {"R0_pc", 8277.0}, {"sigma_R0_pc", 9.0},
                       {"period_yr", 16.0518}, {"e", 0.8839}}},
    };
    if (!nohair.is_null()) j["nohair"] = nohair;
    std::ofstream(out_dir / "s2_fit.json") << j.dump(2) << "\n";
    std::ofstream(out_dir / "ceres_report.txt") << res.report << "\n";
    spdlog::info("wrote {}", (out_dir / "s2_fit.json").string());

    // Gate 4 verdict
    const double nsig =
        std::abs(p.M_msun - 4.297e6) /
        std::sqrt(res.sigma[0] * res.sigma[0] + 0.012e6 * 0.012e6);
    spdlog::info("Gate 4: M within {:.2f} sigma of GRAVITY 2022; chi2red={:.3f}",
                 nsig, res.chi2_red);
    return (res.converged && nsig < 2.0 && res.chi2_red < 2.0) ? 0 : 1;
}
