// Gate 22 — science forecasts for the extension paper: what more S2 data can
// and cannot measure. Every number of Chapter x07 comes from this tool.
//
//   forecast_tool <data_dir> <phase5_dir> <out_dir> [--fisher-only]
//                                                     → <out_dir>/forecasts.json
//   (--fisher-only skips the slow nonlinear bias fits of section 5)
//
// Method. "Truth" is the thesis S2 fit (phase5/s2_fit.json) with the frame
// nuisances (x0, y0, vx, vy, vz) at their prior centres, so noise-free data
// have χ² = 0 at the truth. Synthetic noise-free data are made by the thesis'
// own forward model (S2Model: full GR geodesics, Rømer delay, relativistic
// redshift) at the real Gillessen 2017 epochs with their errors, plus planned
// future epochs. Fisher matrices come from central-difference Jacobians:
// F = JᵀJ for σ-normalised residuals plus the thesis' frame priors; the
// forecast σ = √diag F⁻¹. The bias an unmodelled effect Δd causes is the
// linear response Δθ = F⁻¹Jᵀ(Δd/σ²), checked against a full nonlinear Ceres
// fit (OrbitFitter) of the same data.
//
// Future-data assumptions (stated in the paper; GRAVITY 2020): GRAVITY
// astrometry ≈65 µas per coordinate at 18 epochs per year (54 points in
// 2016.7–2019.7), SINFONI-class radial velocities ≈12 km/s at 5 per year
// (3–6 per year). The extended mass is a Plummer sphere of scale 0.3″, the
// profile GRAVITY 2020 used; its size is quoted, as in GRAVITY 2022, as the
// fraction of M enclosed within S2's apocentre (the model parameter f_ext is
// the sphere's total mass). The spin axis is the model's z axis (the line of
// sight), as everywhere in the thesis.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include "orbit_fitter/OrbitFitter.hpp"
#include "orbit_fitter/SkyProjection.hpp"

using namespace sgr;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

constexpr int NP = 16;   // 13 orbit/frame parameters + ε_q, a*, f_ext
constexpr int kEps = 13, kSpin = 14, kExt = 15;
constexpr double kPlummerMas = 300.0;   // GRAVITY 2020: Plummer scale 0.3″
constexpr double kT0 = 2017.0;          // first planned epoch
// Frame priors exactly as the thesis fitter (Gillessen 2017 §2.3 / Plewa 2015)
constexpr double kSigX0 = 0.2, kSigV0 = 0.1, kSigVz = 5.0;
// Central-difference steps. The thesis' (OrbitFitter::covariance) were set
// for its imaging-era data; at GRAVITY precision over several pericentre
// passages the response of the timing parameters (M, R0, a) and of ε_q is
// visibly nonlinear right after each passage (the star moves ~200 mas/yr on
// a sharply curved path, so a shift of the passage time is not linear in
// position — see --eps-linearity). Those steps are therefore 4–8× smaller;
// the step check below shows the forecasts converged.
constexpr std::array<double, NP> kStep = {5e2,  0.5,  1.25e-3, 2e-5, 2e-3, 2e-3,
                                          2e-3, 2e-5, 2e-3,    2e-3, 1e-3, 1e-3,
                                          0.05, 250.0, 0.02,   5e-5};
const char* kNames[NP] = {"M_msun", "R0_pc", "a_mas", "e", "incl_deg", "Omega_deg",
                          "omega_deg", "t_peri_yr", "x0_mas", "y0_mas", "vx_masyr",
                          "vy_masyr", "vz_kms", "eps_q", "a_star", "f_ext"};

using Theta = std::array<double, NP>;

struct Epochs {
    std::vector<double> ast, rv;
};

// Raw model observables at all epochs: RA, Dec [mas] then RV [km/s].
std::vector<double> predict(const Epochs& ep, const Theta& th) {
    const S2Parameters p = S2Parameters::from_array(th.data());
    S2Model m(th[kSpin], th[kEps]);
    m.set_extended_mass(th[kExt], kPlummerMas);
    // epochs in any order (S2Model::evaluate interpolates each one)
    const double t0 = std::min(*std::min_element(ep.ast.begin(), ep.ast.end()),
                               *std::min_element(ep.rv.begin(), ep.rv.end()));
    const double t1 = std::max(*std::max_element(ep.ast.begin(), ep.ast.end()),
                               *std::max_element(ep.rv.begin(), ep.rv.end()));
    if (!m.integrate(p, t0 - 0.2, t1 + 0.2))
        throw std::runtime_error("forecast_tool: orbit failed to integrate");
    const S2Observables o = m.evaluate(p, ep.ast, ep.rv);
    std::vector<double> out;
    out.reserve(2 * ep.ast.size() + ep.rv.size());
    for (size_t i = 0; i < ep.ast.size(); ++i) {
        out.push_back(o.ra_mas[i]);
        out.push_back(o.dec_mas[i]);
    }
    for (double v : o.vr_kms) out.push_back(v);
    return out;
}

// d(observable)/d(parameter) for the listed columns (others left zero).
std::vector<std::array<double, NP>> jacobian(const Epochs& ep, const Theta& th,
                                             const std::vector<int>& cols,
                                             double step_scale = 1.0) {
    const size_t n = 2 * ep.ast.size() + ep.rv.size();
    std::vector<std::array<double, NP>> J(n);
    for (auto& row : J) row.fill(0.0);
    for (int k : cols) {
        const double h = step_scale * kStep[k];
        Theta tp = th, tm = th;
        tp[k] += h;
        tm[k] -= h;
        const auto yp = predict(ep, tp), ym = predict(ep, tm);
        for (size_t i = 0; i < n; ++i) J[i][k] = (yp[i] - ym[i]) / (2.0 * h);
    }
    return J;
}

// One row of a data set: index into the union prediction and its σ.
struct Row {
    size_t idx;
    double sig;
};

struct Forecast {
    Eigen::MatrixXd C;   // covariance over the chosen columns
    std::vector<int> cols;
    double sigma(int k) const {
        for (size_t a = 0; a < cols.size(); ++a)
            if (cols[a] == k) return std::sqrt(C(a, a));
        return std::nan("");
    }
    double corr(int k, int l) const {
        int a = -1, b = -1;
        for (size_t i = 0; i < cols.size(); ++i) {
            if (cols[i] == k) a = static_cast<int>(i);
            if (cols[i] == l) b = static_cast<int>(i);
        }
        return C(a, b) / std::sqrt(C(a, a) * C(b, b));
    }
};

double prior_sigma(int k) {
    if (k == 8 || k == 9) return kSigX0;
    if (k == 10 || k == 11) return kSigV0;
    if (k == 12) return kSigVz;
    return 0.0;
}

// Fisher matrix over `cols` from the data rows + the frame priors. Columns
// are scaled to unit diagonal before inversion (the M–R0 correlation makes F
// badly scaled in physical units).
Eigen::MatrixXd fisher(const std::vector<std::array<double, NP>>& J,
                       const std::vector<Row>& rows, const std::vector<int>& cols) {
    const int m = static_cast<int>(cols.size());
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(m, m);
    for (const Row& r : rows)
        for (int a = 0; a < m; ++a) {
            const double ja = J[r.idx][cols[a]] / r.sig;
            for (int b = a; b < m; ++b) F(a, b) += ja * J[r.idx][cols[b]] / r.sig;
        }
    for (int a = 0; a < m; ++a) {
        for (int b = 0; b < a; ++b) F(a, b) = F(b, a);
        const double sp = prior_sigma(cols[a]);
        if (sp > 0.0) F(a, a) += 1.0 / (sp * sp);
    }
    return F;
}

Forecast invert(const Eigen::MatrixXd& F, const std::vector<int>& cols) {
    const Eigen::VectorXd d = F.diagonal().cwiseSqrt().cwiseInverse();
    const Eigen::MatrixXd Fs = d.asDiagonal() * F * d.asDiagonal();
    const Eigen::MatrixXd Cs = Fs.ldlt().solve(Eigen::MatrixXd::Identity(F.rows(), F.cols()));
    Forecast f;
    f.C = d.asDiagonal() * Cs * d.asDiagonal();
    f.cols = cols;
    return f;
}

std::vector<int> cols_with(std::initializer_list<int> extra) {
    std::vector<int> c;
    for (int k = 0; k < 13; ++k) c.push_back(k);
    for (int k : extra) c.push_back(k);
    return c;
}

// Linear response of the fit to a data shift Δy: Δθ = F⁻¹Jᵀ(Δy/σ²), and the
// χ² left after the fit absorbs what it can.
struct Bias {
    std::vector<double> dtheta;
    double chi2_before = 0.0, chi2_after = 0.0;
};
Bias linear_bias(const std::vector<std::array<double, NP>>& J, const std::vector<Row>& rows,
                 const std::vector<int>& cols, const std::vector<double>& dy) {
    const Forecast f = invert(fisher(J, rows, cols), cols);
    const int m = static_cast<int>(cols.size());
    Eigen::VectorXd g = Eigen::VectorXd::Zero(m);
    Bias b;
    for (const Row& r : rows) {
        const double z = dy[r.idx] / r.sig;
        b.chi2_before += z * z;
        for (int a = 0; a < m; ++a) g(a) += J[r.idx][cols[a]] / r.sig * z;
    }
    const Eigen::VectorXd dt = f.C * g;
    b.dtheta.assign(dt.data(), dt.data() + m);
    for (const Row& r : rows) {
        double z = dy[r.idx];
        for (int a = 0; a < m; ++a) z -= J[r.idx][cols[a]] * dt(a);
        z /= r.sig;
        b.chi2_after += z * z;
    }
    for (int a = 0; a < m; ++a) {
        const double sp = prior_sigma(cols[a]);
        if (sp > 0.0) b.chi2_after += dt(a) * dt(a) / (sp * sp);
    }
    return b;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <data_dir> <phase5_dir> <out_dir>\n", argv[0]);
        return 2;
    }
    const fs::path data_dir = argv[1], ph5 = argv[2], out = argv[3];
    const bool fisher_only = argc > 4 && std::string(argv[4]) == "--fisher-only";
    fs::create_directories(out);
    const auto clock0 = std::chrono::steady_clock::now();
    auto elapsed = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - clock0).count();
    };

    // ---- truth: the thesis fit, frame nuisances at their prior centres ------
    const json fit = json::parse(std::ifstream(ph5 / "s2_fit.json"));
    Theta truth{};
    for (int k = 0; k < 8; ++k) truth[k] = fit.at(kNames[k]).get<double>();
    // (x0, y0, vx, vy, vz) = 0; ε_q = 0; f_ext = 0; a* set per scenario
    const double sigma_eps_thesis = fit.at("nohair").at("eps_q_sigma").get<double>();
    const double forecast10_thesis =
        fit.at("nohair").at("forecast_sigma_10_more_orbits").get<double>();

    // ---- the real data set: epochs and errors -------------------------------
    const S2Dataset g17 = S2Dataset::load((data_dir / "s2_astrometry.csv").string(),
                                          (data_dir / "s2_radial_velocity.csv").string());
    const double t_last = std::max(g17.astrometry.back().epoch_yr, g17.rv.back().epoch_yr);
    const double span = t_last - g17.astrometry.front().epoch_yr;
    // "Existing quality" continuation: the adaptive-optics-era sampling and
    // median errors of the same data (epochs ≥ 2002.0)
    std::vector<double> sig_ao;
    int n_ao = 0;
    for (const auto& a : g17.astrometry)
        if (a.epoch_yr >= 2002.0) {
            ++n_ao;
            sig_ao.push_back(a.sig_ra_mas);
            sig_ao.push_back(a.sig_dec_mas);
        }
    std::vector<double> sig_rv;
    for (const auto& v : g17.rv) sig_rv.push_back(v.sig_vr_kms);
    const double naco_rate = n_ao / (g17.astrometry.back().epoch_yr - 2002.0);
    const double naco_sig = median(sig_ao);
    const double rvx_rate = g17.rv.size() / (g17.rv.back().epoch_yr - g17.rv.front().epoch_yr);
    const double rvx_sig = median(sig_rv);
    // GRAVITY-era assumptions (GRAVITY 2020; see header)
    const double grav_rate = 18.0, grav_sig_uas = 65.0, grav_rv_rate = 5.0, grav_rv_sig = 12.0;

    const double P_orbit = fit.at("period_yr").get<double>();
    const double T_max = 10.0 * P_orbit;   // up to ten more orbits
    // Plummer mass enclosed within S2's apocentre per unit total mass
    const double r_apo_mas = truth[2] * (1.0 + truth[3]);
    const double k_enc = std::pow(r_apo_mas, 3) /
                         std::pow(r_apo_mas * r_apo_mas + kPlummerMas * kPlummerMas, 1.5);

    // ---- the union of all epochs ---------------------------------------------
    Epochs ep;
    for (const auto& a : g17.astrometry) ep.ast.push_back(a.epoch_yr);
    for (const auto& v : g17.rv) ep.rv.push_back(v.epoch_yr);
    const size_t n_g17_ast = ep.ast.size(), n_g17_rv = ep.rv.size();
    auto grid = [&](double rate) {
        std::vector<double> t;
        for (int j = 0;; ++j) {
            const double tj = kT0 + (j + 0.5) / rate;
            if (tj > kT0 + T_max) break;
            t.push_back(tj);
        }
        return t;
    };
    const auto t_grav = grid(grav_rate), t_naco = grid(naco_rate);
    const auto t_grv_rv = grid(grav_rv_rate), t_nac_rv = grid(rvx_rate);
    const size_t off_grav = ep.ast.size();
    ep.ast.insert(ep.ast.end(), t_grav.begin(), t_grav.end());
    const size_t off_naco = ep.ast.size();
    ep.ast.insert(ep.ast.end(), t_naco.begin(), t_naco.end());
    const size_t off_grv_rv = ep.rv.size();
    ep.rv.insert(ep.rv.end(), t_grv_rv.begin(), t_grv_rv.end());
    const size_t off_nac_rv = ep.rv.size();
    ep.rv.insert(ep.rv.end(), t_nac_rv.begin(), t_nac_rv.end());
    // rows of the prediction vector: (RA, Dec) per astrometric epoch, then RV
    const size_t n_ast = ep.ast.size();
    auto RA = [](size_t i) { return 2 * i; };
    auto DEC = [](size_t i) { return 2 * i + 1; };
    auto RV = [n_ast](size_t j) { return 2 * n_ast + j; };
    std::printf("[forecast] %zu astrometric + %zu RV epochs, %.1f yr beyond %.1f\n",
                ep.ast.size(), ep.rv.size(), T_max, kT0);

    // ---- data sets as row lists -------------------------------------------------
    auto rows_g17 = [&] {
        std::vector<Row> r;
        for (size_t i = 0; i < n_g17_ast; ++i) {
            r.push_back({RA(i), g17.astrometry[i].sig_ra_mas});
            r.push_back({DEC(i), g17.astrometry[i].sig_dec_mas});
        }
        for (size_t j = 0; j < n_g17_rv; ++j) r.push_back({RV(j), g17.rv[j].sig_vr_kms});
        return r;
    };
    // GRAVITY-era (σ_ast in µas) or existing-quality continuation for T years
    auto rows_future = [&](bool gravity, double T, double sig_uas) {
        std::vector<Row> r = rows_g17();
        const size_t a0 = gravity ? off_grav : off_naco;
        const size_t na = gravity ? t_grav.size() : t_naco.size();
        const double sa = gravity ? sig_uas / 1000.0 : naco_sig;
        for (size_t i = a0; i < a0 + na; ++i)
            if (ep.ast[i] < kT0 + T) {
                r.push_back({RA(i), sa});
                r.push_back({DEC(i), sa});
            }
        const size_t v0 = gravity ? off_grv_rv : off_nac_rv;
        const size_t nv = gravity ? t_grv_rv.size() : t_nac_rv.size();
        const double sv = gravity ? grav_rv_sig : rvx_sig;
        for (size_t j = v0; j < v0 + nv; ++j)
            if (ep.rv[j] < kT0 + T) r.push_back({RV(j), sv});
        return r;
    };

    // diagnostic: is the orbit's response to ε_q linear? (--eps-linearity)
    if (argc > 4 && std::string(argv[4]) == "--eps-linearity") {
        Epochs probe;
        for (double t = 1995.0; t < 2050.0; t += 5.0) probe.ast.push_back(t);
        probe.rv = {2002.4, 2018.4, 2034.4};
        Theta th0 = truth;
        const auto y0 = predict(probe, th0);
        for (double h : {125.0, 500.0, 2000.0, 8000.0, 32000.0}) {
            Theta tp = truth, tm = truth;
            tp[kEps] = h;
            tm[kEps] = -h;
            const auto yp = predict(probe, tp), ym = predict(probe, tm);
            std::printf("[lin] h=%7.0f odd/h (RA, mas per unit eps):", h);
            for (size_t i = 0; i < probe.ast.size(); ++i)
                std::printf(" %+.4e", (yp[2 * i] - ym[2 * i]) / (2 * h));
            std::printf("\n[lin]           even (RA, mas):            ");
            for (size_t i = 0; i < probe.ast.size(); ++i)
                std::printf(" %+.2e", yp[2 * i] + ym[2 * i] - 2 * y0[2 * i]);
            std::printf("\n");
        }
        return 0;
    }

    // ---- Jacobians -----------------------------------------------------------------
    Theta thA = truth;   // a* = 0: the thesis' no-hair configuration
    const auto JA = jacobian(ep, thA, cols_with({kEps, kExt}));
    std::printf("[forecast] Jacobian (a* = 0) done, %.1f s\n", elapsed());
    Theta thB = truth;   // a* = 0.94: frame dragging on
    thB[kSpin] = 0.94;
    const auto JB = jacobian(ep, thB, cols_with({kSpin, kExt}));
    std::printf("[forecast] Jacobian (a* = 0.94) done, %.1f s\n", elapsed());

    json J;
    J["assumptions"] = {
        {"truth", "thesis S2 fit (phase5/s2_fit.json); frame nuisances at prior centres"},
        {"plummer_scale_mas", kPlummerMas},
        {"t0_future", kT0},
        {"gravity", {{"ast_per_yr", grav_rate}, {"sig_ast_uas", grav_sig_uas},
                     {"rv_per_yr", grav_rv_rate}, {"sig_rv_kms", grav_rv_sig}}},
        {"existing_quality", {{"ast_per_yr", naco_rate}, {"sig_ast_mas", naco_sig},
                              {"rv_per_yr", rvx_rate}, {"sig_rv_kms", rvx_sig}}},
        {"g17", {{"n_ast", n_g17_ast}, {"n_rv", n_g17_rv}, {"span_yr", span},
                 {"t_last", t_last}}},
        {"period_yr", P_orbit},
        {"apocentre_mas", r_apo_mas},
        {"enclosed_per_total", k_enc},
        {"spin_axis", "line of sight (model z axis)"}};

    // ---- 0. numerical check: the forecasts do not depend on the
    //         finite-difference steps (×1/4 and ×2 the defaults) -----------------
    {
        auto key_sigmas = [&](const std::vector<std::array<double, NP>>& Ja,
                              const std::vector<std::array<double, NP>>& Jb) {
            const auto c14 = cols_with({kEps}), c15 = cols_with({kEps, kExt}),
                       cs = cols_with({kSpin});
            return std::array<double, 4>{
                invert(fisher(Ja, rows_g17(), c14), c14).sigma(kEps),
                invert(fisher(Ja, rows_future(true, 32.0, grav_sig_uas), c14), c14).sigma(kEps),
                invert(fisher(Ja, rows_future(true, 16.0, grav_sig_uas), c15), c15).sigma(kExt),
                invert(fisher(Jb, rows_future(true, 64.0, grav_sig_uas), cs), cs).sigma(kSpin)};
        };
        const auto ref = key_sigmas(JA, JB);
        double worst = 0.0;
        json st = json::array();
        for (double sc : {0.25, 2.0}) {
            const auto Ja = jacobian(ep, thA, cols_with({kEps, kExt}), sc);
            const auto Jb = jacobian(ep, thB, cols_with({kSpin}), sc);
            const auto v = key_sigmas(Ja, Jb);
            json row = {{"step_scale", sc}};
            const char* nm[4] = {"sigma_eps_g17", "sigma_eps_T32", "sigma_fext_T16",
                                 "sigma_astar_T64"};
            for (int q = 0; q < 4; ++q) {
                row[nm[q]] = v[q] / ref[q] - 1.0;
                worst = std::max(worst, std::abs(v[q] / ref[q] - 1.0));
            }
            st.push_back(row);
        }
        // per column: which derivative is step-sensitive?
        json cols_scan = json::array();
        for (int k : cols_with({kEps, kExt})) {
            json cs = {{"param", kNames[k]}};
            for (double sc : {0.25, 2.0}) {
                auto Ja = JA;
                const auto Jk = jacobian(ep, thA, {k}, sc);
                for (size_t i = 0; i < Ja.size(); ++i) Ja[i][k] = Jk[i][k];
                const auto v = key_sigmas(Ja, JB);
                cs[sc < 1.0 ? "x0.25" : "x2"] = {v[1] / ref[1] - 1.0, v[2] / ref[2] - 1.0};
            }
            std::printf("[forecast]   column %-10s: sigma(eps,T32) %+.4f / %+.4f, "
                        "sigma(f,T16) %+.4f / %+.4f  (steps x0.25 / x2)\n",
                        kNames[k], cs["x0.25"][0].get<double>(), cs["x2"][0].get<double>(),
                        cs["x0.25"][1].get<double>(), cs["x2"][1].get<double>());
            cols_scan.push_back(cs);
        }
        J["step_check"] = {{"cases", st}, {"max_rel_change", worst}, {"columns", cols_scan}};
        std::printf("[forecast] step-size check: forecasts change by at most %.2g%%\n",
                    100 * worst);
    }

    // ---- 1. validation against the thesis -----------------------------------------
    {
        const auto r = rows_g17();
        const Forecast f14 = invert(fisher(JA, r, cols_with({kEps})), cols_with({kEps}));
        const Forecast f13 = invert(fisher(JA, r, cols_with({})), cols_with({}));
        J["validation"] = {{"sigma_eps_fisher", f14.sigma(kEps)},
                           {"sigma_eps_thesis", sigma_eps_thesis},
                           {"sigma_M_fisher", f13.sigma(0)},
                           {"sigma_M_thesis", fit.at("sigma_M_msun").get<double>()},
                           {"sigma_R0_fisher", f13.sigma(1)},
                           {"sigma_R0_thesis", fit.at("sigma_R0_pc").get<double>()}};
        std::printf("[forecast] G17 Fisher: sigma(eps_q) = %.4g (thesis profile fit %.4g), "
                    "sigma(M) = %.4g (thesis %.4g)\n",
                    f14.sigma(kEps), sigma_eps_thesis, f13.sigma(0),
                    fit.at("sigma_M_msun").get<double>());
    }

    // ---- 2. sigma(eps_q) versus years -------------------------------------------------
    const std::vector<double> Tgrid = {0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32,
                                       40, 48, 56, 64, 80, 96, 112, 128, 144, 160};
    {
        const auto c = cols_with({kEps});
        const double s0 = invert(fisher(JA, rows_g17(), c), c).sigma(kEps);
        for (bool gravity : {false, true}) {
            json curve = json::array();
            for (double T : Tgrid) {
                if (T > T_max + 1e-9) continue;
                const auto r = rows_future(gravity, T, grav_sig_uas);
                const double s = invert(fisher(JA, r, c), c).sigma(kEps);
                curve.push_back({{"T_yr", T}, {"n_orbits", T / P_orbit}, {"n_rows", r.size()},
                                 {"sigma_eps", s}});
            }
            J["eps_vs_years"][gravity ? "gravity" : "existing_quality"] = curve;
        }
        // the thesis' "ten more orbits" check, at exactly ten orbits
        const auto r10 = rows_future(false, T_max, grav_sig_uas);
        const double s10 = invert(fisher(JA, r10, c), c).sigma(kEps);
        J["ten_orbits"] = {{"sigma_eps_fisher", s10},
                           {"sigma_eps_thesis_sqrtN", forecast10_thesis},
                           {"sigma_eps_g17", s0},
                           {"gain_fisher", s0 / s10},
                           {"gain_sqrtN", s0 / forecast10_thesis}};
        std::printf("[forecast] ten more orbits (existing quality): sigma(eps_q) %.4g -> %.4g "
                    "(x%.1f; thesis 1/sqrt(N): x%.1f)\n",
                    s0, s10, s0 / s10, s0 / forecast10_thesis);
    }

    // ---- 3. Lense–Thirring detectability (a* = 0.94, Kerr), with the
    //         extended mass fixed at zero and marginalised -------------------------
    {
        json lt = json::array();
        for (bool marg : {false, true}) {
            const auto c = marg ? cols_with({kSpin, kExt}) : cols_with({kSpin});
            for (double sig_uas : {65.0, 30.0, 10.0}) {
                json curve = json::array();
                double T3 = std::nan(""), T5 = std::nan(""), prevT = 0.0, prevS = 0.0;
                for (double T : Tgrid) {
                    if (T > T_max + 1e-9) continue;
                    const double s =
                        invert(fisher(JB, rows_future(true, T, sig_uas), c), c).sigma(kSpin);
                    const double S = 0.94 / s;
                    curve.push_back({{"T_yr", T}, {"sigma_astar", s}, {"significance", S}});
                    for (auto [thr, slot] : {std::pair{3.0, &T3}, std::pair{5.0, &T5}})
                        if (std::isnan(*slot) && S >= thr && T > 0.0)
                            *slot = prevT + (thr - prevS) / (S - prevS) * (T - prevT);
                    prevT = T;
                    prevS = S;
                }
                lt.push_back({{"sig_ast_uas", sig_uas}, {"ext_marginalised", marg},
                              {"curve", curve},
                              {"T_3sigma_yr", std::isnan(T3) ? json(nullptr) : json(T3)},
                              {"T_5sigma_yr", std::isnan(T5) ? json(nullptr) : json(T5)}});
                std::printf("[forecast] LT, %2.0f uas%s: a* = 0.94 at 3 sigma after %s yr, "
                            "5 sigma after %s yr\n",
                            sig_uas, marg ? " (ext. mass marginalised)" : "",
                            std::isnan(T3) ? ">160" : std::to_string(T3).c_str(),
                            std::isnan(T5) ? ">160" : std::to_string(T5).c_str());
            }
        }
        J["lense_thirring"] = lt;
    }

    // ---- 4. the extended-mass / quadrupole degeneracy -----------------------------------
    {
        json dg = json::array();
        for (double T : {0.0, 8.0, 16.0, 32.0}) {
            const auto r = T == 0.0 ? rows_g17() : rows_future(true, T, grav_sig_uas);
            const auto c14 = cols_with({kEps}), c15 = cols_with({kEps, kExt}),
                       c14e = cols_with({kExt});
            const Forecast f14 = invert(fisher(JA, r, c14), c14);
            const Forecast f15 = invert(fisher(JA, r, c15), c15);
            const Forecast fe = invert(fisher(JA, r, c14e), c14e);
            const int ie = 13, ix = 14;   // positions of ε_q, f_ext in c15
            dg.push_back({{"T_yr", T},
                          {"sigma_eps_fixed_ext", f14.sigma(kEps)},
                          {"sigma_eps_marg_ext", f15.sigma(kEps)},
                          {"sigma_fext_fixed_eps", fe.sigma(kExt)},
                          {"sigma_fext_marg_eps", f15.sigma(kExt)},
                          {"sigma_fenc_fixed_eps", k_enc * fe.sigma(kExt)},
                          {"sigma_fenc_marg_eps", k_enc * f15.sigma(kExt)},
                          {"corr_eps_fext", f15.corr(kEps, kExt)},
                          {"cov2", {{f15.C(ie, ie), f15.C(ie, ix)}, {f15.C(ix, ie), f15.C(ix, ix)}}}});
            std::printf("[forecast] T=%4.0f: sigma(eps) %.3g (f_ext fixed) / %.3g (marg.), "
                        "sigma(f_enc) %.3g%% (eps fixed) / %.3g%% (marg.), corr %.3f\n",
                        T, f14.sigma(kEps), f15.sigma(kEps), 100 * k_enc * fe.sigma(kExt),
                        100 * k_enc * f15.sigma(kExt), f15.corr(kEps, kExt));
        }
        J["degeneracy"] = dg;
    }

    // ---- 5. bias from an unmodelled extended mass ----------------------------------------
    // Noise-free data made WITH a Plummer extended mass, fitted WITHOUT it by
    // the thesis' own fitter (OrbitFitter, Ceres): (a) the 13-parameter Kerr
    // fit → biases in M and R0 and the χ² left over; (b) ε_q from the thesis'
    // Phase-5 profile — 13-parameter fits at fixed ε_q, the vertex of the
    // parabola through three points, re-centred until the vertex lies inside
    // them. (Ceres' own numerical derivative cannot move ε_q off 0 — its
    // default step there is 1e-6, far below what the orbit can feel — which is
    // why the thesis profiles ε_q.) The linear response F⁻¹Jᵀ(Δd/σ²) is kept
    // alongside as a cross-check: it holds while the biases are ≲ 1σ, and
    // over-predicts the large ones.
    if (!fisher_only) {
        const auto y0 = predict(ep, thA);
        const S2Parameters start = S2Parameters::from_array(thA.data());
        json bias = json::array();
        for (double f_enc : {5e-4, 1e-3, 2e-3}) {
            const double f = f_enc / k_enc;
            Theta tf = thA;
            tf[kExt] = f;
            const auto yf = predict(ep, tf);
            std::vector<double> dy(y0.size());
            for (size_t i = 0; i < dy.size(); ++i) dy[i] = yf[i] - y0[i];
            for (double T : {0.0, 8.0, 16.0, 32.0}) {
                const auto r = T == 0.0 ? rows_g17() : rows_future(true, T, grav_sig_uas);
                const auto c13 = cols_with({}), c14 = cols_with({kEps});
                const Bias b13 = linear_bias(JA, r, c13, dy);
                const Bias b14 = linear_bias(JA, r, c14, dy);
                const Forecast f13 = invert(fisher(JA, r, c13), c13);
                const Forecast f14 = invert(fisher(JA, r, c14), c14);
                const double s_eps = f14.sigma(kEps);

                // the same noise-free data as an S2Dataset for the fitter
                S2Dataset d;
                for (size_t i = 0; i < n_g17_ast; ++i)
                    d.astrometry.push_back({g17.astrometry[i].epoch_yr, yf[RA(i)], yf[DEC(i)],
                                            g17.astrometry[i].sig_ra_mas,
                                            g17.astrometry[i].sig_dec_mas});
                for (size_t i = off_grav; i < off_grav + t_grav.size(); ++i)
                    if (ep.ast[i] < kT0 + T)
                        d.astrometry.push_back({ep.ast[i], yf[RA(i)], yf[DEC(i)],
                                                grav_sig_uas / 1000.0, grav_sig_uas / 1000.0});
                for (size_t j = 0; j < n_g17_rv; ++j)
                    d.rv.push_back({g17.rv[j].epoch_yr, yf[RV(j)], g17.rv[j].sig_vr_kms});
                for (size_t j = off_grv_rv; j < off_grv_rv + t_grv_rv.size(); ++j)
                    if (ep.rv[j] < kT0 + T) d.rv.push_back({ep.rv[j], yf[RV(j)], grav_rv_sig});

                const FitResult k13 = OrbitFitter(d, 0.0, 0.0).fit(start);
                // ε_q profile, starting from the linear prediction
                double c = b14.dtheta[13], v = c, curv = 0.0, chi_min = 0.0;
                int iters = 0;
                for (; iters < 6; ++iters) {
                    std::array<double, 3> chi{};
                    for (int q = 0; q < 3; ++q)
                        chi[q] = OrbitFitter(d, 0.0, c + (q - 1) * s_eps).fit(start).chi2;
                    curv = chi[2] - 2.0 * chi[1] + chi[0];
                    v = c - s_eps * (chi[2] - chi[0]) / (2.0 * curv);
                    chi_min = chi[1] - (chi[2] - chi[0]) * (chi[2] - chi[0]) / (8.0 * curv);
                    if (curv > 0.0 && std::abs(v - c) < 0.25 * s_eps) break;
                    c = v;
                }
                const double s_prof = s_eps * std::sqrt(2.0 / curv);
                bias.push_back({{"f_enc", f_enc},
                                {"f_ext", f},
                                {"T_yr", T},
                                {"chi2_signal", b13.chi2_before},
                                {"dM_msun", k13.params.M_msun - thA[0]},
                                {"dM_over_sigma", (k13.params.M_msun - thA[0]) / f13.sigma(0)},
                                {"dR0_pc", k13.params.R0_pc - thA[1]},
                                {"dR0_over_sigma", (k13.params.R0_pc - thA[1]) / f13.sigma(1)},
                                {"chi2_left_kerr", k13.chi2},
                                {"deps", v},
                                {"sigma_eps_profile", s_prof},
                                {"sigma_eps", s_eps},
                                // significance as an analyst would quote it:
                                // the biased fit's own profile uncertainty
                                {"deps_over_sigma", v / s_prof},
                                {"deps_over_sigma_fisher", v / s_eps},
                                {"chi2_left_nohair", chi_min},
                                {"profile_iterations", iters + 1},
                                {"linear", {{"dM_msun", b13.dtheta[0]},
                                            {"dR0_pc", b13.dtheta[1]},
                                            {"deps", b14.dtheta[13]},
                                            {"chi2_left_kerr", b13.chi2_after}}}});
                std::printf("[forecast] f_enc=%.2f%% T=%4.0f: dM=%+.3g (%.2f sigma; linear %+.3g), "
                            "deps=%+.4g (%.2f sigma; linear %+.4g; profile sigma %.4g), "
                            "chi2 left %.3g\n",
                            100 * f_enc, T, k13.params.M_msun - thA[0],
                            (k13.params.M_msun - thA[0]) / f13.sigma(0), b13.dtheta[0], v,
                            v / s_prof, b14.dtheta[13], s_prof, k13.chi2);
            }
        }
        J["bias"] = bias;
    }

    const auto path = out / "forecasts.json";
    std::ofstream(path) << J.dump(2) << "\n";
    std::printf("forecast_tool: wrote %s (%.1f s)\n", path.string().c_str(), elapsed());
    return 0;
}
