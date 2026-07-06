// Thesis tool (paper/THESIS_PLAN.md §2.1 [CORE]): joint multi-star fit of the
// S-star cluster in a shared potential. One (M, R0) and one reference-frame
// block (x0, y0, vx, vy, vz) are shared by every star; each star carries its
// own six osculating elements. The forward model per star is the SAME full-GR
// timelike-geodesic S2Model used in Phases 4-5 — nothing is re-derived.
//
//   fit_cluster <data_dir> <out_dir> [star ...]     (default: S2 S1 S8 S12 S13)
//
// Outputs (into <out_dir>):
//   cluster_fit.json    joint fit, per-star single fits, cumulative subsets
//   <star>_track.csv    dense sky tracks at the joint best fit (figure data)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "orbit_fitter/OrbitFitter.hpp"

using namespace sgr;
namespace fs = std::filesystem;

namespace {

constexpr int kGlob = 7; // M, R0, x0, y0, vx, vy, vz
constexpr int kElem = 6; // a_mas, e, incl, Omega, omega, t_peri
// frame priors (same as OrbitFitter: Gillessen 2017 §2.3 / Plewa 2015)
constexpr double kSigX0 = 0.2, kSigV0 = 0.1, kSigVz = 5.0;

struct StarStart {
    const char* name;
    double a_mas, e, incl, Omega, omega, t_peri;
};

// Starting guesses from Gillessen et al. (2017) Table 3. These only seed the
// optimiser — every fitted value reported by this tool comes from our own
// chi^2 minimisation against the Table 5 data. A 180-degree probe below
// resolves any orientation-convention mismatch before fitting.
constexpr StarStart kStarts[] = {
    {"S2", 125.5, 0.8839, 134.18, 226.94, 65.51, 2002.33},
    {"S1", 595.0, 0.5560, 119.14, 342.04, 122.30, 2001.80},
    {"S8", 404.7, 0.8031, 74.37, 315.43, 346.70, 1983.64},
    {"S12", 298.7, 0.8883, 33.56, 230.10, 317.90, 1995.59},
    {"S13", 264.1, 0.4250, 24.70, 74.50, 245.20, 2004.86},
};

struct Star {
    std::string name;
    S2Dataset data;
    std::array<double, kElem> elem{};
};

S2Parameters assemble(const double* g, const double* s) {
    S2Parameters p;
    p.M_msun = g[0];
    p.R0_pc = g[1];
    p.x0_mas = g[2];
    p.y0_mas = g[3];
    p.vx_masyr = g[4];
    p.vy_masyr = g[5];
    p.vz_kms = g[6];
    p.a_mas = s[0];
    p.e = s[1];
    p.incl_deg = s[2];
    p.Omega_deg = s[3];
    p.omega_deg = s[4];
    p.t_peri_yr = s[5];
    return p;
}

// Residuals of one star given (global, elements) blocks.
struct StarCost {
    const S2Dataset* data;

    bool operator()(double const* const* blocks, double* residuals) const {
        const S2Parameters p = assemble(blocks[0], blocks[1]);
        const int n_ast = static_cast<int>(data->astrometry.size());
        const int n_rv = static_cast<int>(data->rv.size());
        const int n_res = 2 * n_ast + n_rv;

        double t_min = 1e9, t_max = -1e9;
        for (const auto& a : data->astrometry) {
            t_min = std::min(t_min, a.epoch_yr);
            t_max = std::max(t_max, a.epoch_yr);
        }
        for (const auto& v : data->rv) {
            t_min = std::min(t_min, v.epoch_yr);
            t_max = std::max(t_max, v.epoch_yr);
        }
        S2Model model(0.0, 0.0);
        if (!model.integrate(p, t_min - 0.2, t_max + 0.2)) {
            for (int k = 0; k < n_res; ++k) residuals[k] = 1e4;
            return true;
        }
        std::vector<double> ae, re;
        for (const auto& a : data->astrometry) ae.push_back(a.epoch_yr);
        for (const auto& v : data->rv) re.push_back(v.epoch_yr);
        const S2Observables obs = model.evaluate(p, ae, re);
        if (!obs.valid) {
            for (int k = 0; k < n_res; ++k) residuals[k] = 1e4;
            return true;
        }
        int k = 0;
        for (int i = 0; i < n_ast; ++i) {
            const auto& a = data->astrometry[i];
            residuals[k++] = (obs.ra_mas[i] - a.ra_mas) / a.sig_ra_mas;
            residuals[k++] = (obs.dec_mas[i] - a.dec_mas) / a.sig_dec_mas;
        }
        for (int j = 0; j < n_rv; ++j) {
            const auto& v = data->rv[j];
            residuals[k++] = (obs.vr_kms[j] - v.vr_kms) / v.sig_vr_kms;
        }
        return true;
    }
};

// Reference-frame priors on the shared block (applied once, not per star).
struct FramePrior {
    bool operator()(double const* const* blocks, double* r) const {
        const double* g = blocks[0];
        r[0] = g[2] / kSigX0;
        r[1] = g[3] / kSigX0;
        r[2] = g[4] / kSigV0;
        r[3] = g[5] / kSigV0;
        r[4] = g[6] / kSigVz;
        return true;
    }
};

double star_chi2(const Star& st, const double* g, const double* s) {
    const int n_res = st.data.n_data();
    std::vector<double> r(n_res);
    const double* blocks[2] = {g, s};
    StarCost{&st.data}(blocks, r.data());
    double c = 0.0;
    for (double v : r) c += v * v;
    return c;
}

struct JointResult {
    std::array<double, kGlob> g{};
    std::vector<std::array<double, kElem>> elem;
    double chi2 = 0.0, chi2_red = 0.0;
    double sigma_M = 0.0, sigma_R0 = 0.0;
    int n_data = 0, dof = 0;
    bool converged = false;
};

JointResult fit_joint(std::vector<Star>& stars, bool with_covariance) {
    JointResult out;
    out.g = {4.35e6, 8330.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    out.elem.resize(stars.size());
    for (size_t k = 0; k < stars.size(); ++k) out.elem[k] = stars[k].elem;

    ceres::Problem problem;
    for (size_t k = 0; k < stars.size(); ++k) {
        auto* cost =
            new ceres::DynamicNumericDiffCostFunction<StarCost, ceres::CENTRAL>(
                new StarCost{&stars[k].data});
        cost->AddParameterBlock(kGlob);
        cost->AddParameterBlock(kElem);
        cost->SetNumResiduals(stars[k].data.n_data());
        problem.AddResidualBlock(cost, nullptr, out.g.data(),
                                 out.elem[k].data());
        problem.SetParameterLowerBound(out.elem[k].data(), 1, 0.05); // e
        problem.SetParameterUpperBound(out.elem[k].data(), 1, 0.99);
    }
    auto* prior =
        new ceres::DynamicNumericDiffCostFunction<FramePrior, ceres::CENTRAL>(
            new FramePrior{});
    prior->AddParameterBlock(kGlob);
    prior->SetNumResiduals(5);
    problem.AddResidualBlock(prior, nullptr, out.g.data());
    problem.SetParameterLowerBound(out.g.data(), 0, 1e6);
    problem.SetParameterUpperBound(out.g.data(), 0, 2e7);
    problem.SetParameterLowerBound(out.g.data(), 1, 4000.0);
    problem.SetParameterUpperBound(out.g.data(), 1, 13000.0);

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = 300;
    opts.function_tolerance = 1e-12;
    opts.num_threads = 8;
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    out.converged = summary.termination_type == ceres::CONVERGENCE ||
                    summary.termination_type == ceres::USER_SUCCESS;
    out.chi2 = 2.0 * summary.final_cost;
    for (const auto& s : stars) out.n_data += s.data.n_data();
    const int n_par = kGlob + kElem * static_cast<int>(stars.size());
    out.dof = out.n_data - n_par;
    out.chi2_red = out.chi2 / out.dof;

    if (with_covariance) {
        // Explicit central-difference Jacobian over the concatenated vector
        // (same recipe as OrbitFitter::covariance; the Ceres SVD covariance
        // truncates the M–R0 direction).
        const int N = n_par;
        int n_res_tot = 5; // frame priors
        for (const auto& s : stars) n_res_tot += s.data.n_data();
        std::vector<double> x(N);
        std::copy(out.g.begin(), out.g.end(), x.begin());
        for (size_t k = 0; k < stars.size(); ++k)
            std::copy(out.elem[k].begin(), out.elem[k].end(),
                      x.begin() + kGlob + kElem * k);
        std::vector<double> step(N);
        const double gstep[kGlob] = {2e3, 2.0, 2e-3, 2e-3, 1e-3, 1e-3, 0.05};
        const double estep[kElem] = {5e-3, 2e-5, 2e-3, 2e-3, 2e-3, 1e-4};
        for (int i = 0; i < kGlob; ++i) step[i] = gstep[i];
        for (size_t k = 0; k < stars.size(); ++k)
            for (int i = 0; i < kElem; ++i)
                step[kGlob + kElem * k + i] = estep[i];

        auto residuals_at = [&](const std::vector<double>& xv,
                                std::vector<double>& r) {
            r.clear();
            r.reserve(n_res_tot);
            for (size_t k = 0; k < stars.size(); ++k) {
                std::vector<double> rk(stars[k].data.n_data());
                const double* blocks[2] = {xv.data(),
                                           xv.data() + kGlob + kElem * k};
                StarCost{&stars[k].data}(blocks, rk.data());
                r.insert(r.end(), rk.begin(), rk.end());
            }
            r.push_back(xv[2] / kSigX0);
            r.push_back(xv[3] / kSigX0);
            r.push_back(xv[4] / kSigV0);
            r.push_back(xv[5] / kSigV0);
            r.push_back(xv[6] / kSigVz);
        };
        std::vector<std::vector<double>> J(N);
        for (int k = 0; k < N; ++k) {
            auto xp = x, xm = x;
            xp[k] += step[k];
            xm[k] -= step[k];
            std::vector<double> rp, rm;
            residuals_at(xp, rp);
            residuals_at(xm, rm);
            J[k].resize(rp.size());
            for (size_t i = 0; i < rp.size(); ++i)
                J[k][i] = (rp[i] - rm[i]) / (2.0 * step[k]);
        }
        // invert JtJ by Gauss-Jordan with partial pivoting
        std::vector<double> A(N * N, 0.0), C(N * N, 0.0);
        for (int a = 0; a < N; ++a)
            for (int b = 0; b < N; ++b) {
                double s = 0.0;
                for (size_t i = 0; i < J[a].size(); ++i) s += J[a][i] * J[b][i];
                A[a * N + b] = s;
            }
        for (int i = 0; i < N; ++i) C[i * N + i] = 1.0;
        for (int col = 0; col < N; ++col) {
            int piv = col;
            for (int r = col + 1; r < N; ++r)
                if (std::abs(A[r * N + col]) > std::abs(A[piv * N + col]))
                    piv = r;
            for (int c = 0; c < N; ++c) {
                std::swap(A[col * N + c], A[piv * N + c]);
                std::swap(C[col * N + c], C[piv * N + c]);
            }
            const double d = A[col * N + col];
            for (int c = 0; c < N; ++c) {
                A[col * N + c] /= d;
                C[col * N + c] /= d;
            }
            for (int r = 0; r < N; ++r) {
                if (r == col) continue;
                const double m = A[r * N + col];
                for (int c = 0; c < N; ++c) {
                    A[r * N + c] -= m * A[col * N + c];
                    C[r * N + c] -= m * C[col * N + c];
                }
            }
        }
        out.sigma_M = std::sqrt(std::max(0.0, C[0]));
        out.sigma_R0 = std::sqrt(std::max(0.0, C[N + 1]));
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <data_dir> <out_dir> [star ...]\n",
                     argv[0]);
        return 2;
    }
    const fs::path data_dir = argv[1];
    const fs::path out_dir = argv[2];
    fs::create_directories(out_dir);
    std::vector<std::string> names;
    for (int i = 3; i < argc; ++i) names.emplace_back(argv[i]);
    if (names.empty()) names = {"S2", "S1", "S8", "S12", "S13"};

    // ---- load stars + resolve the 180-degree orientation ambiguity ---------
    std::vector<Star> stars;
    const std::array<double, kGlob> g0{4.35e6, 8330.0, 0, 0, 0, 0, 0};
    for (const auto& n : names) {
        const StarStart* st = nullptr;
        for (const auto& s : kStarts)
            if (n == s.name) st = &s;
        if (!st) {
            std::fprintf(stderr, "no starting elements for %s\n", n.c_str());
            return 2;
        }
        Star star;
        star.name = n;
        std::string base = n;
        std::transform(base.begin(), base.end(), base.begin(), ::tolower);
        star.data = S2Dataset::load(
            (data_dir / (base + "_astrometry.csv")).string(),
            (data_dir / (base + "_radial_velocity.csv")).string());
        star.elem = {st->a_mas, st->e, st->incl, st->Omega, st->omega,
                     st->t_peri};
        // starting-χ² probe over the (Ω, ω) ± 180° convention flips
        double best = 1e300;
        std::array<double, kElem> best_e = star.elem;
        for (double dO : {0.0, 180.0})
            for (double dw : {0.0, 180.0}) {
                auto e = star.elem;
                e[3] = std::fmod(e[3] + dO, 360.0);
                e[4] = std::fmod(e[4] + dw, 360.0);
                const double c = star_chi2(star, g0.data(), e.data());
                if (c < best) {
                    best = c;
                    best_e = e;
                }
            }
        star.elem = best_e;
        spdlog::info("{}: {} ast + {} rv, start chi2 = {:.0f}", n,
                     star.data.astrometry.size(), star.data.rv.size(), best);
        stars.push_back(std::move(star));
    }

    // ---- per-star single fits (13-parameter, own frame) ---------------------
    nlohmann::json per_star = nlohmann::json::array();
    for (auto& st : stars) {
        S2Parameters p0 = assemble(g0.data(), st.elem.data());
        OrbitFitter f(st.data);
        const FitResult r = f.fit(p0);
        spdlog::info("single {}: conv={} M={:.3f}e6 +- {:.3f}e6 chi2red={:.3f}",
                     st.name, r.converged, r.params.M_msun / 1e6,
                     r.sigma[0] / 1e6, r.chi2_red);
        per_star.push_back({{"star", st.name},
                            {"converged", r.converged},
                            {"M_msun", r.params.M_msun},
                            {"sigma_M_msun", r.sigma[0]},
                            {"R0_pc", r.params.R0_pc},
                            {"sigma_R0_pc", r.sigma[1]},
                            {"chi2_red", r.chi2_red},
                            {"n_ast", st.data.astrometry.size()},
                            {"n_rv", st.data.rv.size()},
                            {"a_mas", r.params.a_mas},
                            {"e", r.params.e},
                            {"incl_deg", r.params.incl_deg},
                            {"period_yr", r.period_yr}});
        // seed the joint fit with the refined single-star elements
        if (r.converged)
            st.elem = {r.params.a_mas, r.params.e, r.params.incl_deg,
                       r.params.Omega_deg, r.params.omega_deg,
                       r.params.t_peri_yr};
    }

    // ---- cumulative subsets: M as a function of the number of stars --------
    nlohmann::json subsets = nlohmann::json::array();
    for (size_t n = 1; n <= stars.size(); ++n) {
        std::vector<Star> sub(stars.begin(), stars.begin() + n);
        JointResult r = fit_joint(sub, true);
        std::string list;
        for (const auto& s : sub) list += s.name + " ";
        spdlog::info("joint n={} [{}]: M={:.4f}e6 +- {:.4f}e6, R0={:.0f} +- "
                     "{:.0f}, chi2red={:.3f}",
                     n, list, r.g[0] / 1e6, r.sigma_M / 1e6, r.g[1],
                     r.sigma_R0, r.chi2_red);
        subsets.push_back({{"n_stars", n},
                           {"M_msun", r.g[0]},
                           {"sigma_M_msun", r.sigma_M},
                           {"R0_pc", r.g[1]},
                           {"sigma_R0_pc", r.sigma_R0},
                           {"chi2_red", r.chi2_red},
                           {"converged", r.converged}});
        if (n == stars.size()) {
            // ---- final joint fit: report + dense tracks ---------------------
            nlohmann::json j{{"stars", names},
                             {"M_msun", r.g[0]},
                             {"sigma_M_msun", r.sigma_M},
                             {"R0_pc", r.g[1]},
                             {"sigma_R0_pc", r.sigma_R0},
                             {"x0_mas", r.g[2]},
                             {"y0_mas", r.g[3]},
                             {"vx_masyr", r.g[4]},
                             {"vy_masyr", r.g[5]},
                             {"vz_kms", r.g[6]},
                             {"chi2", r.chi2},
                             {"n_data", r.n_data},
                             {"dof", r.dof},
                             {"chi2_red", r.chi2_red},
                             {"converged", r.converged},
                             {"per_star", per_star},
                             {"subsets", subsets}};
            std::ofstream(out_dir / "cluster_fit.json") << j.dump(2) << "\n";
            spdlog::info("wrote cluster_fit.json");
            for (size_t k = 0; k < stars.size(); ++k) {
                const S2Parameters p = assemble(r.g.data(), r.elem[k].data());
                S2Model model(0.0, 0.0);
                if (!model.integrate(p, 1991.0, 2017.5)) continue;
                std::vector<double> ep;
                for (double t = 1992.0; t <= 2017.0; t += 0.05)
                    ep.push_back(t);
                const S2Observables obs = model.evaluate(p, ep, ep);
                std::string base = stars[k].name;
                std::transform(base.begin(), base.end(), base.begin(),
                               ::tolower);
                std::ofstream csv(out_dir / (base + "_track.csv"));
                csv << "epoch_yr,ra_mas,dec_mas,vr_kms\n";
                for (size_t i = 0; i < ep.size(); ++i)
                    csv << ep[i] << "," << obs.ra_mas[i] << ","
                        << obs.dec_mas[i] << "," << obs.vr_kms[i] << "\n";
            }
            spdlog::info("wrote per-star track CSVs");
            return r.converged ? 0 : 1;
        }
    }
    return 1;
}
