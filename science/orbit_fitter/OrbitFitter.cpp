#include "orbit_fitter/OrbitFitter.hpp"

#include <ceres/ceres.h>

#include <cmath>
#include <fstream>
#include <sstream>

#include "sgr_physics/constants.hpp"

namespace sgr {

namespace {

// Priors on the coordinate-system nuisance parameters (Gillessen 2017 §2.3).
constexpr double kSigX0 = 0.2;   // mas
constexpr double kSigV0 = 0.1;   // mas/yr
constexpr double kSigVz = 5.0;   // km/s

struct S2CostFunctor {
    const S2Dataset* data;
    double a_star, eps_q;
    bool eps_q_is_param = false; // 14-parameter (no-hair) mode

    bool operator()(double const* const* params, double* residuals) const {
        const S2Parameters p = S2Parameters::from_array(params[0]);
        const double eq = eps_q_is_param ? params[0][S2Parameters::kNumParams]
                                         : eps_q;
        const int n_ast = static_cast<int>(data->astrometry.size());
        const int n_rv = static_cast<int>(data->rv.size());
        const int n_res = 2 * n_ast + n_rv + 5;

        S2Model model(a_star, eq);
        double t_min = 1e9, t_max = -1e9;
        for (const auto& a : data->astrometry) {
            t_min = std::min(t_min, a.epoch_yr);
            t_max = std::max(t_max, a.epoch_yr);
        }
        for (const auto& v : data->rv) {
            t_min = std::min(t_min, v.epoch_yr);
            t_max = std::max(t_max, v.epoch_yr);
        }

        if (!model.integrate(p, t_min - 0.2, t_max + 0.2)) {
            for (int k = 0; k < n_res; ++k) residuals[k] = 1e4;
            return true; // huge but finite residual: keeps LM in bounds
        }

        std::vector<double> ast_epochs, rv_epochs;
        ast_epochs.reserve(n_ast);
        rv_epochs.reserve(n_rv);
        for (const auto& a : data->astrometry) ast_epochs.push_back(a.epoch_yr);
        for (const auto& v : data->rv) rv_epochs.push_back(v.epoch_yr);
        const S2Observables obs = model.evaluate(p, ast_epochs, rv_epochs);
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
        // coordinate-system priors
        residuals[k++] = p.x0_mas / kSigX0;
        residuals[k++] = p.y0_mas / kSigX0;
        residuals[k++] = p.vx_masyr / kSigV0;
        residuals[k++] = p.vy_masyr / kSigV0;
        residuals[k++] = p.vz_kms / kSigVz;
        return true;
    }
};

} // namespace

S2Dataset S2Dataset::load(const std::string& astrometry_csv,
                          const std::string& rv_csv) {
    S2Dataset d;
    auto parse = [](const std::string& path, auto callback) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot open " + path);
        std::string line;
        std::getline(f, line); // header
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::vector<double> vals;
            std::string tok;
            while (std::getline(ss, tok, ',')) vals.push_back(std::stod(tok));
            callback(vals);
        }
    };
    parse(astrometry_csv, [&](const std::vector<double>& v) {
        d.astrometry.push_back({v[0], v[1], v[2], v[3], v[4]});
    });
    parse(rv_csv, [&](const std::vector<double>& v) {
        d.rv.push_back({v[0], v[1], v[2]});
    });
    return d;
}

double OrbitFitter::chi2(const S2Parameters& p) const {
    const int n_res = data_->n_data() + 5;
    std::vector<double> r(n_res);
    S2CostFunctor f{data_, a_star_, eps_q_};
    const auto arr = p.to_array();
    const double* blocks[1] = {arr.data()};
    f(blocks, r.data());
    double c = 0.0;
    for (double v : r) c += v * v;
    return c;
}

std::vector<double> OrbitFitter::covariance(
    const S2Parameters& best,
    std::array<double, S2Parameters::kNumParams>& sigma) const {
    constexpr int N = S2Parameters::kNumParams;
    const int n_res = data_->n_data() + 5;
    S2CostFunctor f{data_, a_star_, eps_q_};

    // Physically scaled central-difference steps per parameter:
    // M[Msun], R0[pc], a[mas], e, i, Ω, ω [deg], t_peri[yr], x0,y0[mas],
    // vx,vy[mas/yr], vz[km/s]
    const std::array<double, N> step = {2e3, 2.0, 5e-3, 2e-5, 2e-3, 2e-3, 2e-3,
                                        2e-5, 2e-3, 2e-3, 1e-3, 1e-3, 0.05};

    std::vector<std::vector<double>> J(N, std::vector<double>(n_res));
    const auto x0 = best.to_array();
    for (int k = 0; k < N; ++k) {
        auto xp = x0, xm = x0;
        xp[k] += step[k];
        xm[k] -= step[k];
        std::vector<double> rp(n_res), rm(n_res);
        const double* bp[1] = {xp.data()};
        const double* bm[1] = {xm.data()};
        f(bp, rp.data());
        f(bm, rm.data());
        for (int i = 0; i < n_res; ++i)
            J[k][i] = (rp[i] - rm[i]) / (2.0 * step[k]);
    }

    // JᵀJ (N×N) and inverse via Gauss-Jordan with partial pivoting.
    std::vector<double> A(N * N, 0.0);
    for (int a = 0; a < N; ++a)
        for (int b = 0; b < N; ++b) {
            double s = 0.0;
            for (int i = 0; i < n_res; ++i) s += J[a][i] * J[b][i];
            A[a * N + b] = s;
        }
    std::vector<double> C(N * N, 0.0);
    for (int i = 0; i < N; ++i) C[i * N + i] = 1.0;
    for (int col = 0; col < N; ++col) {
        int piv = col;
        for (int r = col + 1; r < N; ++r)
            if (std::abs(A[r * N + col]) > std::abs(A[piv * N + col])) piv = r;
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
    for (int i = 0; i < N; ++i)
        sigma[i] = std::sqrt(std::max(0.0, C[i * N + i]));
    return C;
}

FitResult OrbitFitter::fit_nohair(const S2Parameters& start, double eps_q_start,
                                  double* eps_q_best,
                                  double* eps_q_sigma) const {
    constexpr int N = S2Parameters::kNumParams;
    const int n_res = data_->n_data() + 5;

    auto* functor = new S2CostFunctor{data_, a_star_, 0.0, /*eps_q_is_param=*/true};
    auto* cost = new ceres::DynamicNumericDiffCostFunction<S2CostFunctor,
                                                           ceres::CENTRAL>(
        functor);
    cost->AddParameterBlock(N + 1);
    cost->SetNumResiduals(n_res);

    std::array<double, N + 1> x{};
    const auto s = start.to_array();
    std::copy(s.begin(), s.end(), x.begin());
    x[N] = eps_q_start;

    ceres::Problem problem;
    problem.AddResidualBlock(cost, nullptr, x.data());
    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = 300;
    opts.function_tolerance = 1e-12;
    opts.num_threads = 8;
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    FitResult out;
    out.params = S2Parameters::from_array(x.data());
    out.report = summary.BriefReport();
    out.converged = summary.termination_type == ceres::CONVERGENCE;
    out.n_residuals = data_->n_data();
    out.chi2 = 2.0 * summary.final_cost;
    out.dof = data_->n_data() - (N + 1);
    out.chi2_red = out.chi2 / out.dof;
    *eps_q_best = x[N];

    // 14×14 covariance from an explicit central-difference Jacobian.
    {
        S2CostFunctor f{data_, a_star_, 0.0, true};
        const std::array<double, N + 1> step = {2e3, 2.0, 5e-3, 2e-5, 2e-3,
                                                2e-3, 2e-3, 2e-5, 2e-3, 2e-3,
                                                1e-3, 1e-3, 0.05, 200.0};
        std::vector<std::vector<double>> J(N + 1, std::vector<double>(n_res));
        for (int k = 0; k < N + 1; ++k) {
            auto xp = x, xm = x;
            xp[k] += step[k];
            xm[k] -= step[k];
            std::vector<double> rp(n_res), rm(n_res);
            const double* bp[1] = {xp.data()};
            const double* bm[1] = {xm.data()};
            f(bp, rp.data());
            f(bm, rm.data());
            for (int i = 0; i < n_res; ++i)
                J[k][i] = (rp[i] - rm[i]) / (2.0 * step[k]);
        }
        // Invert JᵀJ (Gauss-Jordan)
        const int M = N + 1;
        std::vector<double> A(M * M, 0.0), C(M * M, 0.0);
        for (int a = 0; a < M; ++a)
            for (int b = 0; b < M; ++b) {
                double sum = 0.0;
                for (int i = 0; i < n_res; ++i) sum += J[a][i] * J[b][i];
                A[a * M + b] = sum;
            }
        for (int i = 0; i < M; ++i) C[i * M + i] = 1.0;
        for (int col = 0; col < M; ++col) {
            int piv = col;
            for (int r = col + 1; r < M; ++r)
                if (std::abs(A[r * M + col]) > std::abs(A[piv * M + col])) piv = r;
            for (int c = 0; c < M; ++c) {
                std::swap(A[col * M + c], A[piv * M + c]);
                std::swap(C[col * M + c], C[piv * M + c]);
            }
            const double d = A[col * M + col];
            for (int c = 0; c < M; ++c) { A[col * M + c] /= d; C[col * M + c] /= d; }
            for (int r = 0; r < M; ++r) {
                if (r == col) continue;
                const double m = A[r * M + col];
                for (int c = 0; c < M; ++c) {
                    A[r * M + c] -= m * A[col * M + c];
                    C[r * M + c] -= m * C[col * M + c];
                }
            }
        }
        for (int i = 0; i < N; ++i)
            out.sigma[i] = std::sqrt(std::max(0.0, C[i * M + i]));
        *eps_q_sigma = std::sqrt(std::max(0.0, C[N * M + N]));
    }
    return out;
}

FitResult OrbitFitter::fit(const S2Parameters& start) const {
    const int n_res = data_->n_data() + 5;

    auto* functor = new S2CostFunctor{data_, a_star_, eps_q_};
    auto* cost = new ceres::DynamicNumericDiffCostFunction<S2CostFunctor,
                                                           ceres::CENTRAL>(
        functor);
    cost->AddParameterBlock(S2Parameters::kNumParams);
    cost->SetNumResiduals(n_res);

    std::array<double, S2Parameters::kNumParams> x = start.to_array();

    ceres::Problem problem;
    problem.AddResidualBlock(cost, nullptr, x.data());
    problem.SetParameterLowerBound(x.data(), 0, 1e6);    // M
    problem.SetParameterUpperBound(x.data(), 0, 2e7);
    problem.SetParameterLowerBound(x.data(), 1, 4000.0); // R0
    problem.SetParameterUpperBound(x.data(), 1, 13000.0);
    problem.SetParameterLowerBound(x.data(), 3, 0.5);    // e
    problem.SetParameterUpperBound(x.data(), 3, 0.99);

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = 200;
    opts.function_tolerance = 1e-12;
    opts.gradient_tolerance = 1e-12;
    opts.parameter_tolerance = 1e-12;
    opts.num_threads = 8;
    opts.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    FitResult out;
    out.params = S2Parameters::from_array(x.data());
    out.report = summary.FullReport();
    out.converged = summary.termination_type == ceres::CONVERGENCE ||
                    summary.termination_type == ceres::USER_SUCCESS;
    out.n_residuals = data_->n_data();
    out.chi2 = 2.0 * summary.final_cost; // Ceres cost = ½ Σ r²
    out.dof = data_->n_data() - S2Parameters::kNumParams;
    out.chi2_red = out.chi2 / out.dof;

    // 1σ uncertainties from an explicit central-difference Jacobian
    // (see covariance() — the Ceres SVD covariance truncates the strongly
    // correlated M–R0 direction).
    const std::vector<double> cmat = covariance(out.params, out.sigma);

    // Derived period via Kepler: P = 2π√(a³/GM), a_phys = a_mas·R0;
    // propagate σ through the (M, R0, a_mas) covariance block.
    {
        const S2Parameters& p = out.params;
        const double P = 2.0 * 3.14159265358979323846 *
                         std::sqrt(std::pow(p.a_mas / 1000.0 * p.R0_pc, 3) *
                                   std::pow(constants::au_m, 3) /
                                   (p.M_msun * constants::GM_sun_si)) /
                         constants::yr_s;
        out.period_yr = P;
        const double dM = -0.5 * P / p.M_msun;
        const double dR = 1.5 * P / p.R0_pc;
        const double da = 1.5 * P / p.a_mas;
        const int N = S2Parameters::kNumParams;
        auto C = [&](int i, int j) { return cmat[i * N + j]; };
        const double var = dM * dM * C(0, 0) + dR * dR * C(1, 1) +
                           da * da * C(2, 2) + 2 * dM * dR * C(0, 1) +
                           2 * dM * da * C(0, 2) + 2 * dR * da * C(1, 2);
        out.sigma_period_yr = std::sqrt(std::max(0.0, var));
    }
    return out;
}

} // namespace sgr
