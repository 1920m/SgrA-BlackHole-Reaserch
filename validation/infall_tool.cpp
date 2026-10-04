// Gate 24 — falling into Sgr A*: every number the extension paper quotes
// for the fall (no number the code has not produced).
//
//   infall_tool <outdir>          → <outdir>/infall.json
//
// 1. The live renderer's default trip (rain from r0 = 30 M at polar angle
//    60°, a* = 0.94, ingoing Kerr–Schild, sgr_physics/infall.hpp): proper
//    times to the ergosphere, the horizon and the end just above r₋, with
//    the conservation of E, L_z, Q and the 4-velocity norm through r₊.
// 2. Closed-form checks in Schwarzschild (radial fall from rest: the
//    cycloid; rain: τ = (2/3)(r0^1.5 − r^1.5)/√(2M)).
// 3. What the in-faller measures on the way: the frequency of the sky light
//    arriving from straight behind and from the side, and the tidal field
//    (eigenvalues of E_ij; the Kretschmann check against the closed form).
// 4. The dark region on image strips through the centre of the in-faller's
//    view, at several radii outside and inside r₊, bisected on the fp64
//    tracer with the renderer's own camera frame — the reference for the
//    GPU comparison (validation/phase14/compare_infall.py).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "sgr_physics/constants.hpp"
#include "sgr_physics/infall.hpp"
#include "sgr_physics/kerr_schild.hpp"

using namespace sgr;
using nlohmann::json;

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kStripT = 2.0;   // strips cover tan(angle) ∈ [−2, 2] (±63°)

// pixel offset (tx, ty) → direction in the frame (e1 out, e2 θ̂, e3 φ̂) for
// the renderer's default look basis: fwd = −e1, right = e3, up = −e2
std::array<double, 3> pixel_dir(double tx, double ty) {
    const double inv = 1.0 / std::sqrt(1.0 + tx * tx + ty * ty);
    return {-inv, -ty * inv, tx * inv};
}

// frequency ratio ν_seen / ν_∞ of light arriving along local direction n
// (the pixel looking along n receives k = e0 − n·e)
double freq_ratio(const KerrSchild& ks, const Vec4& X, const std::array<Vec4, 4>& e,
                  const std::array<double, 3>& n) {
    Vec4 k{};
    for (int m = 0; m < 4; ++m) k[m] = e[0][m] - n[0] * e[1][m] - n[1] * e[2][m] - n[2] * e[3][m];
    const Metric4<double> g = ks.metric(X);
    double kt = 0.0;
    for (int m = 0; m < 4; ++m) kt += g(0, m) * k[m];
    return 1.0 / (-kt);
}

// edges (tan offsets) of the dark region along one image axis
std::vector<double> strip_edges(const KerrSchild& ks, const Vec4& X,
                                const std::array<Vec4, 4>& e, double r, int axis) {
    KSTraceOptions opt;
    opt.rtol = opt.atol = 1e-11;
    opt.r_escape = std::max(4.0 * r, 200.0);
    auto dark = [&](double t) {
        const auto n = axis == 0 ? pixel_dir(t, 0.0) : pixel_dir(0.0, t);
        return ks_trace(ks, ks_frame_ray(ks, X, e, n), opt).outcome != KSOutcome::Escaped;
    };
    const int N = 600;
    std::vector<double> out;
    bool prev = dark(-kStripT);
    double tp = -kStripT;
    for (int i = 1; i <= N; ++i) {
        const double t = -kStripT + 2.0 * kStripT * i / N;
        const bool d = dark(t);
        if (d != prev) {
            double lo = tp, hi = t;   // dark(lo) == prev
            for (int it = 0; it < 48; ++it) {
                const double mid = 0.5 * (lo + hi);
                if (dark(mid) == prev) lo = mid; else hi = mid;
            }
            out.push_back(0.5 * (lo + hi));
        }
        prev = d;
        tp = t;
    }
    return out;
}
}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path out = argc > 1 ? argv[1] : "validation/extension";
    std::filesystem::create_directories(out);
    const double a = constants::sgrA_spin_default;
    const double tg = constants::sgrA_tg_s, rg_km = constants::sgrA_rg_m / 1000.0;
    json J;

    // ---- 1. the default trip --------------------------------------------------
    InfallStart st;   // rain, r0 = 30, θ0 = 60°: the renderer's defaults
    const InfallTrajectory tr(a, st);
    const KerrSchild& ks = tr.chart();
    const double ct0 = std::cos(st.theta0);
    const double r_ergo = 1.0 + std::sqrt(1.0 - a * a * ct0 * ct0);
    J["trip"] = {{"a_star", a},
                 {"r0", st.r0},
                 {"theta0_deg", st.theta0 * 180.0 / kPi},
                 {"start", "rain (E = 1, L_z = 0, Q = 0)"},
                 {"r_plus", ks.r_plus()},
                 {"r_minus", ks.r_minus()},
                 {"r_stop", tr.samples().back().r},
                 {"r_ergo", r_ergo},
                 {"tau_ergo", tr.tau_at_radius(r_ergo)},
                 {"tau_horizon", tr.tau_horizon()},
                 {"tau_end", tr.tau_end()},
                 {"tau_inside", tr.tau_end() - tr.tau_horizon()},
                 {"tg_s", tg},
                 {"rg_km", rg_km},
                 {"n_samples", tr.samples().size()},
                 {"max_dE", tr.max_dE()},
                 {"max_dL", tr.max_dL()},
                 {"max_dQ", tr.max_dQ()},
                 {"max_norm", tr.max_norm()}};
    std::printf("[infall] rain from %.0f M: ergosphere at tau = %.3f M, horizon %.4f M "
                "(%.1f min), end %.4f M; %.4f M (%.1f s) inside\n",
                st.r0, tr.tau_at_radius(r_ergo), tr.tau_horizon(), tr.tau_horizon() * tg / 60,
                tr.tau_end(), tr.tau_end() - tr.tau_horizon(),
                (tr.tau_end() - tr.tau_horizon()) * tg);
    std::printf("[infall] conservation: dE %.1e, dL %.1e, dQ %.1e, norm %.1e\n", tr.max_dE(),
                tr.max_dL(), tr.max_dQ(), tr.max_norm());

    // ---- 2. closed forms in Schwarzschild ---------------------------------------
    {
        InfallStart rs;
        rs.kind = InfallStartKind::Rest;
        rs.r0 = 10.0;
        const InfallTrajectory t0(0.0, rs);
        const double eta = std::acos(2.0 * 2.0 / rs.r0 - 1.0);
        const double tau_c = std::sqrt(std::pow(rs.r0, 3) / 8.0) * (eta + std::sin(eta));
        InfallStart rn;
        rn.r0 = st.r0;
        const InfallTrajectory t1(0.0, rn);
        const double tau_r = 2.0 / 3.0 * (std::pow(rn.r0, 1.5) - std::pow(2.0, 1.5)) / std::sqrt(2.0);
        J["schwarzschild"] = {{"rest_r0", rs.r0},
                              {"rest_tau_horizon", t0.tau_horizon()},
                              {"rest_tau_closed", tau_c},
                              {"rain_r0", rn.r0},
                              {"rain_tau_horizon", t1.tau_horizon()},
                              {"rain_tau_closed", tau_r},
                              {"max_rel_err", std::max(std::abs(t0.tau_horizon() / tau_c - 1),
                                                       std::abs(t1.tau_horizon() / tau_r - 1))}};
        std::printf("[infall] Schwarzschild: rest from 10 M %.10f (cycloid %.10f), rain from "
                    "30 M %.10f (closed %.10f)\n",
                    t0.tau_horizon(), tau_c, t1.tau_horizon(), tau_r);
    }

    // ---- 3. what the in-faller measures ----------------------------------------
    json along = json::array();
    for (double r : {30.0, 10.0, 4.0, r_ergo, 1.5, ks.r_plus() * 1.0001, 1.2, 1.0, 0.8,
                     tr.samples().back().r + 1e-6}) {
        const double tau = std::abs(r - st.r0) < 1e-9 ? 0.0 : tr.tau_at_radius(r);
        const InfallSample s = tr.at(tau);
        const auto e = infall_tetrad(ks, s.X, s.U);
        const TidalField t = tidal_field(ks, s.X, e);
        const double K = kerr_kretschmann(1.0, a, s.r, s.X[3] / s.r);
        along.push_back({{"r", s.r},
                         {"tau", s.tau},
                         {"theta_deg", s.theta * 180.0 / kPi},
                         {"t_bl", std::isfinite(s.t_bl) ? json(s.t_bl) : json(nullptr)},
                         {"freq_behind", freq_ratio(ks, s.X, e, {1.0, 0.0, 0.0})},
                         {"freq_side", freq_ratio(ks, s.X, e, {0.0, 0.0, 1.0})},
                         {"tidal_eig", {t.eig[0], t.eig[1], t.eig[2]}},
                         {"stretch_2m_ms2", -t.eig[0] * 2.0 / (tg * tg)},
                         {"squeeze_2m_ms2", t.eig[2] * 2.0 / (tg * tg)},
                         {"kretschmann", t.kretschmann},
                         {"kretschmann_closed", K}});
        std::printf("[infall] r = %7.4f: tau %8.4f, sky behind x%.3f, side x%.3f, tidal "
                    "stretch %.3g m/s^2 (2 m), K/K_closed - 1 = %.1e\n",
                    s.r, s.tau, freq_ratio(ks, s.X, e, {1.0, 0.0, 0.0}),
                    freq_ratio(ks, s.X, e, {0.0, 0.0, 1.0}), -t.eig[0] * 2.0 / (tg * tg),
                    t.kretschmann / K - 1.0);
    }
    J["along"] = along;

    // dense profile for the figure: evenly spaced in log(r − r₋)
    {
        json prof = json::array();
        const double rm = ks.r_minus(), r1 = tr.samples().front().r, r2 = tr.samples().back().r;
        const int n = 240;
        for (int i = 0; i < n; ++i) {
            const double u = static_cast<double>(i) / (n - 1);
            const double r = rm + (r1 - rm) * std::pow((r2 - rm) / (r1 - rm), u);
            const double tau = i == 0 ? 0.0 : i == n - 1 ? tr.tau_end() : tr.tau_at_radius(r);
            const InfallSample s = tr.at(tau);
            const auto e = infall_tetrad(ks, s.X, s.U);
            const TidalField t = tidal_field(ks, s.X, e);
            prof.push_back({{"r", s.r},
                            {"tau", s.tau},
                            {"t_bl", std::isfinite(s.t_bl) ? json(s.t_bl) : json(nullptr)},
                            {"freq_behind", freq_ratio(ks, s.X, e, {1.0, 0.0, 0.0})},
                            {"freq_side", freq_ratio(ks, s.X, e, {0.0, 0.0, 1.0})},
                            {"stretch_2m_ms2", -t.eig[0] * 2.0 / (tg * tg)},
                            {"squeeze_2m_ms2", t.eig[2] * 2.0 / (tg * tg)}});
        }
        J["profile"] = prof;
    }

    // ---- 4. the dark region on image strips (GPU reference) ---------------------
    json strips = json::array();
    for (double r : {6.0, 2.0, 1.2, 0.9}) {
        const InfallSample s = tr.at(tr.tau_at_radius(r));
        const auto e = infall_tetrad(ks, s.X, s.U);
        const auto h = strip_edges(ks, s.X, e, s.r, 0), v = strip_edges(ks, s.X, e, s.r, 1);
        // frequency of the sky light seen right at each edge of the dark
        // region ahead (the light that skims it), relative to infinity
        std::vector<double> fe;
        for (double t : h) fe.push_back(freq_ratio(ks, s.X, e, pixel_dir(t, 0.0)));
        strips.push_back({{"r", r}, {"strip_T", kStripT}, {"edges_h", h}, {"edges_v", v},
                          {"freq_at_edges_h", fe}});
        std::printf("[infall] r = %.2f: %zu horizontal and %zu vertical edges; sky at the "
                    "edges x%.3f .. x%.3f\n",
                    r, h.size(), v.size(), fe.empty() ? 0.0 : *std::min_element(fe.begin(), fe.end()),
                    fe.empty() ? 0.0 : *std::max_element(fe.begin(), fe.end()));
    }
    J["strips"] = strips;

    const auto path = out / "infall.json";
    std::ofstream(path) << J.dump(2) << "\n";
    std::printf("infall_tool: wrote %s\n", path.string().c_str());
    return 0;
}
