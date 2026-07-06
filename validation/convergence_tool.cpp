// Thesis data tool (paper/THESIS_PLAN.md §2.1): the convergence / constraint-
// projection evidence for Chapter 4 (numerical methods) and the Chapter 15
// bug-hunt narrative. Re-implements the reference tracer's stepping loop from
// the public physics APIs so the projection step can be toggled WITHOUT
// touching the validated core (which keeps projection unconditionally on).
//
//   convergence_tool <out_dir>
//
// Outputs:
//   projection_boundary.csv  traced Schwarzschild boundary angle vs tolerance,
//                            projection on/off, against the analytic Synge value
//   drift_vs_tol.csv         max conserved-quantity drift along a grazing ray
//                            vs tolerance, projection on/off
//   step_size.csv            accepted step size h(τ) and r(τ) along one
//                            boundary-grazing ray (tol 1e-11, projection on)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include "sgr_physics/integrator.hpp"
#include "sgr_physics/metric.hpp"
#include "sgr_physics/null_geodesics.hpp"
#include "sgr_physics/raytrace_cpu.hpp"

using namespace sgr;
namespace fs = std::filesystem;

namespace {

constexpr double kPi = 3.14159265358979323846;

struct MiniTraceResult {
    bool captured = false;
    bool escaped = false;
    NullConservedErrors max_err{}; // per-field maxima along the ray
    std::vector<std::array<double, 3>> h_log; // (tau, h_used, r) when recording
};

// Classification loop equivalent to RaytraceCpu::trace (no disk), with the
// constraint projection optional and conserved-error tracking.
MiniTraceResult mini_trace(const KerrMetric& metric, const NullRayInit& init,
                           double tol, bool projection, double r_cam,
                           bool record_steps = false) {
    const NullRayConstants& c = init.constants;
    NullState y = init.state;
    auto rhs = [&c](double, const NullState& s, NullState& d) {
        null_geodesic_rhs(c, s, d);
    };
    DormandPrince54<6, decltype(rhs)> rk(rhs, tol, tol);

    const double r_capture = metric.r_horizon() * 1.001 + 1e-6;
    const double r_escape = std::max(2.0 * r_cam, 1500.0);

    MiniTraceResult out;
    double tau = 0.0, h = 1e-3;
    for (int step = 0; step < 120000; ++step) {
        const auto res = rk.try_step(tau, y, h);
        h = std::clamp(res.h_next, 1e-14, 50.0);
        if (!res.accepted) continue;

        if (projection) {
            const double Rp = radial_potential(c, y[1]);
            if (Rp >= 0.0) y[4] = std::copysign(std::sqrt(Rp), y[4]);
            const double Tp = polar_potential(c, y[2]);
            if (Tp >= 0.0) y[5] = std::copysign(std::sqrt(Tp), y[5]);
        }

        const auto err = null_conserved_errors(metric, c, y);
        out.max_err.dE_rel = std::max(out.max_err.dE_rel, err.dE_rel);
        out.max_err.dLz_rel = std::max(out.max_err.dLz_rel, err.dLz_rel);
        out.max_err.dQ_rel = std::max(out.max_err.dQ_rel, err.dQ_rel);
        out.max_err.r_residual = std::max(out.max_err.r_residual, err.r_residual);
        out.max_err.theta_residual =
            std::max(out.max_err.theta_residual, err.theta_residual);
        if (record_steps) out.h_log.push_back({tau, res.h_used, y[1]});

        if (y[1] < r_capture) { out.captured = true; break; }
        if (y[1] > r_escape) { out.escaped = true; break; }
    }
    return out;
}

// Bisect the captured/escaped transition in camera angle along ψ = 0 with the
// mini tracer (same construction as RaytraceCpu::shadow_boundary_angle, with
// tolerance and projection under experimental control).
double boundary_angle(const RaytraceCpu& tracer, const KerrMetric& metric,
                      double tol, bool projection, double angle_max) {
    const double t_half = std::tan(0.5 * tracer.camera().fov_rad);
    auto classify = [&](double alpha) {
        const double x = std::tan(alpha) / t_half;
        const auto r = mini_trace(metric, tracer.camera_ray(x, 0.0), tol,
                                  projection, tracer.camera().r_cam);
        return r.captured; // true = inside the shadow
    };
    double lo = 0.0, hi = angle_max; // captured at lo, escaped at hi
    for (int it = 0; it < 60 && (hi - lo) > 1e-12; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (classify(mid)) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <out_dir>\n", argv[0]);
        return 2;
    }
    const fs::path out = argv[1];
    fs::create_directories(out);

    const KerrMetric metric(1.0, 0.0);
    CameraConfig cam;
    cam.r_cam = 1000.0;
    cam.incl_deg = 90.0;
    cam.fov_rad = 0.032;
    const RaytraceCpu tracer(metric, cam, DiskConfig{});
    const double analytic = schwarzschild_shadow_angle(1.0, cam.r_cam);
    const double t_half = std::tan(0.5 * cam.fov_rad);

    const std::vector<double> tols{1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
                                   1e-10, 1e-11, 1e-12, 1e-13};

    // ---- projection_boundary.csv -------------------------------------------
    {
        std::ofstream csv(out / "projection_boundary.csv");
        csv << "tol,projection,alpha_rad,rel_dev\n";
        for (double tol : tols)
            for (int proj : {1, 0}) {
                const double alpha =
                    boundary_angle(tracer, metric, tol, proj != 0, 0.010);
                csv << tol << ',' << proj << ',' << alpha << ','
                    << (alpha - analytic) / analytic << '\n';
                std::printf("boundary tol=%.0e proj=%d rel_dev=%+.3e\n", tol,
                            proj, (alpha - analytic) / analytic);
            }
    }

    // ---- drift_vs_tol.csv ---------------------------------------------------
    // A grazing escaped ray 0.5% outside the critical angle: long path around
    // the photon sphere and back out to r ~ 2000 M — the worst case for drift.
    {
        std::ofstream csv(out / "drift_vs_tol.csv");
        csv << "tol,projection,dE_rel,dLz_rel,dQ_rel,r_residual,theta_residual\n";
        const double x_graze = std::tan(1.005 * analytic) / t_half;
        const auto init = tracer.camera_ray(x_graze, 0.35);
        for (double tol : tols)
            for (int proj : {1, 0}) {
                const auto r =
                    mini_trace(metric, init, tol, proj != 0, cam.r_cam);
                csv << tol << ',' << proj << ',' << r.max_err.dE_rel << ','
                    << r.max_err.dLz_rel << ',' << r.max_err.dQ_rel << ','
                    << r.max_err.r_residual << ',' << r.max_err.theta_residual
                    << '\n';
            }
    }

    // ---- step_size.csv ------------------------------------------------------
    {
        const double x_graze = std::tan(1.001 * analytic) / t_half;
        const auto r = mini_trace(metric, tracer.camera_ray(x_graze, 0.0),
                                  1e-11, true, cam.r_cam, true);
        std::ofstream csv(out / "step_size.csv");
        csv << "tau,h,r\n";
        for (const auto& row : r.h_log)
            csv << row[0] << ',' << row[1] << ',' << row[2] << '\n';
        std::printf("step_size: %zu accepted steps (escaped=%d)\n",
                    r.h_log.size(), r.escaped ? 1 : 0);
    }

    std::printf("convergence_tool: wrote 3 CSVs to %s\n", out.string().c_str());
    return 0;
}
