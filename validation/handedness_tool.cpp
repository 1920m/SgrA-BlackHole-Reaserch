// Image-handedness correction — every number quoted in the thesis appendix
// "Image handedness: an error found and corrected" (no number
// in the paper that the code has not produced).
//
//   handedness_tool <outdir>      → <outdir>/handedness.json
//
// Compares the corrected camera (backward tangent of the RECEIVED photon,
// −e_t + n·e) with the Phases 1–12 convention (the EMITTED photon, +e_t + n·e,
// CameraConfig::legacy_emitted_photon) using the SAME Carter/Mino integrator,
// and cross-checks the corrected images against the independent Cartesian
// Kerr-Schild Hamiltonian tracer (physics/src/kerr_schild.cpp).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "sgr_physics/disk_model.hpp"
#include "sgr_physics/kerr_schild.hpp"
#include "sgr_physics/raytrace_cpu.hpp"

using namespace sgr;
using nlohmann::json;

namespace {
constexpr double kPi = 3.14159265358979323846;

RaytraceCpu make_tracer(double a, double incl, double fov, double r_out,
                        bool legacy) {
    const KerrMetric m(1.0, a);
    CameraConfig cam;
    cam.r_cam = 1000.0;
    cam.incl_deg = incl;
    cam.fov_rad = fov;
    cam.legacy_emitted_photon = legacy;
    DiskConfig d;
    d.enabled = true;
    d.r_in = m.r_isco(true);
    d.r_out = r_out;
    return RaytraceCpu(m, cam, d);
}

// Bolometric flux summed over the image major axis, left and right halves
// (the Gate 3 measurement: r_cam = 1000 M, i = 80°, fov 0.032, r_out = 18 M).
json beaming(double a, bool legacy) {
    const RaytraceCpu tr = make_tracer(a, 80.0, 0.032, 18.0, legacy);
    const NovikovThorneDisk disk(a, tr.disk().r_in, tr.disk().r_out, 6300.0);
    double left = 0.0, right = 0.0;
    for (int i = 0; i < 400; ++i) {
        const double x = 2.0 * (i + 0.5) / 400.0 - 1.0;
        const TraceResult res = tr.trace(x, 0.0);
        if (res.outcome != RayOutcome::DiskHit) continue;
        (x < 0 ? left : right) += disk.bolometric(res.disk.r, res.disk.g);
    }
    return {{"left", left}, {"right", right}, {"left_over_right", left / right},
            {"brighter_side", left > right ? "left" : "right"}};
}

// Redshift of the disk on either side of the hole (fov 0.06: b ≈ 18 M).
json side_redshift(double a, bool legacy) {
    const RaytraceCpu tr = make_tracer(a, 80.0, 0.06, 40.0, legacy);
    const TraceResult l = tr.trace(-0.6, 0.0), r = tr.trace(+0.6, 0.0);
    return {{"g_left", l.disk.g}, {"g_right", r.disk.g},
            {"r_left", l.disk.r}, {"r_right", r.disk.r}};
}

// Redshift of a KS ray at its first crossing (conserved E, L_z).
double ks_disk_g(const KerrSchild& ks, double r_x, const KSRay& ray) {
    const KerrMetric m(1.0, ks.a());
    const double Om = m.omega_circular(r_x, true);
    const auto g = m.components(r_x, kPi / 2);
    const double ut =
        1.0 / std::sqrt(-(g(0, 0) + 2.0 * Om * g(0, 3) + Om * Om * g(3, 3)));
    return 1.0 / (ut * (ray.P[0] + Om * ks.angular_momentum(ray.X, ray.P)));
}

std::array<double, 3> look(double tx, double ty) {
    const double inv = 1.0 / std::sqrt(1.0 + tx * tx + ty * ty);
    return {-inv, -ty * inv, tx * inv};
}

double bardeen_xi(double r, double a) {
    return (r * r * (3.0 - r) - a * a * (r + 1.0)) / (a * (r - 1.0));
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <outdir>\n", argv[0]);
        return 2;
    }
    const std::filesystem::path outdir = argv[1];
    std::filesystem::create_directories(outdir);
    json out;

    // ---- 1. which limb is brighter, both conventions, a* = 0 and 0.94 -----
    for (double a : {0.0, 0.94}) {
        const std::string key = a == 0.0 ? "a0" : "a094";
        out["beaming"][key]["physical"] = beaming(a, false);
        out["beaming"][key]["legacy"] = beaming(a, true);
        out["side_redshift"][key]["physical"] = side_redshift(a, false);
        out["side_redshift"][key]["legacy"] = side_redshift(a, true);
    }

    // ---- 2. mirror identity: legacy pixel (x, y) == physical (−x, y) -------
    {
        const RaytraceCpu now = make_tracer(0.94, 80.0, 0.06, 40.0, false);
        const RaytraceCpu old = make_tracer(0.94, 80.0, 0.06, 40.0, true);
        double dr = 0.0, dg = 0.0, dphi = 0.0;
        int n = 0;
        for (int i = 0; i <= 12; ++i)
            for (double y : {-0.3, 0.0, 0.3}) {
                const double x = -0.9 + 0.15 * i;
                const TraceResult o = old.trace(x, y), m = now.trace(-x, y);
                if (o.outcome != RayOutcome::DiskHit ||
                    m.outcome != RayOutcome::DiskHit)
                    continue;
                dr = std::max(dr, std::abs(o.disk.r - m.disk.r) / m.disk.r);
                dg = std::max(dg, std::abs(o.disk.g - m.disk.g) / m.disk.g);
                dphi = std::max(dphi, std::abs(o.disk.phi + m.disk.phi));
                ++n;
            }
        out["mirror_identity"] = {{"pixels", n}, {"max_rel_dr", dr},
                                  {"max_rel_dg", dg}, {"max_abs_phi_sum", dphi}};
    }

    // ---- 3. shadow edges, a* = 0.94 edge-on, vs Bardeen ---------------------
    {
        const double a = 0.94;
        const KerrMetric m(1.0, a);
        CameraConfig cam;
        cam.r_cam = 1000.0;
        cam.incl_deg = 90.0;
        cam.fov_rad = 0.032;
        const RaytraceCpu tr(m, cam, DiskConfig{});
        const double t_half = std::tan(0.5 * cam.fov_rad);
        auto edge = [&](double psi) {
            const double s = tr.shadow_boundary_angle(psi, 0.010);
            const auto init = tr.camera_ray(std::tan(s) * std::cos(psi) / t_half,
                                            std::tan(s) * std::sin(psi) / t_half);
            return init.constants.Lz / init.constants.E; // ξ of the edge ray
        };
        const double xi_left = edge(kPi), xi_right = edge(0.0);
        out["shadow_edges"] = {
            {"xi_left", xi_left},
            {"xi_right", xi_right},
            {"bardeen_xi_prograde", bardeen_xi(m.r_photon(true), a)},
            {"bardeen_xi_retrograde", bardeen_xi(m.r_photon(false), a)}};
    }

    // ---- 4. independent Kerr-Schild tracer vs corrected Carter camera ------
    {
        const double a = 0.94, th = 80.0;
        const RaytraceCpu cpu = make_tracer(a, th, 0.06, 40.0, false);
        const KerrSchild ks(1.0, a, -1);
        KSTraceOptions o;
        o.rtol = o.atol = 1e-11;
        const double t_half = std::tan(0.5 * 0.06);
        double dr = 0.0, dg = 0.0, dL = 0.0, dQ = 0.0, dH = 0.0;
        // Rays on the central column (x = 0) pass over the spin axis, where
        // Boyer-Lindquist coordinates — and so the Carter reference — are
        // singular; report the off-axis agreement separately.
        double dr_off = 0.0, dg_off = 0.0;
        json worst;
        int n = 0;
        for (int i = 0; i <= 12; ++i)
            for (double y : {-0.3, 0.0, 0.3}) {
                const double x = -0.9 + 0.15 * i;
                const TraceResult c = cpu.trace(x, y);
                const KSRay ray = ks_camera_ray(ks, 1000.0, th * kPi / 180.0,
                                                look(x * t_half, y * t_half));
                const KSTraceResult k = ks_trace(ks, ray, o);
                dL = std::max(dL, k.max_dL_rel);
                dQ = std::max(dQ, k.max_dQ_rel);
                dH = std::max(dH, k.max_H_rel);
                if (c.outcome != RayOutcome::DiskHit || k.n_cross < 1) continue;
                // the CPU stops at the first crossing INSIDE the annulus
                int j = 0;
                while (j < k.n_cross &&
                       (k.cross_r[j] < cpu.disk().r_in || k.cross_r[j] > cpu.disk().r_out))
                    ++j;
                if (j == k.n_cross) continue;
                const double rel = std::abs(k.cross_r[j] - c.disk.r) / c.disk.r;
                if (rel > dr)
                    worst = {{"x", x}, {"y", y}, {"r_carter", c.disk.r},
                             {"r_ks", k.cross_r[j]}, {"order_carter", c.disk.subring_n},
                             {"order_ks", j}};
                dr = std::max(dr, rel);
                const double relg =
                    std::abs(ks_disk_g(ks, k.cross_r[j], ray) - c.disk.g) / c.disk.g;
                dg = std::max(dg, relg);
                if (std::abs(x) > 1e-6) {
                    dr_off = std::max(dr_off, rel);
                    dg_off = std::max(dg_off, relg);
                }
                ++n;
            }
        out["kerr_schild_crosscheck"] = {{"pixels", n},     {"max_rel_dr", dr},
                                         {"max_rel_dg", dg}, {"max_rel_dLz", dL},
                                         {"max_rel_dQ", dQ}, {"max_rel_H", dH},
                                         {"max_rel_dr_off_axis", dr_off},
                                         {"max_rel_dg_off_axis", dg_off},
                                         {"worst_pixel", worst}};
    }

    const auto path = outdir / "handedness.json";
    std::ofstream(path) << out.dump(2) << "\n";
    spdlog::info("wrote {}", path.string());
    std::printf("%s\n", out.dump(2).c_str());
    return 0;
}
