// Reference render tool (Phases 1–2 gates).
//
//   render_reference schwarzschild <out.png> [width]
//   render_reference kerr94        <out.png> [width]
//
// Renders the CPU reference image, measures the shadow boundary by bisection
// in 32 directions, and writes a JSON report next to the PNG with the measured
// boundary radii (used later for paper figures and the Phase-10 calibration).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "sgr_physics/constants.hpp"
#include "sgr_physics/raytrace_cpu.hpp"

using namespace sgr;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <schwarzschild|kerr94> <out.png> [width]\n",
                     argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const std::string out_png = argv[2];
    const int width = argc > 3 ? std::atoi(argv[3]) : 800;

    double a_star = 0.0;
    if (mode == "kerr94") a_star = 0.94;
    else if (mode != "schwarzschild") {
        std::fprintf(stderr, "unknown mode '%s'\n", mode.c_str());
        return 2;
    }

    const KerrMetric metric(1.0, a_star);
    CameraConfig cam;
    cam.r_cam = 1000.0;
    // 80°: near edge-on (strong frame-dragging asymmetry for Kerr, classic
    // lensed-disk look) while keeping the direct disk image visible — at
    // exactly 90° an infinitely thin disk subtends zero solid angle and only
    // its lensed halo appears.
    cam.incl_deg = 80.0;
    cam.fov_rad = 0.032;
    cam.width = width;
    cam.height = width;

    DiskConfig disk;
    disk.enabled = true;
    disk.r_in = metric.r_isco(true);
    disk.r_out = 14.0;

    spdlog::info("mode={} a*={} r_isco={:.4f} r_hor={:.4f}", mode, a_star,
                 disk.r_in, metric.r_horizon());

    const RaytraceCpu tracer(metric, cam, disk);

    // --- shadow boundary metrology (no disk influence: capture-only) -------
    const RaytraceCpu bare(metric, cam, DiskConfig{});
    nlohmann::json boundary = nlohmann::json::array();
    double min_a = 1e9, max_a = 0.0, sum = 0.0;
    const int n_dirs = 32;
    for (int k = 0; k < n_dirs; ++k) {
        const double psi = 2.0 * 3.14159265358979323846 * k / n_dirs;
        const double alpha = bare.shadow_boundary_angle(psi, 0.009);
        boundary.push_back({{"psi_rad", psi}, {"alpha_rad", alpha}});
        min_a = std::min(min_a, alpha);
        max_a = std::max(max_a, alpha);
        sum += alpha;
        spdlog::debug("psi={:.3f} alpha={:.6e}", psi, alpha);
    }
    const double mean_a = sum / n_dirs;
    // Convert to an equivalent impact-parameter diameter in M (b ≈ α·r for a
    // distant observer) and to μas at the Sgr A* distance.
    const double diam_M = 2.0 * mean_a * cam.r_cam /
                          std::sqrt(1.0 - 2.0 / cam.r_cam);
    const double diam_uas = diam_M * constants::sgrA_uas_per_rg;

    spdlog::info("shadow: mean angular radius {:.6e} rad, spread {:.3e} ({:.3f}%)",
                 mean_a, max_a - min_a, 100.0 * (max_a - min_a) / mean_a);
    spdlog::info("shadow diameter: {:.4f} M  -> {:.2f} uas at Sgr A* distance",
                 diam_M, diam_uas);
    if (a_star == 0.0) {
        const double analytic = schwarzschild_shadow_angle(1.0, cam.r_cam);
        spdlog::info("analytic Schwarzschild radius {:.6e} rad, deviation {:.3e} rel",
                     analytic, std::abs(mean_a - analytic) / analytic);
    }

    // --- full frame ---------------------------------------------------------
    spdlog::info("rendering {}x{} with 2x2 SSAA ...", width, width);
    const ImageRGB img = render_frame(tracer, nullptr, 2);
    if (!write_png_srgb(img, out_png.c_str(), 0.0)) {
        spdlog::error("failed to write {}", out_png);
        return 1;
    }
    spdlog::info("wrote {}", out_png);

    nlohmann::json report{
        {"mode", mode},
        {"a_star", a_star},
        {"r_cam_M", cam.r_cam},
        {"incl_deg", cam.incl_deg},
        {"fov_rad", cam.fov_rad},
        {"shadow_mean_alpha_rad", mean_a},
        {"shadow_min_alpha_rad", min_a},
        {"shadow_max_alpha_rad", max_a},
        {"shadow_diameter_M", diam_M},
        {"shadow_diameter_uas_at_sgrA", diam_uas},
        {"boundary", boundary},
    };
    const std::string out_json = out_png.substr(0, out_png.rfind('.')) + ".json";
    std::ofstream(out_json) << report.dump(2) << "\n";
    spdlog::info("wrote {}", out_json);
    return 0;
}
