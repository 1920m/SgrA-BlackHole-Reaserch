// Phase 10: Sgr A* calibration + spin–inclination degeneracy sweep.
//
//   sweep_tool calibrate <out_dir>              — a*=0.94, i=30°, 32-direction
//                                                 boundary → diameter in μas
//   sweep_tool sweep <out_dir> [na] [ni]        — grid over a* × inclination,
//                                                 4-direction diameters (CSV)
//
// Diameters are measured on the traced critical boundary (bisection in camera
// angle), converted to impact parameters b = r sinα/√(1−2M/r) and to μas with
// the GRAVITY 2022 mass and distance (constants.hpp).
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <execution>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "sgr_physics/constants.hpp"
#include "sgr_physics/raytrace_cpu.hpp"

using namespace sgr;
namespace fs = std::filesystem;

namespace {

double boundary_b_M(const RaytraceCpu& tracer, double psi) {
    const double alpha = tracer.shadow_boundary_angle(psi, 0.010);
    const double rc = tracer.camera().r_cam;
    return std::sin(alpha) * rc / std::sqrt(1.0 - 2.0 / rc);
}

struct Diameters {
    double dx_M, dy_M, mean_M, mean_uas, axis_ratio;
};

Diameters measure(double a_star, double incl_deg, int n_dirs) {
    const KerrMetric metric(1.0, a_star);
    CameraConfig cam;
    cam.r_cam = 1000.0;
    cam.incl_deg = incl_deg;
    cam.fov_rad = 0.032;
    const RaytraceCpu tracer(metric, cam, DiskConfig{});

    // horizontal + vertical extents (n_dirs = 4), or full ring average
    std::vector<double> b(n_dirs);
    for (int k = 0; k < n_dirs; ++k)
        b[k] = boundary_b_M(tracer, 2.0 * 3.14159265358979323846 * k / n_dirs);

    Diameters d{};
    d.dx_M = b[0] + b[n_dirs / 2];
    d.dy_M = b[n_dirs / 4] + b[3 * n_dirs / 4];
    // mean diameter over all sampled directions (opposite pairs)
    double sum = 0.0;
    for (int k = 0; k < n_dirs / 2; ++k) sum += b[k] + b[k + n_dirs / 2];
    d.mean_M = sum / (n_dirs / 2);
    d.mean_uas = d.mean_M * constants::sgrA_uas_per_rg;
    d.axis_ratio = d.dx_M / d.dy_M;
    return d;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <calibrate|sweep> <out_dir> [na] [ni]\n",
                     argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const fs::path out = argv[2];
    fs::create_directories(out);

    if (mode == "calibrate") {
        const Diameters d = measure(constants::sgrA_spin_default,
                                    constants::sgrA_incl_deg, 32);
        spdlog::info("Sgr A* calibration (a*={}, i={} deg):",
                     constants::sgrA_spin_default, constants::sgrA_incl_deg);
        spdlog::info("  d_x = {:.4f} M, d_y = {:.4f} M, mean = {:.4f} M",
                     d.dx_M, d.dy_M, d.mean_M);
        spdlog::info("  mean diameter = {:.2f} uas (EHT 2022 ring: 51.8±2.3, "
                     "shadow: 48.7±7)", d.mean_uas);
        spdlog::info("  axis ratio dx/dy = {:.4f}", d.axis_ratio);
        const bool pass = d.mean_uas > 48.0 && d.mean_uas < 52.0;
        nlohmann::json j{{"a_star", constants::sgrA_spin_default},
                         {"incl_deg", constants::sgrA_incl_deg},
                         {"dx_M", d.dx_M},
                         {"dy_M", d.dy_M},
                         {"mean_M", d.mean_M},
                         {"mean_uas", d.mean_uas},
                         {"axis_ratio", d.axis_ratio},
                         {"uas_per_M", constants::sgrA_uas_per_rg},
                         {"gate10_pass", pass}};
        std::ofstream(out / "calibration.json") << j.dump(2) << "\n";
        spdlog::info("Gate 10: {}", pass ? "PASS (48-52 uas)" : "FAIL");
        return pass ? 0 : 1;
    }

    if (mode == "sweep") {
        const int na = argc > 3 ? std::atoi(argv[3]) : 20;
        const int ni = argc > 4 ? std::atoi(argv[4]) : 20;
        struct Row { double a, inc; Diameters d; };
        std::vector<Row> rows(static_cast<size_t>(na) * ni);
        std::vector<int> idx(rows.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::atomic<int> done{0};
        std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int k) {
            const int ia = k / ni, ii = k % ni;
            const double a = 0.0 + 0.998 * ia / (na - 1);
            const double inc = 1.0 + 89.0 * ii / (ni - 1);
            rows[k] = {a, inc, measure(a, inc, 4)};
            const int n = ++done;
            if (n % 25 == 0) spdlog::info("sweep {}/{}", n, rows.size());
        });
        std::ofstream csv(out / "sweep.csv");
        csv << "a_star,incl_deg,dx_M,dy_M,mean_M,diameter_uas,axis_ratio\n";
        for (const auto& r : rows)
            csv << r.a << "," << r.inc << "," << r.d.dx_M << "," << r.d.dy_M
                << "," << r.d.mean_M << "," << r.d.mean_uas << ","
                << r.d.axis_ratio << "\n";
        spdlog::info("wrote {}", (out / "sweep.csv").string());
        return 0;
    }
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
