// Phase 7 gate: CPU vs GPU pixel comparison (Gate 7).
//
//   compare_shadow <gpu_diagnostic.png> [size] [a_star] [incl_deg]
//
// Recomputes the identical scene with the CPU reference tracer using the
// diagnostic encoding (R: captured, G: disk redshift g/2, B: escape θ/π) and
// reports the mean absolute per-channel difference. Gate: mean < 0.01.
// A handful of near-critical boundary pixels may legitimately classify
// differently between the adaptive CPU integrator and the fixed-substep GPU
// port; they enter the mean with O(1) weight and are reported separately.
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "sgr_physics/raytrace_cpu.hpp"

using namespace sgr;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <gpu.png> [size] [a_star] [incl_deg]\n",
                     argv[0]);
        return 2;
    }
    const std::string gpu_png = argv[1];
    const int size = argc > 2 ? std::atoi(argv[2]) : 64;
    const double a_star = argc > 3 ? std::atof(argv[3]) : 0.94;
    const double incl = argc > 4 ? std::atof(argv[4]) : 80.0;

    int w = 0, h = 0, comp = 0;
    unsigned char* gpu = stbi_load(gpu_png.c_str(), &w, &h, &comp, 3);
    if (!gpu) {
        spdlog::error("cannot load {}", gpu_png);
        return 2;
    }
    if (w != size || h != size) {
        spdlog::error("GPU image is {}x{}, expected {}x{}", w, h, size, size);
        return 2;
    }

    const KerrMetric metric(1.0, a_star);
    CameraConfig cam;
    cam.r_cam = 1000.0;
    cam.incl_deg = incl;
    cam.fov_rad = 0.032;
    cam.width = size;
    cam.height = size;
    DiskConfig disk;
    disk.enabled = true;
    disk.r_in = metric.r_isco(true);
    disk.r_out = 18.0;
    const RaytraceCpu tracer(metric, cam, disk);

    double sum_abs = 0.0, max_abs = 0.0;
    int n_outliers = 0; // pixels with any channel differing by > 0.25
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < size; ++i) {
            const double x = 2.0 * (i + 0.5) / size - 1.0;
            const double y = 1.0 - 2.0 * (j + 0.5) / size;
            const TraceResult res = tracer.trace(x, y);
            double cpu[3] = {0.0, 0.0, 0.0};
            switch (res.outcome) {
                case RayOutcome::Captured:
                case RayOutcome::MaxSteps:
                    cpu[0] = 1.0;
                    break;
                case RayOutcome::DiskHit:
                    cpu[1] = 0.5 * res.disk.g;
                    break;
                case RayOutcome::Escaped:
                    cpu[2] = res.escape_theta / 3.14159265358979323846;
                    break;
            }
            const unsigned char* px = gpu + 3 * (static_cast<size_t>(j) * size + i);
            bool outlier = false;
            for (int c = 0; c < 3; ++c) {
                const double d = std::abs(cpu[c] - px[c] / 255.0);
                sum_abs += d;
                max_abs = std::max(max_abs, d);
                if (d > 0.25) outlier = true;
            }
            if (outlier) ++n_outliers;
        }
    }
    stbi_image_free(gpu);

    const double mean_abs = sum_abs / (3.0 * size * size);
    spdlog::info("CPU vs GPU ({}x{}, a*={}, i={}): mean|d|={:.6f} max|d|={:.4f} "
                 "outlier px={} ({:.3f}%)",
                 size, size, a_star, incl, mean_abs, max_abs, n_outliers,
                 100.0 * n_outliers / (size * size));

    nlohmann::json j{{"size", size},        {"a_star", a_star},
                     {"incl_deg", incl},    {"mean_abs_diff", mean_abs},
                     {"max_abs_diff", max_abs}, {"outlier_pixels", n_outliers},
                     {"gate7_pass", mean_abs < 0.01}};
    std::ofstream(gpu_png + ".compare.json") << j.dump(2) << "\n";

    if (mean_abs < 0.01) {
        spdlog::info("Gate 7 PASS (mean diff {:.6f} < 0.01)", mean_abs);
        return 0;
    }
    spdlog::error("Gate 7 FAIL (mean diff {:.6f} >= 0.01)", mean_abs);
    return 1;
}
