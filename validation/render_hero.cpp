// Hero-plate baker (Phase 9).
//
//   render_hero <out.png> [width] [height] [ssaa] [a_star] [incl_deg]
//               [disk_r_out] [t_peak_k] [exposure_ev]
//
// Historical: this CPU tool baked the first (Phase 9) menu background, a
// Novikov-Thorne Kerr disk. The current plate, engine/assets/hero_plate_4k.png,
// is Sgr A*'s hot flow at 230 GHz rendered by the live engine:
//   sgr_engine --shader disk --offscreen --width 3840 --height 2160 --spp 32
//              --preset 1 --band 230 --rcam 60 --incl 30 --fov 0.42 --aim 0.3 0
//              --screenshot hero_plate_4k.png
// The menu never runs the live renderer; the plate is a full-screen quad.
//
// Beauty-plate tolerances: rtol/atol 1e-9 (visual output, not metrology; the
// gate images and all published numbers use the 1e-11 defaults).
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <numeric>
#include <vector>

#include <spdlog/spdlog.h>

#include "sgr_physics/disk_model.hpp"
#include "sgr_physics/raytrace_cpu.hpp"

using namespace sgr;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <out.png> [width] [height] [ssaa] [a_star] "
                     "[incl_deg] [disk_r_out] [t_peak_k] [exposure_ev]\n",
                     argv[0]);
        return 2;
    }
    const char* out_png = argv[1];
    const int width = argc > 2 ? std::atoi(argv[2]) : 3840;
    const int height = argc > 3 ? std::atoi(argv[3]) : 2160;
    const int ssaa = argc > 4 ? std::atoi(argv[4]) : 2;
    const double a_star = argc > 5 ? std::atof(argv[5]) : 0.60;
    const double incl = argc > 6 ? std::atof(argv[6]) : 84.0;
    const double r_out = argc > 7 ? std::atof(argv[7]) : 22.0;
    const double t_peak = argc > 8 ? std::atof(argv[8]) : 5800.0;
    const double exposure = argc > 9 ? std::atof(argv[9]) : 0.4;

    const KerrMetric metric(1.0, a_star);
    CameraConfig cam;
    cam.r_cam = 1000.0;
    cam.incl_deg = incl;
    // Disk diameter 2*r_out subtends ~2*r_out/r_cam rad; fov sized so the
    // disk spans ~80% of the frame width.
    cam.fov_rad = 2.5 * r_out / cam.r_cam;
    cam.width = width;
    cam.height = height;

    DiskConfig dcfg;
    dcfg.enabled = true;
    dcfg.r_in = metric.r_isco(true);
    dcfg.r_out = r_out;

    TraceOptions topts;
    topts.atol = 1e-9;
    topts.rtol = 1e-9;

    const RaytraceCpu tracer(metric, cam, dcfg, topts);
    const NovikovThorneDisk disk(a_star, dcfg.r_in, dcfg.r_out, t_peak);
    spdlog::info("hero plate: {}x{} ssaa={} a*={} incl={} r_out={} T_peak={}K",
                 width, height, ssaa, a_star, incl, r_out, t_peak);

    // Composition offsets (NDC): black hole at ~57% width, ~40% height, per
    // the design reference — leaves the left column clear for the menu list.
    const double off_x = 0.14, off_y = 0.20;

    ImageRGB img;
    img.resize(width, height);
    std::vector<int> rows(height);
    std::iota(rows.begin(), rows.end(), 0);
    const int ss = std::max(1, ssaa);
    std::atomic<int> done{0};

    std::for_each(std::execution::par, rows.begin(), rows.end(), [&](int j) {
        for (int i = 0; i < width; ++i) {
            float acc[3] = {0.f, 0.f, 0.f};
            for (int sj = 0; sj < ss; ++sj) {
                for (int si = 0; si < ss; ++si) {
                    const double px = i + (si + 0.5) / ss;
                    const double py = j + (sj + 0.5) / ss;
                    const double x = 2.0 * px / width - 1.0 - off_x;
                    const double y = 1.0 - 2.0 * py / height - off_y;
                    const TraceResult res = tracer.trace(x, y);
                    std::array<float, 3> c{0.f, 0.f, 0.f};
                    switch (res.outcome) {
                        case RayOutcome::Escaped:
                            c = shade_starfield(res.escape_theta, res.escape_phi);
                            break;
                        case RayOutcome::DiskHit:
                            c = disk.shade(res.disk.r, res.disk.g);
                            break;
                        default:
                            break; // captured: black
                    }
                    acc[0] += c[0]; acc[1] += c[1]; acc[2] += c[2];
                }
            }
            float* p = img.pixel(i, j);
            const float inv = 1.0f / (ss * ss);
            p[0] = acc[0] * inv; p[1] = acc[1] * inv; p[2] = acc[2] * inv;
        }
        const int d = ++done;
        if (d % 128 == 0) spdlog::info("rows {}/{}", d, height);
    });

    if (!write_png_srgb(img, out_png, exposure)) {
        spdlog::error("failed to write {}", out_png);
        return 1;
    }
    spdlog::info("wrote {}", out_png);
    return 0;
}
