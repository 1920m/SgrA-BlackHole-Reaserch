// CPU twin of the live renderer's PHYSICAL preset (Gate 13, live colour).
//
//   render_disk_cpu <out.png> <width> <height> <a_star> <incl_deg> <r_cam>
//                   <fov_rad> <r_out> <t_peak_k> <exposure_ev>
//
// Same ZAMO pinhole camera, image-plane convention and display transform as
// engine/shaders/disk.comp.glsl (shade_mode 0): opaque Novikov-Thorne disk
// shaded with NovikovThorneDisk::shade (blackbody at g·T), escaped rays black,
// clamp(c·2^EV)^(1/2.2) to 8 bit. Also writes <out>_mask.png (255 = disk hit)
// so validation/phase14/compare_live_colour.py can restrict the comparison to
// disk pixels. Beauty tolerances 1e-9, as render_hero.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <numeric>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>
#include <stb_image_write.h>

#include "sgr_physics/disk_model.hpp"
#include "sgr_physics/raytrace_cpu.hpp"

using namespace sgr;

int main(int argc, char** argv) {
    if (argc < 11) {
        std::fprintf(stderr,
                     "usage: %s <out.png> <width> <height> <a_star> <incl_deg> "
                     "<r_cam> <fov_rad> <r_out> <t_peak_k> <exposure_ev>\n",
                     argv[0]);
        return 2;
    }
    const std::string out_png = argv[1];
    const int width = std::atoi(argv[2]);
    const int height = std::atoi(argv[3]);
    const double a_star = std::atof(argv[4]);
    const double incl = std::atof(argv[5]);
    const double r_cam = std::atof(argv[6]);
    const double fov = std::atof(argv[7]);
    const double r_out = std::atof(argv[8]);
    const double t_peak = std::atof(argv[9]);
    const double exposure = std::atof(argv[10]);

    const KerrMetric metric(1.0, a_star);
    CameraConfig cam;
    cam.r_cam = r_cam;
    cam.incl_deg = incl;
    cam.fov_rad = fov;
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

    ImageRGB img;
    img.resize(width, height);
    std::vector<unsigned char> mask(static_cast<size_t>(width) * height, 0);
    std::vector<int> rows(height);
    std::iota(rows.begin(), rows.end(), 0);
    std::for_each(std::execution::par, rows.begin(), rows.end(), [&](int j) {
        for (int i = 0; i < width; ++i) {
            const double x = 2.0 * (i + 0.5) / width - 1.0;
            const double y = 1.0 - 2.0 * (j + 0.5) / height;
            const TraceResult res = tracer.trace(x, y);
            float* p = img.pixel(i, j);
            if (res.outcome == RayOutcome::DiskHit) {
                const auto c = disk.shade(res.disk.r, res.disk.g);
                p[0] = c[0]; p[1] = c[1]; p[2] = c[2];
                mask[static_cast<size_t>(j) * width + i] = 255;
            }
        }
    });

    if (!write_png_srgb(img, out_png.c_str(), exposure)) {
        spdlog::error("failed to write {}", out_png);
        return 1;
    }
    const std::string mask_png =
        out_png.substr(0, out_png.rfind('.')) + "_mask.png";
    if (!stbi_write_png(mask_png.c_str(), width, height, 1, mask.data(), width)) {
        spdlog::error("failed to write {}", mask_png);
        return 1;
    }
    spdlog::info("wrote {} and {}", out_png, mask_png);
    return 0;
}
