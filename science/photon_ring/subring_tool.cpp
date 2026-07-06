// Phase 3 tool: Novikov-Thorne disk render, Doppler-asymmetry metrology,
// sub-ring isolation images and flux ratios, brightness profile export.
//
//   subring_tool <outdir> [a_star] [incl_deg] [image_size]
//
// Outputs (into <outdir>):
//   disk_nt.png            — colour render, NT disk + starfield
//   subring_n{0,1,2}.png   — per-order bolometric maps (log display scale)
//   subring_report.json    — fluxes, ratios, widths, asymmetry numbers
//   brightness_profile.csv — bolometric I(x) along the image major axis
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "photon_ring/SubRingExtractor.hpp"
#include "sgr_physics/constants.hpp"

using namespace sgr;

namespace {

// Thesis mode (paper/THESIS_PLAN.md §2.1): sub-ring flux ratios and n = 1
// band widths as a function of spin, at fixed inclination.
//   subring_tool --sweep <outdir> [incl_deg] [size]
int run_sweep(const std::filesystem::path& outdir, double incl, int size) {
    std::filesystem::create_directories(outdir);
    std::ofstream csv(outdir / "subring_sweep.csv");
    csv << "a_star,incl_deg,flux_n0,flux_n1,flux_n2,ratio_n0_n1,ratio_n1_n2,"
           "width_n1_psi0_rad,width_n1_psi90_rad,width_n1_psi180_rad,"
           "width_n1_psi270_rad\n";
    for (double a_star : {0.0, 0.3, 0.6, 0.8, 0.94, 0.998}) {
        const KerrMetric metric(1.0, a_star);
        CameraConfig cam;
        cam.r_cam = 1000.0;
        cam.incl_deg = incl;
        cam.fov_rad = 0.032;
        cam.width = size;
        cam.height = size;
        DiskConfig dcfg;
        dcfg.enabled = true;
        dcfg.r_in = metric.r_isco(true);
        dcfg.r_out = 18.0;
        const RaytraceCpu tracer(metric, cam, dcfg);
        const NovikovThorneDisk disk(a_star, dcfg.r_in, dcfg.r_out, 6300.0);
        spdlog::info("sweep a*={}: analysing {}x{} ...", a_star, size, size);
        SubRingExtractor extractor(tracer, disk);
        SubRingStats stats = extractor.analyze(2, size, nullptr);
        // n = 1 band widths along four image-plane directions; a width of 0
        // records "no band found" (occluded by the direct disk image — a real
        // geometric effect at high inclination, documented in the thesis).
        double w[4] = {0.0, 0.0, 0.0, 0.0};
        const double psis[4] = {0.0, 1.5707963267948966, 3.14159265358979323846,
                                4.71238898038469};
        for (int k = 0; k < 4; ++k) {
            const size_t before = stats.widths.size();
            extractor.scan_widths(stats, 1, psis[k], 1.2e-3, 9.5e-3);
            if (stats.widths.size() > before) {
                const auto& b = stats.widths.back();
                w[k] = b.s_outer - b.s_inner;
            }
        }
        const double f0 = stats.bins[0].flux_bolometric;
        const double f1 = stats.bins[1].flux_bolometric;
        const double f2 = stats.bins[2].flux_bolometric;
        csv << a_star << ',' << incl << ',' << f0 << ',' << f1 << ',' << f2
            << ',' << f0 / std::max(1e-30, f1) << ','
            << f1 / std::max(1e-30, f2) << ',' << w[0] << ',' << w[1] << ','
            << w[2] << ',' << w[3] << '\n';
        csv.flush();
        spdlog::info("  ratios n0/n1={:.2f} n1/n2={:.2f}",
                     f0 / std::max(1e-30, f1), f1 / std::max(1e-30, f2));
    }
    spdlog::info("wrote subring_sweep.csv");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <outdir> [a_star] [incl_deg] [size]\n"
                     "       %s --sweep <outdir> [incl_deg] [size]\n",
                     argv[0], argv[0]);
        return 2;
    }
    if (std::string(argv[1]) == "--sweep") {
        if (argc < 3) { std::fprintf(stderr, "--sweep needs <outdir>\n"); return 2; }
        const double incl = argc > 3 ? std::atof(argv[3]) : 80.0;
        const int size = argc > 4 ? std::atoi(argv[4]) : 220;
        return run_sweep(argv[2], incl, size);
    }
    const std::filesystem::path outdir = argv[1];
    std::filesystem::create_directories(outdir);
    const double a_star = argc > 2 ? std::atof(argv[2]) : 0.94;
    const double incl = argc > 3 ? std::atof(argv[3]) : 80.0;
    const int size = argc > 4 ? std::atoi(argv[4]) : 800;

    const KerrMetric metric(1.0, a_star);
    CameraConfig cam;
    cam.r_cam = 1000.0;
    cam.incl_deg = incl;
    cam.fov_rad = 0.032;
    cam.width = size;
    cam.height = size;

    DiskConfig dcfg;
    dcfg.enabled = true;
    dcfg.r_in = metric.r_isco(true);
    dcfg.r_out = 18.0;

    const RaytraceCpu tracer(metric, cam, dcfg);
    const NovikovThorneDisk disk(a_star, dcfg.r_in, dcfg.r_out, 6300.0);
    spdlog::info("a*={} incl={} r_isco={:.4f}", a_star, incl, dcfg.r_in);

    // ---- colour render (Gate 3 image) --------------------------------------
    ShadeFn shade = [&disk](const TraceResult& res, const RaytraceCpu&)
        -> std::array<float, 3> {
        switch (res.outcome) {
            case RayOutcome::Escaped:
                return shade_starfield(res.escape_theta, res.escape_phi);
            case RayOutcome::DiskHit:
                return disk.shade(res.disk.r, res.disk.g);
            default:
                return {0.f, 0.f, 0.f};
        }
    };
    spdlog::info("rendering colour frame {}x{} (2x2 SSAA)...", size, size);
    const ImageRGB frame = render_frame(tracer, shade, 2);
    write_png_srgb(frame, (outdir / "disk_nt.png").string().c_str(), 0.0);
    spdlog::info("wrote disk_nt.png");

    // ---- Doppler beaming asymmetry -----------------------------------------
    // Bolometric flux in the left vs right image halves, plus min/max g.
    double flux_left = 0.0, flux_right = 0.0, g_min = 1e9, g_max = 0.0;
    const int Ns = 400;
    for (int j = 0; j < Ns; ++j) {
        for (int i = 0; i < Ns; ++i) {
            const double x = 2.0 * (i + 0.5) / Ns - 1.0;
            const double y = 1.0 - 2.0 * (j + 0.5) / Ns;
            const TraceResult res = tracer.trace(x, y);
            if (res.outcome != RayOutcome::DiskHit) continue;
            const double I = disk.bolometric(res.disk.r, res.disk.g);
            (x < 0 ? flux_left : flux_right) += I;
            g_min = std::min(g_min, res.disk.g);
            g_max = std::max(g_max, res.disk.g);
        }
    }
    const double asym = std::max(flux_left, flux_right) /
                        std::max(1e-30, std::min(flux_left, flux_right));
    spdlog::info("beaming: flux L={:.4g} R={:.4g} ratio={:.3f}  g in [{:.3f}, {:.3f}]",
                 flux_left, flux_right, asym, g_min, g_max);

    // ---- brightness profile along the major axis (paper figure data) ------
    {
        std::ofstream csv(outdir / "brightness_profile.csv");
        csv << "x_norm,angle_rad,I_bol,g,r_disk,subring_n\n";
        for (int i = 0; i < 2000; ++i) {
            const double x = 2.0 * (i + 0.5) / 2000.0 - 1.0;
            const TraceResult res = tracer.trace(x, 0.0);
            if (res.outcome == RayOutcome::DiskHit) {
                const double I = disk.bolometric(res.disk.r, res.disk.g);
                csv << x << "," << std::tan(0.5 * cam.fov_rad) * x << "," << I
                    << "," << res.disk.g << "," << res.disk.r << ","
                    << res.disk.subring_n << "\n";
            }
        }
    }
    spdlog::info("wrote brightness_profile.csv");

    // ---- sub-ring isolation -------------------------------------------------
    spdlog::info("sub-ring analysis...");
    SubRingExtractor extractor(tracer, disk);
    std::vector<ImageRGB> maps;
    SubRingStats stats = extractor.analyze(2, size, &maps);
    for (int n = 0; n <= 2; ++n) {
        // log display scale for the faint high-order rings
        ImageRGB disp = maps[n];
        for (auto& v : disp.data)
            v = v > 0.f ? static_cast<float>(0.25 * std::log10(1.0 + 1e3 * v)) : 0.f;
        const std::string p = (outdir / ("subring_n" + std::to_string(n) + ".png")).string();
        write_png_srgb(disp, p.c_str(), 0.0);
        spdlog::info("n={}: pixels={} flux={:.6g} <angle>={:.6e} rad", n,
                     stats.bins[n].pixels, stats.bins[n].flux_bolometric,
                     stats.bins[n].mean_image_angle);
    }
    const double r01 = stats.bins[0].flux_bolometric /
                       std::max(1e-30, stats.bins[1].flux_bolometric);
    const double r12 = stats.bins[1].flux_bolometric /
                       std::max(1e-30, stats.bins[2].flux_bolometric);
    spdlog::info("flux ratios: n0/n1 = {:.2f}, n1/n2 = {:.2f}", r01, r12);

    // Band widths along four directions (measures exponential narrowing).
    for (int n : {1, 2})
        for (double psi : {0.0, 1.5707963, 3.14159265, 4.71238898})
            extractor.scan_widths(stats, n, psi, 3.5e-3, 8.0e-3);
    nlohmann::json jw = nlohmann::json::array();
    for (const auto& w : stats.widths) {
        jw.push_back({{"psi", w.psi},
                      {"n", w.n},
                      {"s_inner_rad", w.s_inner},
                      {"s_outer_rad", w.s_outer},
                      {"width_rad", w.s_outer - w.s_inner}});
        spdlog::info("band n={} psi={:.2f}: [{:.6e}, {:.6e}] width {:.3e} rad",
                     w.n, w.psi, w.s_inner, w.s_outer, w.s_outer - w.s_inner);
    }

    nlohmann::json report{
        {"a_star", a_star},
        {"incl_deg", incl},
        {"r_in_M", dcfg.r_in},
        {"r_out_M", dcfg.r_out},
        {"beaming_flux_ratio", asym},
        {"g_min", g_min},
        {"g_max", g_max},
        {"subring_flux",
         {stats.bins[0].flux_bolometric, stats.bins[1].flux_bolometric,
          stats.bins[2].flux_bolometric}},
        {"flux_ratio_n0_n1", r01},
        {"flux_ratio_n1_n2", r12},
        {"band_widths", jw},
    };
    std::ofstream(outdir / "subring_report.json") << report.dump(2) << "\n";
    spdlog::info("wrote subring_report.json");
    return 0;
}
