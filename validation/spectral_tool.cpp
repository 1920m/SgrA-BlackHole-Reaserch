// Gates 13–15 (extension paper, "Spectral rendering"): every number the
// extension quotes on the hypothetical thin disk's colour and brightness.
//
//   spectral_tool <outdir>     → <outdir>/spectral.json
//
// 1. The thin disk Sgr A*'s MEASURED accretion rate would power. Ṁ spans the
//    230 GHz Faraday-rotation bounds 2e-9 … 2e-7 Msun/yr (Marrone et al.
//    2007); for each Ṁ: peak effective temperature (Page-Thorne, SI
//    normalisation), its radius, the blackbody chromaticity there, and the
//    luminosity η Ṁ c² of a disk extending to infinity (η = 1 − E_ISCO).
// 2. Visual versus bolometric Doppler contrast. A camera integrates the
//    Planck spectrum at g·T through the CIE curves; only the bolometric
//    intensity scales as g⁴. Along the image major axis of the Gate 3 view
//    (r_cam = 1000 M, i = 80°, a* = 0.94, r_out = 18 M; Carter science
//    tracer) we sum both over the approaching and receding halves, at
//    6300 K (the thesis reference) and at the T_peak of each Ṁ.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "sgr_physics/constants.hpp"
#include "sgr_physics/disk_model.hpp"
#include "sgr_physics/metric.hpp"
#include "sgr_physics/raytrace_cpu.hpp"

using namespace sgr;
using nlohmann::json;

namespace {

json chroma(double T) {
    const auto xyz = planck_to_xyz(T);
    const double s = xyz[0] + xyz[1] + xyz[2];
    return {{"T", T}, {"x", xyz[0] / s}, {"y", xyz[1] / s}, {"Y_rel6500", xyz[1]}};
}

} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path out = argc > 1 ? argv[1] : "validation/extension";
    std::filesystem::create_directories(out);
    namespace C = constants;
    const double a = 0.94, r_out = 18.0;
    const KerrMetric m(1.0, a);
    const double r_isco = m.r_isco(true);
    const double eta = 1.0 - m.energy_circular(r_isco, true);
    json J;
    J["a_star"] = a;
    J["r_out"] = r_out;
    J["r_isco"] = r_isco;
    J["efficiency"] = eta;

    // ---- 1. the would-be thin disk ----------------------------------------
    const double mdots[3] = {C::sgrA_mdot_min_msun_yr, 2.0e-8, C::sgrA_mdot_max_msun_yr};
    std::vector<double> tpeaks;
    for (double mdot : mdots) {
        const double T = nt_peak_temperature_k(C::sgrA_mass_msun, mdot, a, r_out);
        // radius of the peak (same scan as the physics routine)
        double best_r = r_isco, best_F = 0.0;
        for (int i = 1; i <= 4000; ++i) {
            const double r = r_isco + (r_out - r_isco) * i / 4000.0;
            const double F = nt_flux_si(C::sgrA_mass_msun, mdot, a, r_out, r);
            if (F > best_F) { best_F = F; best_r = r; }
        }
        const double mdot_si = mdot * C::M_sun_kg / C::yr_s;          // kg/s
        const double L_w = eta * mdot_si * C::c_si * C::c_si;           // W
        // Eddington luminosity 4πGMm_p c/σ_T (CODATA 2018 m_p, σ_T)
        const double m_p = 1.67262192369e-27, sigma_T = 6.6524587321e-29;
        const double L_edd = 4.0 * 3.14159265358979323846 * C::sgrA_GM_si * m_p * C::c_si / sigma_T;
        json d = {{"mdot_msun_yr", mdot}, {"T_peak_K", T}, {"r_peak_M", best_r},
                  {"L_W", L_w}, {"L_erg_s", L_w * 1e7}, {"L_over_LEdd", L_w / L_edd},
                  {"L_Edd_erg_s", L_edd * 1e7}, {"chromaticity", chroma(T)}};
        J["thin_disk"].push_back(d);
        tpeaks.push_back(T);
    }

    // ---- 2. visual vs bolometric Doppler contrast -------------------------
    CameraConfig cam;
    cam.r_cam = 1000.0;
    cam.incl_deg = 80.0;
    cam.fov_rad = 0.032;
    DiskConfig dc;
    dc.enabled = true;
    dc.r_in = r_isco;
    dc.r_out = r_out;
    const RaytraceCpu tr(m, cam, dc);
    struct Hit { double x, r, g; };
    std::vector<Hit> hits;
    for (int i = 0; i < 800; ++i) {
        const double x = 2.0 * (i + 0.5) / 800.0 - 1.0;
        const TraceResult res = tr.trace(x, 0.0);
        if (res.outcome == RayOutcome::DiskHit) hits.push_back({x, res.disk.r, res.disk.g});
    }
    double gmin = 1e30, gmax = 0.0;
    for (const Hit& h : hits) { gmin = std::min(gmin, h.g); gmax = std::max(gmax, h.g); }
    J["major_axis"] = {{"hits", hits.size()}, {"g_min", gmin}, {"g_max", gmax}};
    std::vector<double> temps = {6300.0};
    temps.insert(temps.end(), tpeaks.begin(), tpeaks.end());
    for (int k = 0; k <= 40; ++k) temps.push_back(3000.0 * std::pow(20.0, k / 40.0)); // 3000 K … 60000 K
    for (double Tp : temps) {
        const NovikovThorneDisk disk(a, r_isco, r_out, Tp);
        double bl = 0.0, br = 0.0, vl = 0.0, vr = 0.0;
        for (const Hit& h : hits) {
            const double bolo = disk.bolometric(h.r, h.g);
            const double T = h.g * disk.temperature(h.r);
            const double vis = planck_to_xyz(T)[1];
            (h.x < 0.0 ? bl : br) += bolo;
            (h.x < 0.0 ? vl : vr) += vis;
        }
        J["contrast"].push_back({{"T_peak_K", Tp},
                                 {"bolometric_left_over_right", bl / br},
                                 {"visual_left_over_right", vl / vr},
                                 {"approaching_chromaticity", chroma(gmax * Tp)},
                                 {"receding_chromaticity", chroma(gmin * Tp)}});
    }

    const auto path = out / "spectral.json";
    std::ofstream(path) << J.dump(2) << "\n";
    std::printf("spectral_tool: wrote %s\n", path.string().c_str());
    return 0;
}
