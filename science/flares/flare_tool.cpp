// Gate 18 — hot-spot flares of Sgr A*: every number the extension paper
// quotes on them.
//
//   flare_tool <outdir> [n]      → <outdir>/flares.json, flare_*.csv, flare_*.f32
//
// Inputs from the literature (cited in the paper, nothing tuned):
//   * GRAVITY Collaboration 2018 (A&A 618, L10): best-fitting hot-spot orbits
//     R ≈ 7 ± 0.5 R_g, i ≈ 160° ± 10° (clockwise on the sky), P = 40 ± 8 min;
//     loops of max-to-min amplitude ~110–140 µas.
//   * GRAVITY Collaboration 2020 (A&A 635, A143): all three flares
//     consistent with R ≈ 9 R_g at i ≈ 140°; Gaussian emissivity, fiducial
//     size s = 0.5 M; observed loops systematically LARGER than geodesic
//     hot-spot models predict.
// Our spacetime is the thesis' Sgr A*: a* = 0.94 and the GRAVITY 2022 mass
// and distance (µas per M from sgr_physics::constants).
//
// For each (R, i) case the tool traces one fp64 Kerr-Schild ray table
// (science/flares/Hotspot.cpp), then evaluates the slow-light image of the
// orbiting spot at 256 observing times over one orbit: light curve, flux-
// weighted centroid track (the GRAVITY astrometric observable), and its
// size and speed. Echo delays: a short flash (τ = 1 M) at eight orbital
// phases; the arrival-time difference between the direct (n = 0) and first
// lensed (n = 1) images.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "Hotspot.hpp"
#include "sgr_physics/constants.hpp"

using namespace sgr;
using nlohmann::json;

namespace {
constexpr double kPi = 3.14159265358979323846;

struct Case {
    std::string name;
    double R, incl;
    std::string source;
};

// The physical light curve and centroid track are smooth functions of the
// orbital phase; the image grid adds pixel-scale jitter (the thin, unresolved
// lensed n >= 1 arcs crossing pixel boundaries), which shrinks as 1/n. All
// derived observables use the first K harmonics of the one-orbit series; the
// rms of what the filter removes is reported as the numerical noise.
std::vector<double> lowpass(const std::vector<double>& v, int K) {
    const int N = static_cast<int>(v.size());
    std::vector<double> out(N, 0.0);
    for (int k = 0; k <= K; ++k) {
        double re = 0.0, im = 0.0;
        for (int j = 0; j < N; ++j) {
            const double ph = 2.0 * kPi * k * j / N;
            re += v[j] * std::cos(ph);
            im += v[j] * std::sin(ph);
        }
        const double w = (k == 0 ? 1.0 : 2.0) / N;
        for (int j = 0; j < N; ++j) {
            const double ph = 2.0 * kPi * k * j / N;
            out[j] += w * (re * std::cos(ph) + im * std::sin(ph));
        }
    }
    return out;
}

double rms_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(s / a.size());
}

std::vector<float> to_f32(const std::vector<double>& v) {
    std::vector<float> f(v.size());
    for (size_t i = 0; i < v.size(); ++i) f[i] = static_cast<float>(v[i]);
    return f;
}

// Peak time of a sampled curve: parabolic refinement around the maximum.
double peak_time(const std::vector<double>& t, const std::vector<double>& f) {
    const size_t k = std::max_element(f.begin(), f.end()) - f.begin();
    if (k == 0 || k + 1 >= f.size()) return t[k];
    const double y0 = f[k - 1], y1 = f[k], y2 = f[k + 1];
    const double den = y0 - 2.0 * y1 + y2;
    const double off = (den != 0.0) ? 0.5 * (y0 - y2) / den : 0.0;
    return t[k] + off * (t[k + 1] - t[k]);
}

json run_case(const Case& c, int n, const std::filesystem::path& out, double uas_per_M,
              double min_per_M, bool save_images) {
    HotspotParams hp;
    hp.a_star = 0.94;
    hp.R = c.R;
    hp.sigma = 0.5;
    FlareCamera cam;
    cam.r_cam = 1000.0;
    cam.incl_deg = c.incl;
    cam.half_width_M = 14.0;
    cam.n = n;
    std::printf("[flare] %s: tracing %d^2 rays (R = %.2f M, i = %.0f deg)\n",
                c.name.c_str(), n, c.R, c.incl);
    const FlareRayTable T = build_flare_table(hp, cam);
    const Hotspot hs(hp);
    const double P = hs.period();
    // spectral-index sensitivity: alpha = 0.6, the GRAVITY 2018 Appendix C example
    HotspotParams hp6 = hp;
    hp6.alpha = 0.6;
    const Hotspot hs6(hp6);

    // ---- one orbit of steady emission ------------------------------------
    const int NT = 256;
    std::vector<double> t(NT), F(NT), F0(NT), F1(NT), F6(NT), cx(NT), cy(NT);
    std::vector<double> mean_img(static_cast<size_t>(n) * n, 0.0);
    for (int k = 0; k < NT; ++k) {
        t[k] = P * k / NT;
        const auto img = flare_image(T, hs, t[k]);
        const auto img0 = flare_image(T, hs, t[k], 0.0, 0.0, 0x1u);
        const FlareMoments m = flare_moments(T, img);
        F[k] = m.flux;
        F0[k] = flare_moments(T, img0).flux;
        F1[k] = F[k] - F0[k];
        F6[k] = flare_moments(T, flare_image(T, hs6, t[k])).flux;
        cx[k] = m.cx;
        cy[k] = m.cy;
        for (size_t p = 0; p < img.size(); ++p) mean_img[p] += img[p] / NT;
        if (save_images && (k % 64) == 0) {
            const auto path = out / ("flare_" + c.name + "_k" + std::to_string(k) + ".f32");
            const auto fl = to_f32(img);
            std::ofstream(path, std::ios::binary)
                .write(reinterpret_cast<const char*>(fl.data()), fl.size() * sizeof(float));
        }
    }
    if (save_images) {
        const auto fl = to_f32(mean_img);
        std::ofstream(out / ("flare_" + c.name + "_mean.f32"), std::ios::binary)
            .write(reinterpret_cast<const char*>(fl.data()), fl.size() * sizeof(float));
    }
    {
        std::ofstream csv(out / ("flare_" + c.name + ".csv"));
        csv << "t_M,t_min,flux,flux_n0,flux_n1,flux_alpha06,cx_M,cy_M,cx_uas,cy_uas\n";
        for (int k = 0; k < NT; ++k)
            csv << t[k] << "," << t[k] * min_per_M << "," << F[k] << "," << F0[k] << ","
                << F1[k] << "," << F6[k] << "," << cx[k] << "," << cy[k] << ","
                << cx[k] * uas_per_M << "," << cy[k] * uas_per_M << "\n";
    }
    // derived observables from the band-limited series (see lowpass())
    const int K = 12;
    const auto Fs = lowpass(F, K), F6s = lowpass(F6, K);
    const auto cxs = lowpass(cx, K), cys = lowpass(cy, K);
    double Fmean = 0.0;
    for (double v : F) Fmean += v / NT;
    const double noise_flux = rms_diff(F, Fs) / Fmean;
    std::vector<double> rres(NT);
    for (int k = 0; k < NT; ++k) rres[k] = std::hypot(cx[k] - cxs[k], cy[k] - cys[k]);
    const double noise_centroid = rms_diff(rres, std::vector<double>(NT, 0.0));
    const double Fmax = *std::max_element(Fs.begin(), Fs.end());
    const double Fmin = *std::min_element(Fs.begin(), Fs.end());
    const auto [xmin, xmax] = std::minmax_element(cxs.begin(), cxs.end());
    const auto [ymin, ymax] = std::minmax_element(cys.begin(), cys.end());
    // largest separation of any two centroid positions (loop "diameter"),
    // path length, and GRAVITY's own kinematic measure: the largest centroid
    // displacement within a 30-minute window ("~120 uas over ~30 minutes")
    const int w30 = static_cast<int>(std::lround(30.0 / min_per_M / (P / NT)));
    double dmax = 0.0, path = 0.0, chord30 = 0.0;
    for (int a2 = 0; a2 < NT; ++a2) {
        for (int b = a2 + 1; b < NT; ++b)
            dmax = std::max(dmax, std::hypot(cxs[a2] - cxs[b], cys[a2] - cys[b]));
        const int b = (a2 + 1) % NT;
        path += std::hypot(cxs[b] - cxs[a2], cys[b] - cys[a2]);
        const int c30 = (a2 + w30) % NT;
        chord30 = std::max(chord30, std::hypot(cxs[c30] - cxs[a2], cys[c30] - cys[a2]));
    }
    double F0m = 0.0, F1m = 0.0, mcx = 0.0, mcy = 0.0, area2 = 0.0;
    for (int k = 0; k < NT; ++k) {
        F0m += F0[k] / NT;
        F1m += F1[k] / NT;
        mcx += cxs[k] / NT;
        mcy += cys[k] / NT;
        const int b = (k + 1) % NT;
        area2 += cxs[k] * cys[b] - cxs[b] * cys[k];   // shoelace: < 0 clockwise
    }
    const double F6max = *std::max_element(F6s.begin(), F6s.end());
    const double F6min = *std::min_element(F6s.begin(), F6s.end());

    // ---- echoes: flashes at 8 orbital phases --------------------------------
    const double tau = 1.0;
    json echoes = json::array();
    double dmin = 1e30, dmaxe = -1e30, dsum = 0.0;
    int ne = 0;
    for (int q = 0; q < 8; ++q) {
        // flash when the centre's KS azimuth is 2π q/8 (φ = 0: toward the camera)
        const double t_flash = (2.0 * kPi * q / 8.0 - hp.phase0) / hs.omega();
        // direct light from the spot reaches the camera ~r_cam later; scan
        // the window that contains both the n = 0 and n = 1 arrivals
        const double t_lo = t_flash - T.t_cam + 960.0, t_hi = t_flash - T.t_cam + 1120.0;
        const int NS = 1281;
        std::vector<double> ts(NS);
        for (int k = 0; k < NS; ++k) ts[k] = t_lo + (t_hi - t_lo) * k / (NS - 1);
        const auto fc = flare_flash_curves(T, hs, t_flash, tau, ts, 1);
        const std::vector<double>& f0 = fc[0];
        const std::vector<double>& f1 = fc[1];
        const double t0 = peak_time(ts, f0), t1 = peak_time(ts, f1);
        if (q == 0 && save_images) {
            // the flash seen in the direct and first lensed image (figure)
            std::ofstream csv(out / ("flare_" + c.name + "_flash0.csv"));
            csv << "t_obs_M,t_min,flux_n0,flux_n1\n";
            for (int k = 0; k < NS; ++k)
                csv << ts[k] - t_lo << "," << (ts[k] - t_lo) * min_per_M << "," << f0[k]
                    << "," << f1[k] << "\n";
        }
        const double p0 = *std::max_element(f0.begin(), f0.end());
        const double p1 = *std::max_element(f1.begin(), f1.end());
        const double delay = t1 - t0;
        echoes.push_back({{"phase_deg", 45.0 * q}, {"t0_M", t0}, {"t1_M", t1},
                          {"delay_M", delay}, {"delay_min", delay * min_per_M},
                          {"peak_ratio_n1_n0", p1 / p0}});
        dmin = std::min(dmin, delay);
        dmaxe = std::max(dmaxe, delay);
        dsum += delay;
        ++ne;
    }

    json j;
    j["name"] = c.name;
    j["source"] = c.source;
    j["R_M"] = c.R;
    j["incl_deg"] = c.incl;
    j["a_star"] = hp.a_star;
    j["sigma_M"] = hp.sigma;
    j["n_pixels"] = n;
    j["half_width_M"] = cam.half_width_M;
    j["samples"] = T.samples.size();
    j["max_dH"] = T.max_dH;
    j["max_dL_rel"] = T.max_dL;
    j["period_M"] = P;
    j["period_min"] = P * min_per_M;
    j["ut"] = hs.ut();
    j["omega"] = hs.omega();
    j["flux_max_over_min"] = Fmax / Fmin;
    j["flux_max_over_min_alpha06"] = F6max / F6min;
    j["loop_sense"] = area2 < 0.0 ? "clockwise" : "counter-clockwise";
    j["loop_centre_offset_uas"] = std::hypot(mcx, mcy) * uas_per_M;
    j["flux_n1_over_n0_mean"] = F1m / F0m;
    j["centroid_x_range_uas"] = (*xmax - *xmin) * uas_per_M;
    j["centroid_y_range_uas"] = (*ymax - *ymin) * uas_per_M;
    j["centroid_diameter_uas"] = dmax * uas_per_M;
    j["centroid_diameter_M"] = dmax;
    j["centroid_path_uas"] = path * uas_per_M;
    j["centroid_speed_mean_c"] = path / P;
    j["centroid_chord30_uas"] = chord30 * uas_per_M;
    j["centroid_chord30_speed_c"] = chord30 / (w30 * P / NT);
    j["noise_flux_rel"] = noise_flux;
    j["noise_centroid_uas"] = noise_centroid * uas_per_M;
    j["lowpass_harmonics"] = K;
    j["orbit_diameter_uas"] = 2.0 * c.R * uas_per_M;
    j["echo"] = echoes;
    j["echo_delay_min_M"] = dmin;
    j["echo_delay_max_M"] = dmaxe;
    j["echo_delay_mean_M"] = dsum / ne;
    j["echo_delay_mean_min"] = dsum / ne * min_per_M;
    std::printf("[flare] %s: P = %.1f min, loop %.1f uas (orbit %.1f), F max/min %.3f, "
                "echo %.1f-%.1f M\n",
                c.name.c_str(), P * min_per_M, dmax * uas_per_M, 2.0 * c.R * uas_per_M,
                Fmax / Fmin, dmin, dmaxe);
    return j;
}
} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path out = argc > 1 ? argv[1] : "validation/extension";
    const int n = argc > 2 ? std::atoi(argv[2]) : 256;
    std::filesystem::create_directories(out);

    const double uas_per_M = constants::sgrA_uas_per_rg;
    const double min_per_M = constants::sgrA_tg_s / 60.0;

    // Orbit radius matched to the GRAVITY 2018 best-fit period P = 40 min
    // for OUR spacetime (a* = 0.94, GRAVITY 2022 mass): P = 2π(R^{3/2} + a)·t_g.
    const double a = 0.94;
    const double P40 = 40.0 / min_per_M;
    const double R40 = std::pow(P40 / (2.0 * kPi) - a, 2.0 / 3.0);

    const std::vector<Case> cases = {
        {"g18_R7_i160", 7.0, 160.0, "GRAVITY 2018 best fit (R ~ 7 R_g, i ~ 160 deg)"},
        {"g20_R9_i140", 9.0, 140.0, "GRAVITY 2020 global fit (R ~ 9 R_g, i ~ 140 deg)"},
        {"p40_i160", R40, 160.0, "radius whose period is the GRAVITY 2018 P = 40 min"},
    };
    json J;
    J["uas_per_M"] = uas_per_M;
    J["min_per_M"] = min_per_M;
    J["R_for_P40"] = R40;
    for (const auto& c : cases) J["cases"].push_back(run_case(c, n, out, uas_per_M, min_per_M, true));
    // numerical uncertainty: the first case again at half the resolution
    {
        Case half = cases[0];
        half.name += "_half";
        const json h = run_case(half, n / 2, out, uas_per_M, min_per_M, false);
        const json& f = J["cases"][0];
        json rc;
        rc["n_full"] = n;
        rc["n_half"] = n / 2;
        for (const char* k : {"centroid_diameter_uas", "flux_max_over_min",
                              "flux_max_over_min_alpha06", "centroid_chord30_uas",
                              "echo_delay_mean_M", "flux_n1_over_n0_mean"})
            rc[std::string("rel_change_") + k] =
                std::abs(f[k].get<double>() - h[k].get<double>()) / std::abs(f[k].get<double>());
        J["resolution_check"] = rc;
        std::filesystem::remove(out / ("flare_" + half.name + ".csv"));
    }
    const auto path = out / "flares.json";
    std::ofstream(path) << J.dump(2) << "\n";
    std::printf("flare_tool: wrote %s\n", path.string().c_str());
    return 0;
}
