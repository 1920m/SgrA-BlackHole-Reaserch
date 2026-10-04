// Gate 19 — the hot accretion flow of Sgr A* at 230 GHz: every number the
// extension paper quotes on it.
//
//   riaf_tool <outdir> [n]     → <outdir>/riaf.json, riaf_*.f32
//
// 1. Fit the electron-density normalisation n0 of the RIAF (Riaf.hpp) so the
//    230 GHz flux density equals the EHT benchmark, 2.4 Jy (EHT 2022 Paper I,
//    after Paper II and Wielgus et al. 2022).
// 2. Image at that n0; blur with the 20 µas FWHM restoring beam of the EHT
//    CLEAN reconstructions (Paper I); measure the ring the way the EHT's REx
//    does: the centre minimising the spread of the peak-brightness radii
//    over 180 position angles, diameter = twice their mean, fractional width
//    from the radial FWHM, and the brightness asymmetry A (first azimuthal
//    moment of the peak brightness). Compare with Paper I's d = 51.8 ± 2.3
//    µas, W/d ≈ 30–50 %, A ≈ 0.04–0.3.
// 3. Sensitivity: refit and remeasure for T0 ×½ and ×2, β = 3 and 30, and
//    inclinations 10° and 60° (our standard is the EHT's 30°).
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "Riaf.hpp"
#include "sgr_physics/constants.hpp"

using namespace sgr;
using nlohmann::json;

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kTargetJy = 2.4;      // EHT 2022 benchmark at 230 GHz
constexpr double kBeamUas = 20.0;      // EHT CLEAN restoring beam FWHM

struct Setup {
    RiafParams p;
    RiafCamera cam;
};

double image_flux(const Setup& s) {
    namespace C = constants;
    return riaf_flux_jy(riaf_image(RiafModel(s.p), s.cam), s.p.rg_cm, C::sgrA_dist_m * 100.0);
}

// n0 such that F_230 = 2.4 Jy: bisection on log10 n0 (F increases with n0)
double fit_n0(Setup s, int n_fit, double* F_out) {
    s.cam.n = n_fit;
    double lo = 3.0, hi = 11.0;
    for (int it = 0; it < 28; ++it) {
        const double mid = 0.5 * (lo + hi);
        s.p.n0 = std::pow(10.0, mid);
        const double F = image_flux(s);
        if (F < kTargetJy) lo = mid; else hi = mid;
        if (hi - lo < 2e-4) break;
    }
    s.p.n0 = std::pow(10.0, 0.5 * (lo + hi));
    if (F_out) *F_out = image_flux(s);
    return s.p.n0;
}

std::vector<double> blur(const std::vector<double>& im, int n, double sigma_px) {
    const int R = static_cast<int>(std::ceil(4.0 * sigma_px));
    std::vector<double> k(2 * R + 1);
    double ks = 0.0;
    for (int i = -R; i <= R; ++i) ks += (k[i + R] = std::exp(-0.5 * i * i / (sigma_px * sigma_px)));
    for (double& v : k) v /= ks;
    std::vector<double> tmp(im.size(), 0.0), out(im.size(), 0.0);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            double s = 0.0;
            for (int i = -R; i <= R; ++i) {
                const int xx = x + i;
                if (xx >= 0 && xx < n) s += k[i + R] * im[static_cast<size_t>(y) * n + xx];
            }
            tmp[static_cast<size_t>(y) * n + x] = s;
        }
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            double s = 0.0;
            for (int i = -R; i <= R; ++i) {
                const int yy = y + i;
                if (yy >= 0 && yy < n) s += k[i + R] * tmp[static_cast<size_t>(yy) * n + x];
            }
            out[static_cast<size_t>(y) * n + x] = s;
        }
    return out;
}

double sample(const std::vector<double>& im, int n, double px, double py) {
    // bilinear in pixel coordinates (centres at integer + 0.5)
    const double fx = px - 0.5, fy = py - 0.5;
    const int x0 = static_cast<int>(std::floor(fx)), y0 = static_cast<int>(std::floor(fy));
    if (x0 < 0 || y0 < 0 || x0 + 1 >= n || y0 + 1 >= n) return 0.0;
    const double tx = fx - x0, ty = fy - y0;
    auto at = [&](int x, int y) { return im[static_cast<size_t>(y) * n + x]; };
    return (1 - tx) * (1 - ty) * at(x0, y0) + tx * (1 - ty) * at(x0 + 1, y0) +
           (1 - tx) * ty * at(x0, y0 + 1) + tx * ty * at(x0 + 1, y0 + 1);
}

struct Ring {
    double d_uas = 0.0, w_uas = 0.0, A = 0.0, cx_uas = 0.0, cy_uas = 0.0, spread_uas = 0.0;
};

// REx-style ring measurement on an image with pixel size `uas_px`.
Ring measure_ring(const std::vector<double>& im, int n, double uas_px) {
    const int NA = 180;
    const double rmin = 5.0 / uas_px, rmax = 50.0 / uas_px, dr = 0.05;
    auto profile_peaks = [&](double cx, double cy, std::vector<double>& rpk,
                             std::vector<double>& ipk, std::vector<double>& wid) {
        rpk.assign(NA, 0.0);
        ipk.assign(NA, 0.0);
        wid.assign(NA, 0.0);
        for (int k = 0; k < NA; ++k) {
            const double th = 2.0 * kPi * k / NA;
            const double c = std::cos(th), s = std::sin(th);
            std::vector<double> pr;
            for (double r = 0.0; r <= rmax; r += dr) pr.push_back(sample(im, n, cx + r * c, cy - r * s));
            int best = static_cast<int>(rmin / dr);
            for (int q = best; q < static_cast<int>(pr.size()); ++q)
                if (pr[q] > pr[best]) best = q;
            rpk[k] = best * dr;
            ipk[k] = pr[best];
            // FWHM around the peak
            const double half = 0.5 * pr[best];
            int a = best, b = best;
            while (a > 0 && pr[a] > half) --a;
            while (b + 1 < static_cast<int>(pr.size()) && pr[b] > half) ++b;
            wid[k] = (b - a) * dr;
        }
    };
    double best_cx = 0.5 * n, best_cy = 0.5 * n, best_sd = 1e30;
    std::vector<double> rpk, ipk, wid;
    // Coarse-to-fine search for the centre minimising the radius spread,
    // within 10 µas of the black hole. A centre is admissible only if the
    // brightness peak is an interior maximum at every angle: on top of a
    // bright crescent every profile falls monotonically, the "peaks" all sit
    // at the inner search limit and their spread is trivially zero.
    const double box = 10.0 / uas_px;
    double step = box / 4.0, cx0 = 0.5 * n, cy0 = 0.5 * n;
    for (int level = 0; level < 6; ++level) {
        for (int i = -4; i <= 4; ++i)
            for (int j = -4; j <= 4; ++j) {
                const double cx = cx0 + i * step, cy = cy0 + j * step;
                if (std::abs(cx - 0.5 * n) > box || std::abs(cy - 0.5 * n) > box) continue;
                profile_peaks(cx, cy, rpk, ipk, wid);
                bool interior = true;
                for (double r : rpk) interior = interior && (r > rmin + 2.0 * dr);
                if (!interior) continue;
                double m = 0.0, v = 0.0;
                for (double r : rpk) m += r / NA;
                for (double r : rpk) v += (r - m) * (r - m) / NA;
                if (v < best_sd) { best_sd = v; best_cx = cx; best_cy = cy; }
            }
        cx0 = best_cx;
        cy0 = best_cy;
        step *= 0.35;
    }
    profile_peaks(best_cx, best_cy, rpk, ipk, wid);
    Ring R;
    double m = 0.0, w = 0.0, it = 0.0;
    std::complex<double> m1(0.0, 0.0);
    for (int k = 0; k < NA; ++k) {
        m += rpk[k] / NA;
        w += wid[k] / NA;
        it += ipk[k];
        const double th = 2.0 * kPi * k / NA;
        m1 += ipk[k] * std::complex<double>(std::cos(th), std::sin(th));
    }
    R.d_uas = 2.0 * m * uas_px;
    R.w_uas = w * uas_px;
    R.A = std::abs(m1) / it;
    R.cx_uas = (best_cx - 0.5 * n) * uas_px;
    R.cy_uas = (0.5 * n - best_cy) * uas_px;
    R.spread_uas = std::sqrt(best_sd) * uas_px;
    return R;
}

json ring_json(const Ring& r) {
    return {{"d_uas", r.d_uas}, {"w_uas", r.w_uas}, {"w_over_d", r.w_uas / r.d_uas},
            {"A", r.A}, {"centre_uas", {r.cx_uas, r.cy_uas}}, {"radius_spread_uas", r.spread_uas}};
}

// Visibility amplitude of the image along 36 baseline position angles
// (0°–175°), u = 0 … 10 Gλ, by direct Fourier sum; the angle-averaged
// normalised amplitude and its first two local minima. Blurring by a
// Gaussian beam multiplies the visibilities by a Gaussian, so the minima
// do not depend on the beam — the EHT measured them at |u| ≈ 3.0 and
// ≈ 6.5 Gλ (Paper I).
struct VisMinima {
    std::vector<double> u_gl, amp;   // angle-averaged |V|/V(0) (figure)
    // First local minimum of |V| along each of the 36 position angles that
    // shows one (a lopsided image fills in the nulls along its asymmetry
    // axis): how many PAs do, and the median and range of its location.
    int n_pa_with_null = 0;
    double u1_median = 0.0, u1_min = 0.0, u1_max = 0.0;   // [Gλ]
};
VisMinima visibility(const std::vector<double>& im, int n, double rad_px) {
    VisMinima vm;
    const int NU = 201, NPA = 36;
    vm.u_gl.resize(NU);
    vm.amp.assign(NU, 0.0);
    double V0 = 0.0;
    for (double v : im) V0 += v;
    std::vector<double> xs(n);
    for (int i = 0; i < n; ++i) xs[i] = (i + 0.5 - 0.5 * n) * rad_px;
    std::vector<std::vector<double>> per(NPA, std::vector<double>(NU, 0.0));
    for (int k = 0; k < NU; ++k) {
        const double u = 10e9 * k / (NU - 1);
        vm.u_gl[k] = u / 1e9;
        for (int q = 0; q < NPA; ++q) {
            const double pa = kPi * q / NPA;
            const double cu = u * std::cos(pa), su = u * std::sin(pa);
            std::complex<double> V(0.0, 0.0);
            for (int y = 0; y < n; ++y) {
                const double ys = -xs[y];                       // row 0 = top (+y)
                for (int x = 0; x < n; ++x) {
                    const double v = im[static_cast<size_t>(y) * n + x];
                    if (v == 0.0) continue;
                    const double ph = -2.0 * kPi * (cu * xs[x] + su * ys);
                    V += v * std::complex<double>(std::cos(ph), std::sin(ph));
                }
            }
            per[q][k] = std::abs(V) / V0;
            vm.amp[k] += per[q][k] / NPA;
        }
    }
    std::vector<double> u1s;
    for (int q = 0; q < NPA; ++q)
        for (int k = 1; k + 1 < NU; ++k)
            if (per[q][k] < per[q][k - 1] && per[q][k] <= per[q][k + 1]) {
                u1s.push_back(vm.u_gl[k]);
                break;
            }
    vm.n_pa_with_null = static_cast<int>(u1s.size());
    if (!u1s.empty()) {
        std::sort(u1s.begin(), u1s.end());
        vm.u1_median = u1s[u1s.size() / 2];
        vm.u1_min = u1s.front();
        vm.u1_max = u1s.back();
    }
    return vm;
}

void write_f32(const std::filesystem::path& p, const std::vector<double>& v) {
    std::vector<float> f(v.size());
    for (size_t i = 0; i < v.size(); ++i) f[i] = static_cast<float>(v[i]);
    std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char*>(f.data()),
                                              f.size() * sizeof(float));
}

json run(const std::string& name, Setup s, int n, int n_fit, double uas_per_M,
         const std::filesystem::path& out, bool save) {
    namespace C = constants;
    double F_fit = 0.0;
    s.p.n0 = fit_n0(s, n_fit, &F_fit);
    s.cam.n = n;
    const RiafModel model(s.p);
    const RiafImage img = riaf_image(model, s.cam);
    const double F = riaf_flux_jy(img, s.p.rg_cm, C::sgrA_dist_m * 100.0);
    const double uas_px = 2.0 * s.cam.half_width_M / n * uas_per_M;
    const auto bl = blur(img.I, n, kBeamUas / 2.354820045 / uas_px);
    const Ring r_raw = measure_ring(img.I, n, uas_px);
    const Ring r_blur = measure_ring(bl, n, uas_px);
    // peak brightness temperature (Rayleigh–Jeans), unblurred and blurred
    const double nu = s.cam.nu_obs;
    const double rj = cgs::c * cgs::c / (2.0 * cgs::k_B * nu * nu);
    const double Tb = *std::max_element(img.I.begin(), img.I.end()) * rj;
    const double Tb_bl = *std::max_element(bl.begin(), bl.end()) * rj;
    // electron temperature and density at the ISCO in the plane (for the text)
    const RiafFluid fi = model.fluid(model.r_isco() * 1.0001, 0.5 * kPi);
    const double rad_px = 2.0 * s.cam.half_width_M / n * s.p.rg_cm / (C::sgrA_dist_m * 100.0);
    const VisMinima vm = visibility(img.I, n, rad_px);
    if (save) {
        write_f32(out / ("riaf_" + name + ".f32"), img.I);
        write_f32(out / ("riaf_" + name + "_blur.f32"), bl);
        std::ofstream csv(out / ("riaf_" + name + "_vis.csv"));
        csv << "u_glambda,amp\n";
        for (size_t k = 0; k < vm.u_gl.size(); ++k) csv << vm.u_gl[k] << "," << vm.amp[k] << "\n";
    }
    std::printf("[riaf] %-10s n0 = %.3e cm^-3  F = %.3f Jy  d_raw = %.2f  d_blur = %.2f uas  "
                "W/d = %.2f  A = %.3f  first null %.2f Glambda (%d PAs)\n",
                name.c_str(), s.p.n0, F, r_raw.d_uas, r_blur.d_uas, r_blur.w_uas / r_blur.d_uas,
                r_blur.A, vm.u1_median, vm.n_pa_with_null);
    return {{"name", name}, {"incl_deg", s.cam.incl_deg}, {"T0", s.p.T0}, {"beta", s.p.beta},
            {"n0", s.p.n0}, {"flux_fit_jy", F_fit}, {"flux_jy", F}, {"n_pixels", n},
            {"half_width_M", s.cam.half_width_M}, {"uas_per_px", uas_px}, {"max_dH", img.max_dH},
            {"ring_raw", ring_json(r_raw)}, {"ring_blur", ring_json(r_blur)},
            {"vis_null1_median_glambda", vm.u1_median}, {"vis_null1_min_glambda", vm.u1_min},
            {"vis_null1_max_glambda", vm.u1_max}, {"vis_pa_with_null", vm.n_pa_with_null},
            {"Tb_peak_K", Tb}, {"Tb_peak_blur_K", Tb_bl},
            {"isco_n_e", fi.n_e}, {"isco_T_e", fi.T_e}, {"isco_B", fi.B}};
}
} // namespace

int main(int argc, char** argv) {
    namespace C = constants;
    const std::filesystem::path out = argc > 1 ? argv[1] : "validation/extension";
    const int n = argc > 2 ? std::atoi(argv[2]) : 160;
    const int n_fit = std::max(48, n / 2);
    std::filesystem::create_directories(out);

    Setup base;
    base.p.a_star = 0.94;
    base.p.rg_cm = C::sgrA_rg_m * 100.0;
    base.cam.incl_deg = 30.0;
    base.cam.half_width_M = 12.0;
    base.cam.nu_obs = 230e9;
    const double uas_per_M = C::sgrA_uas_per_rg;

    json J;
    J["target_jy"] = kTargetJy;
    J["beam_uas"] = kBeamUas;
    J["nu_obs_hz"] = base.cam.nu_obs;
    J["uas_per_M"] = uas_per_M;
    J["main"] = run("main", base, n, n_fit, uas_per_M, out, true);

    // sensitivity (refit n0 each time, lower resolution)
    const int ns = std::max(64, (n * 3) / 4);
    auto variant = [&](const std::string& nm, auto&& mod) {
        Setup s = base;
        mod(s);
        J["sensitivity"].push_back(run(nm, s, ns, n_fit, uas_per_M, out, false));
    };
    variant("T0_half", [](Setup& s) { s.p.T0 *= 0.5; });
    variant("T0_double", [](Setup& s) { s.p.T0 *= 2.0; });
    variant("beta_3", [](Setup& s) { s.p.beta = 3.0; });
    variant("beta_30", [](Setup& s) { s.p.beta = 30.0; });
    variant("incl_10", [](Setup& s) { s.cam.incl_deg = 10.0; });
    variant("incl_60", [](Setup& s) { s.cam.incl_deg = 60.0; });
    // resolution check of the main case
    J["main_lowres"] = run("main_lowres", base, ns, n_fit, uas_per_M, out, false);

    // ---- Gate 20: polarisation --------------------------------------------
    // Three field geometries (n0 refitted to 2.4 Jy for each: the emissivity
    // depends on the pitch angle), each seen at i = 30° (counter-clockwise
    // flow on the sky) and 150° (clockwise, the sense of the GRAVITY flares).
    // Constant intrinsic polarisation fraction p0: ∠β2 and |β2|/|β1| do not
    // depend on it; |β2|, ⟨|m|⟩ and |m_net| scale with it. No Faraday effects.
    const double p0 = 0.7;
    J["pol_fraction"] = p0;
    const int np = std::max(64, (n * 3) / 4);
    for (FieldGeometry fg : {FieldGeometry::Toroidal, FieldGeometry::Radial, FieldGeometry::Vertical}) {
        const char* fname = fg == FieldGeometry::Toroidal ? "toroidal"
                          : fg == FieldGeometry::Radial ? "radial" : "vertical";
        Setup s = base;
        s.p.field = fg;
        double F = 0.0;
        s.p.n0 = fit_n0(s, n_fit, &F);
        for (double inc : {30.0, 150.0}) {
            s.cam.incl_deg = inc;
            s.cam.n = np;
            const RiafImage img = riaf_image(RiafModel(s.p), s.cam, p0);
            const double uas_px = 2.0 * s.cam.half_width_M / np * uas_per_M;
            const double sig = kBeamUas / 2.354820045 / uas_px;
            const auto Ib = blur(img.I, np, sig), Qb = blur(img.Q, np, sig), Ub = blur(img.U, np, sig);
            auto modes = [&](const std::vector<double>& I, const std::vector<double>& Q,
                             const std::vector<double>& U) {
                const auto b1 = beta_mode(I, Q, U, np, s.cam.half_width_M, 1);
                const auto b2 = beta_mode(I, Q, U, np, s.cam.half_width_M, 2);
                double sI = 0.0, sQ = 0.0, sU = 0.0, sP = 0.0;
                for (size_t q = 0; q < I.size(); ++q) {
                    sI += I[q];
                    sQ += Q[q];
                    sU += U[q];
                    sP += std::hypot(Q[q], U[q]);
                }
                return json{{"beta1_abs", std::abs(b1)}, {"beta2_abs", std::abs(b2)},
                            {"beta2_arg_deg", std::arg(b2) * 180.0 / kPi},
                            {"beta2_over_beta1", std::abs(b2) / std::abs(b1)},
                            {"m_avg", sP / sI}, {"m_net", std::hypot(sQ, sU) / sI}};
            };
            json e = {{"field", fname}, {"incl_deg", inc}, {"n0", s.p.n0}, {"flux_fit_jy", F},
                      {"n_pixels", np}, {"raw", modes(img.I, img.Q, img.U)},
                      {"blurred", modes(Ib, Qb, Ub)}};
            std::printf("[pol]  %-8s i=%3.0f  beta2 %.3f at %+7.1f deg (blurred %.3f at %+7.1f)  "
                        "<|m|> %.3f  m_net %.3f\n",
                        fname, inc, e["raw"]["beta2_abs"].get<double>(),
                        e["raw"]["beta2_arg_deg"].get<double>(),
                        e["blurred"]["beta2_abs"].get<double>(),
                        e["blurred"]["beta2_arg_deg"].get<double>(),
                        e["blurred"]["m_avg"].get<double>(), e["blurred"]["m_net"].get<double>());
            J["polarisation"].push_back(e);
            {
                // maps for the figure (blurred to the EHT beam, as displayed)
                const std::string tag = std::string("riaf_pol_") + fname + "_i" +
                                        std::to_string(static_cast<int>(inc));
                write_f32(out / (tag + "_I.f32"), Ib);
                write_f32(out / (tag + "_Q.f32"), Qb);
                write_f32(out / (tag + "_U.f32"), Ub);
            }
        }
    }

    const auto path = out / "riaf.json";
    std::ofstream(path) << J.dump(2) << "\n";
    std::printf("riaf_tool: wrote %s\n", path.string().c_str());
    return 0;
}
