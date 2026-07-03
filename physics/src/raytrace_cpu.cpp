#include "sgr_physics/raytrace_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <execution>
#include <numeric>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "sgr_physics/integrator.hpp"

namespace sgr {

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Tetrad {
    // Basis vector BL components: e[a]^μ, a ∈ {t, r, θ, φ} (ZAMO frame).
    double et_t, et_phi; // e_t = (∂_t + ω ∂_φ)/α
    double er_r, eth_th, eph_phi;
};

Tetrad zamo_tetrad(const KerrMetric& metric, double r, double theta) {
    const Metric4<double> g = metric.components(r, theta);
    const double omega = -g(0, 3) / g(3, 3);
    const double alpha2 = -(g(0, 0) - g(0, 3) * g(0, 3) / g(3, 3));
    const double alpha = std::sqrt(alpha2);
    Tetrad t{};
    t.et_t = 1.0 / alpha;
    t.et_phi = omega / alpha;
    t.er_r = 1.0 / std::sqrt(g(1, 1));
    t.eth_th = 1.0 / std::sqrt(g(2, 2));
    t.eph_phi = 1.0 / std::sqrt(g(3, 3));
    return t;
}

// Local look direction from tangent-plane offsets: the camera looks along −r̂;
// tx > 0 is image right (+φ̂), ty > 0 is image up (−θ̂).
std::array<double, 3> local_direction(double tx, double ty) {
    const double inv = 1.0 / std::sqrt(1.0 + tx * tx + ty * ty);
    return {-inv, -ty * inv, tx * inv}; // components along (e_r, e_θ, e_φ)
}

} // namespace

double schwarzschild_shadow_angle(double M, double r_obs) {
    return std::asin(std::sqrt(27.0) * M * std::sqrt(1.0 - 2.0 * M / r_obs) / r_obs);
}

NullRayInit RaytraceCpu::camera_ray(double x, double y) const {
    const double aspect = static_cast<double>(cam_.height) / cam_.width;
    const double t_half = std::tan(0.5 * cam_.fov_rad);
    const double tx = t_half * x;
    const double ty = t_half * y * aspect;

    const double theta_cam = cam_.incl_deg * kPi / 180.0;
    const Tetrad tet = zamo_tetrad(metric_, cam_.r_cam, theta_cam);
    const auto n = local_direction(tx, ty);

    // k = e_t + n_r e_r + n_θ e_θ + n_φ e_φ  (local photon energy = 1)
    std::array<double, 4> k{};
    k[0] = tet.et_t;
    k[1] = n[0] * tet.er_r;
    k[2] = n[1] * tet.eth_th;
    k[3] = tet.et_phi + n[2] * tet.eph_phi;

    return make_null_ray(metric_, 0.0, cam_.r_cam, theta_cam, 0.0, k);
}

TraceResult RaytraceCpu::trace(double x, double y) const {
    NullRayInit init = camera_ray(x, y);
    const NullRayConstants& c = init.constants;
    NullState yv = init.state;

    auto rhs = [&c](double, const NullState& s, NullState& d) {
        null_geodesic_rhs(c, s, d);
    };
    DormandPrince54<6, decltype(rhs)> rk(rhs, opts_.atol, opts_.rtol);

    const double r_hor = metric_.r_horizon();
    const double r_capture = r_hor * 1.001 + 1e-6;
    const double r_escape = std::max(2.0 * cam_.r_cam, 1500.0);

    TraceResult out;
    double tau = 0.0, h = 1e-3;
    NullState y_prev = yv;
    double tau_prev = tau;
    int crossings = 0;

    for (int step = 0; step < opts_.max_steps; ++step) {
        y_prev = yv;
        tau_prev = tau;
        const auto res = rk.try_step(tau, yv, h);
        h = std::clamp(res.h_next, 1e-14, opts_.h_max);
        if (!res.accepted) continue;

        // Constraint projection: re-impose Carter's first-order relations
        // p_r² = R(r), p_θ² = Θ(θ), keeping the sign evolved by the
        // second-order dynamics (which handles the turning points). Without
        // this, the O(rtol·r⁴) drift accumulated on the long r ~ 1000 M leg
        // dwarfs the |p_r² − R| ~ 0.1 discrimination needed to classify rays
        // near the critical impact parameter, and the shadow boundary comes
        // out tolerance-dependent (verified by the [.probe] test).
        {
            const double Rp = radial_potential(c, yv[1]);
            if (Rp >= 0.0) yv[4] = std::copysign(std::sqrt(Rp), yv[4]);
            const double Tp = polar_potential(c, yv[2]);
            if (Tp >= 0.0) yv[5] = std::copysign(std::sqrt(Tp), yv[5]);
        }

        // --- equatorial crossing detection (disk plane θ = π/2) ---
        // The 1e-14 dead-band suppresses roundoff flicker for rays confined to
        // the equatorial plane itself (cosθ ~ ±1e-17 noise).
        const double mu_prev = std::cos(y_prev[2]);
        const double mu_now = std::cos(yv[2]);
        if (mu_prev * mu_now < 0.0 &&
            (std::abs(mu_prev) > 1e-14 || std::abs(mu_now) > 1e-14)) {
            // Bisect the sub-step to the crossing.
            double lo = 0.0, hi = tau - tau_prev;
            NullState cross = yv;
            DormandPrince54<6, decltype(rhs)> rkb(rhs, 1e30, 1e30);
            for (int it = 0; it < 40; ++it) {
                const double mid = 0.5 * (lo + hi);
                NullState trial = y_prev;
                double t0 = 0.0;
                rkb.try_step(t0, trial, mid);
                if (std::cos(trial[2]) * mu_prev > 0.0) lo = mid; else hi = mid;
                cross = trial;
            }
            const double r_x = cross[1];
            if (disk_.enabled && disk_.opaque && r_x >= disk_.r_in &&
                r_x <= disk_.r_out) {
                out.outcome = RayOutcome::DiskHit;
                out.disk.r = r_x;
                out.disk.phi = cross[3];
                out.disk.subring_n = crossings;
                // Redshift factor for a prograde circular disk element:
                // g = ν_obs/ν_em = 1 / (u^t (E − Ω L_z)), local camera energy 1.
                const double Om = metric_.omega_circular(r_x, true);
                const Metric4<double> gm = metric_.components(r_x, kPi / 2.0);
                const double denom = -(gm(0, 0) + 2.0 * Om * gm(0, 3) +
                                       Om * Om * gm(3, 3));
                const double ut = 1.0 / std::sqrt(denom);
                out.disk.g = 1.0 / (ut * (c.E - Om * c.Lz));
                out.total_equator_crossings = crossings;
                return out;
            }
            ++crossings;
        }

        if (yv[1] < r_capture) {
            out.outcome = RayOutcome::Captured;
            out.total_equator_crossings = crossings;
            return out;
        }
        if (yv[1] > r_escape && yv[4] > 0.0) {
            out.outcome = RayOutcome::Escaped;
            out.escape_theta = yv[2];
            out.escape_phi = yv[3];
            out.total_equator_crossings = crossings;
            return out;
        }
    }
    out.outcome = RayOutcome::MaxSteps;
    out.total_equator_crossings = crossings;
    return out;
}

double RaytraceCpu::shadow_boundary_angle(double psi, double angle_max) const {
    const double t_half = std::tan(0.5 * cam_.fov_rad);
    const double aspect = static_cast<double>(cam_.height) / cam_.width;
    auto captured_at = [&](double s) {
        const double tx = std::tan(s) * std::cos(psi);
        const double ty = std::tan(s) * std::sin(psi);
        // Invert the pixel mapping used by trace().
        const RayOutcome o = trace(tx / t_half, ty / (t_half * aspect)).outcome;
        return o == RayOutcome::Captured || o == RayOutcome::MaxSteps;
    };
    double lo = 1e-6; // looking (almost) dead centre: must be captured
    double hi = angle_max;
    if (!captured_at(lo)) return 0.0;   // degenerate configuration
    if (captured_at(hi)) return angle_max;
    for (int it = 0; it < 48; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (captured_at(mid)) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// ---------------------------------------------------------------------------
// Shading and frame rendering
// ---------------------------------------------------------------------------
namespace {

// Deterministic hash → [0,1)
inline float hash01(std::uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return static_cast<float>(x) * (1.0f / 4294967296.0f);
}

// Procedural starfield + faint coordinate grid so lensing is visible.
std::array<float, 3> shade_background(double theta, double phi) {
    std::array<float, 3> col{0.004f, 0.005f, 0.009f};

    // celestial grid every 10°, drawn faintly (validation aid: shows lensing)
    const double deg = 180.0 / kPi;
    const double gt = std::fmod(std::abs(theta) * deg, 10.0);
    const double gp = std::fmod(std::abs(phi) * deg, 10.0);
    if (gt < 0.12 || gt > 9.88 || gp < 0.12 || gp > 9.88) {
        col = {0.012f, 0.015f, 0.024f};
    }

    // hashed star lattice on a fine angular grid
    const double cell = 0.35 * kPi / 180.0; // ~0.35° cells
    const auto ci = static_cast<std::int32_t>(std::floor(theta / cell));
    const auto cj = static_cast<std::int32_t>(std::floor(
        std::fmod(std::fmod(phi, 2.0 * kPi) + 2.0 * kPi, 2.0 * kPi) / cell));
    const std::uint32_t hcell = static_cast<std::uint32_t>(ci * 73856093) ^
                                static_cast<std::uint32_t>(cj * 19349663);
    const float r0 = hash01(hcell);
    if (r0 > 0.82f) { // star present in this cell
        const float u = (theta / cell) - std::floor(theta / cell);
        const float v = static_cast<float>(
            std::fmod(std::fmod(phi, 2.0 * kPi) + 2.0 * kPi, 2.0 * kPi) / cell -
            std::floor(std::fmod(std::fmod(phi, 2.0 * kPi) + 2.0 * kPi, 2.0 * kPi) / cell));
        const float su = hash01(hcell * 0x9e3779b9U) * 0.6f + 0.2f;
        const float sv = hash01(hcell * 0x85ebca6bU) * 0.6f + 0.2f;
        const float d2 = (u - su) * (u - su) + (v - sv) * (v - sv);
        const float mag = hash01(hcell * 0xc2b2ae35U);
        const float bright = 0.5f + 3.5f * mag * mag * mag;
        const float sigma2 = 0.0022f * (0.4f + mag);
        const float I = bright * std::exp(-d2 / sigma2);
        const float warm = hash01(hcell * 0x27d4eb2fU);
        col[0] += I * (0.75f + 0.25f * warm);
        col[1] += I * (0.72f + 0.13f * warm);
        col[2] += I * (0.95f - 0.25f * warm);
    }
    return col;
}

// Phase-1 built-in shading: starfield background + uniform warm disk band.
std::array<float, 3> shade_phase1(const TraceResult& res, const RaytraceCpu&) {
    switch (res.outcome) {
        case RayOutcome::Escaped:
            return shade_background(res.escape_theta, res.escape_phi);
        case RayOutcome::DiskHit: {
            // simple emissive band, 1/r² falloff — replaced in Phase 3
            const float f = static_cast<float>(36.0 / (res.disk.r * res.disk.r));
            return {0.9f * f + 0.15f, 0.55f * f + 0.08f, 0.25f * f + 0.03f};
        }
        case RayOutcome::Captured:
        case RayOutcome::MaxSteps:
        default:
            return {0.f, 0.f, 0.f};
    }
}

} // namespace

ImageRGB render_frame(const RaytraceCpu& tracer, ShadeFn shade, int supersample) {
    if (!shade) shade = &shade_phase1;
    const auto& cam = tracer.camera();
    ImageRGB img;
    img.resize(cam.width, cam.height);

    std::vector<int> rows(cam.height);
    std::iota(rows.begin(), rows.end(), 0);
    const int ss = std::max(1, supersample);

    std::for_each(std::execution::par, rows.begin(), rows.end(), [&](int j) {
        for (int i = 0; i < cam.width; ++i) {
            float acc[3] = {0.f, 0.f, 0.f};
            for (int sj = 0; sj < ss; ++sj) {
                for (int si = 0; si < ss; ++si) {
                    const double px = i + (si + 0.5) / ss;
                    const double py = j + (sj + 0.5) / ss;
                    const double x = 2.0 * px / cam.width - 1.0;
                    const double y = 1.0 - 2.0 * py / cam.height;
                    const TraceResult res = tracer.trace(x, y);
                    const auto c = shade(res, tracer);
                    acc[0] += c[0]; acc[1] += c[1]; acc[2] += c[2];
                }
            }
            float* p = img.pixel(i, j);
            const float inv = 1.0f / (ss * ss);
            p[0] = acc[0] * inv; p[1] = acc[1] * inv; p[2] = acc[2] * inv;
        }
    });
    return img;
}

bool write_png_srgb(const ImageRGB& img, const char* path, double exposure_ev) {
    const float gain = static_cast<float>(std::pow(2.0, exposure_ev));
    std::vector<unsigned char> bytes(3u * img.width * img.height);
    for (size_t k = 0; k < bytes.size(); ++k) {
        float v = img.data[k] * gain;
        v = std::clamp(v, 0.0f, 1.0f);
        v = std::pow(v, 1.0f / 2.2f);
        bytes[k] = static_cast<unsigned char>(std::lround(v * 255.0f));
    }
    return stbi_write_png(path, img.width, img.height, 3, bytes.data(),
                          3 * img.width) != 0;
}

} // namespace sgr
