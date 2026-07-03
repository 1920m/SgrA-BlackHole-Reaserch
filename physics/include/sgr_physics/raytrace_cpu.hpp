#pragma once
// CPU reference ray tracer (Phases 1–3). This renderer is the ground truth the
// GPU port of Phase 7 is validated against — clarity and correctness first.
//
// Camera: pinhole at (r_cam, θ_cam) with a ZAMO (zero-angular-momentum
// observer) orthonormal tetrad — reduces to the static observer for a = 0 and
// stays well-defined inside the ergosphere. Rays are launched forward from the
// camera (null geodesics are reversible; where the ray lands is what the pixel
// sees).
//
// Geodesics: Carter/Mino second-order form (null_geodesics.hpp).
//
// Disk: geometrically thin, equatorial. Equator crossings are located by
// bisection on cosθ between accepted steps; the number of crossings *before*
// the one that hits the disk annulus is the photon sub-ring order n
// (Johnson et al. 2020) — n = 0 direct image, n = 1 first lensed image, ...

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

#include "sgr_physics/metric.hpp"
#include "sgr_physics/null_geodesics.hpp"

namespace sgr {

enum class RayOutcome : std::uint8_t {
    Captured,   // crossed (close to) the horizon
    Escaped,    // reached the escape radius; background visible
    DiskHit,    // terminated on the (opaque) disk
    MaxSteps,   // integration budget exhausted (counts as captured for shadow)
};

struct DiskHitInfo {
    double r = 0.0;        // BL radius of the equatorial crossing
    double phi = 0.0;      // BL azimuth of the crossing
    int subring_n = 0;     // equator crossings before this hit (0 = direct)
    double g = 1.0;        // redshift factor ν_obs/ν_emit (filled when disk on)
};

struct TraceResult {
    RayOutcome outcome = RayOutcome::MaxSteps;
    DiskHitInfo disk;                  // valid when outcome == DiskHit
    double escape_theta = 0.0;         // asymptotic direction when Escaped
    double escape_phi = 0.0;
    int total_equator_crossings = 0;   // diagnostic (winding count)
};

struct CameraConfig {
    double r_cam = 1000.0;      // Boyer-Lindquist radius of the camera [M]
    double incl_deg = 90.0;     // inclination from the spin axis (θ_cam)
    double fov_rad = 0.032;     // full horizontal field of view
    int width = 800;
    int height = 800;
};

struct DiskConfig {
    bool enabled = false;
    double r_in = 6.0;          // inner edge (typically r_isco)
    double r_out = 20.0;        // outer edge [M]
    bool opaque = true;
};

struct TraceOptions {
    double atol = 1e-11;
    double rtol = 1e-11;
    int max_steps = 120000;
    double h_max = 50.0;        // Mino-time step cap
};

class RaytraceCpu {
public:
    RaytraceCpu(const KerrMetric& metric, const CameraConfig& cam,
                const DiskConfig& disk, const TraceOptions& opts = {})
        : metric_(metric), cam_(cam), disk_(disk), opts_(opts) {}

    // Trace the ray for normalised image-plane coordinates
    // x ∈ [−1, 1] (right = +φ̂ of the tetrad), y ∈ [−1, 1] (up = −θ̂),
    // scaled so x = ±1 corresponds to ±fov/2 horizontally.
    TraceResult trace(double x, double y) const;

    // Initial (constants, state) for a camera ray — exposed for the GPU
    // comparison harness of Phase 7.
    NullRayInit camera_ray(double x, double y) const;

    // Shadow boundary in camera-angle space along image-plane direction ψ
    // (radians, measured from the +x axis): bisects the captured/escaped
    // transition and returns the angular radius of the boundary as seen by
    // the camera (radians). Machine-accurate, no pixel grid involved.
    double shadow_boundary_angle(double psi, double angle_max) const;

    const KerrMetric& metric() const { return metric_; }
    const CameraConfig& camera() const { return cam_; }
    const DiskConfig& disk() const { return disk_; }

private:
    KerrMetric metric_;
    CameraConfig cam_;
    DiskConfig disk_;
    TraceOptions opts_;
};

// Analytic Schwarzschild shadow angular radius for an observer at r_obs
// (Synge 1966):  sin α = √27 · M · √(1 − 2M/r_obs) / r_obs.
double schwarzschild_shadow_angle(double M, double r_obs);

// ---------------------------------------------------------------------------
// Simple float RGB image + PNG output (stb_image_write in raytrace_cpu.cpp)
// ---------------------------------------------------------------------------
struct ImageRGB {
    int width = 0, height = 0;
    std::vector<float> data; // 3 floats per pixel, row-major, top row first

    void resize(int w, int h) { width = w; height = h; data.assign(3u * w * h, 0.f); }
    float* pixel(int x, int y) { return &data[3u * (static_cast<size_t>(y) * width + x)]; }
    const float* pixel(int x, int y) const {
        return &data[3u * (static_cast<size_t>(y) * width + x)];
    }
};

// Write an 8-bit sRGB PNG (simple gamma 2.2 tonemap, exposure in EV).
bool write_png_srgb(const ImageRGB& img, const char* path, double exposure_ev = 0.0);

// Render a full frame: background starfield + optional disk (Phase 1 uses a
// simple emissive band; Phase 3 swaps in Novikov-Thorne shading via `shade`).
// `shade` maps a TraceResult to linear RGB; pass an empty function for the
// built-in Phase-1 look (procedural starfield + uniform warm disk band).
using ShadeFn =
    std::function<std::array<float, 3>(const TraceResult&, const RaytraceCpu&)>;
ImageRGB render_frame(const RaytraceCpu& tracer, ShadeFn shade = {},
                      int supersample = 1);

// The built-in starfield shading for escaped rays — exposed so custom disk
// shaders can reuse the same background.
std::array<float, 3> shade_starfield(double theta, double phi);

} // namespace sgr
