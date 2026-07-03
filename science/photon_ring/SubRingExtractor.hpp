#pragma once
// Photon-ring substructure extraction (Johnson et al. 2020).
//
// Each ray carries the count n of equatorial-plane crossings *before* the
// crossing that hit the disk: n = 0 is the direct image, n = 1 the first
// lensed image (one extra half-orbit), n = 2 the second, ... Successive
// sub-rings converge exponentially onto the critical curve, demagnified per
// order by e^γ with γ the Lyapunov exponent of the photon shell (γ = π for
// Schwarzschild polar orbits; spin- and position-dependent in Kerr).

#include <array>
#include <string>
#include <vector>

#include "sgr_physics/disk_model.hpp"
#include "sgr_physics/raytrace_cpu.hpp"

namespace sgr {

struct SubRingStats {
    // index = sub-ring order n (orders ≥ bins.size()-1 lumped into the last)
    struct Bin {
        long long pixels = 0;
        double flux_bolometric = 0.0;   // Σ I_bol over image plane (arb. units)
        double mean_image_angle = 0.0;  // flux-weighted ⟨angle from axis⟩ [rad]
    };
    std::vector<Bin> bins;

    // Radial extents [s_min, s_max] of each sub-ring band along an image-plane
    // scan direction ψ (filled by scan_widths).
    struct BandWidth {
        double psi = 0.0;
        int n = 0;
        double s_inner = 0.0, s_outer = 0.0; // camera angles [rad]
    };
    std::vector<BandWidth> widths;
};

class SubRingExtractor {
public:
    SubRingExtractor(const RaytraceCpu& tracer, const NovikovThorneDisk& disk)
        : tracer_(&tracer), disk_(&disk) {}

    // Bolometric maps binned by sub-ring order (n = 0 .. n_max, ≥n_max lumped).
    // Optionally returns per-order grayscale images (log-scaled on write).
    SubRingStats analyze(int n_max, int image_size,
                         std::vector<ImageRGB>* images = nullptr) const;

    // Locate the inner/outer edges of sub-ring band n along direction ψ by
    // dense scan + bisection refinement. Appends into stats.widths.
    void scan_widths(SubRingStats& stats, int n, double psi, double s_lo,
                     double s_hi) const;

private:
    const RaytraceCpu* tracer_;
    const NovikovThorneDisk* disk_;
};

} // namespace sgr
