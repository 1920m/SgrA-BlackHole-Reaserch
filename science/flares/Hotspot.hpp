#pragma once
// Orbiting hot spot near Sgr A* (Gate 18): the time-dependent emitter behind
// the GRAVITY near-infrared flares (GRAVITY Collaboration 2018, A&A 618,
// L10; 2020, A&A 635, A143).
//
// Emitter. A Gaussian blob of emissivity ε ∝ exp(−d²/2s²) — the GRAVITY 2020
// model, fiducial s = 0.5 M — whose centre moves on the prograde circular
// equatorial geodesic of BL radius R: Ω = 1/(R^{3/2} + a) (M = 1), and which
// moves rigidly with the centre's 4-velocity u = u^t (∂_t + Ω ∂_φ). The
// distance d is the comoving (rest-frame) distance at fixed coordinate time,
// d² = (g_μν + u_μ u_ν) Δx^μ Δx^ν with Δx^t = 0, evaluated at the centre.
//
// Kerr-Schild bookkeeping. The spot centre is described by its KS azimuth
// φ_K(t) = φ_0 + Ω t (t = KS time; ∂_t and ∂_φ are the same Killing fields in
// BL and KS, so Ω is the same in both charts) at KS Cartesian position
// x + i y = (R + i s a) e^{i φ_K}, z = 0.
//
// Observed intensity. Optically thin, j_ν ∝ ν^−α. Along a backward ray whose
// momentum is normalised so the camera measures ν = 1 (P·u_cam = 1, as in
// sgr_physics::ks_observer_ray), I_ν/ν³ is conserved and
//     I_obs = ∫ g^{2+α} ε dλ,      g = ν_obs/ν_emit = 1/(u^t (P_t + Ω L_z))
// (P_t and L_z = x P_y − y P_x are constants of the ray, so g is one number
// per ray for a rigidly rotating emitter). α = 0 is the GRAVITY 2020
// prescription F = ∫ D² ε ds.
//
// Slow light. Each backward ray carries its KS coordinate time; light
// reaching the camera at time t_obs left a sample point at t_obs + Δt
// (Δt < 0), when the spot was where it was then.

#include <array>
#include <cstdint>
#include <vector>

#include "sgr_physics/kerr_schild.hpp"

namespace sgr {

struct HotspotParams {
    double a_star = 0.94;
    double R = 7.0;        // orbit radius (Boyer-Lindquist) [M]
    double sigma = 0.5;    // Gaussian size s [M]
    double phase0 = 0.0;   // KS azimuth of the centre at KS time 0 [rad]
    double alpha = 0.0;    // spectral index, j_ν ∝ ν^−α
};

class Hotspot {
public:
    explicit Hotspot(const HotspotParams& p);

    const HotspotParams& params() const { return p_; }
    double omega() const { return Om_; }      // dφ/dt [1/M]
    double ut() const { return ut_; }         // u^t of the orbit
    double period() const;                    // 2π/Ω [M]
    // KS Cartesian position of the centre at KS time t.
    std::array<double, 3> centre(double t) const;
    // Comoving distance² of the KS point X = (t, x, y, z) from the centre
    // at the same KS time.
    double distance2(const Vec4& X) const;
    double emissivity(const Vec4& X) const;   // exp(−d²/2s²)
    // ν_obs/ν_emit for a backward ray with KS covariant momentum P at X.
    double redshift(const Vec4& X, const Vec4& P) const;
    // Comoving spatial metric at the KS-azimuth-0 centre, (h00, h01, h02,
    // h11, h12, h22) — what the GPU twin contracts displacements with.
    std::array<double, 6> comoving_metric() const {
        return {h_[0][0], h_[0][1], h_[0][2], h_[1][1], h_[1][2], h_[2][2]};
    }

private:
    HotspotParams p_;
    KerrSchild ks_;
    double Om_, ut_;
    double h_[3][3];   // comoving spatial metric g + u⊗u at the KS-azimuth-0 centre
};

// Camera for flare images: a ZAMO at BL (r_cam, incl) looking at the hole,
// pixels on a square grid of half-width `half_width_M` in impact-parameter
// units (b = r_cam · tan(angle), the units the thesis calibration converts
// to µas).
struct FlareCamera {
    double r_cam = 1000.0;
    double incl_deg = 160.0;
    double half_width_M = 14.0;
    int n = 128;
};

// Every backward ray's passage through the region where the spot can be
// (|r − R| ≤ 6s, |z| ≤ 6s), sampled densely (spacing ≤ s/5) and stored once:
// the spacetime is stationary, so one table serves every observing time and
// emission model.
struct FlareRayTable {
    struct Sample {
        float dt;           // KS time of the sample minus the camera's [M]
        float x, y, z;      // KS position
        float dlam;         // affine-parameter weight
        uint8_t order;      // passage index: 0 direct image, 1, 2 lensed
    };
    FlareCamera cam;
    HotspotParams geometry;          // R, s, a* the table was built for
    double t_cam = 0.0;              // KS time of the camera event
    std::vector<uint32_t> first;     // per pixel: first sample index (n² + 1)
    std::vector<Sample> samples;
    std::vector<double> P_t, Lz;     // per pixel: ray constants
    std::vector<uint8_t> captured;   // per pixel: ray ended in the hole
    double max_dH = 0.0;             // worst |H|/E² over all rays
    double max_dL = 0.0;             // worst relative L_z drift
};
FlareRayTable build_flare_table(const HotspotParams& geometry,
                                const FlareCamera& cam);

// Image of the spot at observing time t_obs (KS time at the camera), n×n,
// row 0 = top. Optional time profile: a flash exp(−(t_e − t_flash)²/2τ²)
// (τ ≤ 0: steady emission). order_mask selects image orders (bit k = order k).
std::vector<double> flare_image(const FlareRayTable& T, const Hotspot& hs,
                                double t_obs, double t_flash = 0.0,
                                double tau = 0.0, unsigned order_mask = 0xffu);

// Light curves of a short flash — emission ∝ exp(−(t_e − t_flash)²/2τ²) —
// split by image order (0 … max_order), on a UNIFORM observing-time grid.
// Only samples whose emission window overlaps the flash are visited, so
// one call costs about one image.
std::vector<std::vector<double>> flare_flash_curves(const FlareRayTable& T,
                                                    const Hotspot& hs,
                                                    double t_flash, double tau,
                                                    const std::vector<double>& t_obs,
                                                    int max_order = 2);

struct FlareMoments {
    double flux = 0.0;               // Σ I (pixel units; relative)
    double cx = 0.0, cy = 0.0;       // flux-weighted centroid [M], +x right, +y up
};
FlareMoments flare_moments(const FlareRayTable& T, const std::vector<double>& img);

} // namespace sgr
