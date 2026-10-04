#pragma once
// Sgr A* scene presets (Gate 14). Every preset is the SAME spacetime — Sgr A*
// with a* = 0.94 — seen from a different camera and lit by a
// thin Novikov-Thorne disk whose temperature is COMPUTED, not chosen:
// T_peak = nt_peak_temperature_k(M_SgrA, mdot, a*, r_out), with mdot inside
// the Faraday-rotation bounds 2e-9 … 2e-7 Msun/yr (Marrone et al. 2007).
// A thin, radiatively efficient disk is a HYPOTHETICAL for Sgr A* (its real
// flow is a hot RIAF — Gate 19); the presets show what that disk would look
// like, at physically consistent temperatures.
//
// Preset switching changes push constants and display settings only —
// geodesic and metric code paths are untouched. The Interstellar/GARGANTUA
// (a* = 0.6) looks and the 8-bit palette LUTs of Phase 8 are retired; their
// Gate-8 screenshots remain in writeup/screenshots/ as historical records.

#include <array>
#include <cstdint>

namespace sgr {

// Display style bits (disk.comp.glsl shade_mode).
inline constexpr uint32_t kStylePhysical = 0u; // exact opaque model annulus
inline constexpr uint32_t kStyleSoft = 1u;     // translucent rim, band-limited caustics
inline constexpr uint32_t kStyleClouds = 2u;   // volumetric LTE cloud layer

struct DiskPreset {
    const char* name;
    uint32_t style;        // kStyle* bits
    float a_star;
    float inclination_deg;
    float disk_outer_r;
    float exposure_ev;
    float mdot_msun_yr;    // accretion rate → physical T_peak
    float r_cam;           // camera radius [M]
    float turb;            // display filament layer 0..1 (0 = pure Page-Thorne)
    float bloom;           // glare strength (display)
    float wb_kelvin;       // camera white balance (display)
    float saturation;      // display
};

inline constexpr std::array<DiskPreset, 5> kDiskPresets{{
    // name                        style                      a*     incl  r_out  EV    mdot    r_cam turb  bloom  WB      sat
    {"SGR A* - PHYSICAL",          kStylePhysical,            0.94f, 80.0f, 18.0f,  0.0f, 2.0e-8f, 60.0f, 0.00f, 0.03f, 6500.0f, 1.00f},
    {"SGR A* - EARTH VIEW (i=30)", kStyleSoft,                0.94f, 30.0f, 18.0f, -1.0f, 2.0e-8f, 60.0f, 0.45f, 0.06f, 6500.0f, 1.25f},
    {"SGR A* - CLOSE ORBIT",       kStyleSoft | kStyleClouds, 0.94f, 84.0f, 24.0f, -0.3f, 2.0e-8f, 45.0f, 0.80f, 0.12f, 7500.0f, 1.30f},
    {"SGR A* - LOW MDOT (2e-9)",   kStyleSoft,                0.94f, 78.0f, 18.0f,  0.3f, 2.0e-9f, 60.0f, 0.55f, 0.08f, 6500.0f, 1.30f},
    {"SGR A* - HIGH MDOT (2e-7)",  kStyleSoft,                0.94f, 78.0f, 18.0f, -0.3f, 2.0e-7f, 60.0f, 0.55f, 0.08f, 6500.0f, 1.30f},
}};

} // namespace sgr
