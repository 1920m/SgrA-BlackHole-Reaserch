#pragma once
// Visual presets: display LUT + camera/disk parameters.
// Preset switching only changes push-constant parameters and the LUT slot —
// geodesic and metric code paths are untouched (physics unchanged).

#include <array>
#include <cstdint>

namespace sgr {

struct DiskPreset {
    const char* name;
    uint32_t lut_slot;     // bindless sampled-image slot; 0 = physical colours
    float a_star;
    float inclination_deg;
    float disk_outer_r;
    float exposure_ev;
    float t_peak_k;
    float r_cam;           // camera radius [M]; wide disks need more distance
    float turb;            // artistic turbulence 0..1 (0 = pure Page-Thorne)
};

// Slot map (binding 1): 0 disk_colormap (physical), 1 firestorm, 2 gargantua,
// 3 arctic, 4 nebula, 5 cinematic. The turbulence layer multiplies the
// Page-Thorne flux with a mean-1 filament pattern sheared by the real
// Keplerian rotation — it is display-only and set to 0 for the PHYSICAL
// preset (the rule: artistic noise allowed only on top of validated
// physics). CINEMATIC SGR A* additionally enables the volumetric dust
// slab in the shader (shade_mode == 5); `turb` doubles as cloud density.
inline constexpr std::array<DiskPreset, 6> kDiskPresets{{
    {"PHYSICAL (blackbody)", 0, 0.94f, 80.0f, 18.0f, 0.0f, 6300.0f, 60.0f, 0.00f},
    {"FIRESTORM",            1, 0.94f, 80.0f, 18.0f, 0.3f, 6300.0f, 60.0f, 0.65f},
    {"GARGANTUA",            2, 0.60f, 84.0f, 22.0f, 0.4f, 6800.0f, 60.0f, 0.55f},
    {"ARCTIC",               3, 0.97f, 75.0f, 16.0f, 0.2f, 12000.0f, 60.0f, 0.50f},
    {"NEBULA",               4, 0.35f, 70.0f, 28.0f, 0.3f, 5200.0f, 90.0f, 0.60f},
    {"CINEMATIC SGR A*",     5, 0.60f, 85.0f, 30.0f, 0.4f, 6800.0f, 70.0f, 0.85f},
}};

} // namespace sgr
