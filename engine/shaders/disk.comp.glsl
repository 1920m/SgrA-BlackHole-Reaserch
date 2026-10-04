// Live renderer: fp32 Kerr-Schild Hamiltonian geodesics (Gate 16) +
// Novikov-Thorne disk shaded as the visual blackbody at g·T (Gate 13) +
// lensed Milky Way (Gate 15), written as LINEAR radiance into a progressive
// accumulator for the HDR display chain (Gate 14).
//
// Phases 8–15 integrated the Carter/Mino equations in Boyer-Lindquist
// coordinates here; their spin-axis singularity needed clamps and
// heuristics that still left a visible seam, and their camera launched the
// emitted rather than the received photon (a mirror image in Kerr). The
// Cartesian Kerr-Schild form has no pole, is regular across the horizon
// these backward rays reach, and is validated by its fp64 CPU twin
// (physics/src/kerr_schild.cpp, test_kerr_schild.cpp). Science-grade output
// remains the fp64 shader / CPU pair validated in Gate 7.
//
// Bindless: binding 0 = storage images, binding 1 = sampled LUTs
//   slot 0: blackbody radiance table, RGBA32F, built on the CPU from
//           sgr_physics::planck_to_linear_srgb (app.cpp load_luts) —
//           absolute linear sRGB, log T 300 K → 1e5 K, texelFetch only
//   slot 1–4: preset display LUTs (§9): firestorm, gargantua, arctic, nebula
#version 450

layout(local_size_x = 8, local_size_y = 8) in;

// Output: LINEAR radiance into the rgba32f accumulator (slot pc.image_index);
// bloom/tonemap.comp turn it into the display image (Gate 14).
layout(set = 0, binding = 0, rgba32f) uniform image2D hdr_images[16];
layout(set = 0, binding = 1) uniform sampler2D luts[16];

layout(push_constant) uniform PC {
    uint image_index;
    uint width, height;
    float time_s;
    float a_star, r_cam, incl_deg, fov;
    float disk_r_in, disk_r_out, exposure_ev;
    uint shade_mode;   // display style bits: 1 = soft rim (translucent,
                       // band-limited caustics), 2 = volumetric clouds;
                       // 0 = the exact opaque model annulus
    uint y_offset;
    float flux_max;    // max Page-Thorne flux (CPU precomputed for T scaling)
    float t_peak_k;    // peak effective temperature [K]
    float turb;        // artistic turbulence amount 0..1 (0 = pure analytic)
    float cam_phi;     // camera orbit azimuth [rad]
    float time_scale;  // filament advection: Ω [1/M] per wall-clock second
    uint sky_off;      // 1 = fully black background (starfield hidden)
    // Camera basis in the local ZAMO triad (ê_r, ê_θ, ê_φ), built on the CPU
    // in double precision and already orthonormal — the shader must NOT
    // renormalise, because the hardware inversesqrt is approximate (even
    // normalize((-1,0,0)) can return 0.99999994) and that shifts chaotic
    // rays at the shadow edge. Default = look straight at the hole:
    // fwd (-1,0,0), right (0,0,1), up (0,-1,0).
    float fwd_r, fwd_th, fwd_ph;
    float rgt_r, rgt_th, rgt_ph;
    float up_r, up_th, up_ph;
    uint display_flags; // bit0: bolometric (gT)^4 grey disk display
                        // bit1: celestial coordinate grid on the sky
                        // bit2: Milky Way map; bit3: uniform sky (masks)
    uint accum_index;   // progressive sample index (0 = restart)
    float accum_weight; // blend weight of this sample into the mean
    float sky_gain;     // background sky gain (display; incl. camera meter)
} pc;

bool style_soft() { return (pc.shade_mode & 1u) != 0u; }
bool style_vol() { return (pc.shade_mode & 2u) != 0u; }

// Gate 17: the observer's frame, built on the CPU in double precision
// (App::make_frame_params) in Kerr-Schild components — any observer: static
// ZAMO, orbiting, free-falling, hovering (and the G24 in-fall).
layout(set = 0, binding = 2, std430) readonly buffer FrameParamsBuf {
    vec4 cam_x;   // camera event (t, x, y, z), KS
    vec4 e0;      // observer 4-velocity
    vec4 e_fwd;   // spatial legs: look direction, image right, image up
    vec4 e_rgt;
    vec4 e_up;
    float ks_chart; // −1 outgoing (exterior cameras), +1 ingoing
    uint flags;     // bit0 sky Doppler/gravitational shift, bit1 ergoregion,
                    // bit2 hot spot
    uint max_steps; // ray step budget (see MAX_STEPS below; CPU-set)
    // Gate 18 hot spot (fp64 CPU twin: science/flares/Hotspot.cpp)
    vec4 spot0;     // R, s, phase0, display amplitude
    vec4 spot1;     // Omega, u^t, t_obs (camera KS-time offset), alpha
    vec4 spot_h0;   // comoving metric at the KS-azimuth-0 centre:
    vec4 spot_h1;   //   (h00, h01, h02, h11), (h12, h22, x0, y0)
    // Gate 19 hot accretion flow (flags bit3; fp64 twin science/riaf/Riaf.cpp)
    vec4 riaf0;     // n0 [cm^-3], T0 [K], beta, nu_obs [GHz]
    vec4 riaf1;     // E_isco, L_isco, r_isco, r_max of the emitting region
    vec4 riaf2;     // r_g [cm] (constants.hpp), unused ×3
    // Gate 21 EXOTIC (flags bit4) — the Dneg wormhole, NOT Sgr A*:
    vec4 wh0;       // throat radius ρ, half-length a, lensing M, camera ℓ
} fp;

const float PI = 3.14159265;

// ---- Kerr-Schild Hamiltonian kernel (Gate 16) --------------------------------
// fp32 port of physics/src/kerr_schild.cpp (validated there against the
// Carter/Mino path, the Schwarzschild sqrt(27) limit and §4 conservation).
// Cartesian Kerr-Schild in the OUTGOING chart (s = −1), M = 1:
//   g^μν = η^μν − f l^μ l^ν,  f = 2r³/(r⁴ + a²z²),
//   l_μ = (1, s(rx + s a y)/(r²+a²), s(ry − s a x)/(r²+a²), s z/r).
// No coordinate pole (the BL spin-axis "zipper" and all its clamps are gone)
// and the past horizon — where backward-traced captured rays go — is a
// regular surface, so capture is simply r < r₊.
// State: x (Cartesian position), P = (p_t, p_x, p_y, p_z). p_t and
// L_z = x p_y − y p_x are exact constants (stationary, axisymmetric).
float KS_S = -1.0;   // set from fp.ks_chart in main()

float ks_radius(vec3 x) {
    float a = pc.a_star;
    float rho2 = dot(x, x) - a * a;
    float disc = sqrt(rho2 * rho2 + 4.0 * a * a * x.z * x.z);
    float r2 = (rho2 >= 0.0) ? 0.5 * (rho2 + disc)
                             : 2.0 * a * a * x.z * x.z / max(disc - rho2, 1e-30);
    return sqrt(max(r2, 1e-12));
}

void ks_rhs(vec3 x, vec4 P, out vec3 dx, out vec3 dp, out float dt) {
    float a = pc.a_star;
    float r = ks_radius(x);
    float r2 = r * r;
    float W = r2 * r2 + a * a * x.z * x.z;
    float D = r2 + a * a;
    float f = 2.0 * r2 * r / W;
    float Nx = r * x.x + KS_S * a * x.y, Ny = r * x.y - KS_S * a * x.x;
    vec3 l = vec3(KS_S * Nx / D, KS_S * Ny / D, KS_S * x.z / r);
    float lp = -P.x + dot(l, P.yzw);                // l^μ p_μ, l^t = −1
    dx = P.yzw - f * l * lp;
    dt = -P.x + f * lp;                             // dt/dλ = g^tν p_ν (slow light)
    vec3 ri = r * (r2 * x + vec3(0.0, 0.0, a * a * x.z)) / W;
    vec3 dW = 4.0 * r2 * r * ri + vec3(0.0, 0.0, 2.0 * a * a * x.z);
    // ∂f written without W² (r⁸ overflows fp32 beyond r ~ 6e4)
    vec3 fi = 2.0 * (3.0 * r2 * ri - r2 * r * (dW / W)) / W;
    vec3 dD = 2.0 * r * ri;
    vec3 dlx = KS_S * ((ri * x.x + vec3(r, KS_S * a, 0.0)) * D - Nx * dD) / (D * D);
    vec3 dly = KS_S * ((ri * x.y + vec3(-KS_S * a, r, 0.0)) * D - Ny * dD) / (D * D);
    vec3 dlz = KS_S * (vec3(0.0, 0.0, r) - x.z * ri) / r2;
    vec3 dlp = P.y * dlx + P.z * dly + P.w * dlz;
    dp = 0.5 * fi * lp * lp + f * lp * dlp;
}

// RK4 with the first stage supplied (the caller already evaluated it for
// step-size control). Also advances the coordinate time t along the ray
// (backward: t decreases), which the hot spot's slow light needs.
void ks_rk4_k1(inout vec3 x, inout vec4 P, inout float t, float h, vec3 dx1,
               vec3 dp1, float dt1) {
    vec3 dx2, dp2, dx3, dp3, dx4, dp4;
    float dt2, dt3, dt4;
    ks_rhs(x + 0.5 * h * dx1, vec4(P.x, P.yzw + 0.5 * h * dp1), dx2, dp2, dt2);
    ks_rhs(x + 0.5 * h * dx2, vec4(P.x, P.yzw + 0.5 * h * dp2), dx3, dp3, dt3);
    ks_rhs(x + h * dx3, vec4(P.x, P.yzw + h * dp3), dx4, dp4, dt4);
    x += h / 6.0 * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4);
    P.yzw += h / 6.0 * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4);
    t += h / 6.0 * (dt1 + 2.0 * dt2 + 2.0 * dt3 + dt4);
}

void ks_rk4(inout vec3 x, inout vec4 P, float h) {
    vec3 dx1, dp1, dx2, dp2, dx3, dp3, dx4, dp4;
    float dt_;
    ks_rhs(x, P, dx1, dp1, dt_);
    ks_rhs(x + 0.5 * h * dx1, vec4(P.x, P.yzw + 0.5 * h * dp1), dx2, dp2, dt_);
    ks_rhs(x + 0.5 * h * dx2, vec4(P.x, P.yzw + 0.5 * h * dp2), dx3, dp3, dt_);
    ks_rhs(x + h * dx3, vec4(P.x, P.yzw + h * dp3), dx4, dp4, dt_);
    x += h / 6.0 * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4);
    P.yzw += h / 6.0 * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4);
}

// Φ(r) = ∫ a/Δ dr (→ 0 at infinity): φ_KS = φ_BL + s Φ(r).
float ks_phi_shift(float r) {
    float a = pc.a_star;
    float d = sqrt(max(1.0 - a * a, 1e-8));
    return (a == 0.0) ? 0.0
                      : a / (2.0 * d) * log(abs((r - 1.0 - d) / (r - 1.0 + d)));
}

// u^t of the prograde circular equatorial orbit at r (BL; equal in KS).
float circular_ut(float r, float Om) {
    float a = pc.a_star;
    float dl = r * r - 2.0 * r + a * a;
    float AA = (r * r + a * a) * (r * r + a * a) - a * a * dl;
    float gtt = -(1.0 - 2.0 / r);
    float gtp = -2.0 * a / r;
    float gpp = AA / (r * r);
    float dn = -(gtt + 2.0 * Om * gtp + Om * Om * gpp);
    return (dn > 0.0) ? 1.0 / sqrt(dn) : 0.0;
}

// ---- Page-Thorne relative flux (mirrors physics/src/disk_model.cpp) --------
float pt_flux(float r, float a, float r_in) {
    if (r <= r_in * (1.0 + 1e-6) || r > pc.disk_r_out) return 0.0;
    float x = sqrt(r);
    float x0 = sqrt(r_in);
    float denom = x * x * x - 3.0 * x + 2.0 * a;
    if (denom <= 0.0) return 0.0;
    float ac = acos(clamp(a, -1.0, 1.0));
    float x1 = 2.0 * cos((ac - PI) / 3.0);
    float x2 = 2.0 * cos((ac + PI) / 3.0);
    float x3 = -2.0 * cos(ac / 3.0);
    float br = x - x0 - 1.5 * a * log(x / x0);
    if (abs(x1) > 1e-5)
        br -= 3.0 * (x1 - a) * (x1 - a) / (x1 * (x1 - x2) * (x1 - x3)) *
              log((x - x1) / (x0 - x1));
    if (abs(x2) > 1e-5)
        br -= 3.0 * (x2 - a) * (x2 - a) / (x2 * (x2 - x1) * (x2 - x3)) *
              log((x - x2) / (x0 - x2));
    if (abs(x3) > 1e-5)
        br -= 3.0 * (x3 - a) * (x3 - a) / (x3 * (x3 - x1) * (x3 - x2)) *
              log((x - x3) / (x0 - x3));
    // F ∝ bracket / (x⁴ (x³ − 3x + 2a)) — Newtonian limit r⁻³, exactly as
    // NovikovThorneDisk::flux (test_disk_model checks its large-r form).
    // An extra 1/x² here once made the live disk fall as r⁻⁴ and, against
    // the CPU-computed flux_max, run ~35% cooler than T_peak.
    return max(0.0, br / (x * x * x * x * denom));
}

// Observed linear-sRGB radiance of a blackbody seen at temperature T_obs.
// Since g³·B_{ν/g}(T) = B_ν(gT), a redshifted blackbody IS a blackbody at
// g·T: this one lookup carries both the Doppler/gravitational colour shift
// and the beaming (its bolometric integral is the g⁴ law). Visual radiance
// follows the CIE-weighted visible band, not T⁴ — that is what a camera sees.
const float BB_T_MIN = 300.0, BB_T_MAX = 100000.0; // mirror app.cpp kBb*
const int BB_N = 1024;
vec3 blackbody_radiance(float T_obs) {
    if ((pc.display_flags & 1u) != 0u)            // bolometric teaching view
        return vec3(pow(max(T_obs, 0.0) / 6500.0, 4.0));
    if (T_obs <= BB_T_MIN) return vec3(0.0);      // < 1e-20 of 6500 K: black
    float u = (log(min(T_obs, BB_T_MAX)) - log(BB_T_MIN)) /
              (log(BB_T_MAX) - log(BB_T_MIN)) * float(BB_N - 1);
    int i0 = min(int(u), BB_N - 2);
    float w = u - float(i0);
    return mix(texelFetch(luts[0], ivec2(i0, 0), 0).rgb,
               texelFetch(luts[0], ivec2(i0 + 1, 0), 0).rgb, w);
}

// ---- artistic turbulence layer (display only; physics untouched) ------------
// Mean-1 multiplicative filament pattern in (log r, φ), advected by the REAL
// relativistic circular-orbit angular velocity Ω = 1/(r^{3/2} + a): the
// differential rotation shears the noise into the trailing spiral filaments
// seen in MHD disk simulations. turb = 0 recovers the exact Page-Thorne disk.
// Lattice hash, periodic in y with period `per` (the azimuthal direction —
// a non-periodic pattern would seam at φ = ±π, which projects exactly onto
// the image's central column and reads as a vertical zipper).
float hash2p(vec2 i, float per) {
    i.y = mod(i.y, per);
    uvec2 q = uvec2(ivec2(i) + 8192);
    uint h = q.x * 1597334677u ^ q.y * 3812015801u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return float(h) * (1.0 / 4294967296.0);
}
float vnoise(vec2 p, float per) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash2p(i, per), hash2p(i + vec2(1, 0), per), u.x),
               mix(hash2p(i + vec2(0, 1), per), hash2p(i + vec2(1, 1), per),
                   u.x),
               u.y);
}
// 3-D value noise for the CINEMATIC volumetric dust: isotropic (round
// cells in world space) and sampled at Cartesian points on the orbit,
// so 2π-periodicity in φ is automatic and seam-free.
float hash3(vec3 i) {
    uvec3 q = uvec3(ivec3(i) + 512);
    uint h = q.x * 1597334677u ^ q.y * 3812015801u ^ q.z * 2798796415u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return float(h) * (1.0 / 4294967296.0);
}
float vnoise3(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash3(i), hash3(i + vec3(1, 0, 0)), u.x),
                   mix(hash3(i + vec3(0, 1, 0)),
                       hash3(i + vec3(1, 1, 0)), u.x), u.y),
               mix(mix(hash3(i + vec3(0, 0, 1)),
                       hash3(i + vec3(1, 0, 1)), u.x),
                   mix(hash3(i + vec3(0, 1, 1)),
                       hash3(i + vec3(1, 1, 1)), u.x), u.y),
               u.z);
}
float fbm3(vec3 p) {
    float n = 0.0, w = 0.5, ws = 0.0;
    for (int i = 0; i < 3; ++i) {
        n += w * vnoise3(p);
        ws += w;
        p = p * 2.03 + vec3(11.5, 7.7, 5.3);
        w *= 0.55;
    }
    return n / ws;   // mean 0.5
}

// `fade` band-limits the pattern where the lens map compresses the disk
// image below the pixel footprint (higher-order sub-images; the polar
// caustic column): fading toward the mean-1 disk IS the correct
// band-limited average of the texture, not a blur hack. The lane depth
// clamps at turb = 1 — beyond that the mix() extrapolates, lanes swing
// negative before the square, and the pattern degenerates into the
// high-contrast sawtooth the fade exists to prevent.
float turb_factor(float r, float phi, float fade) {
    float t = clamp(pc.turb, 0.0, 1.0) * clamp(fade, 0.0, 1.0);
    if (t <= 0.0) return 1.0;
    float Om = 1.0 / (r * sqrt(r) + pc.a_star);   // Kerr circular orbit
    // Pattern phase = azimuth + STATIC shear winding + live advection.
    // The winding term (120·Ω, several turns at the ISCO tapering to a
    // fraction of a turn at the rim) is what turbulence looks like after
    // the differential rotation has stretched it for many orbits — fine
    // trailing streamlines instead of fresh isotropic patches.
    // time_scale = rot_speed / (GM/c³): at rot_speed 1 this is the true
    // wall-clock rotation of Sgr A* (ISCO orbit ≈ 8.5 min — patience).
    float ph = phi + 120.0 * Om - Om * pc.time_s * pc.time_scale;
    // strongly anisotropic: fine radial striations, long orbital coherence.
    // Azimuthal lattice frequency 8/2π with the period doubling per octave
    // keeps the pattern exactly 2π-periodic (no seam on the axis column).
    const float kTwoPi = 6.28318530718;
    float per = 8.0;
    vec2 p = vec2(log(r) * 22.0, ph * (per / kTwoPi));
    float n = 0.0, w = 0.5, wsum = 0.0;
    for (int i = 0; i < 5; ++i) {
        n += w * vnoise(p, per);
        wsum += w;
        p = vec2(p.x * 1.93 + 17.13, p.y * 2.0);
        per *= 2.0;
        w *= 0.60;
    }
    n /= wsum;                                      // mean 0.5
    // Lane depth ~3.5 stops: enough for strong streamlines, shallow enough
    // that the magnified caustic image of the lanes stays smooth.
    float f = mix(1.0, 0.30 + 1.40 * n, t);         // mean stays 1
    return f * f;
}

// ---- starfield (matches CPU) ------------------------------------------------
float hash01(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return float(x) * (1.0 / 4294967296.0);
}
vec3 shade_starfield(float theta, float phi) {
    // Procedural fallback when the Milky Way map is absent: true black
    // between point stars (the grid overlay lives in shade_sky).
    vec3 col = vec3(0.0);
    float cell = 0.35 * PI / 180.0;
    float phw = mod(mod(phi, 2.0 * PI) + 2.0 * PI, 2.0 * PI);
    uint hcell = uint(int(floor(theta / cell)) * 73856093) ^
                 uint(int(floor(phw / cell)) * 19349663);
    if (hash01(hcell) > 0.82) {
        float u = fract(theta / cell), v = fract(phw / cell);
        float su = hash01(hcell * 0x9e3779b9u) * 0.6 + 0.2;
        float sv = hash01(hcell * 0x85ebca6bu) * 0.6 + 0.2;
        float d2 = (u - su) * (u - su) + (v - sv) * (v - sv);
        float mag = hash01(hcell * 0xc2b2ae35u);
        float I = (0.5 + 3.5 * mag * mag * mag) *
                  exp(-d2 / (0.0022 * (0.4 + mag)));
        float warm = hash01(hcell * 0x27d4eb2fu);
        col += I * vec3(0.75 + 0.25 * warm, 0.72 + 0.13 * warm,
                        0.95 - 0.25 * warm);
    }
    return col;
}

// ---- Gate 15: the real Milky Way behind Sgr A* ------------------------------
// luts[1] = NASA SVS Deep Star Maps 2020 (Gaia DR2 + Hipparcos-2 + Tycho-2),
// galactic plate carrée, u = 0.5 − l/2π, v = 0.5 − b/π, linear radiance.
// It is the sky as seen FROM EARTH, used as the celestial sphere at infinity
// (stars within a few kpc would sit elsewhere seen from the Galactic Centre —
// labelled in the UI and the paper).
//
// Orientation: the BL spin axis n̂ is placed at inclination I_EARTH to the
// line of sight toward Earth (i ≈ 30°), tilted toward
// position angle SPIN_PA on the sky measured from galactic north toward
// increasing l. The PA is NOT constrained by current data — 0 is a stated
// assumption. BL φ = 0 is the meridian containing Earth, so a camera at
// θ = I_EARTH, cam_phi = 0 looks at Sgr A* exactly as we do, with the
// Galactic-Centre star clouds (l = 0) behind it.
// Galactic Cartesian: x̂ Sun → GC, ŷ toward l = 90°, ẑ toward the NGP.
const float I_EARTH = 30.0 * 3.14159265 / 180.0;
const float SPIN_PA = 0.0;
vec3 bl_dir_to_galactic(vec3 d) {
    float ci = cos(I_EARTH), si = sin(I_EARTH);
    float cp = cos(SPIN_PA), sp = sin(SPIN_PA);
    vec3 e = vec3(-1.0, 0.0, 0.0);          // Sgr A* → Earth
    vec3 s = vec3(0.0, sp, cp);             // sky-plane tilt direction
    vec3 zb = ci * e + si * s;              // spin axis
    vec3 xb = si * e - ci * s;              // φ = 0 meridian (toward Earth)
    vec3 yb = vec3(0.0, -cp, sp);           // ẑ_BL × x̂_BL
    return d.x * xb + d.y * yb + d.z * zb;
}
vec3 shade_sky(float theta, float phi) {
    // validation: radiance 1 from every direction, so captured rays (exact
    // zeros) separate cleanly from the sky in shadow-mask comparisons
    if ((pc.display_flags & 8u) != 0u) return vec3(pc.sky_gain);
    vec3 col;
    if ((pc.display_flags & 4u) != 0u) {
        vec3 d = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
        vec3 g = bl_dir_to_galactic(d);
        float l = atan(g.y, g.x);
        float b = asin(clamp(g.z, -1.0, 1.0));
        vec2 uv = vec2(0.5 - l / (2.0 * PI), 0.5 - b / PI);
        // Mip from the (unlensed) pixel footprint vs the map texel; strongly
        // lensed regions alias per sample, which progressive AA averages.
        float px_ang = 2.0 * tan(0.5 * pc.fov) / float(pc.width);
        float texel_ang = 2.0 * PI / float(textureSize(luts[1], 0).x);
        float lod = max(log2(px_ang / texel_ang), 0.0);
        col = textureLod(luts[1], uv, lod).rgb;
    } else {
        col = 0.5 * shade_starfield(theta, phi);   // procedural fallback
    }
    col *= pc.sky_gain;
    if ((pc.display_flags & 2u) != 0u) {             // celestial grid overlay
        float deg = 180.0 / PI;
        float gt = mod(abs(theta) * deg, 10.0);
        float gp = mod(abs(phi) * deg, 10.0);
        if (gt < 0.12 || gt > 9.88 || gp < 0.12 || gp > 9.88)
            col += vec3(0.012, 0.015, 0.024) * pc.sky_gain * 8.0;
    }
    return col;
}

// Linear-sRGB emission of the disk surface at BL radius r_x, azimuth phi_x,
// redshift factor g: the visual radiance of a blackbody at g·T_eff (CPU
// twin: NovikovThorneDisk::shade).
vec3 disk_emission(float r_x, float phi_x, float g, float fade) {
    float F = pt_flux(r_x, pc.a_star, pc.disk_r_in);
    F *= turb_factor(r_x, phi_x, fade);
    // Artistic presets: exponential (stops-based) emissivity taper over the
    // outer rim — a linear fade barely moves the logarithmic display grade.
    // (Display only; the PHYSICAL path keeps the sharp model annulus.)
    if (style_soft()) {
        float rn = (r_x - pc.disk_r_in) /
                   max(pc.disk_r_out - pc.disk_r_in, 1e-3);
        F *= exp2(-15.0 * smoothstep(0.55, 1.0, rn));
    }
    float T = pc.t_peak_k * pow(max(F, 0.0) / pc.flux_max, 0.25);
    return blackbody_radiance(g * T);
}

// ---- Gate 18: orbiting hot spot ---------------------------------------------
// The emitter GRAVITY fitted to the 2018 near-infrared flares (GRAVITY 2018,
// 2020): a Gaussian blob, ε ∝ exp(−d²/2s²), on the prograde circular
// geodesic of radius R, moving rigidly with the centre's u = u^t(∂_t + Ω∂_φ).
// d is the comoving distance, (g + u⊗u)(Δx, Δx) at fixed KS time, with the
// metric precomputed at the KS-azimuth-0 centre (rotations about z are
// isometries of KS Cartesian coordinates). Observed intensity, optically
// thin, j_ν ∝ ν^−α: I = ∫ g^{2+α} ε dλ with the camera-normalised momentum.
// Exactly the CPU twin science/flares/Hotspot.cpp (fp64).
bool spot_on() { return (fp.flags & 4u) != 0u; }
float spot_emissivity(vec3 x, float t_e) {
    float ph = fp.spot0.z + fp.spot1.x * t_e;
    float c = cos(ph), sn = sin(ph);
    vec2 c0 = fp.spot_h1.zw;
    vec2 cen = vec2(c * c0.x - sn * c0.y, sn * c0.x + c * c0.y);
    vec2 dxy = x.xy - cen;
    vec3 d = vec3(c * dxy.x + sn * dxy.y, -sn * dxy.x + c * dxy.y, x.z);
    float d2 = fp.spot_h0.x * d.x * d.x + fp.spot_h0.w * d.y * d.y +
               fp.spot_h1.y * d.z * d.z +
               2.0 * (fp.spot_h0.y * d.x * d.y + fp.spot_h0.z * d.x * d.z +
                      fp.spot_h1.x * d.y * d.z);
    float s = fp.spot0.y;
    return exp(-0.5 * d2 / (s * s));
}

// ---- Gate 19: the hot accretion flow at a radio frequency ------------------------
// The semi-analytic RIAF of science/riaf/Riaf.hpp (Broderick & Loeb 2006
// profiles, β-equipartition toroidal field, Keplerian / plunging flow),
// thermal synchrotron (Leung et al. 2011 fit, as in grmonty) and Kirchhoff
// absorption, composited front-to-back:
//   I_obs = Σ [B_{ν_f}(T_e)/ε³] (1 − e^{−Δτ}) e^{−τ},  ν_f = ε ν_obs,
//   Δτ = α_{ν_f} ε Δλ r_g.
// Units are arranged for fp32: frequencies in GHz, I in cgs.
bool riaf_on() { return (fp.flags & 8u) != 0u; }
const float THETA_PER_K = 1.68637e-10;  // k/(m_e c²) [1/K]
const float NUC_PER_G = 2.799249e6;     // e/(2π m_e c) [Hz/G]
const float JPREF = 1.13967e-29;        // √2 π e²/(3c), cgs
const float PL_PREF = 3.07236e-19;      // 2k (1e9 Hz)²/c²: B_ν = PL_PREF T ν_G² x/(e^x−1)
const float H_OVER_K_G = 0.0479924;     // h (1e9 Hz)/k [K]
const float MPC2 = 1.50328e-3;          // m_p c² [erg]

// Modified Bessel K0, K1 (Numerical Recipes polynomial fits), K2 by recurrence
float bess_i0(float x) {
    float y = (x / 3.75) * (x / 3.75);
    return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 +
           y * (0.0360768 + y * 0.0045813)))));
}
float bess_i1(float x) {
    float y = (x / 3.75) * (x / 3.75);
    return x * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934 +
           y * (0.02658733 + y * (0.00301532 + y * 0.00032411))))));
}
float bess_k2(float x) {
    float k0, k1;
    if (x <= 2.0) {
        float y = 0.25 * x * x;
        k0 = -log(0.5 * x) * bess_i0(x) + (-0.57721566 + y * (0.42278420 + y * (0.23069756 +
             y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.0000074))))));
        k1 = log(0.5 * x) * bess_i1(x) + (1.0 / x) * (1.0 + y * (0.15443144 + y * (-0.67278579 +
             y * (-0.18156897 + y * (-0.01919402 + y * (-0.00110404 + y * (-0.00004686)))))));
    } else {
        float y = 2.0 / x;
        float e = exp(-x) / sqrt(x);
        k0 = e * (1.25331414 + y * (-0.07832358 + y * (0.02189568 + y * (-0.01062446 +
             y * (0.00587872 + y * (-0.00251540 + y * 0.00053208))))));
        k1 = e * (1.25331414 + y * (0.23498619 + y * (-0.03655620 + y * (0.01504268 +
             y * (-0.00780353 + y * (0.00325614 + y * (-0.00068245)))))));
    }
    return k0 + 2.0 / x * k1;
}
// thermal synchrotron emissivity, ν in GHz (cgs output)
float riaf_jnu(float nu_g, float n_e, float T_e, float B, float sin_b) {
    float th = THETA_PER_K * T_e;
    if (th < 0.05 || n_e <= 0.0 || B <= 0.0 || sin_b <= 0.0) return 0.0;
    float nu_s = (2.0 / 9.0) * NUC_PER_G * B * th * th * sin_b;      // Hz
    float X = nu_g * 1e9 / nu_s;
    if (!(X < 1e12)) return 0.0;
    float xp = pow(X, 1.0 / 3.0);
    float xx = sqrt(X) + 1.88774862536 * sqrt(xp);
    return JPREF * n_e * nu_s / bess_k2(1.0 / th) * xx * xx * exp(-xp);
}
float riaf_planck(float nu_g, float T) {
    float x = H_OVER_K_G * nu_g / T;
    if (x > 80.0) return 0.0;
    float f = (x < 1e-3) ? 1.0 - 0.5 * x : x / (exp(x) - 1.0);
    return PL_PREF * T * nu_g * nu_g * f;
}
// One transfer sample at KS point x (BL radius r) with step h along the
// backward ray P (camera-normalised). Accumulates intensity and depth.
void riaf_sample(vec3 x, vec4 P, float Lz, float r, float h, float r_hor,
                 inout float I_acc, inout float tau) {
    float a = pc.a_star;
    float cth = clamp(x.z / r, -1.0, 1.0);
    float sth = sqrt(max(1.0 - cth * cth, 0.0));
    float rho = r * sth, z = r * cth;
    if (r <= 1.01 * r_hor || rho < 1e-3) return;
    float n_e = fp.riaf0.x * pow(rho, -1.1) * exp(-0.5 * z * z / (rho * rho));
    float T_e = fp.riaf0.y * pow(r, -0.84);
    float B = sqrt(8.0 * PI * n_e * MPC2 * (2.0 / (12.0 * r)) / fp.riaf0.z);
    // Boyer-Lindquist metric
    float S = r * r + a * a * cth * cth;
    float Dl = r * r - 2.0 * r + a * a;
    float AA = (r * r + a * a) * (r * r + a * a) - a * a * Dl * sth * sth;
    float gtt = -(1.0 - 2.0 * r / S), gtp = -2.0 * a * r * sth * sth / S;
    float grr = S / Dl, gpp = AA * sth * sth / S;
    // flow 4-velocity (BL): Keplerian outside the ISCO, plunge inside
    vec3 u = vec3(0.0);          // (u^t, u^r, u^φ)
    if (r >= fp.riaf1.z) {
        float Om = 1.0 / (pow(rho, 1.5) + a);
        float nrm = -(gtt + 2.0 * Om * gtp + Om * Om * gpp);
        if (nrm > 0.0) u = vec3(1.0, 0.0, Om) / sqrt(nrm);
    } else {
        // equatorial plunge with the ISCO's E, L: its u^r and Ω at this r
        float De = Dl, Se = r * r;
        float Ae = (r * r + a * a) * (r * r + a * a) - a * a * De;
        float ett = -(1.0 - 2.0 / r), etp = -2.0 * a / r, epp = Ae / Se;
        float det = ett * epp - etp * etp;
        float itt = epp / det, itp = -etp / det, ipp = ett / det;
        float ut_l = -fp.riaf1.x, up_l = fp.riaf1.y;
        float blk = itt * ut_l * ut_l + 2.0 * itp * ut_l * up_l + ipp * up_l * up_l;
        float ur = -sqrt(max((-1.0 - blk) * De / Se, 0.0));
        float Om = (itp * ut_l + ipp * up_l) / (itt * ut_l + itp * up_l);
        float den = gtt + 2.0 * Om * gtp + Om * Om * gpp;
        if (!(den < 0.0)) {
            Om = -gtp / gpp;
            den = gtt + 2.0 * Om * gtp + Om * Om * gpp;
        }
        float ut = sqrt((-1.0 - grr * ur * ur) / den);
        u = vec3(ut, ur, Om * ut);
    }
    if (u.x <= 0.0) {
        float om = -gtp / gpp;
        float al = sqrt(max(-(gtt - gtp * gtp / gpp), 1e-12));
        u = vec3(1.0 / al, 0.0, om / al);
    }
    // toroidal field b ∝ (λ, 0, 0, 1), b·u = 0
    float u_t = gtt * u.x + gtp * u.z, u_p = gtp * u.x + gpp * u.z;
    float lam = -u_p / u_t;
    float nb = gtt * lam * lam + 2.0 * gtp * lam + gpp;
    if (!(nb > 0.0)) return;
    vec2 b = vec2(lam, 1.0) / sqrt(nb);      // (b^t, b^φ)
    // the backward momentum in BL components (covector): P_t, P_r, P_φ
    float pk = atan(x.y, x.x) - atan(KS_S * a, r);
    float cp = cos(pk), sp = sin(pk);
    float dpk_dr = KS_S * a / Dl;
    float dx_dpk = (-r * sp - KS_S * a * cp) * sth;
    float dy_dpk = (r * cp - KS_S * a * sp) * sth;
    float Pr = P.x * KS_S * 2.0 * r / Dl + P.y * (cp * sth + dx_dpk * dpk_dr) +
               P.z * (sp * sth + dy_dpk * dpk_dr) + P.w * cth;
    float eps = P.x * u.x + Pr * u.y + Lz * u.z;
    if (!(eps > 0.0) || eps > 1e4) return;   // deep redshift near r+: no emission
    float Pb = P.x * b.x + Lz * b.y;
    float cb = clamp(-Pb / eps, -1.0, 1.0);
    float nu_f = fp.riaf0.w * eps;
    float jn = riaf_jnu(nu_f, n_e, T_e, B, sqrt(1.0 - cb * cb));
    float Bn = riaf_planck(nu_f, T_e);
    if (!(jn > 0.0) || !(Bn > 0.0)) return;
    float dtau = jn / Bn * eps * h * fp.riaf2.x;
    I_acc += exp(-tau) * (Bn / (eps * eps * eps)) * (1.0 - exp(-dtau));
    tau += dtau;
}

// ---- Gate 21: EXOTIC SPACETIME, NOT SGR A* — the Dneg wormhole ------------
// James, von Tunzelmann, Franklin & Thorne (2015): ds² = −dt² + dℓ² +
// r(ℓ)²dΩ², r = ρ for |ℓ| ≤ a, ρ + M[x atan x − ½ ln(1+x²)] beyond,
// x = 2(|ℓ|−a)/(πM). Spherical symmetry keeps every ray in a plane through
// the centre, so each is integrated in its own plane (angle ψ from the
// camera's position): dℓ/dt = p_ℓ, dψ/dt = B/r², dp_ℓ/dt = B² r'/r³ — the
// planar form of their ray equations (A.7), with no polar singularity. The
// fp64 twin integrates the full (A.7) (science/wormhole, wormhole_tool).
// A ray traced back to ℓ → +∞ shows our sky (the Milky Way map); to
// ℓ → −∞ the other universe (the same map turned around and recoloured:
// illustrative). The camera sits at sky position (θ, φ) = (incl, cam_phi)
// — mouse drag orbits the wormhole exactly as it orbits Sgr A*.
bool wormhole_on() { return (fp.flags & 16u) != 0u; }
float wh_r(float l) {
    float d = abs(l) - fp.wh0.y;
    if (d <= 0.0) return fp.wh0.x;
    float x = 2.0 * d / (PI * fp.wh0.z);
    return fp.wh0.x + fp.wh0.z * (x * atan(x) - 0.5 * log(1.0 + x * x));
}
float wh_drdl(float l) {
    float d = abs(l) - fp.wh0.y;
    if (d <= 0.0) return 0.0;
    return sign(l) * (2.0 / PI) * atan(2.0 * d / (PI * fp.wh0.z));
}
vec3 shade_wormhole(float px_f, float py_f) {
    float xs = 2.0 * px_f / float(pc.width) - 1.0;
    float ys = 1.0 - 2.0 * py_f / float(pc.height);
    float aspect = float(pc.height) / float(pc.width);
    float thalf = tan(0.5 * pc.fov);
    float tx = thalf * xs, ty = thalf * ys * aspect;
    // pixel direction in the camera's local triad (ê_ℓ, ê_θ, ê_φ): the same
    // look basis as the Kerr camera, with ê_r → ê_ℓ; then the JVFT camera
    // basis e_x = ê_ℓ, e_y = ê_φ, e_z = −ê_θ. N = where the light comes from
    // (the backward-traced ray moves along N).
    vec3 n = normalize(vec3(pc.fwd_r, pc.fwd_th, pc.fwd_ph) +
                       tx * vec3(pc.rgt_r, pc.rgt_th, pc.rgt_ph) +
                       ty * vec3(pc.up_r, pc.up_th, pc.up_ph));
    vec3 N = vec3(n.x, n.z, -n.y);
    float lc = fp.wh0.w;
    float rc = wh_r(lc);
    vec3 c1 = vec3(1.0, 0.0, 0.0);                 // the camera's position
    vec3 perp = N - N.x * c1;
    float sp = length(perp);
    vec3 c2 = (sp > 1e-7) ? perp / sp : vec3(0.0, 1.0, 0.0);
    float B = rc * sp;                             // impact parameter
    float l = lc, psi = 0.0, pl = N.x;
    float lfar = max(40.0, 4.0 * abs(lc));
    float lmouth = fp.wh0.y + 10.0 * fp.wh0.z;     // where r(ℓ) bends
    for (int k = 0; k < 6000; ++k) {
        float r = wh_r(l);
        float h = 0.02 * max(abs(l), r);
        if (abs(l) < lmouth) h = min(h, 0.2 * fp.wh0.z);
        // RK4 on (ℓ, ψ, p_ℓ)
        float k1l = pl, k1p = B / (r * r), k1q = B * B * wh_drdl(l) / (r * r * r);
        float l2 = l + 0.5 * h * k1l, q2 = pl + 0.5 * h * k1q, r2 = wh_r(l2);
        float k2l = q2, k2p = B / (r2 * r2), k2q = B * B * wh_drdl(l2) / (r2 * r2 * r2);
        float l3 = l + 0.5 * h * k2l, q3 = pl + 0.5 * h * k2q, r3 = wh_r(l3);
        float k3l = q3, k3p = B / (r3 * r3), k3q = B * B * wh_drdl(l3) / (r3 * r3 * r3);
        float l4 = l + h * k3l, q4 = pl + h * k3q, r4 = wh_r(l4);
        float k4l = q4, k4p = B / (r4 * r4), k4q = B * B * wh_drdl(l4) / (r4 * r4 * r4);
        l += h / 6.0 * (k1l + 2.0 * k2l + 2.0 * k3l + k4l);
        psi += h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
        pl += h / 6.0 * (k1q + 2.0 * k2q + 2.0 * k3q + k4q);
        if (abs(l) > lfar && l * pl > 0.0) break;
    }
    // the rest of the outward sweep in the nearly flat far zone: the
    // straight line's asin(B/r) (as in the CPU twin)
    psi += asin(clamp(B / wh_r(l), 0.0, 1.0));
    vec3 u = cos(psi) * c1 + sin(psi) * c2;        // (e_x, e_y, e_z) components
    float thc = pc.incl_deg * PI / 180.0, phc = pc.cam_phi;
    vec3 Rh = vec3(sin(thc) * cos(phc), sin(thc) * sin(phc), cos(thc));
    vec3 Th = vec3(cos(thc) * cos(phc), cos(thc) * sin(phc), -sin(thc));
    vec3 Ph = vec3(-sin(phc), cos(phc), 0.0);
    vec3 G = u.x * Rh + u.y * Ph - u.z * Th;       // direction on the sky
    float th = acos(clamp(G.z, -1.0, 1.0));
    float ph = atan(G.y, G.x);
    if ((pc.display_flags & 8u) != 0u)             // validation: our side 1, other 0
        return (l > 0.0) ? vec3(1.0) : vec3(0.0);
    if (l > 0.0) return shade_sky(th, ph);
    // the other universe: our map turned around, colour channels swapped
    // so the dust lanes turn blue (illustrative — its sky is unknowable)
    return shade_sky(PI - th, ph + PI).bgr;
}

// Step budget per ray. Rays near the critical curve wind around the photon
// shell before escaping or falling in, and near the high-spin prograde
// photon orbit (r ≈ 1.43 M at a* = 0.94) the orbit is only weakly unstable:
// a ray one pixel from the shadow edge winds several times at ~10³ steps per
// orbit. A ray that exhausts the budget counts as captured, so a short
// budget pushes the prograde shadow edge outward (G17 strip test: 17 px at
// 1600 steps, edge-on). Only those few rays ever use the extra steps.
// The budget arrives in FrameParams (6000) so the validation can re-run the
// old value (validation/phase14/compare_observers.py).
int MAX_STEPS = 6000;

// Trace one camera ray (sub-pixel coordinates). Disk emission accumulates
// front-to-back with per-crossing opacity (the soft-style rim is translucent,
// so stars and lensed inner structure show through the faded edge); the sky
// radiance is returned separately. Returns the remaining transmittance.
//
// Handedness: the pixel looking along n RECEIVES k = e_t − n·e; we trace
// P = −k backward. (Phases 1–12 launched +e_t + n·e, the photon the camera
// would emit — in Kerr that is the spin-reversed, i.e. mirrored, image;
// physics/tests/test_kerr_schild.cpp pins both facts.)
float shade_sample(float px_f, float py_f, float jit, out vec3 disk_out,
                   out vec3 sky_out, out float sq_out, out float ergo_out) {
    float a = pc.a_star;
    float r0 = pc.r_cam;
    float th0 = pc.incl_deg * PI / 180.0;
    float xs = 2.0 * px_f / float(pc.width) - 1.0;
    float ys = 1.0 - 2.0 * py_f / float(pc.height);
    float aspect = float(pc.height) / float(pc.width);
    float thalf = tan(0.5 * pc.fov);
    float tx = thalf * xs, ty = thalf * ys * aspect;

    float s0 = sin(th0), c0 = cos(th0);
    // Pixel direction n in the observer's frame; the pixel records
    // k_rec = e0 − n^i e_i, we trace V = −k_rec = −e0 + n^i e_i backward.
    float inv = 1.0 / sqrt(1.0 + tx * tx + ty * ty);
    vec4 V = -fp.e0 + (fp.e_fwd + tx * fp.e_rgt + ty * fp.e_up) * inv;
    vec3 x = fp.cam_x.yzw;
    // lower with g = η + f l⊗l at the camera
    float rc = ks_radius(x);
    float Wc = rc * rc * rc * rc + a * a * x.z * x.z;
    float fc = 2.0 * rc * rc * rc / Wc;
    float Dc = rc * rc + a * a;
    vec4 lc = vec4(1.0, KS_S * (rc * x.x + KS_S * a * x.y) / Dc,
                   KS_S * (rc * x.y - KS_S * a * x.x) / Dc, KS_S * x.z / rc);
    vec4 P = vec4(-V.x, V.yzw) + fc * lc * dot(lc, V);
    float Lz = x.x * P.z - x.y * P.y;      // conserved

    float r_hor = 1.0 + sqrt(max(0.0, 1.0 - a * a));
    float r_escape = max(2.0 * r0, 300.0);

    int outcome = 0;           // 0 flying, 1 captured, 2 escaped, 3 opaque
    bool ergo_visit = false;
    vec3 acc = vec3(0.0);
    float Tr = 1.0;
    int ncross = 0, nz = 0;
    float veil = 0.0;
    float min_r = r0;
    vec3 esc_dir = vec3(0.0, 0.0, 1.0);

    // Edge-on views: grazing rays ride the disk plane — tighten the base step.
    // (No per-sample step jitter: the BL kernel needed it to break up
    // banding, but here it made each progressive sample's escape direction
    // wobble by more than a sky texel and smeared stars into arcs. A fixed
    // step pattern gives a smooth, sub-texel deterministic error instead.)
    float kstep = mix(1.0, 0.45, smoothstep(0.30, 0.10, abs(c0)));

    float t_ray = 0.0;   // KS time along the backward ray, relative to the camera
    float riaf_I = 0.0, riaf_tau = 0.0;   // Gate 19 radio intensity [cgs], depth
    for (int step = 0; step < MAX_STEPS && outcome == 0; ++step) {
        vec3 xp = x;
        vec4 Pp = P;
        float r = ks_radius(x);
        vec3 dx0, dp0;
        float dt0;
        ks_rhs(x, P, dx0, dp0, dt0);
        float speed = max(length(dx0), 1e-6);
        // ~4% of r per step; finer inside the photon region and the plane
        float frac = 0.03 * kstep * mix(0.5, 1.0, smoothstep(2.0, 6.0, r));
        if (abs(x.z) < 0.15 * r && r < pc.disk_r_out + 6.0) frac = min(frac, 0.02);
        if (style_vol() && r < pc.disk_r_out * 1.05 && abs(x.z) < 0.4 * r)
            frac = min(frac, 0.015);   // sample the cloud slab finely
        float h = frac * r / speed;
        if (riaf_on() && r < fp.riaf1.w * 1.2)
            h = min(h, 0.03 * r / speed);   // resolve the plasma (as the CPU twin)
        if (spot_on()) {
            // never jump into or across the hot spot's orbit, and sample
            // its Gaussian at <= s/5 spacing (as the CPU twin does)
            float roi = 6.0 * fp.spot0.y;
            float dist = max(max(abs(r - fp.spot0.x) - roi, abs(x.z) - roi), 0.0);
            h = min(h, max(0.2 * fp.spot0.y, 0.5 * dist) / speed);
        }
        ks_rk4_k1(x, P, t_ray, h, dx0, dp0, dt0);
        float rn = ks_radius(x);
        min_r = min(min_r, rn);

        // ---- hot spot: slow-light emission at this sample ----------------
        if (spot_on() && outcome == 0) {
            float roi = 6.0 * fp.spot0.y;
            if (abs(rn - fp.spot0.x) <= roi && abs(x.z) <= roi) {
                float t_e = fp.cam_x.x + fp.spot1.z + t_ray;
                float e = spot_emissivity(x, t_e);
                if (e > 1e-7) {
                    float gsp = 1.0 / (fp.spot1.y * (P.x + fp.spot1.x * Lz));
                    if (gsp > 0.0) {
                        float I = fp.spot0.w * pow(gsp, 2.0 + fp.spot1.w) * e * h;
                        // false colour (2.2 µm synchrotron has no visible
                        // colour); grey for the CPU/GPU comparison
                        vec3 tint = ((pc.display_flags & 16u) != 0u)
                                        ? vec3(1.0) : vec3(1.0, 0.42, 0.12);
                        acc += Tr * I * tint;
                    }
                }
            }
        }

        // ---- Gate 19: hot accretion flow (radio band) ----------------------
        if (riaf_on() && outcome == 0 && rn < fp.riaf1.w) {
            riaf_sample(x, P, Lz, rn, h, r_hor, riaf_I, riaf_tau);
            if (riaf_tau > 30.0) outcome = 3;   // opaque: nothing behind counts
        }

        // ---- ergoregion visit (flags bit1; display overlay) ----------------
        // r_E(θ) = M + √(M² − a²cos²θ): inside it nothing can stay at rest
        // relative to the distant stars. Every captured ray crosses it, so
        // the overlay marks only light that went INSIDE and still escaped —
        // the frame-dragged crescent on the prograde side of the shadow.
        if ((fp.flags & 2u) != 0u) {
            float czn = x.z / max(rn, 1e-6);
            if (rn < 1.0 + sqrt(max(1.0 - a * a * czn * czn, 0.0))) ergo_visit = true;
        }

        // ---- volumetric cloud layer (style bit 2; display only) ----------
        // An LTE slab orbiting at the real Ω(r): source function = local
        // blackbody at g·T, so emission and absorption share one optical
        // depth. Deleting this block restores the exact thin-disk image.
        if (style_vol()) {
            vec3 xm = 0.5 * (xp + x);
            float Rc = max(length(xm.xy), 1e-3);
            float zc = xm.z;
            float Hc = 0.30 + 0.07 * Rc;
            if (Rc > pc.disk_r_in * 0.85 && Rc < pc.disk_r_out * 1.05 &&
                abs(zc) < 2.5 * Hc) {
                float env = exp(-0.5 * (zc / Hc) * (zc / Hc));
                env *= smoothstep(pc.disk_r_in * 0.85, pc.disk_r_in * 1.10, Rc) *
                       (1.0 - smoothstep(pc.disk_r_out * 0.70, pc.disk_r_out * 1.05, Rc));
                float Om = 1.0 / (Rc * sqrt(Rc) + a);
                float phm = atan(xm.y, xm.x) + pc.cam_phi;
                const float L_RE = 140.0;
                float Aadv = pc.time_s * pc.time_scale;
                float nsum = 0.0;
                for (int k = 0; k < 2; ++k) {
                    float lifec = (Aadv + float(k) * 0.5 * L_RE) / L_RE;
                    float age = fract(lifec) * L_RE;
                    float epoch = floor(lifec);
                    float wk = 0.5 - 0.5 * cos(6.2831853 * age / L_RE);
                    float phk = phm + 18.0 * Om - Om * age;
                    vec3 q = vec3(Rc * cos(phk), Rc * sin(phk), zc * 1.8) * 0.35 +
                             vec3(epoch * 13.71 + float(k) * 61.3, epoch * 7.31,
                                  epoch * 5.17);
                    nsum += wk * fbm3(q);
                }
                float dens = clamp(pc.turb, 0.0, 1.0) * env *
                             pow(smoothstep(0.44, 0.80, nsum), 1.5) * 1.7;
                if (dens > 1e-4) {
                    float ut = circular_ut(Rc, Om);
                    if (ut > 0.0) {
                        float g_v = 1.0 / (ut * (P.x + Om * Lz));
                        float Fv = pt_flux(Rc, a, pc.disk_r_in);
                        float Tv = pc.t_peak_k * pow(max(Fv, 0.0) / pc.flux_max, 0.25);
                        float Tobs = clamp(g_v, 0.05, 4.0) * max(Tv, 900.0);
                        float dtau = 0.8 * dens * length(x - xp) / Rc;
                        float att = exp(-dtau);
                        acc += Tr * (1.0 - att) * blackbody_radiance(Tobs);
                        Tr *= att;
                        if (Tr < 0.05) outcome = 3;
                    }
                }
            }
        }

        // ---- equatorial crossing z = 0 ------------------------------------
        if (xp.z * x.z < 0.0 && outcome == 0) {
            ++nz;
            float lo = 0.0, hi = 1.0;   // bisect the step fraction (RK4)
            vec3 xc = x;
            for (int it = 0; it < 12; ++it) {
                float mid = 0.5 * (lo + hi);
                vec3 xt = xp;
                vec4 Pt = Pp;
                ks_rk4(xt, Pt, mid * h);
                if (xt.z * xp.z > 0.0) lo = mid; else hi = mid;
                xc = xt;
            }
            float r_x = sqrt(max(dot(xc.xy, xc.xy) - a * a, 0.0));
            if (!riaf_on() && r_x >= pc.disk_r_in && r_x <= pc.disk_r_out) {
                float Om = 1.0 / (r_x * sqrt(r_x) + a);
                float ut = circular_ut(r_x, Om);
                // g = (P·u_cam)/(P·u_disk), P·u_cam = 1 by construction
                float g_x = 1.0 / (ut * (P.x + Om * Lz));
                // BL azimuth (texture phase): φ_K = atan2(y,x) − arg(r + i s a),
                // φ_BL = φ_K − s Φ(r)
                float ph_bl = atan(xc.y, xc.x) - atan(KS_S * a, r_x) -
                              KS_S * ks_phi_shift(r_x);
                float o = 1.0;
                if (style_soft()) {
                    float rnrm = (r_x - pc.disk_r_in) /
                                 max(pc.disk_r_out - pc.disk_r_in, 1e-3);
                    o = smoothstep(0.98, 0.60, rnrm);
                }
                // higher-order images are demagnified ~e^-π per crossing:
                // fade the filament texture toward its mean (band-limit)
                float fade = mix(mix(1.0, 0.30, min(veil, 1.0)), 0.0,
                                 clamp(veil - 1.0, 0.0, 1.0));
                acc += Tr * o * disk_emission(r_x, ph_bl + pc.cam_phi, g_x, fade);
                Tr *= 1.0 - o;
                veil += o;
                ++ncross;
                if (Tr < 0.05 || ncross >= 6) outcome = 3;
            }
        }

        // Capture = the backward ray reaches the PAST horizon. The outgoing
        // chart (exterior cameras) crosses it regularly, so r < r₊ is
        // capture. The ingoing chart (Gate 24 in-fall) is regular across the
        // FUTURE horizon instead: a camera inside r₊ sees the outside
        // universe through it (its backward rays leave outward), while a
        // backward ray heading inward from outside is bound for the past
        // horizon, which this chart reaches only as t → −∞.
        if (outcome == 0) {
            if (KS_S < 0.0) {
                if (rn < r_hor) outcome = 1;
            } else if (rn < r_hor + 0.02 && rn < r) {
                outcome = 1;
            }
        }
        if (outcome == 0 && rn > r_escape) {
            vec3 dxe, dpe;
            float dte;
            ks_rhs(x, P, dxe, dpe, dte);
            if (dot(x, dxe) > 0.0) {
                // far and receding: its direction of motion is the sky
                // direction the light came from (no parallax)
                esc_dir = normalize(dxe);
                outcome = 2;
            }
        }
    }
    if (outcome == 0) outcome = 1; // budget exhausted (deep photon-ring winding)

    // Higher-order (n ≥ 2) images: ask main() for a few extra jittered
    // samples so the thin photon-ring arcs are not point-sampled.
    sq_out = (nz >= 2 && min_r < 5.0) ? 0.30 : 0.0;
    ergo_out = (ergo_visit && outcome != 1) ? 1.0 : 0.0;

    if (riaf_on()) {
        // Rayleigh-Jeans brightness temperature; false colour (afmhot-like,
        // 1 at 5e10 K) or grey Tb/1e10 K for the CPU/GPU comparison
        float Tb = riaf_I / (PL_PREF * fp.riaf0.w * fp.riaf0.w);
        if ((pc.display_flags & 16u) != 0u) {
            acc += vec3(Tb / 1e10);
        } else {
            float v = Tb / 5e10;
            acc += vec3(clamp(2.0 * v, 0.0, 1.0), clamp(2.0 * v - 0.5, 0.0, 1.0),
                        clamp(2.0 * v - 1.0, 0.0, 1.0));
        }
    }
    disk_out = acc;
    if (outcome == 2) {
        float esc_th = acos(clamp(esc_dir.z, -1.0, 1.0));
        float esc_ph = atan(esc_dir.y, esc_dir.x);
        vec3 sky = shade_sky(esc_th, esc_ph + pc.cam_phi);
        // Received starlight shifted by g = ν_cam/ν_∞ = 1/P_t (the
        // backward P_t is the received photon's energy at infinity per unit
        // camera-frame energy): gravitational blueshift deep in the well,
        // Doppler + beaming for a moving observer. The map is RGB, not a
        // spectrum: each pixel is shifted as a 6500 K blackbody (stated
        // approximation) — colour and brightness from one ratio, B(gT)/B(T).
        if ((fp.flags & 1u) != 0u) {
            float gs = (P.x > 1e-6) ? 1.0 / P.x : 0.0;
            vec3 ref = max(blackbody_radiance(6500.0), vec3(1e-12));
            sky *= blackbody_radiance(6500.0 * gs) / ref;
        }
        sky_out = Tr * sky;
    } else {
        sky_out = vec3(0.0);
    }
    return Tr;
}

void main() {
    uvec2 pix = uvec2(gl_GlobalInvocationID.x,
                      gl_GlobalInvocationID.y + pc.y_offset);
    if (pix.x >= pc.width || pix.y >= pc.height) return;
    KS_S = fp.ks_chart;
    MAX_STEPS = int(fp.max_steps);

    // Progressive anti-aliasing: sample n of a static view is jittered by
    // the R2 low-discrepancy sequence (Roberts 2018) and folded into the
    // running mean in the accumulator. n = 0 is the un-jittered pixel
    // centre, so a single-sample render is exactly the historical image.
    const float R2_A1 = 0.7548776662466927, R2_A2 = 0.5698402909980532;
    vec2 jit2 = fract(vec2(0.5) + float(pc.accum_index) * vec2(R2_A1, R2_A2)) - 0.5;
    float px_f = float(pix.x) + 0.5 + jit2.x;
    float py_f = float(pix.y) + 0.5 + jit2.y;
    uint pix_seed = uint(pix.x) * 7919u ^ uint(pix.y) * 104729u ^
                    pc.accum_index * 2246822519u;

    // Adaptive supersampling, triggered by RAY GEOMETRY rather than a
    // fixed screen band. The caustic-squeezed families — the axis
    // columns at oblique views, the pole-caustic fan at near-polar
    // views — image a huge swathe of disk into each pixel, and one ray
    // per pixel point-samples that as speckle. The first sample reports
    // its squeeze depth; squeezed pixels get 4 or 8 further jittered
    // samples, everything else keeps single-sample cost. A screen-space
    // band cannot do this job: the polar-view caustic fan does not sit
    // on the image's central column.
    if (wormhole_on()) {
        vec3 cw = shade_wormhole(px_f, py_f);
        if (!all(lessThan(abs(cw), vec3(1e30)))) cw = vec3(0.0);
        if (pc.accum_index > 0u) {
            vec3 prev = imageLoad(hdr_images[pc.image_index], ivec2(pix)).rgb;
            cw = mix(prev, cw, pc.accum_weight);
        }
        imageStore(hdr_images[pc.image_index], ivec2(pix), vec4(cw, 1.0));
        return;
    }
    vec3 disk_acc, sky_acc;
    float sq0, ergo_acc;
    {
        float jj = hash01(pix_seed);
        vec3 d, sk;
        shade_sample(px_f, py_f, jj, d, sk, sq0, ergo_acc);
        disk_acc = d;
        sky_acc = sk;
    }
    float wsum = 1.0;
    float w_ex = smoothstep(0.02, 0.25, sq0);
    if (w_ex > 0.0) {
        int ns_extra = (sq0 > 0.60) ? 7 : 3;
        for (int s = 1; s <= ns_extra; ++s) {
            vec2 o = (s < 4)
                ? (vec2(float(s & 1), float(s >> 1)) - 0.5) * 0.5
                : (vec2(float(s & 1), float((s - 4) >> 1)) - 0.5) * 0.25;
            float jj = hash01(pix_seed ^ uint(s) * 15485863u);
            vec3 d, sk;
            float sq_d, er_d;
            shade_sample(px_f + o.x, py_f + o.y, jj, d, sk, sq_d, er_d);
            disk_acc += w_ex * d;
            sky_acc += w_ex * sk;
            ergo_acc += w_ex * er_d;
        }
        wsum += float(ns_extra) * w_ex;
    }
    // Linear radiance (disk + lensed sky). Exposure, bloom, white balance
    // and the tone curve are applied downstream in tonemap.comp.
    vec3 sky_c = (pc.sky_off != 0u) ? vec3(0.0) : sky_acc / wsum;
    vec3 cur = disk_acc / wsum + sky_c;
    if (ergo_acc > 0.0) {
        // ergoregion overlay: blend toward a blue of at least display level
        float lum = dot(cur, vec3(0.2126, 0.7152, 0.0722));
        cur = mix(cur, vec3(0.20, 0.55, 1.0) * max(lum, 0.25 * pc.sky_gain),
                  0.65 * ergo_acc / wsum);
    }
    if (!all(lessThan(abs(cur), vec3(1e30)))) cur = vec3(0.0); // NaN/Inf guard
    if (pc.accum_index > 0u) {
        vec3 prev = imageLoad(hdr_images[pc.image_index], ivec2(pix)).rgb;
        cur = mix(prev, cur, pc.accum_weight);
    }
    imageStore(hdr_images[pc.image_index], ivec2(pix), vec4(cur, 1.0));
}
