// Live full renderer (Phase 8): fp32 Carter/Mino geodesics + Novikov-Thorne
// disk + blackbody colormap + artistic preset LUT + starfield.
//
// Single precision is adequate here because the live camera sits near the
// hole (r_cam ≈ 40–120 M) where the radial potential spans ~1e7 rather than
// the 1e12 of the far validation camera; the per-step constraint projection
// (p_r = ±√R, p_θ = ±√Θ) again pins the on-shell relations. Science-grade
// output remains the fp64 shader / CPU pair validated in Gate 7.
//
// Bindless: binding 0 = storage images, binding 1 = sampled LUTs
//   slot 0: physical blackbody colormap (logT 500 K → 60000 K)
//   slot 1–4: preset display LUTs (§9): firestorm, gargantua, arctic, nebula
#version 450

layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D storage_images[16];
layout(set = 0, binding = 1) uniform sampler2D luts[16];

layout(push_constant) uniform PC {
    uint image_index;
    uint width, height;
    float time_s;
    float a_star, r_cam, incl_deg, fov;
    float disk_r_in, disk_r_out, exposure_ev;
    uint shade_mode;   // here: preset LUT slot (0 = physical blackbody only)
    uint y_offset;
    float flux_max;    // max Page-Thorne flux (CPU precomputed for T scaling)
    float t_peak_k;    // peak effective temperature [K]
    float turb;        // artistic turbulence amount 0..1 (0 = pure analytic)
    float cam_phi;     // camera orbit azimuth [rad]
    float time_scale;  // filament advection: Ω [1/M] per wall-clock second
    uint sky_off;      // 1 = fully black background (starfield hidden)
} pc;

const float PI = 3.14159265;

// Temporary diagnostic: colour disk hits by ray classification instead of
// physical shading (R = crossing order, G = pole proximity, B = feather
// weight). Never enabled in a committed build.
#define DEBUG_CLASSIFY 0

// ---- Carter potentials (fp32) ----------------------------------------------
struct Consts { float a, E, Lz, Q; };

float radial_potential(Consts cc, float r) {
    float P = cc.E * (r * r + cc.a * cc.a) - cc.a * cc.Lz;
    float delta = r * r - 2.0 * r + cc.a * cc.a;
    float K = (cc.Lz - cc.a * cc.E) * (cc.Lz - cc.a * cc.E) + cc.Q;
    return P * P - delta * K;
}
float radial_potential_deriv(Consts cc, float r) {
    float P = cc.E * (r * r + cc.a * cc.a) - cc.a * cc.Lz;
    float K = (cc.Lz - cc.a * cc.E) * (cc.Lz - cc.a * cc.E) + cc.Q;
    return 4.0 * cc.E * P * r - 2.0 * (r - 1.0) * K;
}
float polar_potential(Consts cc, float sth, float cth) {
    return cc.Q + cth * cth * (cc.a * cc.a * cc.E * cc.E -
                               cc.Lz * cc.Lz / (sth * sth));
}
float polar_potential_deriv(Consts cc, float sth, float cth) {
    return -cc.a * cc.a * cc.E * cc.E * 2.0 * sth * cth +
           2.0 * cc.Lz * cc.Lz * cth / (sth * sth * sth);
}

void rhs(Consts cc, float y[6], out float dy[6]) {
    float sth = sin(y[2]), cth = cos(y[2]);
    // fp32 polar clamp: the θ-force carries Lz²/sin³θ — at the fp64
    // science path's 1e-8 clamp that is resolvable, at fp32 with a fixed
    // step budget it detonates (1/s³ ~ 1e18) and scatters phantom
    // crossings along the exact axis column. 2e-3 rad ≈ 0.1°: displacing
    // the polar turning point of a near-axis ray by a tenth of a degree
    // is invisible; the numerical blow-up was not.
    if (abs(sth) < 2e-3) sth = (sth >= 0.0) ? 2e-3 : -2e-3;
    float r = y[1];
    float delta = r * r - 2.0 * r + cc.a * cc.a;
    float P = cc.E * (r * r + cc.a * cc.a) - cc.a * cc.Lz;
    dy[0] = 0.0; // coordinate time unused in the live path
    dy[1] = y[4];
    dy[2] = y[5];
    dy[3] = -(cc.a * cc.E - cc.Lz / (sth * sth)) + cc.a * P / delta;
    dy[4] = 0.5 * radial_potential_deriv(cc, r);
    dy[5] = 0.5 * polar_potential_deriv(cc, sth, cth);
}

void rk4_step(Consts cc, inout float y[6], float h) {
    float k1[6], k2[6], k3[6], k4[6], t[6];
    rhs(cc, y, k1);
    for (int i = 0; i < 6; ++i) t[i] = y[i] + 0.5 * h * k1[i];
    rhs(cc, t, k2);
    for (int i = 0; i < 6; ++i) t[i] = y[i] + 0.5 * h * k2[i];
    rhs(cc, t, k3);
    for (int i = 0; i < 6; ++i) t[i] = y[i] + h * k3[i];
    rhs(cc, t, k4);
    for (int i = 0; i < 6; ++i)
        y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
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
    return max(0.0, br / (x * x * x * x * denom * x * x));
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
    vec3 col = vec3(0.004, 0.005, 0.009);
    float deg = 180.0 / PI;
    float gt = mod(abs(theta) * deg, 10.0);
    float gp = mod(abs(phi) * deg, 10.0);
    if (gt < 0.12 || gt > 9.88 || gp < 0.12 || gp > 9.88)
        col = vec3(0.012, 0.015, 0.024);
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

// Linear-sRGB emission of the disk surface at BL radius r_x, azimuth phi_x,
// redshift factor g (blackbody at g·T_eff; bolometric (gT)^4 brightness).
vec3 disk_emission(float r_x, float phi_x, float g, float fade) {
    float F = pt_flux(r_x, pc.a_star, pc.disk_r_in);
    F *= turb_factor(r_x, phi_x, fade);
    // Artistic presets: exponential (stops-based) emissivity taper over the
    // outer rim — a linear fade barely moves the logarithmic display grade.
    // (Display only; the PHYSICAL path keeps the sharp model annulus.)
    if (pc.shade_mode > 0u) {
        float rn = (r_x - pc.disk_r_in) /
                   max(pc.disk_r_out - pc.disk_r_in, 1e-3);
        F *= exp2(-15.0 * smoothstep(0.55, 1.0, rn));
    }
    float T = pc.t_peak_k * pow(max(F, 0.0) / pc.flux_max, 0.25);
    float Tobs = g * T;
    float u = clamp((log(max(Tobs, 501.0)) - log(500.0)) /
                    (log(60000.0) - log(500.0)), 0.0, 1.0);
    vec3 hue = texture(luts[0], vec2(u, 0.5)).rgb;
    return hue * pow(Tobs / 6500.0, 4.0);
}

// Trace one camera ray (sub-pixel coordinates). Disk emission accumulates
// front-to-back with per-crossing opacity (the artistic rim is translucent,
// so stars and lensed inner structure show through the faded edge); the sky
// radiance is returned separately so main() can grade only the disk part.
// Returns the remaining transmittance toward the background.
float shade_sample(float px_f, float py_f, float jit, out vec3 disk_out,
                   out vec3 sky_out, out float sq_out) {
    float a = pc.a_star;
    float r0 = pc.r_cam;
    float th0 = pc.incl_deg * PI / 180.0;
    float x = 2.0 * px_f / float(pc.width) - 1.0;
    float yy = 1.0 - 2.0 * py_f / float(pc.height);
    float aspect = float(pc.height) / float(pc.width);
    float thalf = tan(0.5 * pc.fov);
    float tx = thalf * x, ty = thalf * yy * aspect;

    float s0 = sin(th0), c0 = cos(th0);
    float sig = r0 * r0 + a * a * c0 * c0;
    float del = r0 * r0 - 2.0 * r0 + a * a;
    float A = (r0 * r0 + a * a) * (r0 * r0 + a * a) - a * a * del * s0 * s0;
    float g_tt = -(1.0 - 2.0 * r0 / sig);
    float g_tp = -2.0 * a * r0 * s0 * s0 / sig;
    float g_rr = sig / del;
    float g_thth = sig;
    float g_pp = A * s0 * s0 / sig;
    float omg = -g_tp / g_pp;
    float alpha = sqrt(-(g_tt - g_tp * g_tp / g_pp));

    float inv = 1.0 / sqrt(1.0 + tx * tx + ty * ty);
    float n_r = -inv, n_th = -ty * inv, n_ph = tx * inv;

    float kt = 1.0 / alpha;
    float kr = n_r / sqrt(g_rr);
    float kth = n_th / sqrt(g_thth);
    float kph = omg / alpha + n_ph / sqrt(g_pp);

    Consts cc;
    cc.a = a;
    cc.E = -(g_tt * kt + g_tp * kph);
    cc.Lz = g_tp * kt + g_pp * kph;
    float pth0 = sig * kth;
    cc.Q = pth0 * pth0 + c0 * c0 * (cc.Lz * cc.Lz / (s0 * s0) - a * a * cc.E * cc.E);

    // Near-axial snap, SUB-PIXEL threshold: a ray with |Lz| << sqrt(Q)
    // skirts the axis at sin(theta_min) = |Lz|/sqrt(Q), where its
    // azimuth winds at Lz/sin²θ per unit λ — thousands of radians the
    // fp32 fixed-budget path cannot resolve. With Lz = 0 the polar
    // passage integrates exactly (θ runs through the pole, φ picks up
    // the correct π); the trajectory error is the detour itself. At
    // 2.5e-3·sqrt(Q) the rewritten family is ~1 px wide (an earlier
    // 3e-2 threshold rewrote a ~30 px band and left a visible trace).
    if (abs(cc.Lz) < 2.5e-3 * sqrt(max(cc.Q, 0.0))) cc.Lz = 0.0;

    float y[6];
    y[0] = 0.0; y[1] = r0; y[2] = th0; y[3] = 0.0;
    y[4] = sig * kr; y[5] = pth0;

    float r_hor = 1.0 + sqrt(max(0.0, 1.0 - a * a));
    float r_capture = r_hor * 1.005 + 1e-4;
    float r_escape = max(2.0 * r0, 300.0);

    int outcome = 0;
    vec3 acc = vec3(0.0);   // accumulated disk emission (front to back)
    float Tr = 1.0;         // transmittance toward the background
    int ncross = 0;         // translucent crossings so far (bounded)
    float veil = 0.0;       // opacity-weighted crossing depth: the
                            // display shaping below keys on this instead
                            // of the integer ncross, because a ray that
                            // grazed a 1%-opacity rim wisp is NOT
                            // meaningfully "behind the disk" — stepping
                            // the texture fade on the integer painted a
                            // hard horizontal seam across the bowl
    int cross_step = -1;    // step index of the first crossing
    float esc_th = 0.0, esc_ph = 0.0;
    float min_sth = 1.0;     // closest approach to the spin axis DURING
                             // the flight near the hole (r < r_cam/2):
                             // rays with small min|sinθ| went over a
                             // pole and their disk image is squeezed
                             // into the axis caustic. Seeded at 1 and
                             // radius-gated: seeding with the CAMERA's
                             // own sinθ classified every ray of a
                             // near-polar view as "squeezed" from step
                             // zero (the face-on "crayon" regression) —
                             // sitting near the axis at the camera end
                             // compresses nothing.
    int nturn = 0;           // polar turnings at high latitude: light that
                             // zig-zags over the pole repeatedly images the
                             // disk at flux ~e^(-2*pi*n) — real, but far
                             // below one pixel and one display stop; kept
                             // to two windings, suppressed beyond
    float prev_p5 = pth0;

    // Edge-on refinement: with the camera within ~10° of the disk plane
    // the whole central band of rays grazes the plane for most of its
    // path, and phase error accumulated at the relaxed base step makes
    // annulus hits resolve pseudo-randomly (speckle on the bowl rims; a
    // global 3× refinement probe renders them clean). Ramp the base
    // step down smoothly with viewing inclination — face-on views pay
    // nothing, edge-on views trade auto-resolution for correctness.
    float kstep = mix(1.0, 0.40, smoothstep(0.30, 0.10, abs(c0)));

    float prev_c = c0;
    float prev[6];
    for (int step = 0; step < 1200 && outcome == 0; ++step) {
        for (int i = 0; i < 6; ++i) prev[i] = y[i];
        // After the first translucent crossing the ray only supplies the
        // background behind the veil — step twice as fast and stop caring
        // after 500 further steps (keeps rim-piercing rays from burning the
        // whole budget at small radii). The CINEMATIC volumetric mode
        // never relaxes: coarse steps sample the fog in visible chunks.
        float relax = (ncross > 0 && pc.shade_mode != 5u) ? 2.0 : 1.0;
        // per-ray step dither: neighbouring rays otherwise share an
        // IDENTICAL deterministic step sequence, so every quantisation
        // boundary (budget cliff, marginal dip detection) lands at the
        // same place for whole patches of pixels and images as coherent
        // blocks; ±10% of per-ray phase turns that into per-pixel noise
        relax *= 0.90 + 0.20 * jit;
        float h = relax * kstep * 0.04 * y[1] / (abs(y[4]) + 1e-4);
        h = min(h, relax * kstep * 0.10 / (abs(y[5]) + 1.0));
        h = min(h, relax * kstep * 0.08);
        // polar-axis guard: the θ-equation stiffens as sinθ → 0 (Lz²/sin³θ);
        // under-resolved steps there alias into a dotted column of phantom
        // disk crossings on the spin axis. The wall grows as 1/sin³θ, so
        // the clamp must tighten QUADRATICALLY, not linearly.
        {
            float sguard = sin(y[2]), cguard = cos(y[2]);
            float sa = abs(sguard);
            h = min(h, 0.004 + 0.35 * sa * sa);
            // Stiffness-adaptive polar clamp: the θ-force Θ' carries
            // Lz²/sin³θ and can exceed the static clamp's resolving
            // power by an order of magnitude for rays TURNING near the
            // axis (the dashed-ladder artefact family). Bound the
            // momentum change per step relative to the momentum scale:
            // h ≲ |p_θ|/|Θ'|. The stiff segment is brief (~tens of
            // steps), so the budget is unaffected.
            if (sa < 0.35) {
                float sc = (sa < 2e-3) ? ((sguard >= 0.0) ? 2e-3 : -2e-3)
                                       : sguard;
                float tp = abs(polar_potential_deriv(cc, sc, cguard));
                // FLOORED: at the polar turning wall p_θ → 0 while Θ'
                // explodes, and an unfloored clamp stalls the ray in
                // microscopic steps until the budget dies — which imaged
                // as budget-quantised dark rungs on the axis column. The
                // reflection guard below handles the wall analytically;
                // steps below the floor buy nothing.
                h = min(h, max(0.35 * (abs(y[5]) + 0.5) / (tp + 1.0),
                               2.5e-3));
                // stochastic step phase: decorrelates any residual
                // quantisation between neighbouring rays so the axis-band
                // supersampling averages it smooth
                h *= 0.80 + 0.40 * fract(jit + 0.6180339887 * float(step));
            }
            // Plane-grazing refinement: rays that ride within
            // |cosθ| < 0.15 of the disk plane while radially inside the
            // (padded) annulus are the edge-on bowl-rim family — at the
            // relaxed base step their marginal dips and grazing hits
            // resolve pseudo-randomly (the speckle/staircase artefacts;
            // a 3× GLOBAL refinement probe renders them clean). Cap the
            // step for exactly that family instead of paying 3×
            // everywhere; also resolve their polar oscillation
            // half-period π/√(Lz²−a²E²) so no dip through the plane is
            // ever stepped over whole.
            if (abs(cguard) < 0.15 && y[1] > pc.disk_r_in - 2.0 &&
                y[1] < pc.disk_r_out + 6.0) {
                h = min(h, 0.045);
                float om2 = cc.Lz * cc.Lz - cc.a * cc.a * cc.E * cc.E;
                if (om2 > 1.0) h = min(h, 1.0 / sqrt(om2));
            }
        }
        float h_used = h;
        rk4_step(cc, y, h);
        // Polar turning-point reflection: if the step tunnelled into the
        // forbidden cone (Θ < 0 — the fp32 fixed-budget integrator cannot
        // resolve the 1/sin³θ wall however small the step), reflect the
        // polar motion at the wall instead of continuing with a corrupted
        // trajectory. This is what produced the ladder of phantom
        // crossings on the axis column: tunnelled rays re-crossed the
        // equator at pseudo-random radii.
        {
            float sref = sin(y[2]), cref = cos(y[2]);
            if (abs(sref) < 2e-3) sref = (sref >= 0.0) ? 2e-3 : -2e-3;
            if (polar_potential(cc, sref, cref) < 0.0) {
                y[2] = prev[2];
                y[5] = -prev[5];
            }
        }
        if (cross_step >= 0 && outcome == 0 && step - cross_step > 900)
            outcome = 1; // background budget exhausted: fade to black
                         // (900: at edge-on inclinations the path from a
                         // rim-piercing first crossing to the far bowl is
                         // long, and a lower cliff quantised into black
                         // checkerboard blocks on the bowl rims)

        float Rp = radial_potential(cc, y[1]);
        if (Rp >= 0.0) y[4] = (y[4] >= 0.0) ? sqrt(Rp) : -sqrt(Rp);
        float sth = sin(y[2]), cth = cos(y[2]);
        if (y[1] < 0.5 * r0) min_sth = min(min_sth, abs(sth));
        if (y[5] * prev_p5 < 0.0 && abs(cth) > 0.5) ++nturn;
        prev_p5 = y[5];
        if (abs(sth) > 1e-6) {
            float Tp = polar_potential(cc, sth, cth);
            if (Tp >= 0.0) y[5] = (y[5] >= 0.0) ? sqrt(Tp) : -sqrt(Tp);
        }

        // ---- CINEMATIC volumetric dust (shade_mode == 5; display only) ----
        // A flared slab of procedural dust orbiting at the real Ω(r),
        // sampled once per integrator step while the ray is inside it.
        // Geodesics, redshift and the Page-Thorne midplane are untouched:
        // the clouds are emissive fog composited front-to-back like the
        // translucent rim, and deleting this block restores the exact
        // thin-disk image.
        if (pc.shade_mode == 5u && outcome == 0) {
            float rm  = 0.5 * (prev[1] + y[1]);
            float thm = 0.5 * (prev[2] + y[2]);
            float zc  = rm * cos(thm);
            float Rc  = max(rm * abs(sin(thm)), 1e-3);
            float Hc  = 0.30 + 0.07 * Rc;               // flared height
            if (Rc > pc.disk_r_in * 0.85 && Rc < pc.disk_r_out * 1.05 &&
                abs(zc) < 2.5 * Hc) {
                float env = exp(-0.5 * (zc / Hc) * (zc / Hc));
                // radial containment: banks thin out toward the rim
                // instead of combing past it edge-on
                env *= smoothstep(pc.disk_r_in * 0.85,
                                  pc.disk_r_in * 1.10, Rc) *
                       (1.0 - smoothstep(pc.disk_r_out * 0.90,
                                         pc.disk_r_out * 1.05, Rc));
                // Cloud banks: isotropic 3-D value noise sampled at the
                // ORBITING world position. The previous 2-D lattice had
                // quarter-turn azimuthal cells against ~10%-of-radius
                // radial cells (~15:1 stretched): it could only ever
                // draw lines. Round world-space cells read as puffy
                // banks from any viewpoint and decorrelate rim-tangent
                // rays without per-ray jitter (no more combing).
                float Om  = 1.0 / (Rc * sqrt(Rc) + pc.a_star);
                float phm = 0.5 * (prev[3] + y[3]) + pc.cam_phi;
                // Advection by the exact Ω(r) shears ANY fixed pattern
                // into strings after a few differential turns (at
                // rotation 1000× that is seconds of wall clock). So the
                // field lives in two staggered phases that each re-form
                // every L_RE of coordinate time and cross-fade (flow
                // noise): each phase is advected by the exact Ω over
                // its own age only — orbiting banks forever, never a
                // stringified steady state.
                const float L_RE = 140.0;   // re-form period [M]
                float Aadv = pc.time_s * pc.time_scale;
                float nsum = 0.0;
                for (int k = 0; k < 2; ++k) {
                    float lifec = (Aadv + float(k) * 0.5 * L_RE) / L_RE;
                    float age   = fract(lifec) * L_RE;
                    float epoch = floor(lifec);
                    float wk    = 0.5 -
                                  0.5 * cos(6.2831853 * age / L_RE);
                    float phk   = phm + 18.0 * Om - Om * age;
                    vec3  q     = vec3(Rc * cos(phk), Rc * sin(phk),
                                       zc * 1.8) * 0.35 +
                                  vec3(epoch * 13.71 + float(k) * 61.3,
                                       epoch * 7.31, epoch * 5.17);
                    nsum += wk * fbm3(q);
                }
                // blobby threshold: separated banks with real gaps
                // (stars and disk visible between clumps), not mist
                float dens = clamp(pc.turb, 0.0, 1.0) * env *
                             pow(smoothstep(0.44, 0.80, nsum), 1.5) *
                             1.7;
                if (dens > 1e-4) {
                    // local colour: blackbody at the orbiting gas's
                    // redshifted temperature, exactly like the midplane
                    float Fv = pt_flux(Rc, pc.a_star, pc.disk_r_in);
                    float sg = Rc * Rc;
                    float dl = Rc * Rc - 2.0 * Rc + a * a;
                    float AA = (Rc * Rc + a * a) * (Rc * Rc + a * a) -
                               a * a * dl;
                    float gtt = -(1.0 - 2.0 * Rc / sg);
                    float gtp = -2.0 * a * Rc / sg;
                    float gpp = AA / sg;
                    float dn = -(gtt + 2.0 * Om * gtp + Om * Om * gpp);
                    if (dn > 0.0) {
                        float g_v = 1.0 / ((1.0 / sqrt(dn)) *
                                           (cc.E - Om * cc.Lz));
                        float Tv = pc.t_peak_k *
                                   pow(max(Fv, 0.0) / pc.flux_max, 0.25);
                        float Tobs = clamp(g_v, 0.05, 4.0) * max(Tv, 900.0);
                        float uu = clamp((log(max(Tobs, 501.0)) -
                                          log(500.0)) /
                                         (log(60000.0) - log(500.0)),
                                         0.0, 1.0);
                        vec3 huev = texture(luts[0], vec2(uu, 0.5)).rgb;
                        float dtau = 0.8 * dens * h * rm;
                        float em = 1.0 - exp(-dtau);
                        acc += Tr * em * huev *
                               pow(Tobs / 6500.0, 4.0) * 0.8;
                        Tr *= exp(-0.7 * dtau);     // dust is semi-opaque
                        if (Tr < 0.05) outcome = 3;
                    }
                }
            }
        }

        // Radial quick-reject: only bisect crossings that can touch the
        // annulus. Without this, translucent grazing rays skimming the disk
        // plane pay the 18-step bisection on every oscillation and the
        // frame rate collapses by an order of magnitude. Deeply
        // axis-squeezed rays (display presets) bypass the reject: their
        // crossings count as caustic coverage wherever they land (the
        // band-limited hit test below).
        bool may_hit =
            (max(prev[1], y[1]) >= pc.disk_r_in - 0.5 &&
             min(prev[1], y[1]) <= pc.disk_r_out + 0.5) ||
            (pc.shade_mode > 0u && min_sth < 0.006);
        if (may_hit && prev_c * cth < 0.0 &&
            (abs(prev_c) > 1e-7 || abs(cth) > 1e-7)) {
            float cr[6];
            for (int i = 0; i < 6; ++i) cr[i] = y[i];
            if (abs(prev_c) > 5e-3 || abs(cth) > 5e-3) {
                // bisect the equatorial crossing
                float lo = 0.0, hi = 1.0; // fraction of the last step
                // h_used is the EXACT step that produced prev → y,
                // including the adaptive clamps and the stochastic
                // phase; re-deriving it from the static clamps alone
                // (as this used to) re-integrated the fraction with a
                // slightly different step and landed at the wrong state
                float hh = h_used;
                for (int it = 0; it < 12; ++it) {
                    float mid = 0.5 * (lo + hi);
                    float tr[6];
                    for (int i = 0; i < 6; ++i) tr[i] = prev[i];
                    rk4_step(cc, tr, mid * hh);
                    if (cos(tr[2]) * prev_c > 0.0) lo = mid; else hi = mid;
                    for (int i = 0; i < 6; ++i) cr[i] = tr[i];
                }
            } else {
                // grazing in-plane: no bisection (rim-skimming rays used
                // to collapse the frame rate re-bisecting every
                // oscillation) — but do NOT take the raw endpoint either:
                // that quantises the crossing radius by the step length,
                // which imaged the annulus edges as hit/miss staircase
                // blocks at edge-on views. A linear zero of cosθ across
                // the step is free and lands within the step error.
                float dc = prev_c - cth;
                float tz = (abs(dc) > 1e-12)
                               ? clamp(prev_c / dc, 0.0, 1.0) : 0.5;
                for (int i = 0; i < 6; ++i)
                    cr[i] = mix(prev[i], y[i], tz);
            }
            float r_x = cr[1];
            // Axis-squeeze depth: min_sth is radius-gated and seeded
            // at 1 (see its declaration), so it measures genuine polar
            // passages near the hole and the absolute thresholds below
            // hold at every viewing inclination.
            float u_ax = min_sth;
            // For the DEEPEST squeeze the crossing radius sweeps the
            // whole equatorial plane, so half the samples miss the
            // annulus and no amount of averaging beats the binomial
            // noise under the 24-stop display lift. Band-limit the hit
            // test itself: any equator crossing counts as caustic
            // coverage at fractional opacity (display presets only).
            float deep = (pc.shade_mode > 0u)
                ? 1.0 - smoothstep(0.003, 0.006, u_ax) : 0.0;
            bool in_ann = r_x >= pc.disk_r_in && r_x <= pc.disk_r_out;
            // coverage only near the annulus: a pole-passer crossing the
            // far plane (r >> r_out) genuinely sees no disk there
            if ((in_ann || (deep > 0.0 && r_x < 1.5 * pc.disk_r_out)) &&
                nturn < 3) {
                // The deepest pole-passers image the ENTIRE annulus
                // inside one pixel: any point sample of F(r_x) is a
                // lottery ticket (the grainy axis needle). Shade them
                // at a fixed representative radius instead — a stable
                // stand-in for the annulus integral the pixel actually
                // sees. Display presets only.
                // GRADED pull, covering the whole squeezed family: at
                // near-polar viewing the family is a wide image fan and
                // any point-sampled or ordering structure inside it
                // (sub-image rungs, annulus-edge lottery) is fp32 noise
                // below the resolvable scale — the pulled radius gives
                // the fan its smooth band-limited glow, and because the
                // pulled radius is interior the first crossing opaquely
                // dominates (no ordering rungs).
                float rep = (pc.shade_mode > 0u)
                    ? 1.0 - smoothstep(0.003, 0.05, u_ax) : 0.0;
                float r_sh = mix(r_x,
                                 mix(pc.disk_r_in, pc.disk_r_out, 0.50),
                                 rep);
                float Om = 1.0 / (r_sh * sqrt(r_sh) + a);
                float sg = r_sh * r_sh;
                float dl = r_sh * r_sh - 2.0 * r_sh + a * a;
                float AA = (r_sh * r_sh + a * a) * (r_sh * r_sh + a * a) -
                           a * a * dl;
                float gtt = -(1.0 - 2.0 * r_sh / sg);
                float gtp = -2.0 * a * r_sh / sg;
                float gpp = AA / sg;
                float dn = -(gtt + 2.0 * Om * gtp + Om * Om * gpp);
                float g_x = 1.0 / ((1.0 / sqrt(dn)) * (cc.E - Om * cc.Lz));

                // Opacity: the PHYSICAL preset keeps the opaque model disk;
                // artistic presets turn translucent over the faded rim so
                // the disk dissolves into space instead of cutting a black
                // silhouette out of the starfield.
                float o = 1.0;
                if (pc.shade_mode > 0u) {
                    float rn = (r_sh - pc.disk_r_in) /
                               max(pc.disk_r_out - pc.disk_r_in, 1e-3);
                    o = smoothstep(0.98, 0.60, rn);
                }
                // De-aliasing of squeezed images (the axis-column rungs
                // and the dotted sub-image patches). Two mechanisms:
                //  - TEXTURE: pixels there span a huge swathe of disk
                //    surface, so the filament pattern must fade toward
                //    its mean (the correct band-limited average);
                //  - GEOMETRY: the crossing radius sweeps the whole
                //    annulus across a few pixel rows, so the hard
                //    r_in/r_out edges image as a hit/miss barcode —
                //    soften the annulus window for those rays only.
                // "Squeezed" = the ray went over a pole (its image is
                // compressed into the axis column) or is a higher-order
                // sub-image (demagnified ~e^-pi per extra crossing).
                float fade = mix(mix(1.0, 0.30, min(veil, 1.0)),
                                 0.0, clamp(veil - 1.0, 0.0, 1.0));
                // Pole-passers carry an unresolvable azimuth error (the
                // φ-winding above), so their filament-texture phase is
                // garbage — fading the texture to its mean is both the
                // honest band-limited average of a sub-pixel-squeezed
                // image AND the cure for the dotted axis column.
                fade *= smoothstep(0.015, 0.08, u_ax);
                // Soft annulus window for SQUEEZED images (display
                // presets only; PHYSICAL keeps the raw model annulus).
                // Squeezed = higher-order sub-image (demagnified ~e^-π
                // per extra crossing) OR a pole-passing ray (its disk
                // image is compressed into the axis column). For those
                // rays the crossing radius sweeps the whole annulus
                // across a few pixels, so the hard r_in/r_out edges
                // image as a hit/miss barcode; softening the window is
                // the band-limited average of the edge, same argument
                // as the texture fade. Interior hits (w_edge = 1) —
                // including the whole magnified far-side arch — are
                // untouched.
                float span = max(pc.disk_r_out - pc.disk_r_in, 1e-3);
                float w_edge =
                    smoothstep(pc.disk_r_in,
                               pc.disk_r_in + 0.10 * span, r_sh) *
                    (1.0 - smoothstep(pc.disk_r_out - 0.30 * span,
                                      pc.disk_r_out, r_sh));
                float squeeze = max(min(veil, 1.0),
                                    1.0 - smoothstep(0.015, 0.10,
                                                     u_ax));
                if (pc.shade_mode > 0u) o *= mix(1.0, w_edge, squeeze);
                // out-of-annulus caustic coverage: the crossing stands
                // in for the fraction of the swept plane that carries
                // disk light, not for a certain hit
                if (!in_ann) o *= 0.35 * deep;
#if DEBUG_CLASSIFY
                acc = vec3(float(ncross) / 3.0,
                           1.0 - smoothstep(0.0, 0.15, min_sth),
                           w_edge);
                outcome = 3;
#else
                acc += Tr * o *
                       disk_emission(r_sh, cr[3] + pc.cam_phi, g_x, fade);
                Tr *= 1.0 - o;
#endif
                veil += o;
                ++ncross;
                if (cross_step < 0) cross_step = step;
                // stop when effectively opaque, or after enough translucent
                // passes that further contributions are negligible
                if (Tr < 0.05 || ncross >= 6) outcome = 3;
            }
        }
        prev_c = cth;

        if (outcome == 0 && y[1] < r_capture) outcome = 1;
        if (outcome == 0 && y[1] > r_escape && y[4] > 0.0) {
            outcome = 2;
            esc_th = y[2];
            esc_ph = y[3];
        }
    }
    if (outcome == 0) outcome = 1;

    // report the axis-squeeze depth so main() can supersample exactly
    // the caustic-squeezed pixels, wherever they land on screen
    sq_out = 1.0 - smoothstep(0.015, 0.15, min_sth);

    disk_out = acc;
    // camera azimuth rotates the whole sky with the orbit; the sky shines
    // through whatever transmittance the disk crossings left over
    sky_out = (outcome == 2)
                  ? Tr * shade_starfield(esc_th, esc_ph + pc.cam_phi)
                  : vec3(0.0);
    return Tr;
}

void main() {
    uvec2 pix = uvec2(gl_GlobalInvocationID.x,
                      gl_GlobalInvocationID.y + pc.y_offset);
    if (pix.x >= pc.width || pix.y >= pc.height) return;

    float px_f = float(pix.x) + 0.5;
    float py_f = float(pix.y) + 0.5;

    // Adaptive supersampling, triggered by RAY GEOMETRY rather than a
    // fixed screen band. The caustic-squeezed families — the axis
    // columns at oblique views, the pole-caustic fan at near-polar
    // views — image a huge swathe of disk into each pixel, and one ray
    // per pixel point-samples that as speckle. The first sample reports
    // its squeeze depth; squeezed pixels get 4 or 8 further jittered
    // samples, everything else keeps single-sample cost. A screen-space
    // band cannot do this job: the polar-view caustic fan does not sit
    // on the image's central column.
    vec3 disk_acc, sky_acc;
    float sq0;
    {
        float jj = hash01(uint(pix.x) * 7919u ^ uint(pix.y) * 104729u);
        vec3 d, sk;
        shade_sample(px_f, py_f, jj, d, sk, sq0);
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
            float jj = hash01(uint(pix.x) * 7919u ^ uint(pix.y) * 104729u ^
                              uint(s) * 15485863u);
            vec3 d, sk;
            float sq_d;
            shade_sample(px_f + o.x, py_f + o.y, jj, d, sk, sq_d);
            disk_acc += w_ex * d;
            sky_acc += w_ex * sk;
        }
        wsum += float(ns_extra) * w_ex;
    }
#if DEBUG_CLASSIFY
    imageStore(storage_images[pc.image_index], ivec2(pix),
               vec4(clamp(disk_acc / wsum, 0.0, 1.0), 1.0));
    if (true) return;
#endif
    float ev = exp2(pc.exposure_ev);
    vec3 disk_c = disk_acc / wsum * ev;
    vec3 sky_c = sky_acc / wsum * ev;
    if (pc.sky_off != 0u) sky_c = vec3(0.0); // black-background toggle

    // artistic preset grade (display only; §9 — physics untouched)
    if (pc.shade_mode > 0u) {
        // palette remap on COMPRESSED log display luminance: the physical
        // (gT)^4 brightness spans ~4 decades across the disk, which a linear
        // display crushes to black outside the beamed side. A wide log
        // window (the same trick photographic tone mapping uses) keeps the
        // whole disk inside the LUT ramp so the filaments read everywhere.
        // The LUT PNGs are sRGB-encoded: the sample IS the display colour.
        // The luma floor sits BELOW the 24-stop window so a fully faded
        // pixel maps to exactly 0 = the LUT's true black; the (attenuated)
        // sky adds on top, so the translucent rim shows stars through it.
        float luma = dot(disk_c, vec3(0.2126, 0.7152, 0.0722));
        float lu = clamp(1.0 + log2(max(luma, 1e-9)) / 24.0, 0.0, 1.0);
        // ±0.75-texel dither: at dim exposures whole disk regions live
        // in the bottom few entries of the 256-entry ramp, and smooth
        // gradients quantise into visible terraces ("crayon" bands) —
        // per-pixel dither trades the bands for invisible grain
        lu = clamp(lu + (hash01(pix.x * 2654435761u ^ pix.y * 40503u) -
                         0.5) * 0.006, 0.0, 1.0);
        vec3 graded = texture(luts[pc.shade_mode], vec2(lu, 0.5)).rgb;
        vec3 sky_disp = pow(clamp(sky_c, 0.0, 1.0), vec3(1.0 / 2.2));
        imageStore(storage_images[pc.image_index], ivec2(pix),
                   vec4(clamp(graded + sky_disp, 0.0, 1.0), 1.0));
        return;
    }
    vec3 col = pow(clamp(disk_c + sky_c, 0.0, 1.0), vec3(1.0 / 2.2));
    imageStore(storage_images[pc.image_index], ivec2(pix), vec4(col, 1.0));
}
