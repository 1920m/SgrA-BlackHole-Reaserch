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
                   out vec3 sky_out) {
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

    // Near-axial snap: a ray with |Lz| << sqrt(Q) skirts the spin axis at
    // sin(theta_min) = |Lz|/sqrt(Q) — for |Lz| < 3% sqrt(Q) that detour is
    // under 0.03 rad, far below a pixel, but its 1/sin^3(theta) wall is
    // exactly what the fp32 fixed-budget integrator cannot resolve (the
    // dashed-column artefact). Setting Lz = 0 removes the wall from the
    // dynamics identically; the trajectory error is the sub-pixel detour.
    if (abs(cc.Lz) < 0.03 * sqrt(max(cc.Q, 0.0))) cc.Lz = 0.0;

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
    int cross_step = -1;    // step index of the first crossing
    float esc_th = 0.0, esc_ph = 0.0;
    float min_sth = abs(s0); // closest approach to the spin axis so far:
                             // rays with small min|sinθ| went over a pole,
                             // and their disk image is squeezed into the
                             // axis column (the caustic)
    int nturn = 0;           // polar turnings at high latitude: light that
                             // zig-zags over the pole repeatedly images the
                             // disk at flux ~e^(-2*pi*n) — real, but far
                             // below one pixel and one display stop; kept
                             // to two windings, suppressed beyond
    float prev_p5 = pth0;
    float turn_sth = 1.0;    // smallest |sinθ| at which the polar motion
                             // REVERSED: rays that turn inside the axis
                             // cone are the ones the fp32 path quantises
                             // into the dashed column; rays that pass
                             // over the pole without turning render fine

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
        float h = relax * 0.04 * y[1] / (abs(y[4]) + 1e-4);
        h = min(h, relax * 0.10 / (abs(y[5]) + 1.0));
        h = min(h, relax * 0.08);
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
                h = min(h, 0.35 * (abs(y[5]) + 0.5) / (tp + 1.0));
                // stochastic step phase: decorrelates any residual
                // quantisation between neighbouring rays so the axis-band
                // supersampling averages it smooth
                h *= 0.80 + 0.40 * fract(jit + 0.6180339887 * float(step));
            }
        }
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
        if (cross_step >= 0 && outcome == 0 &&
            step - cross_step > (pc.shade_mode == 5u ? 900 : 500))
            outcome = 1; // background budget exhausted: fade to black

        float Rp = radial_potential(cc, y[1]);
        if (Rp >= 0.0) y[4] = (y[4] >= 0.0) ? sqrt(Rp) : -sqrt(Rp);
        float sth = sin(y[2]), cth = cos(y[2]);
        min_sth = min(min_sth, abs(sth));
        if (y[5] * prev_p5 < 0.0 && abs(cth) > 0.5) {
            ++nturn;
            turn_sth = min(turn_sth, abs(sth));
        }
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
            float Hc  = 0.22 + 0.055 * Rc;              // flared height
            if (Rc > pc.disk_r_in * 0.85 && Rc < pc.disk_r_out * 1.05 &&
                abs(zc) < 2.5 * Hc) {
                float env = exp(-0.5 * (zc / Hc) * (zc / Hc));
                // radial containment: banks thin out toward the rim
                // instead of combing past it edge-on
                env *= smoothstep(pc.disk_r_in * 0.85,
                                  pc.disk_r_in * 1.10, Rc) *
                       (1.0 - smoothstep(pc.disk_r_out * 0.90,
                                         pc.disk_r_out * 1.05, Rc));
                // wisps: the equatorial filament lattice broken up by a
                // second lattice in (log r, z/H) so the fog reads as 3-D
                // cloud banks, not a painted plane; both advect with Ω.
                float Om  = 1.0 / (Rc * sqrt(Rc) + pc.a_star);
                float phv = (y[3] + pc.cam_phi) + 120.0 * Om -
                            Om * pc.time_s * pc.time_scale;
                const float kTwoPi2 = 6.28318530718;
                // long azimuthal coherence (quarter-turn cells) and only
                // two octaves: cloud BANKS, not confetti — the fine
                // detail comes from the vertical breakup below. A small
                // per-ray phase jitter decorrelates the lattice between
                // neighbouring rays so edge-on rims do not comb.
                float perv = 4.0;
                vec2  pv   = vec2(log(Rc) * 9.0,
                                  (phv + 0.05 * jit) * (perv / kTwoPi2));
                float n1 = 0.0, wv = 0.60, wsumv = 0.0;
                for (int i = 0; i < 2; ++i) {
                    n1 += wv * vnoise(pv, perv);
                    wsumv += wv;
                    pv = vec2(pv.x * 1.93 + 17.13, pv.y * 2.0);
                    perv *= 2.0; wv *= 0.55;
                }
                n1 /= wsumv;
                float n2 = vnoise(vec2(log(Rc) * 7.0 + 31.7,
                                       zc / Hc * 1.4 + 0.35 * phv), 1e6);
                // steep threshold: separated cloud banks with real gaps
                // (stars and disk visible between clumps), not uniform
                // mist
                float dens = clamp(pc.turb, 0.0, 1.0) * env *
                             pow(max(2.2 * (0.9 * n1 + 0.5 * n2 - 0.45),
                                     0.0), 1.8);
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
        // frame rate collapses by an order of magnitude.
        bool may_hit =
            max(prev[1], y[1]) >= pc.disk_r_in - 0.5 &&
            min(prev[1], y[1]) <= pc.disk_r_out + 0.5;
        if (may_hit && prev_c * cth < 0.0 &&
            (abs(prev_c) > 1e-7 || abs(cth) > 1e-7)) {
            float cr[6];
            for (int i = 0; i < 6; ++i) cr[i] = y[i];
            if (abs(prev_c) > 5e-3 || abs(cth) > 5e-3) {
                // bisect the equatorial crossing
                float lo = 0.0, hi = 1.0; // fraction of the last step
                // reconstruct the step size (MUST mirror the controller
                // above exactly, including the polar clamp, or the
                // re-integrated fraction lands at the wrong state)
                float hh = relax * 0.04 * prev[1] / (abs(prev[4]) + 1e-4);
                hh = min(hh, relax * 0.10 / (abs(prev[5]) + 1.0));
                hh = min(hh, relax * 0.08);
                {
                    float sg2 = abs(sin(prev[2]));
                    hh = min(hh, 0.004 + 0.35 * sg2 * sg2);
                }
                for (int it = 0; it < 12; ++it) {
                    float mid = 0.5 * (lo + hi);
                    float tr[6];
                    for (int i = 0; i < 6; ++i) tr[i] = prev[i];
                    rk4_step(cc, tr, mid * hh);
                    if (cos(tr[2]) * prev_c > 0.0) lo = mid; else hi = mid;
                    for (int i = 0; i < 6; ++i) cr[i] = tr[i];
                }
            }
            // else: grazing in-plane — the endpoint IS the crossing to
            // step accuracy; bisecting every oscillation is what used to
            // collapse the frame rate on rim-skimming rays
            float r_x = cr[1];
            if (r_x >= pc.disk_r_in && r_x <= pc.disk_r_out &&
                nturn < 3) {
                float Om = 1.0 / (r_x * sqrt(r_x) + a);
                float sg = r_x * r_x;
                float dl = r_x * r_x - 2.0 * r_x + a * a;
                float AA = (r_x * r_x + a * a) * (r_x * r_x + a * a) - a * a * dl;
                float gtt = -(1.0 - 2.0 * r_x / sg);
                float gtp = -2.0 * a * r_x / sg;
                float gpp = AA / sg;
                float dn = -(gtt + 2.0 * Om * gtp + Om * Om * gpp);
                float g_x = 1.0 / ((1.0 / sqrt(dn)) * (cc.E - Om * cc.Lz));

                // Opacity: the PHYSICAL preset keeps the opaque model disk;
                // artistic presets turn translucent over the faded rim so
                // the disk dissolves into space instead of cutting a black
                // silhouette out of the starfield.
                float o = 1.0;
                if (pc.shade_mode > 0u) {
                    float rn = (r_x - pc.disk_r_in) /
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
                float fade = (ncross == 0) ? 1.0
                             : ((ncross == 1) ? 0.30 : 0.0);
                // Polar-cone feather (display presets only): rays that
                // dove within ~7 degrees of the spin axis take a polar
                // detour the fp32 fixed-budget integrator quantises into
                // a dashed axis column. The genuine flux of those images
                // is orders below the disk (the fp64 reference shows no
                // visible column at these geometries in linear display);
                // it only reads here because the 24-stop log grade lifts
                // it. Feather it out rather than display its noise.
                // Every near-axis ray is one of three kinds: snapped
                // axial (Lz -> 0 above; integrates cleanly, KEEPS its
                // light — it paints the arch continuously across the
                // axis), a turning ray whose polar turning point sits in
                // the numerically fragile cone (suppressed — these are
                // the quantised dash makers), or a ray turning farther
                // out (untouched). Pass-over rays have turn_sth = 1, so
                // feathering on the TURNING latitude never dims them.
                float w_pole = 1.0;
                if (ncross >= 1) {
                    // higher-order sub-images: keep, but band-limit the
                    // texture (demagnified ~e^-pi per extra crossing)
                    float span =
                        max(pc.disk_r_out - pc.disk_r_in, 1e-3);
                    float w_edge =
                        smoothstep(pc.disk_r_in,
                                   pc.disk_r_in + 0.10 * span, r_x) *
                        (1.0 - smoothstep(pc.disk_r_out - 0.30 * span,
                                          pc.disk_r_out, r_x));
                    o *= w_edge;
                }
                o *= w_pole;
#if DEBUG_CLASSIFY
                acc = vec3(float(ncross) / 3.0,
                           1.0 - smoothstep(0.0, 0.15, min_sth),
                           w_pole);
                outcome = 3;
#else
                acc += Tr * o *
                       disk_emission(r_x, cr[3] + pc.cam_phi, g_x, fade);
                Tr *= 1.0 - o;
#endif
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

    // The spin axis projects onto the central image column, where the
    // lensed polar caustic of the far-side disk is ~1 px wide and aliases
    // badly at one ray per pixel. Supersample those columns (2×2); the
    // band scales with resolution so upscaled internal targets stay
    // covered.
    float ax_px = abs(px_f - 0.5 * float(pc.width));
    float axis_halfwidth = max(6.0, 0.025 * float(pc.width));
    bool near_axis = ax_px < axis_halfwidth;

    // With the stiffness-adaptive polar clamp the axis columns integrate
    // cleanly except for the exactly-axial (Lz-snapped) family, whose
    // remaining 1-2 px trace the display presets bridge from the first
    // clean column each side. PHYSICAL stays raw by policy.
    const float kBridge = 10.0;
    bool bridging = (pc.shade_mode > 0u) && (ax_px < kBridge);
    vec3 disk_acc = vec3(0.0), sky_acc = vec3(0.0);
    int ns = near_axis ? 4 : 1;
    if (bridging) {
        vec3 dl = vec3(0.0), dr = vec3(0.0);
        vec3 sl = vec3(0.0), sr = vec3(0.0);
        float xl = 0.5 * float(pc.width) - kBridge;
        float xr = 0.5 * float(pc.width) + kBridge;
        for (int s = 0; s < 4; ++s) {
            vec2 o = (vec2(float(s & 1), float(s >> 1)) - 0.5) * 0.5;
            float jl = hash01(uint(pix.x) * 7919u ^ uint(pix.y) * 104729u ^
                              uint(s) * 15485863u);
            vec3 d, sk;
            shade_sample(xl + o.x, py_f + o.y, jl, d, sk);
            dl += d; sl += sk;
            shade_sample(xr + o.x, py_f + o.y, fract(jl + 0.5), d, sk);
            dr += d; sr += sk;
        }
        float t = clamp(0.5 + 0.5 * (px_f - 0.5 * float(pc.width)) /
                        kBridge, 0.0, 1.0);
        disk_acc = mix(dl, dr, t);
        sky_acc = mix(sl, sr, t);
        ns = 4;
    } else {
        for (int s = 0; s < ns; ++s) {
            vec2 o = (ns == 1) ? vec2(0.0)
                     : (vec2(float(s & 1), float(s >> 1)) - 0.5) * 0.5;
            float jj = hash01(uint(pix.x) * 7919u ^ uint(pix.y) * 104729u ^
                              uint(s) * 15485863u);
            vec3 d, sk;
            shade_sample(px_f + o.x, py_f + o.y, jj, d, sk);
            disk_acc += d;
            sky_acc += sk;
        }
    }
#if DEBUG_CLASSIFY
    imageStore(storage_images[pc.image_index], ivec2(pix),
               vec4(clamp(disk_acc / float(ns), 0.0, 1.0), 1.0));
    if (true) return;
#endif
    float ev = exp2(pc.exposure_ev);
    vec3 disk_c = disk_acc / float(ns) * ev;
    vec3 sky_c = sky_acc / float(ns) * ev;

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
        vec3 graded = texture(luts[pc.shade_mode], vec2(lu, 0.5)).rgb;
        vec3 sky_disp = pow(clamp(sky_c, 0.0, 1.0), vec3(1.0 / 2.2));
        imageStore(storage_images[pc.image_index], ivec2(pix),
                   vec4(clamp(graded + sky_disp, 0.0, 1.0), 1.0));
        return;
    }
    vec3 col = pow(clamp(disk_c + sky_c, 0.0, 1.0), vec3(1.0 / 2.2));
    imageStore(storage_images[pc.image_index], ivec2(pix), vec4(col, 1.0));
}
