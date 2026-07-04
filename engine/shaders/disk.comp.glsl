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
    if (abs(sth) < 1e-6) sth = (sth >= 0.0) ? 1e-6 : -1e-6;
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
float hash2(vec2 i) {
    uvec2 q = uvec2(ivec2(i) + 8192);
    uint h = q.x * 1597334677u ^ q.y * 3812015801u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return float(h) * (1.0 / 4294967296.0);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash2(i), hash2(i + vec2(1, 0)), u.x),
               mix(hash2(i + vec2(0, 1)), hash2(i + vec2(1, 1)), u.x), u.y);
}
float turb_factor(float r, float phi) {
    if (pc.turb <= 0.0) return 1.0;
    float Om = 1.0 / (r * sqrt(r) + pc.a_star);   // Kerr circular orbit
    // time_scale = rot_speed / (GM/c³): at rot_speed 1 this is the true
    // wall-clock rotation of Sgr A* (ISCO orbit ≈ 8.5 min — patience).
    float ph = phi - Om * pc.time_s * pc.time_scale;
    // filaments: broad lanes radially (log r), stretched along the orbit
    vec2 p = vec2(log(r) * 9.0, ph * 3.0);
    float n = 0.0, w = 0.5, wsum = 0.0;
    for (int i = 0; i < 5; ++i) {
        n += w * vnoise(p);
        wsum += w;
        p = p * vec2(2.03, 1.91) + vec2(17.13, 9.71);
        w *= 0.58;
    }
    n /= wsum;                                     // mean 0.5
    float f = mix(1.0, 0.25 + 1.5 * n, pc.turb);   // mean stays 1
    return f * f;                                  // deepen the contrast
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

void main() {
    uvec2 pix = uvec2(gl_GlobalInvocationID.x,
                      gl_GlobalInvocationID.y + pc.y_offset);
    if (pix.x >= pc.width || pix.y >= pc.height) return;

    float a = pc.a_star;
    float r0 = pc.r_cam;
    float th0 = pc.incl_deg * PI / 180.0;
    float x = 2.0 * (float(pix.x) + 0.5) / float(pc.width) - 1.0;
    float yy = 1.0 - 2.0 * (float(pix.y) + 0.5) / float(pc.height);
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

    float y[6];
    y[0] = 0.0; y[1] = r0; y[2] = th0; y[3] = 0.0;
    y[4] = sig * kr; y[5] = pth0;

    float r_hor = 1.0 + sqrt(max(0.0, 1.0 - a * a));
    float r_capture = r_hor * 1.005 + 1e-4;
    float r_escape = max(2.0 * r0, 300.0);

    int outcome = 0;
    float disk_r = 0.0, disk_g = 1.0, disk_ph = 0.0;
    float esc_th = 0.0, esc_ph = 0.0;

    float prev_c = c0;
    float prev[6];
    for (int step = 0; step < 1200 && outcome == 0; ++step) {
        for (int i = 0; i < 6; ++i) prev[i] = y[i];
        float h = 0.04 * y[1] / (abs(y[4]) + 1e-4);
        h = min(h, 0.10 / (abs(y[5]) + 1.0));
        h = min(h, 0.08);
        // polar-axis guard: the θ-equation stiffens as sinθ → 0 (Lz²/sin³θ);
        // under-resolved steps there alias into a dotted column of phantom
        // disk crossings on the spin axis
        h = min(h, 0.015 + 0.25 * abs(sin(y[2])));
        rk4_step(cc, y, h);

        float Rp = radial_potential(cc, y[1]);
        if (Rp >= 0.0) y[4] = (y[4] >= 0.0) ? sqrt(Rp) : -sqrt(Rp);
        float sth = sin(y[2]), cth = cos(y[2]);
        if (abs(sth) > 1e-6) {
            float Tp = polar_potential(cc, sth, cth);
            if (Tp >= 0.0) y[5] = (y[5] >= 0.0) ? sqrt(Tp) : -sqrt(Tp);
        }

        if (prev_c * cth < 0.0 && (abs(prev_c) > 1e-7 || abs(cth) > 1e-7)) {
            // bisect the equatorial crossing
            float lo = 0.0, hi = 1.0; // fraction of the last step
            float cr[6];
            for (int i = 0; i < 6; ++i) cr[i] = y[i];
            // reconstruct the step size taken (same controller, prev state)
            float hh = 0.04 * prev[1] / (abs(prev[4]) + 1e-4);
            hh = min(hh, 0.10 / (abs(prev[5]) + 1.0));
            hh = min(hh, 0.08);
            hh = min(hh, 0.015 + 0.25 * abs(sin(prev[2])));
            for (int it = 0; it < 18; ++it) {
                float mid = 0.5 * (lo + hi);
                float tr[6];
                for (int i = 0; i < 6; ++i) tr[i] = prev[i];
                rk4_step(cc, tr, mid * hh);
                if (cos(tr[2]) * prev_c > 0.0) lo = mid; else hi = mid;
                for (int i = 0; i < 6; ++i) cr[i] = tr[i];
            }
            float r_x = cr[1];
            if (r_x >= pc.disk_r_in && r_x <= pc.disk_r_out) {
                outcome = 3;
                disk_r = r_x;
                disk_ph = cr[3];
                float Om = 1.0 / (r_x * sqrt(r_x) + a);
                float sg = r_x * r_x;
                float dl = r_x * r_x - 2.0 * r_x + a * a;
                float AA = (r_x * r_x + a * a) * (r_x * r_x + a * a) - a * a * dl;
                float gtt = -(1.0 - 2.0 * r_x / sg);
                float gtp = -2.0 * a * r_x / sg;
                float gpp = AA / sg;
                float dn = -(gtt + 2.0 * Om * gtp + Om * Om * gpp);
                disk_g = 1.0 / ((1.0 / sqrt(dn)) * (cc.E - Om * cc.Lz));
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

    vec3 col = vec3(0.0);
    if (outcome == 2) {
        // camera azimuth rotates the whole sky with the orbit
        col = shade_starfield(esc_th, esc_ph + pc.cam_phi);
    } else if (outcome == 3) {
        float F = pt_flux(disk_r, a, pc.disk_r_in);
        F *= turb_factor(disk_r, disk_ph + pc.cam_phi);
        float T = pc.t_peak_k * pow(max(F, 0.0) / pc.flux_max, 0.25);
        float Tobs = disk_g * T;
        // colour: physical blackbody colormap, log-T parameterised
        float u = clamp((log(max(Tobs, 501.0)) - log(500.0)) /
                        (log(60000.0) - log(500.0)), 0.0, 1.0);
        vec3 hue = texture(luts[0], vec2(u, 0.5)).rgb;
        // brightness: bolometric (g·T)⁴ scale
        float L = pow(Tobs / 6500.0, 4.0);
        col = hue * L;
    }
    col *= exp2(pc.exposure_ev);
    // artistic preset grade (display only; §9 — physics untouched)
    if (pc.shade_mode > 0u && outcome == 3) {
        // palette remap on COMPRESSED log display luminance: the physical
        // (gT)^4 brightness spans ~4 decades across the disk, which a linear
        // display crushes to black outside the beamed side. A ~14-stop log
        // window (the same trick photographic tone mapping uses) keeps the
        // whole disk inside the LUT ramp so the filaments read everywhere.
        // The LUT PNGs are sRGB-encoded: the sample IS the display colour.
        float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
        float lu = clamp(1.0 + log2(max(luma, 1e-7)) / 24.0, 0.0, 1.0);
        vec3 graded = texture(luts[pc.shade_mode], vec2(lu, 0.5)).rgb;
        imageStore(storage_images[pc.image_index], ivec2(pix),
                   vec4(graded, 1.0));
        return;
    }
    col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
    imageStore(storage_images[pc.image_index], ivec2(pix), vec4(col, 1.0));
}
