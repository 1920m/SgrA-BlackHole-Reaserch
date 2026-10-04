// GPU port of the Carter/Mino null-geodesic integrator (Phase 7).
//
// Mirrors physics/src/{null_geodesics,raytrace_cpu}.cpp exactly:
//   • ZAMO-tetrad pinhole camera
//   • second-order Mino-time Carter equations (d²r/dτ² = R'/2, d²θ/dτ² = Θ'/2)
//   • RK4 fixed-substep with adaptive halving + per-step constraint
//     projection p_r = ±√R, p_θ = ±√Θ (kills far-field drift; see CPU notes)
//   • equatorial-crossing bisection for the thin disk, sub-ring counting
//
// Precision: double-precision core (the CPU↔GPU gate runs the camera at
// r = 1000 M where p_r² ~ 1e12 dwarfs the 0.1-scale critical discrimination —
// hopeless in fp32). GLSL lacks double trig, so dsin/dcos are implemented via
// range reduction + Taylor (≤1e-15 abs error). The fp32 "live" variant with a
// near camera arrives with Phase 8 presets.
//
// shade_mode: 0 = beauty (starfield + simple disk), 1 = diagnostic encoding
// (R: captured, G: disk redshift factor g/2, B: escape θ/π) used by
// validation/compare_shadow.
#version 450
#extension GL_ARB_gpu_shader_fp64 : require

layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D storage_images[16];

layout(push_constant) uniform PC {
    uint image_index;
    uint width, height;
    float time_s;
    float a_star, r_cam, incl_deg, fov;
    float disk_r_in, disk_r_out, exposure_ev;
    uint shade_mode;
    uint y_offset; // banded dispatch (offscreen mode: TDR-safe submissions)
} pc;

const double PI = 3.14159265358979323846LF;

// ---- double-precision trig (range reduction + Taylor, |err| < 1e-15) ------
double dsin_core(double x) { // |x| <= pi/4
    double x2 = x * x;
    return x * (1.0LF + x2 * (-1.0LF/6.0LF + x2 * (1.0LF/120.0LF +
           x2 * (-1.0LF/5040.0LF + x2 * (1.0LF/362880.0LF +
           x2 * (-1.0LF/39916800.0LF))))));
}
double dcos_core(double x) {
    double x2 = x * x;
    return 1.0LF + x2 * (-0.5LF + x2 * (1.0LF/24.0LF + x2 * (-1.0LF/720.0LF +
           x2 * (1.0LF/40320.0LF + x2 * (-1.0LF/3628800.0LF)))));
}
void dsincos(double x, out double s, out double c) {
    // reduce to [-pi/4, pi/4] with quadrant tracking
    double q = floor(x / (PI * 0.5LF) + 0.5LF);
    int iq = int(mod(q, 4.0LF) + 4.0LF) & 3;
    double r = x - q * (PI * 0.5LF);
    double sr = dsin_core(r), cr = dcos_core(r);
    if (iq == 0) { s = sr;  c = cr;  }
    else if (iq == 1) { s = cr;  c = -sr; }
    else if (iq == 2) { s = -sr; c = -cr; }
    else { s = -cr; c = sr; }
}

// ---- Carter potentials ------------------------------------------------------
struct Consts { double M, a, E, Lz, Q; };

double radial_potential(Consts cc, double r) {
    double P = cc.E * (r * r + cc.a * cc.a) - cc.a * cc.Lz;
    double delta = r * r - 2.0LF * cc.M * r + cc.a * cc.a;
    double K = (cc.Lz - cc.a * cc.E) * (cc.Lz - cc.a * cc.E) + cc.Q;
    return P * P - delta * K;
}
double radial_potential_deriv(Consts cc, double r) {
    double P = cc.E * (r * r + cc.a * cc.a) - cc.a * cc.Lz;
    double K = (cc.Lz - cc.a * cc.E) * (cc.Lz - cc.a * cc.E) + cc.Q;
    return 4.0LF * cc.E * P * r - 2.0LF * (r - cc.M) * K;
}
double polar_potential(Consts cc, double sth, double cth) {
    double s2 = sth * sth;
    return cc.Q + cth * cth * (cc.a * cc.a * cc.E * cc.E - cc.Lz * cc.Lz / s2);
}
double polar_potential_deriv(Consts cc, double sth, double cth) {
    return -cc.a * cc.a * cc.E * cc.E * 2.0LF * sth * cth +
           2.0LF * cc.Lz * cc.Lz * cth / (sth * sth * sth);
}

// state: t, r, th, ph, pr, pth (Mino time)
void rhs(Consts cc, double y[6], out double dy[6]) {
    double sth, cth;
    dsincos(y[2], sth, cth);
    if (abs(sth) < 1e-8LF) sth = (sth >= 0.0LF) ? 1e-8LF : -1e-8LF;
    double s2 = sth * sth;
    double r = y[1];
    double delta = r * r - 2.0LF * cc.M * r + cc.a * cc.a;
    double P = cc.E * (r * r + cc.a * cc.a) - cc.a * cc.Lz;
    dy[0] = -cc.a * (cc.a * cc.E * s2 - cc.Lz) + (r * r + cc.a * cc.a) * P / delta;
    dy[1] = y[4];
    dy[2] = y[5];
    dy[3] = -(cc.a * cc.E - cc.Lz / s2) + cc.a * P / delta;
    dy[4] = 0.5LF * radial_potential_deriv(cc, r);
    dy[5] = 0.5LF * polar_potential_deriv(cc, sth, cth);
}

void rk4_step(Consts cc, inout double y[6], double h) {
    double k1[6], k2[6], k3[6], k4[6], t[6];
    rhs(cc, y, k1);
    for (int i = 0; i < 6; ++i) t[i] = y[i] + 0.5LF * h * k1[i];
    rhs(cc, t, k2);
    for (int i = 0; i < 6; ++i) t[i] = y[i] + 0.5LF * h * k2[i];
    rhs(cc, t, k3);
    for (int i = 0; i < 6; ++i) t[i] = y[i] + h * k3[i];
    rhs(cc, t, k4);
    for (int i = 0; i < 6; ++i)
        y[i] += h / 6.0LF * (k1[i] + 2.0LF * k2[i] + 2.0LF * k3[i] + k4[i]);
}

// ---- starfield (beauty mode) — mirrors CPU hash01/star lattice -------------
float hash01(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return float(x) * (1.0 / 4294967296.0);
}
vec3 shade_starfield(float theta, float phi) {
    vec3 col = vec3(0.004, 0.005, 0.009);
    float deg = 180.0 / 3.14159265;
    float gt = mod(abs(theta) * deg, 10.0);
    float gp = mod(abs(phi) * deg, 10.0);
    if (gt < 0.12 || gt > 9.88 || gp < 0.12 || gp > 9.88)
        col = vec3(0.012, 0.015, 0.024);
    float cell = 0.35 * 3.14159265 / 180.0;
    float phw = mod(mod(phi, 6.2831853) + 6.2831853, 6.2831853);
    int ci = int(floor(theta / cell));
    int cj = int(floor(phw / cell));
    uint hcell = uint(ci * 73856093) ^ uint(cj * 19349663);
    float r0 = hash01(hcell);
    if (r0 > 0.82) {
        float u = fract(theta / cell);
        float v = fract(phw / cell);
        float su = hash01(hcell * 0x9e3779b9u) * 0.6 + 0.2;
        float sv = hash01(hcell * 0x85ebca6bu) * 0.6 + 0.2;
        float d2 = (u - su) * (u - su) + (v - sv) * (v - sv);
        float mag = hash01(hcell * 0xc2b2ae35u);
        float bright = 0.5 + 3.5 * mag * mag * mag;
        float sigma2 = 0.0022 * (0.4 + mag);
        float I = bright * exp(-d2 / sigma2);
        float warm = hash01(hcell * 0x27d4eb2fu);
        col += I * vec3(0.75 + 0.25 * warm, 0.72 + 0.13 * warm, 0.95 - 0.25 * warm);
    }
    return col;
}

void main() {
    uvec2 pix = uvec2(gl_GlobalInvocationID.x,
                      gl_GlobalInvocationID.y + pc.y_offset);
    if (pix.x >= pc.width || pix.y >= pc.height) return;

    // ---- camera ray (ZAMO tetrad), mirrors RaytraceCpu::camera_ray ---------
    double M = 1.0LF;
    double a = double(pc.a_star);
    double r0 = double(pc.r_cam);
    double th0 = double(pc.incl_deg) * PI / 180.0LF;
    double x = 2.0LF * (double(pix.x) + 0.5LF) / double(pc.width) - 1.0LF;
    double yy = 1.0LF - 2.0LF * (double(pix.y) + 0.5LF) / double(pc.height);
    double aspect = double(pc.height) / double(pc.width);
    double thalf = double(tan(0.5 * pc.fov));
    double tx = thalf * x;
    double ty = thalf * yy * aspect;

    double s0, c0;
    dsincos(th0, s0, c0);
    // metric pieces at camera
    double sig = r0 * r0 + a * a * c0 * c0;
    double del = r0 * r0 - 2.0LF * M * r0 + a * a;
    double A = (r0 * r0 + a * a) * (r0 * r0 + a * a) - a * a * del * s0 * s0;
    double g_tt = -(1.0LF - 2.0LF * M * r0 / sig);
    double g_tp = -2.0LF * M * a * r0 * s0 * s0 / sig;
    double g_rr = sig / del;
    double g_thth = sig;
    double g_pp = A * s0 * s0 / sig;
    double omega = -g_tp / g_pp;
    double alpha = sqrt(-(g_tt - g_tp * g_tp / g_pp));

    double inv = 1.0LF / sqrt(1.0LF + tx * tx + ty * ty);
    double n_r = -inv, n_th = -ty * inv, n_ph = tx * inv;

    // Backward tangent of the RECEIVED photon, −k_rec = −e_t + n·e (mirrors
    // RaytraceCpu::camera_ray; Phases 1–12 launched +e_t + n·e, which in Kerr
    // renders the mirror image — thesis appendix "Image handedness").
    double kt = -1.0LF / alpha;
    double kr = n_r / sqrt(g_rr);
    double kth = n_th / sqrt(g_thth);
    double kph = -omega / alpha + n_ph / sqrt(g_pp);

    Consts cc;
    cc.M = M; cc.a = a;
    cc.E = -(g_tt * kt + g_tp * kph);
    cc.Lz = g_tp * kt + g_pp * kph;
    double pth0 = sig * kth;
    cc.Q = pth0 * pth0 + c0 * c0 * (cc.Lz * cc.Lz / (s0 * s0) - a * a * cc.E * cc.E);

    double y[6];
    y[0] = 0.0LF; y[1] = r0; y[2] = th0; y[3] = 0.0LF;
    y[4] = sig * kr; y[5] = pth0;

    double r_hor = M + sqrt(M * M - a * a);
    double r_capture = r_hor * 1.001LF + 1e-6LF;
    double r_escape = max(2.0LF * r0, 1500.0LF);
    double disk_in = double(pc.disk_r_in), disk_out = double(pc.disk_r_out);

    int outcome = 0; // 0 in flight, 1 captured, 2 escaped, 3 disk
    double disk_r = 0.0LF, disk_g = 1.0LF;
    int crossings = 0;
    double esc_th = 0.0LF, esc_ph = 0.0LF;

    // Proportional step control in Mino time: Δr ≈ 2% of r per step
    // (h = 0.02 r/|p_r|) and Δθ ≤ 0.05 (h ≤ 0.05/|p_θ|). Constraint
    // projection below keeps the on-shell relations exact, so RK4's local
    // error only perturbs the trajectory phase at O(h⁴) ~ 1e-7/step.
    double prev[6];
    for (int step = 0; step < 8000 && outcome == 0; ++step) {
        for (int i = 0; i < 6; ++i) prev[i] = y[i];
        double rr = y[1];
        double h = 0.02LF * rr / (abs(y[4]) + 1e-6LF);
        h = min(h, 0.05LF / (abs(y[5]) + 1.0LF));
        h = min(h, 0.01LF);
        rk4_step(cc, y, h);

        // constraint projection (see CPU tracer notes)
        double Rp = radial_potential(cc, y[1]);
        if (Rp >= 0.0LF) y[4] = (y[4] >= 0.0LF) ? sqrt(Rp) : -sqrt(Rp);
        double sth, cth;
        dsincos(y[2], sth, cth);
        if (abs(sth) > 1e-8LF) {
            double Tp = polar_potential(cc, sth, cth);
            if (Tp >= 0.0LF) y[5] = (y[5] >= 0.0LF) ? sqrt(Tp) : -sqrt(Tp);
        }

        // equatorial crossing (θ = π/2): bisect on cos θ
        double c_prev, s_prev;
        dsincos(prev[2], s_prev, c_prev);
        if (c_prev * cth < 0.0LF && (abs(c_prev) > 1e-14LF || abs(cth) > 1e-14LF)) {
            double lo = 0.0LF, hi = h;
            double cross[6];
            for (int i = 0; i < 6; ++i) cross[i] = y[i];
            for (int it = 0; it < 30; ++it) {
                double mid = 0.5LF * (lo + hi);
                double trial[6];
                for (int i = 0; i < 6; ++i) trial[i] = prev[i];
                rk4_step(cc, trial, mid);
                double st2, ct2;
                dsincos(trial[2], st2, ct2);
                if (ct2 * c_prev > 0.0LF) lo = mid; else hi = mid;
                for (int i = 0; i < 6; ++i) cross[i] = trial[i];
            }
            double r_x = cross[1];
            if (r_x >= disk_in && r_x <= disk_out) {
                outcome = 3;
                disk_r = r_x;
                // redshift factor for the received photon, prograde disk:
                // g = 1 / (u^t (Ω L_z − E)) with the backward constants
                double sM = sqrt(M);
                double Om = sM / (r_x * sqrt(r_x) + a * sM);
                double sg = r_x * r_x; // Σ at θ=π/2
                double dl = r_x * r_x - 2.0LF * M * r_x + a * a;
                double AA = (r_x * r_x + a * a) * (r_x * r_x + a * a) - a * a * dl;
                double gtt = -(1.0LF - 2.0LF * M * r_x / sg);
                double gtp = -2.0LF * M * a * r_x / sg;
                double gpp = AA / sg;
                double denom = -(gtt + 2.0LF * Om * gtp + Om * Om * gpp);
                double ut = 1.0LF / sqrt(denom);
                disk_g = 1.0LF / (ut * (Om * cc.Lz - cc.E));
            } else {
                crossings++;
            }
        }

        if (outcome == 0 && y[1] < r_capture) outcome = 1;
        if (outcome == 0 && y[1] > r_escape && y[4] > 0.0LF) {
            outcome = 2;
            esc_th = y[2];
            esc_ph = y[3];
        }
    }
    if (outcome == 0) outcome = 1; // budget exhausted → captured (near-critical)

    vec3 col;
    if (pc.shade_mode == 1u) {
        // diagnostic encoding for the CPU/GPU comparison
        col = vec3(outcome == 1 ? 1.0 : 0.0,
                   outcome == 3 ? float(disk_g) * 0.5 : 0.0,
                   outcome == 2 ? float(esc_th) / 3.14159265 : 0.0);
    } else {
        if (outcome == 2) col = shade_starfield(float(esc_th), float(esc_ph));
        else if (outcome == 3) {
            float f = float(36.0LF / (disk_r * disk_r));
            float g4 = float(disk_g * disk_g * disk_g * disk_g);
            col = vec3(0.9 * f + 0.15, 0.55 * f + 0.08, 0.25 * f + 0.03) * g4;
        } else col = vec3(0.0);
        col *= exp2(pc.exposure_ev);
        col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
    }
    imageStore(storage_images[pc.image_index], ivec2(pix), vec4(col, 1.0));
}
