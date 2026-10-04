// Display transform (Gate 14): linear radiance → 8-bit display image.
//
// Everything here is DISPLAY ONLY — the radiance accumulator it reads is the
// physical image (Planck × CIE at g·T, lensed, beamed). Order:
//   bloom mix (glare) → exposure → camera white balance → saturation →
//   tone curve → output encoding (+ dither).
// op 0 = LINEAR/SCIENCE: clamp(c·2^EV)^(1/2.2) with no bloom, grade or
//        dither — the exact transform of the CPU reference write_png_srgb,
//        so GPU/CPU comparisons and Science Mode stay quantitative.
// op 1 = AgX (Sobotka; minimal polynomial fit by B. Wrensch) — filmic
//        highlight roll-off that keeps hue in over-exposed regions.
// op 2 = ACES filmic (Narkowicz 2016 fit) + sRGB OETF.
#version 450

layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0, rgba32f) uniform readonly image2D img32[16];
layout(set = 0, binding = 0, rgba16f) uniform readonly image2D img16[16];
layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img8[16];

layout(push_constant) uniform PC {
    uint hdr_slot, bloom_slot, ldr_slot;
    uint w, h, bloom_w, bloom_h;
    uint op;
    float exposure;        // linear gain 2^EV
    float bloom_strength;  // 0 = off
    float bloom_norm;      // 1 / number of bloom levels summed
    float saturation;
    float wb_r, wb_g, wb_b;
    uint y_offset;         // offscreen row bands
} pc;

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

vec3 ld_bloom(ivec2 p) {
    p = clamp(p, ivec2(0), ivec2(int(pc.bloom_w) - 1, int(pc.bloom_h) - 1));
    return imageLoad(img16[pc.bloom_slot], p).rgb;
}
vec3 bloom_at(vec2 pix_centre) {
    vec2 t = pix_centre * vec2(pc.bloom_w, pc.bloom_h) / vec2(pc.w, pc.h) - 0.5;
    vec2 f = floor(t);
    vec2 wt = t - f;
    ivec2 i = ivec2(f);
    return mix(mix(ld_bloom(i), ld_bloom(i + ivec2(1, 0)), wt.x),
               mix(ld_bloom(i + ivec2(0, 1)), ld_bloom(i + ivec2(1, 1)), wt.x),
               wt.y);
}

// ---- AgX -------------------------------------------------------------------
vec3 agx_contrast(vec3 x) {
    vec3 x2 = x * x, x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x +
           0.4298 * x2 + 0.1191 * x - 0.00232;
}
vec3 agx_display(vec3 c) {
    const mat3 inset = mat3(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                            0.0784335999999992, 0.878468636469772, 0.0784336,
                            0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    const mat3 outset = mat3(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                             -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                             -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    const float min_ev = -12.47393, max_ev = 4.026069;
    vec3 v = inset * max(c, vec3(1e-10));
    v = clamp(log2(v), min_ev, max_ev);
    v = agx_contrast((v - min_ev) / (max_ev - min_ev));
    // The AgX curve output is already display-encoded (2.2 reference); the
    // outset matrix restores saturation in that encoding.
    return outset * v;
}

// ---- ACES (Narkowicz) + sRGB ----------------------------------------------
vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
vec3 srgb_oetf(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055,
               step(vec3(0.0031308), c));
}

float ign(vec2 p) { // interleaved gradient noise (Jimenez 2014)
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

void main() {
    uvec2 pix = uvec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y + pc.y_offset);
    if (pix.x >= pc.w || pix.y >= pc.h) return;
    vec3 c = imageLoad(img32[pc.hdr_slot], ivec2(pix)).rgb;
    if (!all(lessThan(abs(c), vec3(1e30)))) c = vec3(0.0);

    vec3 o;
    if (pc.op == 0u) {
        o = pow(clamp(c * pc.exposure, 0.0, 1.0), vec3(1.0 / 2.2));
    } else {
        if (pc.bloom_strength > 0.0)
            c = mix(c, bloom_at(vec2(pix) + 0.5) * pc.bloom_norm, pc.bloom_strength);
        c *= pc.exposure;
        c *= vec3(pc.wb_r, pc.wb_g, pc.wb_b);
        float l = dot(c, LUMA);
        c = max(mix(vec3(l), c, pc.saturation), vec3(0.0));
        o = (pc.op == 1u) ? agx_display(c) : srgb_oetf(aces(c));
        o += (ign(vec2(pix)) - 0.5) / 255.0;   // break up 8-bit banding
    }
    imageStore(img8[pc.ldr_slot], ivec2(pix), vec4(clamp(o, 0.0, 1.0), 1.0));
}
