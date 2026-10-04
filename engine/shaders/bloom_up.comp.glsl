// HDR bloom — upsample-and-accumulate pass (Gate 14; display only).
//
// Runs coarse → fine: level k += 3×3 tent ([1 2 1]⊗[1 2 1] / 16) bilinear
// upsample of level k+1, which already holds the accumulated chain below it.
// After the last pass level 1 holds the sum of all levels, each a wider blur
// of the scene; tonemap.comp divides by the level count, so the result is
// a normalised, energy-preserving glare kernel (Jimenez 2014, CoD:AW style).
#version 450

layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0, rgba16f) uniform image2D img16[16];

layout(push_constant) uniform PC {
    uint src_slot, dst_slot;   // src = coarser level k+1, dst = level k
    uint src_w, src_h, dst_w, dst_h;
    float radius;              // tent half-width in source texels
} pc;

vec3 ld(ivec2 p) {
    p = clamp(p, ivec2(0), ivec2(int(pc.src_w) - 1, int(pc.src_h) - 1));
    return imageLoad(img16[pc.src_slot], p).rgb;
}

vec3 bilinear(vec2 t) {   // t in source texel-centre coordinates
    vec2 f = floor(t);
    vec2 w = t - f;
    ivec2 i = ivec2(f);
    return mix(mix(ld(i), ld(i + ivec2(1, 0)), w.x),
               mix(ld(i + ivec2(0, 1)), ld(i + ivec2(1, 1)), w.x), w.y);
}

void main() {
    ivec2 d = ivec2(gl_GlobalInvocationID.xy);
    if (d.x >= int(pc.dst_w) || d.y >= int(pc.dst_h)) return;
    vec2 t = (vec2(d) + 0.5) * vec2(pc.src_w, pc.src_h) /
                 vec2(pc.dst_w, pc.dst_h) - 0.5;
    float r = pc.radius;
    vec3 up = (bilinear(t + vec2(-r, -r)) + 2.0 * bilinear(t + vec2(0, -r)) +
               bilinear(t + vec2(r, -r)) +
               2.0 * bilinear(t + vec2(-r, 0)) + 4.0 * bilinear(t) +
               2.0 * bilinear(t + vec2(r, 0)) +
               bilinear(t + vec2(-r, r)) + 2.0 * bilinear(t + vec2(0, r)) +
               bilinear(t + vec2(r, r))) / 16.0;
    vec3 cur = imageLoad(img16[pc.dst_slot], d).rgb;
    imageStore(img16[pc.dst_slot], d, vec4(cur + up, 1.0));
}
