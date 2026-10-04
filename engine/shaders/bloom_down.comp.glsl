// HDR bloom — downsample pass (Gate 14; display only, physics untouched).
//
// Level k (rgba16f) = energy-preserving 4×4 tent filter ([1 3 3 1]⊗[1 3 3 1]
// / 64, i.e. bilinear-of-bilinear) of level k−1 at twice the resolution.
// The first pass reads the rgba32f radiance accumulator and uses a Karis
// (1/(1+luma)) weighted average so isolated hot pixels — stars, one-pixel
// caustic sparkles — cannot flicker into large glare blobs.
#version 450

layout(local_size_x = 8, local_size_y = 8) in;

// Same bindless binding viewed at the two formats the post chain uses; each
// slot is only ever accessed through the declaration matching its view.
layout(set = 0, binding = 0, rgba32f) uniform readonly image2D img32[16];
layout(set = 0, binding = 0, rgba16f) uniform image2D img16[16];

layout(push_constant) uniform PC {
    uint src_slot, dst_slot;
    uint src_w, src_h, dst_w, dst_h;
    uint src_is_hdr;   // 1: source is the rgba32f accumulator (first pass)
} pc;

vec3 load_src(ivec2 p) {
    p = clamp(p, ivec2(0), ivec2(int(pc.src_w) - 1, int(pc.src_h) - 1));
    vec3 c = (pc.src_is_hdr != 0u) ? imageLoad(img32[pc.src_slot], p).rgb
                                   : imageLoad(img16[pc.src_slot], p).rgb;
    // a single non-finite pixel would smear across the whole bloom pyramid
    return all(lessThan(abs(c), vec3(6.0e4))) ? max(c, vec3(0.0)) : vec3(0.0);
}

void main() {
    ivec2 d = ivec2(gl_GlobalInvocationID.xy);
    if (d.x >= int(pc.dst_w) || d.y >= int(pc.dst_h)) return;
    const float w1[4] = float[4](1.0, 3.0, 3.0, 1.0);
    ivec2 s0 = 2 * d - ivec2(1);
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            vec3 c = load_src(s0 + ivec2(i, j));
            float w = w1[i] * w1[j];
            if (pc.src_is_hdr != 0u)
                w /= 1.0 + dot(c, vec3(0.2126, 0.7152, 0.0722));
            acc += w * c;
            wsum += w;
        }
    }
    imageStore(img16[pc.dst_slot], d, vec4(acc / wsum, 1.0));
}
