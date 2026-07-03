// Phase 6 gate shader: solid colour (with a soft vertical gradient so the
// blit orientation is visually verifiable) written to the bindless storage
// image selected by push constant.
#version 450

layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D storage_images[16];

layout(push_constant) uniform PC {
    uint image_index;
    uint width, height;
    float time_s;
    float a_star, r_cam, incl_deg, fov;
    float disk_r_in, disk_r_out, exposure_ev;
    float pad0;
} pc;

void main() {
    uvec2 p = gl_GlobalInvocationID.xy;
    if (p.x >= pc.width || p.y >= pc.height) return;
    vec2 uv = vec2(p) / vec2(pc.width, pc.height);
    vec3 c = vec3(0.06, 0.35, 0.78) * (0.65 + 0.35 * uv.y);
    imageStore(storage_images[pc.image_index], ivec2(p), vec4(c, 1.0));
}
