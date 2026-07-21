#pragma once
// Camera geometry shared by the compute shader and the ImGui overlays.
//
// The renderer places the camera at Boyer-Lindquist (r_cam, incl_deg, phi)
// and traces rays from the local ZAMO triad (e_r, e_theta, e_phi). Both the
// ray basis (push constants) and the S-star sprite projection must agree
// exactly, so both go through this one header.
//
// Conventions (matching disk.comp.glsl):
//   * the camera sits at BL azimuth 0 in the integration frame; the world is
//     rotated by cam_phi, so a world azimuth phi_w maps to phi_w - cam_phi;
//   * screen +x is the ray basis "right" vector, screen +y is "up";
//   * the historical aim is forward = -e_r, right = +e_phi, up = -e_theta,
//     which this header reproduces EXACTLY (the components are 0 and +-1, so
//     no rounding enters) when look_at is the origin.
//
// Everything here is display geometry only: it decides where rays start and
// which way they point, never how they propagate.

#include <cmath>
#include <array>

namespace sgr {

using Vec3 = std::array<double, 3>;

inline Vec3 v_sub(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
inline double v_dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline Vec3 v_cross(const Vec3& a, const Vec3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}
inline double v_len(const Vec3& a) { return std::sqrt(v_dot(a, a)); }
inline Vec3 v_unit(const Vec3& a) {
    const double L = v_len(a);
    return (L > 0.0) ? Vec3{a[0] / L, a[1] / L, a[2] / L} : Vec3{0.0, 0.0, 0.0};
}

// Camera frame in BH-centred Cartesian coordinates (z = spin axis) plus the
// local triad and the ray basis expressed in triad components.
struct CameraFrame {
    Vec3 pos{};                      // camera position [M]
    Vec3 er{}, eth{}, eph{};         // local orthonormal triad
    // ray basis in TRIAD components (what the shader consumes)
    Vec3 fwd{-1.0, 0.0, 0.0};
    Vec3 rgt{0.0, 0.0, 1.0};
    Vec3 up{0.0, -1.0, 0.0};
};

// Build the frame for a camera at (r_cam, incl_deg, cam_phi) aimed at
// `target` (BH-centred Cartesian). Passing the origin reproduces the
// historical basis exactly.
inline CameraFrame camera_frame(double r_cam, double incl_deg, double cam_phi,
                                const Vec3& target = {0.0, 0.0, 0.0}) {
    CameraFrame c;
    const double th = incl_deg * 3.14159265358979323846 / 180.0;
    const double s = std::sin(th), co = std::cos(th);
    const double sp = std::sin(cam_phi), cp = std::cos(cam_phi);
    c.er = {s * cp, s * sp, co};
    c.eth = {co * cp, co * sp, -s};
    c.eph = {-sp, cp, 0.0};
    c.pos = {r_cam * c.er[0], r_cam * c.er[1], r_cam * c.er[2]};

    // Aim: default (target at the origin) is exactly -e_r, so take that
    // branch verbatim and keep the components exact.
    const Vec3 d = v_sub(target, c.pos);
    const bool at_origin = (target[0] == 0.0 && target[1] == 0.0 &&
                            target[2] == 0.0);
    if (at_origin || v_len(d) < 1e-12) {
        c.fwd = {-1.0, 0.0, 0.0};
        c.rgt = {0.0, 0.0, 1.0};
        c.up = {0.0, -1.0, 0.0};
        return c;
    }
    const Vec3 u = v_unit(d);
    // forward in triad components
    c.fwd = {v_dot(u, c.er), v_dot(u, c.eth), v_dot(u, c.eph)};
    c.fwd = v_unit(c.fwd);
    // screen up reference: -e_theta (toward the spin axis), with a fallback
    // when the aim is nearly along it
    Vec3 upref{0.0, -1.0, 0.0};
    if (std::abs(v_dot(c.fwd, upref)) > 0.999) upref = {1.0, 0.0, 0.0};
    c.rgt = v_unit(v_cross(c.fwd, upref));
    c.up = v_cross(c.rgt, c.fwd);   // already unit: orthonormal by construction
    return c;
}

// Project a BH-centred Cartesian point to screen pixels for a frame built by
// camera_frame(). Returns false when the point is behind the camera or far
// outside the frustum. `vis` reports occultation by the silhouette (1 = clear,
// 0 = hidden) using the Schwarzschild critical impact parameter.
struct Projected {
    double x = 0.0, y = 0.0;   // pixels
    double dist = 0.0;         // camera distance [M]
    double vis = 1.0;
    bool ok = false;
};

inline Projected project_point(const CameraFrame& c, const Vec3& p, double fov,
                               double width, double height) {
    Projected o;
    const Vec3 d = v_sub(p, c.pos);
    const Vec3 t{v_dot(d, c.er), v_dot(d, c.eth), v_dot(d, c.eph)};
    const double zf = v_dot(t, c.fwd);
    if (zf <= 1e-9) return o;                      // behind the camera
    const double xr = v_dot(t, c.rgt), yu = v_dot(t, c.up);
    const double thalf = std::tan(0.5 * fov);
    const double aspect = height / width;
    const double sx = (xr / zf) / thalf;
    const double sy = (yu / zf) / (thalf * aspect);
    if (std::abs(sx) > 6.0 || std::abs(sy) > 6.0) return o;
    o.x = (sx + 1.0) * 0.5 * width;
    o.y = (1.0 - sy) * 0.5 * height;
    o.dist = v_len(d);
    // occultation: closest approach of the camera->point segment to the hole
    if (o.dist > 1e-9) {
        const Vec3 u{d[0] / o.dist, d[1] / o.dist, d[2] / o.dist};
        const double tstar = -v_dot(c.pos, u);
        if (tstar > 0.0 && tstar < o.dist) {
            const Vec3 q{c.pos[0] + tstar * u[0], c.pos[1] + tstar * u[1],
                         c.pos[2] + tstar * u[2]};
            const double b = v_len(q);
            const double bc = 3.0 * std::sqrt(3.0);   // 3*sqrt(3) M
            o.vis = (b - bc) / (0.45 * bc);
            o.vis = (o.vis < 0.0) ? 0.0 : ((o.vis > 1.0) ? 1.0 : o.vis);
        }
    }
    o.ok = true;
    return o;
}

// Cartesian (BH frame) -> Boyer-Lindquist-style spherical angles used by the
// renderer's camera parameters. (For a != 0 BL radius is oblate-spheroidal;
// the camera placement uses the spherical convention the shader assumes.)
inline void cartesian_to_camera(const Vec3& p, double& r_cam, double& incl_deg,
                                double& cam_phi) {
    const double r = v_len(p);
    r_cam = r;
    incl_deg = (r > 0.0)
                   ? std::acos(std::max(-1.0, std::min(1.0, p[2] / r))) *
                         180.0 / 3.14159265358979323846
                   : 90.0;
    cam_phi = std::atan2(p[1], p[0]);
}

} // namespace sgr
