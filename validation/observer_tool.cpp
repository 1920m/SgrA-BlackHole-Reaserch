// Gate 17 — observer-dependent phenomena: every number the extension paper
// quotes for them (no number the code has not produced).
//
//   observer_tool <outdir>       → <outdir>/observers.json
//
// 1. Kinematics of the standard observers (sgr_physics/observers.hpp — the
//    same functions the live renderer builds its camera from): speed
//    relative to the ZAMO, Lorentz factor, clock rate dτ/dt, lapse, frame
//    dragging, at the radii the G17 renders use.
// 2. The shadow each observer sees: angular positions of its left/right and
//    top/bottom edges, bisected on the Kerr-Schild tracer (fp64). At a* = 0
//    they are checked against Synge's formula + special-relativistic
//    aberration; at a* = 0.94 they are the reference the GPU renders are
//    compared with (validation/phase14/compare_observers.py).
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "sgr_physics/constants.hpp"
#include "sgr_physics/kerr_schild.hpp"
#include "sgr_physics/metric.hpp"
#include "sgr_physics/observers.hpp"

using namespace sgr;
using nlohmann::json;

namespace {
constexpr double kPi = 3.14159265358979323846;

const char* kind_name(ObserverKind k) {
    switch (k) {
        case ObserverKind::Zamo: return "zamo";
        case ObserverKind::Orbiting: return "orbit";
        case ObserverKind::Infall: return "infall";
        case ObserverKind::Hovering: return "hover";
        default: return "moving";
    }
}

bool captured(const KerrSchild& ks, double r, double th, const ObserverFrame& f,
              const std::array<double, 3>& n) {
    KSTraceOptions opt;
    opt.rtol = opt.atol = 1e-11;
    opt.r_escape = std::max(4.0 * r, 200.0);
    return ks_trace(ks, ks_observer_ray(ks, r, th, f, n), opt).outcome ==
           KSOutcome::Captured;
}

// Image-plane direction of a pinhole pixel offset (tx, ty) = tan of the
// angles from the look axis, in the camera basis fwd = −e_r̂', right = e_φ̂',
// up = −e_θ̂' (the renderer's default basis, CameraBasis.hpp).
std::array<double, 3> pixel_dir(double tx, double ty) {
    const double inv = 1.0 / std::sqrt(1.0 + tx * tx + ty * ty);
    return {-inv, -ty * inv, tx * inv};
}

// Shadow along an image axis through the image centre (axis 0 = horizontal,
// 1 = vertical): the signed pixel offsets (tan of the angle from the look
// axis) of its two edges. A coarse scan finds the captured interval — under
// strong aberration it need not contain the image centre — then each edge
// is bisected.
std::array<double, 2> edges(const KerrSchild& ks, double r, double th,
                            const ObserverFrame& f, int axis) {
    auto cap = [&](double t) {
        const auto n = axis == 0 ? pixel_dir(t, 0.0) : pixel_dir(0.0, t);
        return captured(ks, r, th, f, n);
    };
    const int N = 400;
    const double T = 3.0;   // ±72° from the look axis
    int first = -1, last = -1;
    for (int i = 0; i <= N; ++i) {
        if (cap(-T + 2.0 * T * i / N)) {
            if (first < 0) first = i;
            last = i;
        }
    }
    if (first < 0) return {std::nan(""), std::nan("")};
    auto bisect = [&](double in, double out) {
        for (int it = 0; it < 48; ++it) {
            const double mid = 0.5 * (in + out);
            if (cap(mid)) in = mid; else out = mid;
        }
        return 0.5 * (in + out);
    };
    const double step = 2.0 * T / N;
    return {bisect(-T + step * first, -T + step * (first - 1)),
            bisect(-T + step * last, -T + step * (last + 1))};
}

json kinematics(const KerrMetric& m, ObserverKind kind, double r, double th) {
    const ObserverFrame f = observer_frame(m, kind, r, th);
    return {{"observer", kind_name(kind)}, {"r", r}, {"theta_deg", th * 180.0 / kPi},
            {"valid", f.valid}, {"speed", f.speed}, {"gamma", f.gamma},
            {"dtau_dt", f.valid ? f.dtau_dt() : std::nan("")},
            {"alpha", f.alpha}, {"omega", f.omega}};
}

json shadow(const KerrSchild& ks, const KerrMetric& m, ObserverKind kind,
            double r, double incl_deg) {
    const double th = incl_deg * kPi / 180.0;
    const ObserverFrame f = observer_frame(m, kind, r, th);
    json j = {{"observer", kind_name(kind)}, {"a", m.spin_star()}, {"r", r},
              {"incl_deg", incl_deg}, {"valid", f.valid}, {"speed", f.speed}};
    if (!f.valid) return j;
    const auto h = edges(ks, r, th, f, 0);
    const auto v = edges(ks, r, th, f, 1);
    // signed image-plane offsets (tan of the angle; +x right, +y up)
    j["x_left"] = h[0];
    j["x_right"] = h[1];
    j["y_down"] = v[0];
    j["y_up"] = v[1];
    return j;
}
} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path out = argc > 1 ? argv[1] : "validation/extension";
    std::filesystem::create_directories(out);
    json J;

    // ---- 1. kinematics -------------------------------------------------------
    json kin = json::array();
    for (double a : {0.0, 0.94}) {
        const KerrMetric m(1.0, a);
        for (double r : {60.0, 15.0, 6.0})
            for (auto kind : {ObserverKind::Zamo, ObserverKind::Orbiting,
                              ObserverKind::Infall, ObserverKind::Hovering}) {
                json k = kinematics(m, kind, r, 0.5 * kPi);
                k["a"] = a;
                kin.push_back(k);
            }
        // ergosphere: the hovering observer's last radius, and the rain
        // observer at the horizon
        json erg;
        erg["a"] = a;
        erg["r_ergo_eq"] = m.r_ergosphere(0.5 * kPi);
        erg["r_plus"] = m.r_horizon();
        erg["r_photon_pro"] = m.r_photon(true);
        erg["r_photon_retro"] = m.r_photon(false);
        erg["hover_valid_1.001rE"] =
            observer_frame(m, ObserverKind::Hovering, 1.001 * m.r_ergosphere(0.5 * kPi), 0.5 * kPi).valid;
        erg["hover_valid_0.999rE"] =
            observer_frame(m, ObserverKind::Hovering, 0.999 * m.r_ergosphere(0.5 * kPi), 0.5 * kPi).valid;
        erg["infall_speed_at_1.001rp"] =
            observer_frame(m, ObserverKind::Infall, 1.001 * m.r_horizon(), 0.5 * kPi).speed;
        J["ergosphere"].push_back(erg);
    }
    J["kinematics"] = kin;

    // ---- 2. the shadow each observer sees -----------------------------------
    // (a) a* = 0 at r = 10 M on the equator: analytic check.
    {
        const KerrSchild ks(1.0, 0.0, -1);
        const KerrMetric m(1.0, 0.0);
        const double r = 10.0;
        const double psi_s = std::asin(3.0 * std::sqrt(3.0) * std::sqrt(1.0 - 2.0 / r) / r);
        json chk = json::array();
        for (auto kind : {ObserverKind::Zamo, ObserverKind::Infall, ObserverKind::Orbiting}) {
            const ObserverFrame f = observer_frame(m, kind, r, 0.5 * kPi);
            json s = shadow(ks, m, kind, r, 90.0);
            // analytic edges: aberrate the static-frame edge directions
            // n = (−cos ψ, 0, ±sin ψ) (r̂, θ̂, φ̂) by the observer velocity
            auto aberrate = [&](std::array<double, 3> n) {
                const double v2 = f.speed * f.speed;
                if (v2 == 0.0) return n;
                const std::array<double, 3> vh{f.v[0] / f.speed, f.v[1] / f.speed, f.v[2] / f.speed};
                const double c = n[0] * vh[0] + n[1] * vh[1] + n[2] * vh[2];
                // n' = [n + (γ−1)(n·v̂)v̂ + γ v] / [γ (1 + v·n)]   (look direction)
                const double g = f.gamma;
                std::array<double, 3> np{};
                for (int i = 0; i < 3; ++i)
                    np[i] = (n[i] + (g - 1.0) * c * vh[i] + g * f.speed * vh[i]) /
                            (g * (1.0 + f.speed * c));
                return np;
            };
            const auto nr = aberrate({-std::cos(psi_s), 0.0, std::sin(psi_s)});
            const auto nl = aberrate({-std::cos(psi_s), 0.0, -std::sin(psi_s)});
            s["analytic_x_right"] = nr[2] / -nr[0];
            s["analytic_x_left"] = nl[2] / -nl[0];
            s["psi_static_deg"] = psi_s * 180.0 / kPi;
            chk.push_back(s);
        }
        J["schwarzschild_check"] = chk;
    }
    // (b) the G17 render configurations, a* = 0.94 (GPU comparison targets)
    {
        json g17 = json::array();
        for (double a : {0.0, 0.94}) {
            const KerrSchild ks(1.0, a, -1);
            const KerrMetric m(1.0, a);
            for (auto kind : {ObserverKind::Zamo, ObserverKind::Infall,
                              ObserverKind::Orbiting, ObserverKind::Hovering})
                g17.push_back(shadow(ks, m, kind, 10.0, 90.0));
            for (auto kind : {ObserverKind::Zamo, ObserverKind::Infall,
                              ObserverKind::Orbiting})
                g17.push_back(shadow(ks, m, kind, 25.0, 60.0));
        }
        J["shadow_edges"] = g17;
    }

    const auto path = out / "observers.json";
    std::ofstream(path) << J.dump(2) << "\n";
    std::printf("observer_tool: wrote %s\n", path.string().c_str());
    return 0;
}
