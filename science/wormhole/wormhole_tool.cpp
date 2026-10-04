// Gate 21 — EXOTIC SPACETIME, NOT SGR A*: the numbers the extension paper's
// wormhole appendix quotes.
//
//   wormhole_tool <outdir>     → <outdir>/wormhole.json
//
// The Interstellar wormhole of James et al. (2015): length 2a = 0.01ρ, lensing
// width W = 0.05ρ. Checks: W from the embedding diagram against the closed
// form; conservation along rays; the edge of the other universe's disk
// against sin α = ρ/r(ℓ_c); and the Einstein ring — the circle on the
// camera's sky where the image of the far side of our own universe is
// infinitely magnified — located as the direction whose ray is deflected by
// exactly π (it comes from the point behind the wormhole).
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "Wormhole.hpp"

using namespace sgr;
using nlohmann::json;

namespace {
constexpr double kPi = 3.14159265358979323846;
}

int main(int argc, char** argv) {
    const std::filesystem::path out = argc > 1 ? argv[1] : "validation/extension";
    std::filesystem::create_directories(out);
    DnegParams p;
    p.rho = 1.0;
    p.a = 0.005;
    p.M = 0.05 / DnegWormhole::lensing_width_closed_form(1.0);
    const DnegWormhole w(p);
    json J;
    J["rho"] = p.rho;
    J["a"] = p.a;
    J["M"] = p.M;
    J["W_numeric"] = w.lensing_width_numeric();
    J["W_closed"] = DnegWormhole::lensing_width_closed_form(p.M);
    J["W_over_M"] = DnegWormhole::lensing_width_closed_form(1.0);

    double worst_B2 = 0.0, worst_H = 0.0, worst_ring_shift = 0.0;
    for (double l_c : {2.0, 6.0, 15.0}) {
        double l_far = 400.0;
        auto trace = [&](double alpha) {
            const std::array<double, 3> N{-std::cos(alpha), std::sin(alpha), 0.0};
            return w.trace(w.camera_ray(l_c, 0.5 * kPi, 0.0, N), l_far);
        };
        // edge of the other universe's disk
        double lo = 1e-4, hi = 0.5 * kPi * 0.999;
        for (int it = 0; it < 50; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (trace(mid).universe < 0) lo = mid; else hi = mid;
        }
        const double edge = 0.5 * (lo + hi);
        // Einstein ring: our-universe rays whose deflection takes them to the
        // point directly behind the wormhole, φ_∞ = π (camera at φ = 0)
        auto phi_inf = [&](double alpha) {
            const auto o = trace(alpha);
            worst_B2 = std::max(worst_B2, o.max_dB2);
            worst_H = std::max(worst_H, o.max_H);
            return o.universe > 0 ? std::fmod(std::abs(o.ph), 2.0 * kPi) : -1.0;
        };
        // deflection decreases with α outside the edge: bracket φ_∞ = π
        auto find_ring = [&]() {
            double rlo = edge * 1.0001, rhi = 0.5 * kPi * 0.999;
            if (!(phi_inf(rlo) > kPi && phi_inf(rhi) < kPi)) return std::nan("");
            for (int it = 0; it < 50; ++it) {
                const double mid = 0.5 * (rlo + rhi);
                if (phi_inf(mid) > kPi) rlo = mid; else rhi = mid;
            }
            return 0.5 * (rlo + rhi);
        };
        const double ring = find_ring();
        // convergence in the stopping distance (the asymptotic completion
        // in DnegWormhole::trace makes the result independent of it)
        l_far = 4000.0;
        const double ring_far = find_ring();
        l_far = 400.0;
        worst_ring_shift = std::max(worst_ring_shift, std::abs(ring_far - ring) * 180.0 / kPi);
        J["cameras"].push_back({{"l_c", l_c}, {"r_c", w.r(l_c)},
                                {"edge_deg", edge * 180.0 / kPi},
                                {"edge_analytic_deg", std::asin(p.rho / w.r(l_c)) * 180.0 / kPi},
                                {"einstein_ring_deg", ring * 180.0 / kPi}});
        std::printf("[wormhole] l_c = %5.1f: edge %.8f deg (analytic %.8f), Einstein ring %.4f deg\n",
                    l_c, edge * 180.0 / kPi, std::asin(p.rho / w.r(l_c)) * 180.0 / kPi,
                    ring * 180.0 / kPi);
    }
    J["max_dB2"] = worst_B2;
    J["ring_lfar_shift_deg"] = worst_ring_shift;   // l_far 400 vs 4000
    J["max_H"] = worst_H;
    const auto path = out / "wormhole.json";
    std::ofstream(path) << J.dump(2) << "\n";
    std::printf("[wormhole] Einstein ring shift l_far 400 -> 4000: %.2e deg\n", worst_ring_shift);
    std::printf("wormhole_tool: wrote %s\n", path.string().c_str());
    return 0;
}
