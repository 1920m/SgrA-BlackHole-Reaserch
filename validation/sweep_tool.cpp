// Phase 10: Sgr A* calibration + spin–inclination degeneracy sweep.
//
//   sweep_tool calibrate <out_dir>              — a*=0.94, i=30°, 32-direction
//                                                 boundary → diameter in μas
//   sweep_tool sweep <out_dir> [na] [ni]        — grid over a* × inclination,
//                                                 4-direction diameters (CSV)
//
// Diameters are measured on the traced critical boundary (bisection in camera
// angle), converted to impact parameters b = r sinα/√(1−2M/r) and to μas with
// the GRAVITY 2022 mass and distance (constants.hpp).
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <execution>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "sgr_physics/constants.hpp"
#include "sgr_physics/raytrace_cpu.hpp"

using namespace sgr;
namespace fs = std::filesystem;

namespace {

double boundary_b_M(const RaytraceCpu& tracer, double psi) {
    const double alpha = tracer.shadow_boundary_angle(psi, 0.010);
    const double rc = tracer.camera().r_cam;
    return std::sin(alpha) * rc / std::sqrt(1.0 - 2.0 / rc);
}

struct Diameters {
    double dx_M, dy_M, mean_M, mean_uas, axis_ratio, disp_x_M;
};

Diameters measure(double a_star, double incl_deg, int n_dirs) {
    const KerrMetric metric(1.0, a_star);
    CameraConfig cam;
    cam.r_cam = 1000.0;
    cam.incl_deg = incl_deg;
    cam.fov_rad = 0.032;
    const RaytraceCpu tracer(metric, cam, DiskConfig{});

    // horizontal + vertical extents (n_dirs = 4), or full ring average
    std::vector<double> b(n_dirs);
    for (int k = 0; k < n_dirs; ++k)
        b[k] = boundary_b_M(tracer, 2.0 * 3.14159265358979323846 * k / n_dirs);

    Diameters d{};
    d.dx_M = b[0] + b[n_dirs / 2];
    d.dy_M = b[n_dirs / 4] + b[3 * n_dirs / 4];
    // mean diameter over all sampled directions (opposite pairs)
    double sum = 0.0;
    for (int k = 0; k < n_dirs / 2; ++k) sum += b[k] + b[k + n_dirs / 2];
    d.mean_M = sum / (n_dirs / 2);
    d.mean_uas = d.mean_M * constants::sgrA_uas_per_rg;
    d.axis_ratio = d.dx_M / d.dy_M;
    // horizontal centroid displacement of the boundary (spin signature:
    // the shadow shifts away from the prograde side)
    d.disp_x_M = 0.5 * (b[0] - b[n_dirs / 2]);
    return d;
}

// ---------------------------------------------------------------------------
// Spherical photon orbits and Lyapunov exponents (thesis Ch 5 / Ch 7).
// Bardeen (1973) conserved quantities on the photon shell r = r̃ (E = 1):
//   ξ(r) = [r²(3M−r) − a²(r+M)] / [a(r−M)],
//   η(r) = r³[4a²M − r(r−3M)²] / [a²(r−M)²].
// Instability: in Mino time δr̈ = ½R″(r̃)δr with R″ = 8E²r² + 4EP − 2K,
// so the e-fold rate is √(½R″); the demagnification exponent per polar
// half-orbit is γ = √(½R″(r̃)) · Δτ_θ with Δτ_θ = ∫dθ/√Θ between turning
// points (γ = π exactly for Schwarzschild, all orbits).
// ---------------------------------------------------------------------------
double bardeen_xi(double a, double r) {
    return (r * r * (3.0 - r) - a * a * (r + 1.0)) / (a * (r - 1.0));
}
double bardeen_eta(double a, double r) {
    const double d = a * (r - 1.0);
    return r * r * r * (4.0 * a * a - r * (r - 3.0) * (r - 3.0)) / (d * d);
}

// Mino-time polar half-period ∫dθ/√Θ. In u = cosθ the polar quartic is
// U(u) = η + u²(a² − η − ξ²) − a²u⁴ = a²(v₁ − u²)(u² − v₂) with v = u²
// roots v₁ ∈ (0,1], v₂ ≤ 0, giving
// Δτ_θ = (2/a)∫₀^{π/2} dψ / √(v₁ sin²ψ − v₂)   (u = √v₁ sinψ).
double mino_polar_half_period(double a, double xi, double eta) {
    if (a < 1e-8) return 3.14159265358979323846 / std::sqrt(eta + xi * xi);
    const double A4 = -a * a, A2 = a * a - eta - xi * xi, A0 = eta;
    const double disc = std::sqrt(A2 * A2 - 4.0 * A4 * A0);
    const double v1 = (-A2 - disc) / (2.0 * A4); // positive root (A4 < 0)
    const double v2 = (-A2 + disc) / (2.0 * A4); // non-positive root
    // Simpson over ψ ∈ [0, π/2]
    const int N = 400;
    const double hpsi = 0.5 * 3.14159265358979323846 / N;
    double s = 0.0;
    for (int k = 0; k <= N; ++k) {
        const double sp = std::sin(k * hpsi);
        const double f = 1.0 / std::sqrt(v1 * sp * sp - v2);
        s += f * ((k == 0 || k == N) ? 1.0 : (k % 2 ? 4.0 : 2.0));
    }
    return (2.0 / a) * s * hpsi / 3.0;
}

double lyapunov_gamma(double a, double r) {
    if (a < 1e-8) return 3.14159265358979323846; // Schwarzschild: γ = π exactly
    const double xi = bardeen_xi(a, r), eta = bardeen_eta(a, r);
    const double P = (r * r + a * a) - a * xi;
    const double K = (xi - a) * (xi - a) + eta;
    const double Rpp = 8.0 * r * r + 4.0 * P - 2.0 * K;
    return std::sqrt(0.5 * Rpp) * mino_polar_half_period(a, xi, eta);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <calibrate|sweep> <out_dir> [na] [ni]\n",
                     argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const fs::path out = argv[2];
    fs::create_directories(out);

    if (mode == "calibrate") {
        const Diameters d = measure(constants::sgrA_spin_default,
                                    constants::sgrA_incl_deg, 32);
        spdlog::info("Sgr A* calibration (a*={}, i={} deg):",
                     constants::sgrA_spin_default, constants::sgrA_incl_deg);
        spdlog::info("  d_x = {:.4f} M, d_y = {:.4f} M, mean = {:.4f} M",
                     d.dx_M, d.dy_M, d.mean_M);
        spdlog::info("  mean diameter = {:.2f} uas (EHT 2022 ring: 51.8±2.3, "
                     "shadow: 48.7±7)", d.mean_uas);
        spdlog::info("  axis ratio dx/dy = {:.4f}", d.axis_ratio);
        const bool pass = d.mean_uas > 48.0 && d.mean_uas < 52.0;
        nlohmann::json j{{"a_star", constants::sgrA_spin_default},
                         {"incl_deg", constants::sgrA_incl_deg},
                         {"dx_M", d.dx_M},
                         {"dy_M", d.dy_M},
                         {"mean_M", d.mean_M},
                         {"mean_uas", d.mean_uas},
                         {"axis_ratio", d.axis_ratio},
                         {"uas_per_M", constants::sgrA_uas_per_rg},
                         {"gate10_pass", pass}};
        std::ofstream(out / "calibration.json") << j.dump(2) << "\n";
        spdlog::info("Gate 10: {}", pass ? "PASS (48-52 uas)" : "FAIL");
        return pass ? 0 : 1;
    }

    if (mode == "sweep") {
        const int na = argc > 3 ? std::atoi(argv[3]) : 20;
        const int ni = argc > 4 ? std::atoi(argv[4]) : 20;
        struct Row { double a, inc; Diameters d; };
        std::vector<Row> rows(static_cast<size_t>(na) * ni);
        std::vector<int> idx(rows.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::atomic<int> done{0};
        std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int k) {
            const int ia = k / ni, ii = k % ni;
            const double a = 0.0 + 0.998 * ia / (na - 1);
            const double inc = 1.0 + 89.0 * ii / (ni - 1);
            rows[k] = {a, inc, measure(a, inc, 4)};
            const int n = ++done;
            if (n % 25 == 0) spdlog::info("sweep {}/{}", n, rows.size());
        });
        std::ofstream csv(out / "sweep.csv");
        csv << "a_star,incl_deg,dx_M,dy_M,mean_M,diameter_uas,axis_ratio,"
               "disp_x_M\n";
        for (const auto& r : rows)
            csv << r.a << "," << r.inc << "," << r.d.dx_M << "," << r.d.dy_M
                << "," << r.d.mean_M << "," << r.d.mean_uas << ","
                << r.d.axis_ratio << "," << r.d.disp_x_M << "\n";
        spdlog::info("wrote {}", (out / "sweep.csv").string());
        return 0;
    }

    if (mode == "photonorbits") {
        // spherical_orbits.csv: the (ξ, η, γ) family over the photon shell at
        // several spins (thesis Ch 5 figure).
        {
            std::ofstream csv(out / "spherical_orbits.csv");
            csv << "a_star,r_tilde,xi,eta,gamma\n";
            for (double a : {0.2, 0.5, 0.94, 0.998}) {
                const KerrMetric m(1.0, a);
                const double r0 = m.r_photon(true) * (1.0 + 1e-6);
                const double r1 = m.r_photon(false) * (1.0 - 1e-6);
                for (int i = 0; i <= 120; ++i) {
                    const double r = r0 + (r1 - r0) * i / 120.0;
                    const double eta = bardeen_eta(a, r);
                    if (eta < 0.0) continue;
                    csv << a << ',' << r << ',' << bardeen_xi(a, r) << ','
                        << eta << ',' << lyapunov_gamma(a, r) << '\n';
                }
            }
        }
        // lyapunov_vs_spin.csv: γ of the polar orbit (ξ = 0) and of the
        // prograde/retrograde equatorial-limit orbits, over the spin range.
        {
            std::ofstream csv(out / "lyapunov_vs_spin.csv");
            csv << "a_star,r_polar,gamma_polar,gamma_pro,gamma_retro\n";
            for (int i = 0; i <= 100; ++i) {
                const double a = 0.998 * i / 100.0;
                const KerrMetric m(1.0, a);
                double rp = m.r_photon(true), rr = m.r_photon(false);
                // polar orbit: bisect ξ(r) = 0 on [r_pro, r_retro]
                double lo = rp * (1 + 1e-9), hi = rr * (1 - 1e-9);
                if (a < 1e-8) { lo = hi = 3.0; }
                else
                    for (int it = 0; it < 80; ++it) {
                        const double mid = 0.5 * (lo + hi);
                        if (bardeen_xi(a, mid) > 0.0) lo = mid; else hi = mid;
                    }
                const double rpol = 0.5 * (lo + hi);
                csv << a << ',' << rpol << ',' << lyapunov_gamma(a, rpol) << ','
                    << lyapunov_gamma(a, rp * (1.0 + 1e-6)) << ','
                    << lyapunov_gamma(a, rr * (1.0 - 1e-6)) << '\n';
            }
        }
        spdlog::info("wrote spherical_orbits.csv + lyapunov_vs_spin.csv");
        return 0;
    }
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
