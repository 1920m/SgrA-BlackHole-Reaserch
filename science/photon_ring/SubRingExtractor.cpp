#include "photon_ring/SubRingExtractor.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <execution>
#include <mutex>
#include <numeric>

namespace sgr {

SubRingStats SubRingExtractor::analyze(int n_max, int image_size,
                                       std::vector<ImageRGB>* images) const {
    SubRingStats stats;
    stats.bins.resize(n_max + 1);
    if (images) {
        images->assign(n_max + 1, ImageRGB{});
        for (auto& im : *images) im.resize(image_size, image_size);
    }

    const auto& cam = tracer_->camera();
    const double t_half = std::tan(0.5 * cam.fov_rad);

    std::vector<int> rows(image_size);
    std::iota(rows.begin(), rows.end(), 0);
    std::mutex mtx;

    std::for_each(std::execution::par, rows.begin(), rows.end(), [&](int j) {
        std::vector<SubRingStats::Bin> local(n_max + 1);
        for (int i = 0; i < image_size; ++i) {
            const double x = 2.0 * (i + 0.5) / image_size - 1.0;
            const double y = 1.0 - 2.0 * (j + 0.5) / image_size;
            const TraceResult res = tracer_->trace(x, y);
            if (res.outcome != RayOutcome::DiskHit) continue;
            const int n = std::min(res.disk.subring_n, n_max);
            const double I = disk_->bolometric(res.disk.r, res.disk.g);
            local[n].pixels++;
            local[n].flux_bolometric += I;
            const double tx = t_half * x, ty = t_half * y;
            local[n].mean_image_angle += I * std::atan(std::hypot(tx, ty));
            if (images) {
                float* p = (*images)[n].pixel(i, j);
                p[0] = p[1] = p[2] = static_cast<float>(I);
            }
        }
        std::scoped_lock lk(mtx);
        for (int n = 0; n <= n_max; ++n) {
            stats.bins[n].pixels += local[n].pixels;
            stats.bins[n].flux_bolometric += local[n].flux_bolometric;
            stats.bins[n].mean_image_angle += local[n].mean_image_angle;
        }
    });

    for (auto& b : stats.bins)
        if (b.flux_bolometric > 0.0) b.mean_image_angle /= b.flux_bolometric;
    return stats;
}

void SubRingExtractor::scan_widths(SubRingStats& stats, int n, double psi,
                                   double s_lo, double s_hi) const {
    const auto& cam = tracer_->camera();
    const double t_half = std::tan(0.5 * cam.fov_rad);
    const double aspect = static_cast<double>(cam.height) / cam.width;

    auto order_at = [&](double s) -> int {
        const double tx = std::tan(s) * std::cos(psi);
        const double ty = std::tan(s) * std::sin(psi);
        const TraceResult res =
            tracer_->trace(tx / t_half, ty / (t_half * aspect));
        return res.outcome == RayOutcome::DiskHit ? res.disk.subring_n : -1;
    };

    // Dense scan to bracket the band, then bisect both edges.
    const int N = 1200;
    double first = -1.0, last = -1.0;
    int prev_order = order_at(s_lo);
    double prev_s = s_lo;
    for (int i = 1; i <= N; ++i) {
        const double s = s_lo + (s_hi - s_lo) * i / N;
        const int o = order_at(s);
        if (o == n && first < 0.0) {
            // refine inner edge in [prev_s, s]
            double lo = prev_s, hi = s;
            for (int it = 0; it < 30; ++it) {
                const double mid = 0.5 * (lo + hi);
                if (order_at(mid) == n) hi = mid; else lo = mid;
            }
            first = 0.5 * (lo + hi);
        }
        if (o == n) last = s;
        if (o != n && last > 0.0 && prev_order == n) {
            double lo = prev_s, hi = s;
            for (int it = 0; it < 30; ++it) {
                const double mid = 0.5 * (lo + hi);
                if (order_at(mid) == n) lo = mid; else hi = mid;
            }
            last = 0.5 * (lo + hi);
        }
        prev_order = o;
        prev_s = s;
    }
    if (first > 0.0 && last > first) {
        stats.widths.push_back({psi, n, first, last});
    }
}

} // namespace sgr
