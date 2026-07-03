// CPU geodesic throughput baseline (Google Benchmark) — the reference numbers
// the GPU port of Phase 7 is compared against (Phase 11).
#include <benchmark/benchmark.h>

#include <cmath>

#include "sgr_physics/integrator.hpp"
#include "sgr_physics/metric.hpp"
#include "sgr_physics/null_geodesics.hpp"
#include "sgr_physics/timelike_geodesics.hpp"

using namespace sgr;

static void BM_NullGeodesicStep(benchmark::State& state) {
    const KerrMetric metric(1.0, 0.94);
    NullRayConstants c;
    c.M = 1.0; c.a = 0.94; c.E = 1.0; c.Lz = 7.0; c.Q = 30.0;
    NullState y{};
    y[1] = 50.0; y[2] = 1.3;
    y[4] = -std::sqrt(radial_potential(c, 50.0));
    y[5] = std::sqrt(polar_potential(c, 1.3));

    auto rhs = [&c](double, const NullState& s, NullState& d) {
        null_geodesic_rhs(c, s, d);
    };
    DormandPrince54<6, decltype(rhs)> rk(rhs, 1e-10, 1e-10);

    double tau = 0.0, h = 1e-3;
    NullState y0 = y;
    for (auto _ : state) {
        const auto res = rk.try_step(tau, y, h);
        h = res.h_next;
        if (y[1] < 1.5 || y[1] > 400.0) { y = y0; tau = 0.0; h = 1e-3; }
        benchmark::DoNotOptimize(y);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_NullGeodesicStep);

static void BM_ChristoffelEval(benchmark::State& state) {
    const KerrMetric metric(1.0, 0.94);
    double r = 10.0;
    for (auto _ : state) {
        const Christoffel gam = christoffel(metric, r, 1.1);
        benchmark::DoNotOptimize(gam);
        r = 10.0 + std::fmod(r, 1.0) * 0.001; // defeat CSE across iterations
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ChristoffelEval);

static void BM_TimelikeGeodesicStep(benchmark::State& state) {
    const KerrMetric metric(1.0, 0.94);
    TimelikeGeodesic geo(metric);
    const Metric4<double> g0 = metric.components(20.0, 1.4);
    TimelikeState y{};
    y[1] = 20.0; y[2] = 1.4;
    y[6] = 0.02 / std::sqrt(g0(2, 2));
    y[7] = 0.22 / std::sqrt(g0(3, 3));
    y[4] = TimelikeGeodesic::solve_ut(g0, 0.0, y[6], y[7]);
    TimelikeStepper stepper{&geo, 1e-10, 1e-10};

    const TimelikeState y0 = y;
    double tau = 0.0, h = 0.1;
    for (auto _ : state) {
        stepper.step(tau, y, h);
        if (y[1] < 3.0 || y[1] > 100.0) { y = y0; tau = 0.0; h = 0.1; }
        benchmark::DoNotOptimize(y);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_TimelikeGeodesicStep);
