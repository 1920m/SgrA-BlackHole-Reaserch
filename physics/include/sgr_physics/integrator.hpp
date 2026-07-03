#pragma once
// Adaptive Dormand-Prince RK5(4) integrator (a.k.a. RK45 / DOPRI5).
// Header-only, fixed-dimension template, no allocation in the hot loop.
//
// Error control: mixed absolute/relative tolerance with PI-style step
// adaptation. The 5th-order solution is propagated (local extrapolation).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace sgr {

template <std::size_t N>
using StateVec = std::array<double, N>;

template <std::size_t N>
struct StepResult {
    bool accepted = false;
    double h_used = 0.0;   // step size actually taken (if accepted)
    double h_next = 0.0;   // suggested next step size
    double error = 0.0;    // scaled error estimate of the attempted step
};

// RHS concept: void rhs(double x, const StateVec<N>& y, StateVec<N>& dydx)
template <std::size_t N, typename RHS>
class DormandPrince54 {
public:
    explicit DormandPrince54(RHS rhs, double atol = 1e-12, double rtol = 1e-12)
        : rhs_(std::move(rhs)), atol_(atol), rtol_(rtol) {}

    double atol() const { return atol_; }
    double rtol() const { return rtol_; }
    void set_tolerances(double atol, double rtol) { atol_ = atol; rtol_ = rtol; }

    // Attempt a single step of size h from (x, y). On acceptance, x and y are
    // advanced in place. Always returns a suggested next step size.
    StepResult<N> try_step(double& x, StateVec<N>& y, double h) {
        static constexpr double c2 = 1.0 / 5.0, c3 = 3.0 / 10.0, c4 = 4.0 / 5.0,
                                c5 = 8.0 / 9.0;
        static constexpr double a21 = 1.0 / 5.0;
        static constexpr double a31 = 3.0 / 40.0, a32 = 9.0 / 40.0;
        static constexpr double a41 = 44.0 / 45.0, a42 = -56.0 / 15.0, a43 = 32.0 / 9.0;
        static constexpr double a51 = 19372.0 / 6561.0, a52 = -25360.0 / 2187.0,
                                a53 = 64448.0 / 6561.0, a54 = -212.0 / 729.0;
        static constexpr double a61 = 9017.0 / 3168.0, a62 = -355.0 / 33.0,
                                a63 = 46732.0 / 5247.0, a64 = 49.0 / 176.0,
                                a65 = -5103.0 / 18656.0;
        // 5th-order weights (b) and embedded 4th-order weights (b*)
        static constexpr double b1 = 35.0 / 384.0, b3 = 500.0 / 1113.0,
                                b4 = 125.0 / 192.0, b5 = -2187.0 / 6784.0,
                                b6 = 11.0 / 84.0;
        static constexpr double e1 = 71.0 / 57600.0, e3 = -71.0 / 16695.0,
                                e4 = 71.0 / 1920.0, e5 = -17253.0 / 339200.0,
                                e6 = 22.0 / 525.0, e7 = -1.0 / 40.0;

        StateVec<N> k1, k2, k3, k4, k5, k6, k7, yt, y5;
        rhs_(x, y, k1);
        for (std::size_t i = 0; i < N; ++i) yt[i] = y[i] + h * a21 * k1[i];
        rhs_(x + c2 * h, yt, k2);
        for (std::size_t i = 0; i < N; ++i)
            yt[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i]);
        rhs_(x + c3 * h, yt, k3);
        for (std::size_t i = 0; i < N; ++i)
            yt[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
        rhs_(x + c4 * h, yt, k4);
        for (std::size_t i = 0; i < N; ++i)
            yt[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
        rhs_(x + c5 * h, yt, k5);
        for (std::size_t i = 0; i < N; ++i)
            yt[i] = y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] +
                                a64 * k4[i] + a65 * k5[i]);
        rhs_(x + h, yt, k6);
        for (std::size_t i = 0; i < N; ++i)
            y5[i] = y[i] + h * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] +
                                b5 * k5[i] + b6 * k6[i]);
        rhs_(x + h, y5, k7); // FSAL stage, used only in the error estimate here

        double err = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            const double ei = h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] +
                                   e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
            const double sc =
                atol_ + rtol_ * std::max(std::abs(y[i]), std::abs(y5[i]));
            const double q = ei / sc;
            err += q * q;
        }
        err = std::sqrt(err / static_cast<double>(N));

        StepResult<N> res;
        res.error = err;
        const double safety = 0.9;
        double factor = (err > 0.0) ? safety * std::pow(err, -0.2) : 5.0;
        factor = std::clamp(factor, 0.2, 5.0);
        res.h_next = h * factor;

        if (err <= 1.0) {
            res.accepted = true;
            res.h_used = h;
            x += h;
            y = y5;
        }
        return res;
    }

private:
    RHS rhs_;
    double atol_, rtol_;
};

} // namespace sgr
