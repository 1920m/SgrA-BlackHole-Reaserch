#pragma once
// Timelike geodesics via the second-order geodesic equation
//
//     dx^μ/dτ = u^μ,     du^λ/dτ = −Γ^λ_{μν} u^μ u^ν
//
// with Christoffel symbols obtained from the generic IMetric interface
// (complex-step differentiation, machine precision). This integrator works for
// *any* stationary axisymmetric metric — in particular the Hartle-Thorne /
// quasi-Kerr metric, which is NOT Carter-separable, so S-star orbits in the
// no-hair module must go through this path.
//
// Conserved diagnostics:
//   E  = −u_t          (stationarity)          drift < 1e-6 relative
//   L_z =  u_φ         (axisymmetry)           drift < 1e-6 relative
//   g_μν u^μ u^ν = −1  (normalisation)         drift < 1e-8 absolute
//   Q  (Kerr only)     (Carter separability)   drift < 1e-6 relative

#include <array>

#include "sgr_physics/integrator.hpp"
#include "sgr_physics/metric.hpp"

namespace sgr {

// State: y = (t, r, θ, φ, u^t, u^r, u^θ, u^φ), proper time parameterisation.
using TimelikeState = StateVec<8>;

class TimelikeGeodesic {
public:
    explicit TimelikeGeodesic(const IMetric& metric) : metric_(&metric) {}

    void rhs(const TimelikeState& y, TimelikeState& dy) const;

    // Complete u^t from the spatial components (u^r, u^θ, u^φ) so that
    // g_μν u^μ u^ν = −1, taking the future-directed root.
    static double solve_ut(const Metric4<double>& g, double ur, double utheta,
                           double uphi);

    // Conserved-quantity diagnostics at a state.
    double energy(const TimelikeState& y) const;   // −u_t
    double angmom(const TimelikeState& y) const;   //  u_φ
    double norm(const TimelikeState& y) const;     //  g_μν u^μ u^ν  (→ −1)
    // Carter constant, meaningful for Kerr only (pass the Kerr spin a).
    double carter_q(const TimelikeState& y, double a) const;

private:
    const IMetric* metric_;
};

// Adaptive integration driver: advance one accepted RK45 step (retrying with
// smaller h on rejection), returns the accepted step size; h is updated to the
// suggestion for the next step.
struct TimelikeStepper {
    const TimelikeGeodesic* geo;
    double atol = 1e-12;
    double rtol = 1e-12;

    // Advances (tau, y) by one accepted step; h carries the trial step in and
    // the suggested next step out.
    void step(double& tau, TimelikeState& y, double& h) const;
};

} // namespace sgr
