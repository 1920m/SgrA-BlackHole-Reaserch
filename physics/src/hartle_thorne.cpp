#include "sgr_physics/hartle_thorne.hpp"

#include <cmath>

namespace sgr {
namespace {

inline double real_part(double x) { return x; }
inline double real_part(const std::complex<double>& x) { return x.real(); }

// Large-r series of F1, F2 in u = M/r, generated symbolically (sympy) from the
// exact GB06 (A.4) expressions; coefficients are exact rationals rounded to
// double, powers u^3 .. u^26. Verified against the closed form to <1e-13
// relative in the overlap window r ∈ [8M, 12M] (test_hartle_thorne_limit).
constexpr double kF1Series[] = {
    -1.00000000000000000e+00, -3.00000000000000000e+00, -7.14285714285714324e+00,
    -1.57142857142857135e+01, -3.33333333333333357e+01, -6.93333333333333286e+01,
    -1.42545454545454533e+02, -2.90909090909090935e+02, -5.90769230769230717e+02,
    -1.19560439560439568e+03, -2.41371428571428578e+03, -4.86400000000000000e+03,
    -9.78823529411764684e+03, -1.96768627450980384e+04, -3.95228070175438625e+04,
    -7.93330526315789466e+04, -1.59158857142857130e+05, -3.19168831168831151e+05,
    -6.39817865612648195e+05, -1.28222608695652173e+06, -2.56901120000000019e+06,
    -5.14608836923076957e+06, -1.03065162393162400e+07, -2.06386387301587313e+07};
constexpr double kF2Series[] = {
    1.00000000000000000e+00, 2.50000000000000000e+00, 5.14285714285714324e+00,
    1.00000000000000000e+01, 1.90476190476190474e+01, 3.60000000000000000e+01,
    6.78787878787878753e+01, 1.28000000000000000e+02, 2.41678321678321680e+02,
    4.57142857142857167e+02, 8.66461538461538453e+02, 1.64571428571428578e+03,
    3.13223529411764684e+03, 5.97333333333333303e+03, 1.14130030959752330e+04,
    2.18453333333333321e+04, 4.18839097744360915e+04, 8.04305454545454559e+04,
    1.54681242236024846e+05, 2.97890909090909117e+05, 5.74437286956521682e+05,
    1.10907076923076925e+06, 2.14375537777777761e+06, 4.14821274725274742e+06};

// Switch radius between closed form and series: below ~9M the closed form has
// acceptable cancellation error (<1e-12 relative); above, the two O(r) terms
// cancel to O(M³/r³) and the series (machine precise there) must be used.
constexpr double kSeriesSwitch = 9.0;
constexpr int kNumCoeffs = 24;

template <typename T>
T horner_u3(const double* coeffs, T u) {
    T acc = T(coeffs[kNumCoeffs - 1]);
    for (int i = kNumCoeffs - 2; i >= 0; --i) acc = acc * u + T(coeffs[i]);
    return acc * u * u * u;
}

template <typename T>
T F1_impl(T r, double M) {
    if (real_part(r) > kSeriesSwitch * M) return horner_u3<T>(kF1Series, T(M) / r);
    using std::log;
    const T m = T(M);
    return -5.0 * (r - m) * (2.0 * m * m + 6.0 * m * r - 3.0 * r * r) /
               (8.0 * m * r * (r - 2.0 * m)) -
           15.0 * r * (r - 2.0 * m) / (16.0 * m * m) * log(r / (r - 2.0 * m));
}

template <typename T>
T F2_impl(T r, double M) {
    if (real_part(r) > kSeriesSwitch * M) return horner_u3<T>(kF2Series, T(M) / r);
    using std::log;
    const T m = T(M);
    return 5.0 * (2.0 * m * m - 3.0 * m * r - 3.0 * r * r) / (8.0 * m * r) +
           15.0 * (r * r - 2.0 * m * m) / (16.0 * m * m) * log(r / (r - 2.0 * m));
}

} // namespace

double HartleThorneMetric::F1(double r) const { return F1_impl<double>(r, mass()); }
double HartleThorneMetric::F2(double r) const { return F2_impl<double>(r, mass()); }

template <typename T>
Metric4<T> HartleThorneMetric::components_impl(T r, T theta) const {
    const double M = kerr_.mass();
    Metric4<T> g = kerr_components<T>(r, theta, M, kerr_.spin());
    if (eps_q_ == 0.0) return g; // exact Kerr by construction

    using std::cos;
    using std::sin;
    const T ct = cos(theta), st = sin(theta);
    const T ang = T(1) - T(3) * ct * ct; // (1 − 3cos²θ) = −2 P₂(cosθ)
    const T f = T(1) - T(2.0 * M) / r;   // Schwarzschild lapse factor
    const T f1 = F1_impl<T>(r, M);
    const T f2 = F2_impl<T>(r, M);

    // Sign convention, anchored empirically (see test_hartle_thorne_limit
    // "[precession]"): with the + sign below the far-field Newtonian limit is
    // δΦ = −δQ P₂(cosθ)/r³ with δQ = +ε_q M³, i.e. exactly the project's
    // Q_total = −M a² + ε_q M³, and the measured pericenter shift matches
    // δ(Δω) = −3π ε_q (M/p)². (GB06 list h^{ab} as the perturbation of the
    // inverse metric, so the covariant deviation enters with the opposite
    // sign to a naive index-lowering of their eq. (15).)
    const double e = eps_q_;
    g(0, 0) += T(e) * f * ang * f1;
    g(1, 1) += T(e) * ang * f1 / f;
    g(2, 2) += T(e) * (-r * r * ang * f2);
    g(3, 3) += T(e) * (-r * r * st * st * ang * f2);
    return g;
}

Metric4<double> HartleThorneMetric::components(double r, double theta) const {
    return components_impl<double>(r, theta);
}
Metric4<std::complex<double>> HartleThorneMetric::components(
    std::complex<double> r, std::complex<double> theta) const {
    return components_impl<std::complex<double>>(r, theta);
}

} // namespace sgr
