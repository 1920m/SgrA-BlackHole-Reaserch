#include "Wormhole.hpp"

#include <algorithm>
#include <cmath>

#include "sgr_physics/integrator.hpp"

namespace sgr {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

double DnegWormhole::r(double l) const {
    const double d = std::abs(l) - p_.a;
    if (d <= 0.0) return p_.rho;
    const double x = 2.0 * d / (kPi * p_.M);
    return p_.rho + p_.M * (x * std::atan(x) - 0.5 * std::log1p(x * x));
}

double DnegWormhole::drdl(double l) const {
    const double d = std::abs(l) - p_.a;
    if (d <= 0.0) return 0.0;
    const double x = 2.0 * d / (kPi * p_.M);
    return (l >= 0.0 ? 1.0 : -1.0) * (2.0 / kPi) * std::atan(x);
}

// W: the radial distance from the mouth to where the embedding surface
// z(ℓ) = ∫ √(1 − (dr/dℓ)²) dℓ is inclined at 45° (dr/dℓ = 1/√2) — which is
// how JVFT's closed form arises (their footnote 19).
double DnegWormhole::lensing_width_numeric() const {
    double lo = p_.a, hi = p_.a + 1000.0 * p_.M;
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (drdl(mid) < 1.0 / std::sqrt(2.0)) lo = mid; else hi = mid;
    }
    return r(0.5 * (lo + hi)) - p_.rho;
}

double DnegWormhole::lensing_width_closed_form(double M) {
    const double u = kPi / (2.0 * std::sqrt(2.0));
    return M * (-std::log(1.0 / std::cos(u)) + u * std::tan(u));
}

DnegWormhole::Ray DnegWormhole::camera_ray(double l_c, double th_c, double ph_c,
                                           const std::array<double, 3>& N) const {
    // JVFT (A.9): the incoming ray propagates along n = −N in the camera basis
    // e_x = e_ℓ, e_y = e_φ, e_z = −e_θ; backward tracing follows −n = N.
    const double nl = -N[0], nph = -N[1], nth = N[2];
    const double rc = r(l_c), st = std::sin(th_c);
    Ray y;
    y.l = l_c;
    y.th = th_c;
    y.ph = ph_c;
    y.pl = nl;
    y.pth = rc * nth;
    y.b = rc * st * nph;
    y.B2 = rc * rc * (nth * nth + nph * nph);
    return y;
}

void DnegWormhole::rhs(const Ray& y, double d[5]) const {
    const double rr = r(y.l), st = std::sin(y.th), ct = std::cos(y.th);
    d[0] = y.pl;
    d[1] = y.pth / (rr * rr);
    d[2] = y.b / (rr * rr * st * st);
    d[3] = y.B2 * drdl(y.l) / (rr * rr * rr);
    d[4] = y.b * y.b * ct / (rr * rr * st * st * st);
}

DnegWormhole::Outcome DnegWormhole::trace(Ray y0, double l_far) const {
    // integrate backward in t: t runs negative
    using S = StateVec<5>;
    const double b = y0.b, B2 = y0.B2;
    auto f = [&](double, const S& s, S& ds) {
        Ray y{s[0], s[1], s[2], s[3], s[4], b, B2};
        double d[5];
        rhs(y, d);
        for (int k = 0; k < 5; ++k) ds[k] = -d[k];   // backward
    };
    DormandPrince54<5, decltype(f)> rk(f, 1e-12, 1e-12);
    S s{y0.l, y0.th, y0.ph, y0.pl, y0.pth};
    Outcome o;
    double t = 0.0, h = 1e-3;
    for (int step = 0; step < 2000000; ++step) {
        const auto res = rk.try_step(t, s, h);
        h = std::min(res.h_next, 0.2 * std::max(std::abs(s[0]), r(s[0])));
        if (!res.accepted) continue;
        const double rr = r(s[0]), st = std::sin(s[1]);
        // conserved: B² = p_θ² + b²/sin²θ and the null condition
        // H ∝ −1 + p_ℓ² + B²/r² = 0
        o.max_dB2 = std::max(o.max_dB2, std::abs(s[4] * s[4] + b * b / (st * st) - B2) /
                                            std::max(B2, 1e-30));
        o.max_H = std::max(o.max_H, std::abs(-1.0 + s[3] * s[3] + B2 / (rr * rr)));
        // moving away from the wormhole (backward in time: ℓ grows outward
        // when −p_ℓ has the sign of ℓ) and far out
        if (std::abs(s[0]) > l_far && s[0] * (-s[3]) > 0.0) break;
    }
    o.universe = s[0] > 0.0 ? +1 : -1;
    // Asymptotic direction. At |ℓ| = l_far the ray still has asin(B/r) of
    // its sweep along its great circle to go (B/r ≈ 2e-3 at l_far = 400 —
    // not negligible for the Einstein ring); out there r' = 1 − O(M/ℓ), so
    // completing it as a straight line errs by O(B M/ℓ²).
    const double rr = r(s[0]), st = std::sin(s[1]), ct = std::cos(s[1]);
    const double sph = std::sin(s[2]), cph = std::cos(s[2]);
    const double P[3] = {st * cph, st * sph, ct};
    const double eth[3] = {ct * cph, ct * sph, -st};
    const double eph[3] = {-sph, cph, 0.0};
    const double B = std::sqrt(B2);
    const double dl = std::asin(std::min(B / rr, 1.0));
    double D[3];
    for (int k = 0; k < 3; ++k) {
        // unit tangent of the BACKWARD motion on the sphere
        const double T = B > 0.0 ? -(s[4] * eth[k] + (b / st) * eph[k]) / B : 0.0;
        D[k] = std::cos(dl) * P[k] + std::sin(dl) * T;
    }
    o.th = std::acos(std::clamp(D[2], -1.0, 1.0));
    // keep the winding of the integrated φ (the correction is small)
    const double dphi = std::remainder(std::atan2(D[1], D[0]) - s[2], 2.0 * kPi);
    o.ph = s[2] + dphi;
    return o;
}

} // namespace sgr
