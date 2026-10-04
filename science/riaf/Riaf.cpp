#include "Riaf.hpp"

#include <algorithm>
#include <cmath>
#include <execution>
#include <numeric>

#include "sgr_physics/integrator.hpp"

namespace sgr {

namespace {
constexpr double kPi = 3.14159265358979323846;

double dot4(const Metric4<double>& g, const Vec4& u, const Vec4& w) {
    double s = 0.0;
    for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 4; ++n) s += g(m, n) * u[m] * w[n];
    return s;
}
} // namespace

RiafModel::RiafModel(const RiafParams& p) : p_(p), m_(1.0, p.a_star) {
    r_isco_ = m_.r_isco(true);
    E_isco_ = m_.energy_circular(r_isco_, true);
    L_isco_ = m_.angmom_circular(r_isco_, true);
}

RiafFluid RiafModel::fluid(double r, double th) const {
    RiafFluid f;
    const double st = std::sin(th), ct = std::cos(th);
    const double rho = r * st, z = r * ct;
    if (r <= m_.r_horizon() * 1.01 || rho < 1e-3) return f;
    f.n_e = p_.n0 * std::pow(rho, p_.p_n) * std::exp(-0.5 * z * z / (rho * rho));
    f.T_e = p_.T0 * std::pow(r, p_.p_T);
    // B²/8π = β⁻¹ n_e m_p c² r_S/(12 r), r_S = 2M
    f.B = std::sqrt(8.0 * kPi * f.n_e * cgs::m_p * cgs::c * cgs::c * (2.0 / (12.0 * r)) / p_.beta);

    const Metric4<double> g = m_.components(r, th);
    const double a = m_.spin();
    Vec4 u{};
    if (p_.static_flow) {
        // tests: at rest relative to the distant stars (outside the ergosphere)
        if (g(0, 0) < 0.0) u = {1.0 / std::sqrt(-g(0, 0)), 0.0, 0.0, 0.0};
    } else if (r >= r_isco_) {
        // Keplerian circulation at the cylindrical radius
        const double Om = 1.0 / (std::pow(rho, 1.5) + a);
        const double nrm = -(g(0, 0) + 2.0 * Om * g(0, 3) + Om * Om * g(3, 3));
        if (nrm > 0.0) {
            const double ut = 1.0 / std::sqrt(nrm);
            u = {ut, 0.0, 0.0, Om * ut};
        }
    } else {
        // Inside the ISCO the gas plunges. On the equator that is the geodesic
        // with the ISCO's E and L; off the plane those constants are not
        // kinematically allowed, so every point takes the equatorial plunge's
        // radial velocity u^r and angular velocity Ω = u^φ/u^t at the same r,
        // no polar motion, and u^t from the local normalisation (exactly the
        // geodesic on the equator).
        const Metric4<double> ge = m_.components(r, 0.5 * kPi);
        const Metric4<double> gi = invert_bl_block(ge);
        const double ut_l = -E_isco_, up_l = L_isco_;
        const double blk = gi(0, 0) * ut_l * ut_l + 2.0 * gi(0, 3) * ut_l * up_l + gi(3, 3) * up_l * up_l;
        const double ur = -std::sqrt(std::max((-1.0 - blk) / ge(1, 1), 0.0));
        double Om = (gi(0, 3) * ut_l + gi(3, 3) * up_l) / (gi(0, 0) * ut_l + gi(0, 3) * up_l);
        double den = g(0, 0) + 2.0 * Om * g(0, 3) + Om * Om * g(3, 3);
        if (!(den < 0.0)) {
            // near the axis that rotation is not timelike: co-rotate with the
            // ZAMO instead (den = −α²)
            Om = -g(0, 3) / g(3, 3);
            den = g(0, 0) + 2.0 * Om * g(0, 3) + Om * Om * g(3, 3);
        }
        const double ut = std::sqrt((-1.0 - g(1, 1) * ur * ur) / den);
        u = {ut, ur, 0.0, Om * ut};
    }
    if (u[0] <= 0.0) {
        // not timelike here (near the axis): the ZAMO, where n_e ≈ 0 anyway
        const double om = -g(0, 3) / g(3, 3);
        const double al = std::sqrt(-(g(0, 0) - g(0, 3) * g(0, 3) / g(3, 3)));
        u = {1.0 / al, 0.0, 0.0, om / al};
    }
    f.u = u;
    if (p_.field == FieldGeometry::Toroidal) {
        // toroidal field: b ∝ (λ, 0, 0, 1), λ = −u_φ/u_t so that b·u = 0
        const double u_t = g(0, 0) * u[0] + g(0, 3) * u[3];
        const double u_p = g(0, 3) * u[0] + g(3, 3) * u[3];
        const double lam = -u_p / u_t;
        const double nb = g(0, 0) * lam * lam + 2.0 * g(0, 3) * lam + g(3, 3);
        if (!(nb > 0.0)) return f;
        const double ib = 1.0 / std::sqrt(nb);
        f.b = {lam * ib, 0.0, 0.0, ib};
    } else {
        // radial ∂_r or vertical ∂_z = cosθ ∂_r − (sinθ/r) ∂_θ, projected
        // orthogonal to the flow, b = v + (u·v) u, and normalised
        const Vec4 v = (p_.field == FieldGeometry::Radial) ? Vec4{0.0, 1.0, 0.0, 0.0}
                                                           : Vec4{0.0, ct, -st / r, 0.0};
        const double uv = dot4(g, u, v);
        Vec4 b{};
        for (int m = 0; m < 4; ++m) b[m] = v[m] + uv * u[m];
        const double nb = dot4(g, b, b);
        if (!(nb > 0.0)) return f;
        for (double& x : b) x /= std::sqrt(nb);
        f.b = b;
    }
    f.valid = std::isfinite(f.n_e) && std::isfinite(f.B);
    return f;
}

std::complex<double> walker_penrose(double a, double r, double th, const Vec4& k,
                                    const Vec4& f) {
    const double st = std::sin(th), ct = std::cos(th);
    const double A = (k[0] * f[1] - k[1] * f[0]) + a * st * st * (k[1] * f[3] - k[3] * f[1]);
    const double B = ((r * r + a * a) * (k[3] * f[2] - k[2] * f[3]) -
                      a * (k[0] * f[2] - k[2] * f[0])) * st;
    return std::complex<double>(A, -B) * std::complex<double>(r, -a * ct);
}

Vec4 levi_civita_cross(double a, double r, double th, const Vec4& x, const Vec4& y,
                       const Vec4& z) {
    // f^μ = ε^{μνρσ} x_ν y_ρ z_σ with ε^{0123} = −1/√−g (ε_{0123} = +√−g)
    auto det3 = [&](int i, int j, int k) {
        return x[i] * (y[j] * z[k] - y[k] * z[j]) - x[j] * (y[i] * z[k] - y[k] * z[i]) +
               x[k] * (y[i] * z[j] - y[j] * z[i]);
    };
    const double sg = (r * r + a * a * std::cos(th) * std::cos(th)) * std::sin(th);
    const double s = -1.0 / sg;
    return {s * det3(1, 2, 3), -s * det3(0, 2, 3), s * det3(0, 1, 3), -s * det3(0, 1, 2)};
}

double RiafModel::j_nu(double nu, double n_e, double T_e, double B, double sin_th) {
    const double Th = cgs::k_B * T_e / (cgs::m_e * cgs::c * cgs::c);
    if (Th < 0.05 || n_e <= 0.0 || B <= 0.0 || sin_th <= 0.0) return 0.0;
    const double nu_c = cgs::e * B / (2.0 * kPi * cgs::m_e * cgs::c);
    const double nu_s = (2.0 / 9.0) * nu_c * Th * Th * sin_th;
    const double X = nu / nu_s;
    if (!(X < 1e12)) return 0.0;
    const double K2 = std::cyl_bessel_k(2.0, 1.0 / Th);
    const double xp = std::cbrt(X);
    const double xx = std::sqrt(X) + 1.88774862536 * std::sqrt(xp);   // 2^{11/12} X^{1/6}
    return std::sqrt(2.0) * kPi * cgs::e * cgs::e * n_e * nu_s / (3.0 * cgs::c * K2) *
           xx * xx * std::exp(-xp);
}

double RiafModel::planck(double nu, double T) {
    const double x = cgs::h * nu / (cgs::k_B * T);
    if (x > 700.0) return 0.0;
    return 2.0 * cgs::h * nu * nu * nu / (cgs::c * cgs::c) / std::expm1(x);
}

void TransferAccumulator::add(double nu_f, double T_e, double alpha, double dl_cm) {
    if (!(alpha > 0.0) || !(dl_cm > 0.0)) return;
    const double dtau = alpha * dl_cm;
    const double S = RiafModel::planck(nu_f, T_e) / (nu_f * nu_f * nu_f);
    inv += std::exp(-tau) * S * (-std::expm1(-dtau));
    tau += dtau;
}

RiafImage riaf_image(const RiafModel& model, const RiafCamera& cam, double pol_fraction) {
    RiafImage img;
    img.n = cam.n;
    img.half_width_M = cam.half_width_M;
    img.I.assign(static_cast<size_t>(cam.n) * cam.n, 0.0);
    const bool pol = pol_fraction > 0.0;
    if (pol) {
        img.Q.assign(img.I.size(), 0.0);
        img.U.assign(img.I.size(), 0.0);
    }
    const KerrSchild ks(1.0, model.params().a_star, -1);
    const KerrMetric mbl(1.0, model.params().a_star);
    const double a = mbl.spin();
    // camera ZAMO tetrad (BL components) for the polarisation basis
    const Metric4<double> gc = mbl.components(cam.r_cam, cam.incl_deg * kPi / 180.0);
    const double om_c = -gc(0, 3) / gc(3, 3);
    const double al_c = std::sqrt(-(gc(0, 0) - gc(0, 3) * gc(0, 3) / gc(3, 3)));
    const Vec4 e_t{1.0 / al_c, 0.0, 0.0, om_c / al_c};
    const Vec4 e_i[3] = {{0.0, 1.0 / std::sqrt(gc(1, 1)), 0.0, 0.0},
                         {0.0, 0.0, 1.0 / std::sqrt(gc(2, 2)), 0.0},
                         {0.0, 0.0, 0.0, 1.0 / std::sqrt(gc(3, 3))}};
    auto cross2 = [](std::complex<double> p, std::complex<double> q) {
        return p.real() * q.imag() - p.imag() * q.real();
    };
    const double th_cam = cam.incl_deg * kPi / 180.0;
    const double rg_cm = model.params().rg_cm;
    std::vector<int> rows(cam.n);
    std::iota(rows.begin(), rows.end(), 0);
    std::vector<double> dH(cam.n, 0.0);

    std::for_each(std::execution::par, rows.begin(), rows.end(), [&](int j) {
        using State = StateVec<8>;
        auto f = [&ks](double, const State& y, State& dy) {
            const Vec4 X{y[0], y[1], y[2], y[3]}, P{y[4], y[5], y[6], y[7]};
            Vec4 dX, dP;
            ks.rhs(X, P, dX, dP);
            for (int i = 0; i < 4; ++i) { dy[i] = dX[i]; dy[i + 4] = dP[i]; }
        };
        for (int i = 0; i < cam.n; ++i) {
            const double bx = cam.half_width_M * (2.0 * (i + 0.5) / cam.n - 1.0);
            const double by = cam.half_width_M * (1.0 - 2.0 * (j + 0.5) / cam.n);
            const double tx = bx / cam.r_cam, ty = by / cam.r_cam;
            const double inv = 1.0 / std::sqrt(1.0 + tx * tx + ty * ty);
            const std::array<double, 3> nd{-inv, -ty * inv, tx * inv};
            const KSRay ray = ks_camera_ray(ks, cam.r_cam, th_cam, nd);
            // polarisation basis at the camera: image right/up made exactly
            // orthogonal to this pixel's direction, and their κ with the
            // received photon k = e_t − n·e
            std::complex<double> kx, ky;
            if (pol) {
                auto dot3 = [](const std::array<double, 3>& p, const std::array<double, 3>& q) {
                    return p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
                };
                std::array<double, 3> e1{0.0, 0.0, 1.0}, e2{0.0, -1.0, 0.0};
                double d = dot3(e1, nd);
                for (int c = 0; c < 3; ++c) e1[c] -= d * nd[c];
                d = std::sqrt(dot3(e1, e1));
                for (double& c : e1) c /= d;
                const double d2n = dot3(e2, nd), d2e = dot3(e2, e1);
                for (int c = 0; c < 3; ++c) e2[c] -= d2n * nd[c] + d2e * e1[c];
                d = std::sqrt(dot3(e2, e2));
                for (double& c : e2) c /= d;
                Vec4 kc{}, fx{}, fy{};
                for (int m = 0; m < 4; ++m) {
                    kc[m] = e_t[m];
                    for (int c = 0; c < 3; ++c) {
                        kc[m] -= nd[c] * e_i[c][m];
                        fx[m] += e1[c] * e_i[c][m];
                        fy[m] += e2[c] * e_i[c][m];
                    }
                }
                kx = walker_penrose(a, cam.r_cam, th_cam, kc, fx);
                ky = walker_penrose(a, cam.r_cam, th_cam, kc, fy);
            }
            double Qacc = 0.0, Uacc = 0.0;
            DormandPrince54<8, decltype(f)> rk(f, 1e-10, 1e-10);
            State y{};
            for (int k = 0; k < 4; ++k) { y[k] = ray.X[k]; y[k + 4] = ray.P[k]; }
            const double E0 = ray.P[0];
            TransferAccumulator acc;
            double lam = 0.0, h = 1.0;
            for (int step = 0; step < 200000; ++step) {
                const Vec4 X{y[0], y[1], y[2], y[3]}, P{y[4], y[5], y[6], y[7]};
                const double r = ks.radius(X[1], X[2], X[3]);
                Vec4 dX, dP;
                ks.rhs(X, P, dX, dP);
                const double speed = std::sqrt(dX[1] * dX[1] + dX[2] * dX[2] + dX[3] * dX[3]);
                // resolve the plasma: ≤ 3% of r per step inside the emitting region
                const double Lmax = (r < cam.r_max * 1.2) ? 0.03 * r : std::max(0.5 * (r - cam.r_max), 0.03 * r);
                const auto res = rk.try_step(lam, y, std::min(h, Lmax / std::max(speed, 1e-30)));
                h = res.h_next;
                if (!res.accepted) continue;
                const Vec4 Xn{y[0], y[1], y[2], y[3]}, Pn{y[4], y[5], y[6], y[7]};
                const double rn = ks.radius(Xn[1], Xn[2], Xn[3]);
                dH[j] = std::max(dH[j], std::abs(ks.hamiltonian(Xn, Pn)) / (E0 * E0));
                if (rn < cam.r_max) {
                    const Vec4 bl = ks.bl_coords(Xn);
                    const RiafFluid fl = model.fluid(bl[1], bl[2]);
                    if (fl.valid && fl.n_e > 0.0) {
                        const Vec4 u = ks.vector_from_bl(bl[1], bl[2], bl[3], fl.u);
                        const Vec4 b = ks.vector_from_bl(bl[1], bl[2], bl[3], fl.b);
                        double eps = 0.0, pb = 0.0;
                        for (int m = 0; m < 4; ++m) { eps += Pn[m] * u[m]; pb += Pn[m] * b[m]; }
                        if (eps > 0.0 && std::isfinite(eps)) {
                            const double nu_f = cam.nu_obs * eps;
                            const double cth = std::clamp(-pb / eps, -1.0, 1.0);
                            const double sth = std::sqrt(1.0 - cth * cth);
                            const double jn = RiafModel::j_nu(nu_f, fl.n_e, fl.T_e, fl.B, sth);
                            const double Bn = RiafModel::planck(nu_f, fl.T_e);
                            if (jn > 0.0 && Bn > 0.0) {
                                const double before = acc.inv;
                                acc.add(nu_f, fl.T_e, jn / Bn, eps * res.h_used * rg_cm);
                                if (pol) {
                                    // emitted E ⊥ k and B in the fluid frame:
                                    // f = ε(u, k, b); its direction at the
                                    // camera from the Walker–Penrose constant
                                    const Metric4<double> g = mbl.components(bl[1], bl[2]);
                                    const Metric4<double> gi = invert_bl_block(g);
                                    const Vec4 Pb = ks.covector_to_bl(Xn, Pn);
                                    Vec4 kl{}, k{}, ul{}, bll{};
                                    for (int m = 0; m < 4; ++m) kl[m] = -Pb[m];
                                    for (int m = 0; m < 4; ++m)
                                        for (int q = 0; q < 4; ++q) {
                                            k[m] += gi(m, q) * kl[q];
                                            ul[m] += g(m, q) * fl.u[q];
                                            bll[m] += g(m, q) * fl.b[q];
                                        }
                                    const Vec4 fe = levi_civita_cross(a, bl[1], bl[2], ul, kl, bll);
                                    const std::complex<double> ke = walker_penrose(a, bl[1], bl[2], k, fe);
                                    const double chi = std::atan2(-cross2(kx, ke), cross2(ky, ke));
                                    // sky EVPA = χ − 90° (from image-up toward
                                    // image-left): e^{2iχ_sky} = −e^{2iχ}
                                    const double dI = (acc.inv - before) * pol_fraction;
                                    Qacc -= dI * std::cos(2.0 * chi);
                                    Uacc -= dI * std::sin(2.0 * chi);
                                }
                            }
                        }
                    }
                }
                if (acc.tau > 40.0) break;                 // opaque: nothing behind counts
                if (rn < ks.r_plus()) break;
                if (rn > 1.2 * cam.r_cam) {
                    ks.rhs(Xn, Pn, dX, dP);
                    if (Xn[1] * dX[1] + Xn[2] * dX[2] + Xn[3] * dX[3] > 0.0) break;
                }
            }
            const size_t pix = static_cast<size_t>(j) * cam.n + i;
            img.I[pix] = acc.intensity(cam.nu_obs);
            if (pol) {
                const double nu3 = cam.nu_obs * cam.nu_obs * cam.nu_obs;
                img.Q[pix] = Qacc * nu3;
                img.U[pix] = Uacc * nu3;
            }
        }
    });
    img.max_dH = *std::max_element(dH.begin(), dH.end());
    return img;
}

std::complex<double> beta_mode(const std::vector<double>& I, const std::vector<double>& Q,
                               const std::vector<double>& U, int n, double half, int m) {
    std::complex<double> acc(0.0, 0.0);
    double It = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const size_t p = static_cast<size_t>(j) * n + i;
            const double bx = half * (2.0 * (i + 0.5) / n - 1.0);
            const double by = half * (1.0 - 2.0 * (j + 0.5) / n);
            const double phi = std::atan2(-bx, by);
            acc += std::complex<double>(Q[p], U[p]) *
                   std::complex<double>(std::cos(m * phi), -std::sin(m * phi));
            It += I[p];
        }
    return acc / It;
}

double riaf_flux_jy(const RiafImage& img, double rg_cm, double dist_cm) {
    const double pix = 2.0 * img.half_width_M / img.n * rg_cm / dist_cm;   // rad
    double F = 0.0;
    for (double v : img.I) F += v;
    return F * pix * pix / 1e-23;
}

} // namespace sgr
