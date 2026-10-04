#pragma once
// EXOTIC SPACETIME — NOT A MODEL OF SGR A* (Gate 21).
//
// The Double Negative ("Dneg") traversable wormhole of James, von Tunzelmann,
// Franklin & Thorne (2015, Am. J. Phys. 83, 486; after Morris & Thorne 1988),
// used here as a test of the renderer's generality on a non-Kerr metric:
//   ds² = −dt² + dℓ² + r(ℓ)² (dθ² + sin²θ dφ²),
//   r(ℓ) = ρ                                          |ℓ| ≤ a,
//   r(ℓ) = ρ + M [x arctan x − ½ ln(1 + x²)],  x = 2(|ℓ| − a)/(πM),  |ℓ| > a.
// Rays (their eqs. A.7, with p_t = −1 so the affine parameter is t):
//   dℓ/dt = p_ℓ,  dθ/dt = p_θ/r²,  dφ/dt = b/(r² sin²θ),
//   dp_ℓ/dt = B² (dr/dℓ)/r³,  dp_θ/dt = b² cosθ/(r² sin³θ),
// with constants b = p_φ and B² = p_θ² + b²/sin²θ (spherical symmetry). A ray
// traced back to ℓ → +∞ comes from "our" universe, to ℓ → −∞ from the other.

#include <array>

namespace sgr {

struct DnegParams {
    double rho = 1.0;   // throat radius
    double a = 0.005;   // half-length of the cylindrical interior
    double M = 0.05 / 1.42953;   // lensing parameter: W = 1.42953 M (JVFT 2015)
};

class DnegWormhole {
public:
    explicit DnegWormhole(const DnegParams& p) : p_(p) {}
    const DnegParams& params() const { return p_; }
    double r(double l) const;
    double drdl(double l) const;
    // Lensing width W = z-extent of the embedding surface's flare, from
    // z(ℓ) = ∫ √(1 − (dr/dℓ)²) dℓ evaluated where dr/dℓ = 1/√2 ... (see .cpp)
    double lensing_width_numeric() const;
    static double lensing_width_closed_form(double M);   // 1.42953 M

    struct Ray {
        double l, th, ph, pl, pth;   // state
        double b, B2;                // constants
    };
    // Camera at (ℓ_c, θ_c, φ_c) with the JVFT local basis e_x = e_ℓ,
    // e_y = e_φ, e_z = −e_θ; N = the direction on the camera's sky the light
    // ARRIVES FROM (unit, components along e_x, e_y, e_z).
    Ray camera_ray(double l_c, double th_c, double ph_c,
                   const std::array<double, 3>& N) const;
    void rhs(const Ray& y, double d[5]) const;
    struct Outcome {
        int universe = 0;            // +1 our side (ℓ → +∞), −1 the other
        double th = 0.0, ph = 0.0;   // asymptotic direction on that sky
        double max_db = 0.0, max_dB2 = 0.0, max_H = 0.0;
    };
    // Backward integration (DOPRI) until |ℓ| exceeds l_far and is growing.
    Outcome trace(Ray y, double l_far = 200.0) const;

private:
    DnegParams p_;
};

} // namespace sgr
