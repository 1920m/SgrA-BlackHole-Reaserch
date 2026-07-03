#include "sgr_physics/constants.hpp"

#include <cstdio>
#include <string>

namespace sgr::constants {

// Human-readable calibration summary, logged by tools and reproduced in the
// paper's reproducibility appendix.
std::string describe() {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "Sgr A*: M = %.4g Msun, d = %.0f pc, a* = %.2f, i = %.0f deg | "
                  "r_g = %.4g m (%.4g AU), 1 r_g = %.4f uas on sky",
                  sgrA_mass_msun, sgrA_distance_pc, sgrA_spin_default,
                  sgrA_incl_deg, sgrA_rg_m, sgrA_rg_au, sgrA_uas_per_rg);
    return std::string(buf);
}

} // namespace sgr::constants
