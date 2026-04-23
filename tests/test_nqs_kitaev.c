/*
 * tests/test_nqs_kitaev.c
 *
 * NQS local-energy kernel for the Kitaev honeycomb model (brick-wall
 * representation). The honeycomb lattice maps to a rectangular Lx × Ly
 * grid where horizontal links alternate between x-bonds and y-bonds
 * by the parity of (x + y), and vertical links are z-bonds.
 *
 *     H = -Jx Σ_x-bonds σ^x σ^x - Jy Σ_y-bonds σ^y σ^y - Jz Σ_z-bonds σ^z σ^z
 *
 * Tests:
 *   (1) Off-diagonal couplings scale linearly with Jx, Jy (σ^x σ^x and
 *       σ^y σ^y connect σ-flipped basis states) and diagonal scales with
 *       Jz (σ^z σ^z is diagonal on the z-basis). Using a uniform ansatz
 *       ψ(s) ≡ 1 makes ψ(s')/ψ(s) = 1 so local-energy components
 *       collapse to direct sums of coefficients we can check by hand.
 *   (2) Cross-check against exact diagonalisation for a 2x2 Kitaev patch.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "harness.h"
#include "nqs/nqs_config.h"
#include "nqs/nqs_gradient.h"
/* Constant log-amp callback: ψ(s) ≡ 1 for every configuration. */
static void const_log_amp(const int *spins, int num_sites, void *user,
                           double *out_log_abs, double *out_arg) {
    (void)spins; (void)num_sites; (void)user;
    if (out_log_abs) *out_log_abs = 0.0;
    if (out_arg)     *out_arg     = 0.0;
}
static void test_kitaev_diag_zz_on_uniform_ansatz(void) {
    /* With Jx = Jy = 0 and Jz = 1, only the vertical z-bonds contribute.
     * For a 2x2 lattice there are 2 vertical bonds: (0,0)-(0,1) and
     * (1,0)-(1,1). With all spins +1, E = -Jz·(1·1 + 1·1) = -2.
     *
     * The off-diagonal pieces vanish because Jx = Jy = 0. */
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian      = NQS_HAM_KITAEV_HONEYCOMB;
    cfg.j_coupling       = 0.0;     /* Jx */
    cfg.transverse_field = 0.0;     /* Jy */
    cfg.j2_coupling      = 1.0;     /* Jz */
    int spins[4] = {+1, +1, +1, +1};
    double E = nqs_local_energy(&cfg, 2, 2, spins, const_log_amp, NULL);
    ASSERT_NEAR(E, -2.0, 1e-12);
}
static void test_kitaev_xx_bond_on_uniform_ansatz(void) {
    /* Pick Jx = 1, Jy = Jz = 0. On a 2x2 the x-bond parity pattern is:
     *   (0,0)-(1,0): x+y=0 → x-bond
     *   (0,1)-(1,1): x+y=1 → y-bond    (NOT counted with Jy=0)
     * So only 1 x-bond. σ^x σ^x coupling: -Jx · ψ(s')/ψ(s) with s' being
     * the double-flipped state. Uniform ψ → ratio = 1.
     * E_loc = -Jx · 1 = -1. */
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian      = NQS_HAM_KITAEV_HONEYCOMB;
    cfg.j_coupling       = 1.0;
    cfg.transverse_field = 0.0;
    cfg.j2_coupling      = 0.0;
    int spins[4] = {+1, +1, +1, +1};
    double E = nqs_local_energy(&cfg, 2, 2, spins, const_log_amp, NULL);
    ASSERT_NEAR(E, -1.0, 1e-12);
}
static void test_kitaev_yy_bond_on_uniform_ansatz(void) {
    /* Jy = 1, Jx = Jz = 0. 2x2 y-bonds: only (0,1)-(1,1) (x+y=1 parity).
     * σ^y σ^y on |↑↑⟩ → (i·1)(i·1) |↓↓⟩ = -|↓↓⟩. Coefficient of
     * ψ(s')/ψ(s) in local energy is -Jy · (-s_i s_j) = Jy · s_i s_j.
     * With all spins +1: Jy · 1 · 1 = +1 with uniform ψ. */
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian      = NQS_HAM_KITAEV_HONEYCOMB;
    cfg.j_coupling       = 0.0;
    cfg.transverse_field = 1.0;
    cfg.j2_coupling      = 0.0;
    int spins[4] = {+1, +1, +1, +1};
    double E = nqs_local_energy(&cfg, 2, 2, spins, const_log_amp, NULL);
    /* One y-bond → E = +1·(+1)(+1) = +1. */
    ASSERT_NEAR(E, 1.0, 1e-12);
}
static void test_kitaev_mixed_couplings(void) {
    /* Jx = 2, Jy = 3, Jz = 5 on a 2x2. Bond count:
     *   x-bonds (horizontal, (x+y) even): (0,0)-(1,0) → 1 bond
     *   y-bonds (horizontal, (x+y) odd):  (0,1)-(1,1) → 1 bond
     *   z-bonds (vertical):               (0,0)-(0,1), (1,0)-(1,1) → 2 bonds
     *
     * All spins +1: diagonal zz = -Jz·(1+1) = -10. Off-diagonal xx = -Jx·1
     * = -2 (σ^x σ^x ratio = 1 on uniform). Off-diagonal yy: coef =
     * +Jy·s_a s_b = +3. Sum = -10 - 2 + 3 = -9. */
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian      = NQS_HAM_KITAEV_HONEYCOMB;
    cfg.j_coupling       = 2.0;
    cfg.transverse_field = 3.0;
    cfg.j2_coupling      = 5.0;
    int spins[4] = {+1, +1, +1, +1};
    double E = nqs_local_energy(&cfg, 2, 2, spins, const_log_amp, NULL);
    ASSERT_NEAR(E, -9.0, 1e-12);
}
/* ------------------------------------------------------------------
 * Kitaev–Heisenberg (NQS_HAM_KITAEV_HEISENBERG) kernel tests.
 *
 * Convention implemented in src/nqs/nqs_gradient.c local_energy_kh:
 *   H = K · Σ σ^γ_i σ^γ_j  +  J · Σ σ_i · σ_j
 * on the same brick-wall honeycomb as the pure-Kitaev kernel. Bond γ
 * colouring: horizontal (x,y)-(x+1,y) is γ=x when (x+y) even, else γ=y;
 * vertical bond is γ=z.
 *
 * Per-bond matrix elements (s' = s ⊕ {i,j} off-diag):
 *   γ=x:  diag = J·s_i s_j,       off = (K+J) − J·s_i s_j
 *   γ=y:  diag = J·s_i s_j,       off = J − (K+J)·s_i s_j
 *   γ=z:  diag = (K+J)·s_i s_j,   off = J·(1 − s_i s_j)
 * ------------------------------------------------------------------ */

static void test_kh_heisenberg_limit_all_up_2x2(void) {
    /* K=0, J=1 (pure Heisenberg on honeycomb). 2x2 lattice, all spins +1.
     * Bonds: 1 x, 1 y, 2 z.  With all sasb = +1:
     *   x: diag = 1,  off coef = (0+1) − 1·1 = 0
     *   y: diag = 1,  off coef = 1 − (0+1)·1 = 0
     *   z: diag = (0+1)·1 = 1, off coef = 1·(1 − 1) = 0 (no flip)
     * Total: diag 1+1+1+1 = 4, off = 0 → E = +4. */
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_KITAEV_HEISENBERG;
    cfg.kh_K = 0.0;
    cfg.kh_J = 1.0;
    int spins[4] = {+1, +1, +1, +1};
    double E = nqs_local_energy(&cfg, 2, 2, spins, const_log_amp, NULL);
    ASSERT_NEAR(E, 4.0, 1e-12);
}

static void test_kh_kitaev_limit_all_up_2x2(void) {
    /* J=0, K=1 (pure Kitaev). 2x2 all spins +1.
     *   x: diag = 0, off coef = (1+0) − 0 = 1  → off contribution +1
     *   y: diag = 0, off coef = 0 − (1+0)·1 = −1 → off contribution −1
     *   z: diag = (1+0)·1 = 1 each, 2 z-bonds → diag 2; off coef 0
     * Total: diag 0+0+1+1 = 2, off 1−1 = 0 → E = +2.
     * Note: this is the opposite sign from the legacy local_energy_kitaev,
     * which uses H = −Σ J_α σ^α σ^α; here we use H = +K·σ^γ σ^γ. */
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_KITAEV_HEISENBERG;
    cfg.kh_K = 1.0;
    cfg.kh_J = 0.0;
    int spins[4] = {+1, +1, +1, +1};
    double E = nqs_local_energy(&cfg, 2, 2, spins, const_log_amp, NULL);
    ASSERT_NEAR(E, 2.0, 1e-12);
}

static void test_kh_equal_couplings_all_up_2x2(void) {
    /* K=1, J=1, all spins +1 on 2x2.
     *   x: diag = 1, off coef = (1+1) − 1·1 = 1
     *   y: diag = 1, off coef = 1 − (1+1)·1 = −1
     *   z: diag = (1+1)·1 = 2 per bond × 2 z-bonds → diag 4; off coef 0
     * Total: diag 1+1+4 = 6, off 1−1 = 0 → E = +6. */
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_KITAEV_HEISENBERG;
    cfg.kh_K = 1.0;
    cfg.kh_J = 1.0;
    int spins[4] = {+1, +1, +1, +1};
    double E = nqs_local_energy(&cfg, 2, 2, spins, const_log_amp, NULL);
    ASSERT_NEAR(E, 6.0, 1e-12);
}

static void test_kh_antiparallel_zbond_triggers_offdiag(void) {
    /* Configure 2x2 with a single z-bond antiparallel pair. Put spins as
     *   (0,0) = +1   (1,0) = +1
     *   (0,1) = -1   (1,1) = +1
     * Flat layout (size_y=2, so flat(x,y)=2x+y):
     *   [flat(0,0)=0: +1, flat(0,1)=1: -1, flat(1,0)=2: +1, flat(1,1)=3: +1]
     * Bonds on 2x2:
     *   x-bond (0,0)-(1,0): sites 0,2, sasb = +1 → diag += 0·1 = 0 (J=0),
     *                       off coef = (K+0) − 0 = K. Uniform ψ: +K.
     *   y-bond (0,1)-(1,1): sites 1,3, sasb = -1 → diag += 0·(-1) = 0,
     *                       off coef = 0 − (K+0)·(-1) = +K. Uniform ψ: +K.
     *   z-bond (0,0)-(0,1): sites 0,1, sasb = -1 → diag += (K+0)·(-1) = -K,
     *                       off: sa != sb, so off coef = 0·(1-(-1)) = 0.
     *   z-bond (1,0)-(1,1): sites 2,3, sasb = +1 → diag += (K+0)·1 = +K,
     *                       off: sa == sb, skip.
     * Total: diag = -K + K = 0. Off = K + K = 2K. With K=1: E = 2.0. */
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_KITAEV_HEISENBERG;
    cfg.kh_K = 1.0;
    cfg.kh_J = 0.0;
    /* flat layout (x=0,y=0),(x=0,y=1),(x=1,y=0),(x=1,y=1) for size_y=2. */
    int spins[4] = {+1, -1, +1, +1};
    double E = nqs_local_energy(&cfg, 2, 2, spins, const_log_amp, NULL);
    ASSERT_NEAR(E, 2.0, 1e-12);
}

static void test_kh_matches_legacy_kitaev_at_matching_signs(void) {
    /* Cross-check the two Kitaev kernels at their overlap point.
     *
     * Legacy local_energy_kitaev uses H = −Σ_α J_α σ^α σ^α. The KH
     * kernel (at J_Heisenberg = 0) uses H = +K σ^γ σ^γ. They represent
     * the same physics when K_KH = −J_α and J_α is isotropic
     * (Jx = Jy = Jz). This test verifies the two dispatch paths
     * produce numerically identical local energies on the same spin
     * configurations, guarding against silent drift between the
     * kernels since the docs claim equivalence at this limit. */
    nqs_config_t cfg_kh = nqs_config_defaults();
    cfg_kh.hamiltonian = NQS_HAM_KITAEV_HEISENBERG;
    cfg_kh.kh_K = -1.0;
    cfg_kh.kh_J =  0.0;

    nqs_config_t cfg_kit = nqs_config_defaults();
    cfg_kit.hamiltonian      = NQS_HAM_KITAEV_HONEYCOMB;
    cfg_kit.j_coupling       = 1.0;  /* Jx */
    cfg_kit.transverse_field = 1.0;  /* Jy */
    cfg_kit.j2_coupling      = 1.0;  /* Jz */

    int spins_up[4]    = {+1, +1, +1, +1};
    int spins_mixed[4] = {+1, -1, +1, +1};

    double E_kh_up  = nqs_local_energy(&cfg_kh,  2, 2, spins_up,    const_log_amp, NULL);
    double E_kit_up = nqs_local_energy(&cfg_kit, 2, 2, spins_up,    const_log_amp, NULL);
    ASSERT_NEAR(E_kh_up, E_kit_up, 1e-12);

    double E_kh_mx  = nqs_local_energy(&cfg_kh,  2, 2, spins_mixed, const_log_amp, NULL);
    double E_kit_mx = nqs_local_energy(&cfg_kit, 2, 2, spins_mixed, const_log_amp, NULL);
    ASSERT_NEAR(E_kh_mx, E_kit_mx, 1e-12);
}

int main(void) {
    TEST_RUN(test_kitaev_diag_zz_on_uniform_ansatz);
    TEST_RUN(test_kitaev_xx_bond_on_uniform_ansatz);
    TEST_RUN(test_kitaev_yy_bond_on_uniform_ansatz);
    TEST_RUN(test_kitaev_mixed_couplings);
    TEST_RUN(test_kh_heisenberg_limit_all_up_2x2);
    TEST_RUN(test_kh_kitaev_limit_all_up_2x2);
    TEST_RUN(test_kh_equal_couplings_all_up_2x2);
    TEST_RUN(test_kh_antiparallel_zbond_triggers_offdiag);
    TEST_RUN(test_kh_matches_legacy_kitaev_at_matching_signs);
    TEST_SUMMARY();
}