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
int main(void) {
    TEST_RUN(test_kitaev_diag_zz_on_uniform_ansatz);
    TEST_RUN(test_kitaev_xx_bond_on_uniform_ansatz);
    TEST_RUN(test_kitaev_yy_bond_on_uniform_ansatz);
    TEST_RUN(test_kitaev_mixed_couplings);
    TEST_SUMMARY();
}