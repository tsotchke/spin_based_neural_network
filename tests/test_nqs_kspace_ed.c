/*
 * tests/test_nqs_kspace_ed.c
 *
 * Validates libirrep-backed full-Hilbert-space Heisenberg ED against
 * known analytic results and the homegrown nqs_lanczos.
 *
 * Gated on SPIN_NN_HAS_IRREP; enable with
 *     git submodule update --init --recursive
 *     make IRREP_ENABLE=1 test_nqs_kspace_ed
 */
#include <math.h>
#include <stdio.h>
#include "harness.h"

#ifdef SPIN_NN_HAS_IRREP
#include "nqs/nqs_kspace_ed.h"

/* Two-site Heisenberg: H = J S_1·S_2.  Hilbert space is the
 * 4-dimensional spin-½ × spin-½, eigenvalues are
 *   triplet:    +J/4 (3-fold)
 *   singlet:    -3J/4 (ground state)
 * for J = +1 antiferromagnetic. */
static void test_two_site_heisenberg_matches_analytic_singlet(void) {
    int bi[1] = { 0 }, bj[1] = { 1 };
    double evals[2] = { 0.0, 0.0 };
    int rc = nqs_kspace_ed_heisenberg(/*num_sites*/ 2, /*num_bonds*/ 1,
                                       bi, bj, /*J*/ 1.0,
                                       /*k_wanted*/ 2, /*max_iters*/ 30,
                                       evals);
    ASSERT_EQ_INT(rc, 0);
    printf("# N=2 Heisenberg: E_0 = %.10f, E_1 = %.10f\n", evals[0], evals[1]);
    /* singlet = -3/4, lowest triplet = +1/4 */
    ASSERT_NEAR(evals[0], -0.75, 1e-10);
    ASSERT_NEAR(evals[1],  0.25, 1e-10);
}

/* Four-site Heisenberg ring (PBC): bonds (0,1), (1,2), (2,3), (3,0).
 * Bethe-ansatz / direct ED gives E_0 = -2J for J = 1. */
static void test_four_site_ring_matches_bethe_ansatz(void) {
    int bi[4] = { 0, 1, 2, 3 };
    int bj[4] = { 1, 2, 3, 0 };
    double evals[2] = { 0.0, 0.0 };
    int rc = nqs_kspace_ed_heisenberg(4, 4, bi, bj, 1.0,
                                       /*k*/ 2, /*max_iters*/ 60, evals);
    ASSERT_EQ_INT(rc, 0);
    printf("# N=4 Heisenberg ring: E_0 = %.10f (analytic -2.0)\n", evals[0]);
    ASSERT_NEAR(evals[0], -2.0, 1e-9);
}

/* Six-site Heisenberg ring (PBC): bonds (i, i+1 mod 6) for i = 0..5.
 * E_0 = -1 - 2 cos(π/6) - 2 cos(π/3) = ?  Direct ED reference:
 *   E_0 ≈ -2.80277563773199 for J = 1 (Bethe ansatz, N=6 ring). */
static void test_six_site_ring_matches_known_reference(void) {
    int bi[6] = { 0, 1, 2, 3, 4, 5 };
    int bj[6] = { 1, 2, 3, 4, 5, 0 };
    double evals[3] = { 0.0, 0.0, 0.0 };
    int rc = nqs_kspace_ed_heisenberg(6, 6, bi, bj, 1.0,
                                       3, /*max_iters*/ 80, evals);
    ASSERT_EQ_INT(rc, 0);
    printf("# N=6 Heisenberg ring: E_0 = %.10f (Bethe -2.802776)\n", evals[0]);
    ASSERT_NEAR(evals[0], -2.802775637732, 1e-8);
    ASSERT_TRUE(evals[1] >= evals[0] - 1e-12);
    ASSERT_TRUE(evals[2] >= evals[1] - 1e-12);
}

/* Kagome 2×2 (Γ, A₁) sector ED.  Full Hilbert space at N = 12 is
 * 2^12 = 4096; the (Γ, A₁) sector is sector_dim = 30 — a 137×
 * reduction.  At N = 27 (L = 3) the analogous reduction is from
 * 2^27 ≈ 1.3·10^8 to a sector dimension on the order of 10^5,
 * which is the regime where the kagome AFM physics actually opens.
 *
 * On the homegrown full-Hilbert-space spectrum (per
 * CHANGELOG.md §0.4.2), the lowest two eigenvalues on this cluster
 * are E_0 = −5.44487522 (global GS) and E_1 = −5.32839240 (first
 * excited).  The global GS lives in a DIFFERENT irrep (likely (Γ, A₂)
 * or finite-momentum) — it is NOT in (Γ, A₁) for kagome 2×2 PBC —
 * so the (Γ, A₁) sector ED's lowest is the second eigenvalue in the
 * homegrown spectrum, E_1 = −5.32839240.  This is the right cross-
 * validation: the sector-restricted lowest must equal the lowest
 * (Γ, A₁) eigenvalue identified by the homegrown solver, not the
 * global minimum.  Match to 1e-6 (Lanczos convergence floor). */
static void test_kagome_2x2_gamma_a1_matches_homegrown_E1(void) {
    double evals[2] = { 0.0, 0.0 };
    int rc = nqs_kspace_ed_kagome_at_gamma(/*L*/ 2, /*J*/ 1.0,
                                            /*k_wanted*/ 2,
                                            /*max_iters*/ 80, evals);
    ASSERT_EQ_INT(rc, 0);
    printf("# kagome L=2 (Γ, A₁) lowest = %.10f "
           "(homegrown spectrum: E_0 = −5.44487522 in another irrep, "
           "E_1 = −5.32839240 in (Γ, A₁))\n", evals[0]);
    ASSERT_NEAR(evals[0], -5.32839240, 1e-6);
    ASSERT_TRUE(evals[1] >= evals[0] - 1e-12);
}

/* Scan all 1D irreps at Γ on kagome 2×2 PBC.  The global ground state
 * E_0 = -5.44487522 from the homegrown full-Hilbert-space spectrum
 * must surface as the minimum across {A_1, A_2, B_1, B_2}.  The irrep
 * containing it is the symmetry of the kagome AFM ground state on
 * this cluster — physically informative for the spin-liquid
 * classification protocol. */
static void test_kagome_2x2_gs_lives_in_some_1d_irrep_at_gamma(void) {
    double global_e0 = +1e300, global_e1 = +1e300;
    nqs_kspace_irrep_t gs_irrep = NQS_KSPACE_IRREP_A1;
    double per_irrep[4] = { +1e300, +1e300, +1e300, +1e300 };
    int rc = nqs_kspace_ed_kagome_scan_gamma_1d_irreps(/*L*/ 2, /*J*/ 1.0,
                                                       /*max_iters*/ 80,
                                                       &global_e0, &global_e1,
                                                       &gs_irrep, per_irrep);
    ASSERT_EQ_INT(rc, 0);
    static const char *names[4] = { "A_1", "A_2", "B_1", "B_2" };
    printf("# kagome 2×2 scan at Γ:  A_1=%.6f  A_2=%.6f  B_1=%.6f  B_2=%.6f\n",
           per_irrep[0], per_irrep[1], per_irrep[2], per_irrep[3]);
    printf("# global E_0 = %.10f  (homegrown −5.44487522)  in %s\n"
           "# global E_1 = %.10f  gap Δ = %.6f\n",
           global_e0, names[(int)gs_irrep],
           global_e1, global_e1 - global_e0);
    ASSERT_NEAR(global_e0, -5.44487522, 1e-6);
    ASSERT_TRUE(global_e1 >= global_e0 - 1e-12);
}

int main(void) {
    TEST_RUN(test_two_site_heisenberg_matches_analytic_singlet);
    TEST_RUN(test_four_site_ring_matches_bethe_ansatz);
    TEST_RUN(test_six_site_ring_matches_known_reference);
    TEST_RUN(test_kagome_2x2_gamma_a1_matches_homegrown_E1);
    TEST_RUN(test_kagome_2x2_gs_lives_in_some_1d_irrep_at_gamma);
    TEST_SUMMARY();
}

#else /* !SPIN_NN_HAS_IRREP */
static void test_skipped_without_irrep(void) {
    printf("# skipped: built without -DSPIN_NN_HAS_IRREP\n");
}
int main(void) {
    TEST_RUN(test_skipped_without_irrep);
    TEST_SUMMARY();
}
#endif
