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

int main(void) {
    TEST_RUN(test_two_site_heisenberg_matches_analytic_singlet);
    TEST_RUN(test_four_site_ring_matches_bethe_ansatz);
    TEST_RUN(test_six_site_ring_matches_known_reference);
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
