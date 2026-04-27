/*
 * tests/test_libirrep_rdm.c
 *
 * Validates the libirrep-bridge partial-trace / von-Neumann /
 * Renyi-entropy wrappers against analytic 2-qubit references.
 *
 * Singlet:  |ψ⟩ = (|01⟩ − |10⟩) / √2
 *   ρ_A = Tr_B |ψ⟩⟨ψ| = ½ I            (maximally mixed)
 *   S   = log 2 ≈ 0.69314718
 *   S_2 = log 2 (Renyi α=2 of a maximally-mixed 2×2 ρ)
 *
 * Bell |Φ+⟩ = (|00⟩ + |11⟩) / √2 — same ρ_A, same entropies.
 *
 * Product state |00⟩:
 *   ρ_A = diag(1, 0)                   (pure)
 *   S   = 0
 */
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "harness.h"
#include "libirrep_bridge.h"

#ifdef SPIN_NN_HAS_IRREP

static void test_singlet_partial_trace_is_maximally_mixed(void) {
    /* Basis: i = 0..3 → site_0 digit + 2 · site_1 digit.  So
     * |00⟩=0, |10⟩=1, |01⟩=2, |11⟩=3.  Singlet = (|01⟩ − |10⟩) / √2. */
    double inv_sqrt2 = 1.0 / sqrt(2.0);
    double _Complex psi[4] = { 0.0, -inv_sqrt2, +inv_sqrt2, 0.0 };
    int sites_A[1] = { 0 };
    double _Complex rho_A[4] = { 0 };
    int rc = libirrep_bridge_partial_trace_spin_half(/*num_sites*/ 2, psi,
                                                      sites_A, /*nA*/ 1,
                                                      rho_A);
    ASSERT_EQ_INT(rc, 0);
    /* ρ_A should be ½ I.  Diagonal entries 0.5, off-diagonals 0. */
    ASSERT_NEAR(creal(rho_A[0]), 0.5, 1e-12);
    ASSERT_NEAR(creal(rho_A[3]), 0.5, 1e-12);
    ASSERT_NEAR(cabs(rho_A[1]), 0.0, 1e-12);
    ASSERT_NEAR(cabs(rho_A[2]), 0.0, 1e-12);

    double S = 0.0;
    rc = libirrep_bridge_entropy_vonneumann(rho_A, /*n*/ 2, &S);
    ASSERT_EQ_INT(rc, 0);
    printf("# singlet vN entropy: %.10f (analytic log 2 = %.10f)\n",
           S, log(2.0));
    ASSERT_NEAR(S, log(2.0), 1e-10);

    double S2 = 0.0;
    rc = libirrep_bridge_entropy_renyi(rho_A, 2, /*alpha*/ 2.0, &S2);
    ASSERT_EQ_INT(rc, 0);
    /* Renyi-2 of ½ I (2×2): S_2 = -log(Tr ρ²) = -log(2 · 0.25) = log 2. */
    ASSERT_NEAR(S2, log(2.0), 1e-10);
}

static void test_product_state_partial_trace_is_pure(void) {
    /* |00⟩ — psi[0] = 1, others 0. */
    double _Complex psi[4] = { 1.0, 0.0, 0.0, 0.0 };
    int sites_A[1] = { 0 };
    double _Complex rho_A[4] = { 0 };
    int rc = libirrep_bridge_partial_trace_spin_half(2, psi, sites_A, 1, rho_A);
    ASSERT_EQ_INT(rc, 0);
    /* ρ_A = |0⟩⟨0| = diag(1, 0). */
    ASSERT_NEAR(creal(rho_A[0]), 1.0, 1e-12);
    ASSERT_NEAR(cabs(rho_A[3]), 0.0, 1e-12);

    double S = 0.0;
    libirrep_bridge_entropy_vonneumann(rho_A, 2, &S);
    printf("# product-state vN entropy: %.3e (analytic 0)\n", S);
    ASSERT_TRUE(fabs(S) < 1e-10);
}

/* Singlet projection: |ψ⟩ = (|01⟩ − |10⟩)/√2 is already a J=0 state,
 * so projection onto J=0 leaves it invariant (up to numerical noise).
 * The projected norm² should be 1.  Triplet input (|00⟩) has zero
 * J=0 component; projection should leave a vanishing-norm output. */
static void test_singlet_input_survives_J_eq_0_projection(void) {
    double inv_sqrt2 = 1.0 / sqrt(2.0);
    double _Complex psi[4]     = { 0.0, -inv_sqrt2, +inv_sqrt2, 0.0 };
    double _Complex psi_out[4] = { 0 };
    int rc = libirrep_bridge_spin_project_singlet(/*N*/ 2,
                                                    /*nα*/ 4,
                                                    /*nβ*/ 4,
                                                    /*nγ*/ 4,
                                                    psi, psi_out);
    ASSERT_EQ_INT(rc, 0);
    /* Output should be the singlet state (up to overall complex factor)
     * with norm² ≈ 1. */
    double norm2 = 0.0;
    for (int i = 0; i < 4; i++) norm2 += creal(psi_out[i]) * creal(psi_out[i])
                                       + cimag(psi_out[i]) * cimag(psi_out[i]);
    printf("# J=0 projection on singlet: ||P|ψ⟩||² = %.6f\n", norm2);
    ASSERT_TRUE(norm2 > 0.999 && norm2 < 1.001);
}

static void test_triplet_input_vanishes_under_J_eq_0_projection(void) {
    double _Complex psi[4]     = { 1.0, 0.0, 0.0, 0.0 };  /* |00⟩ — pure triplet */
    double _Complex psi_out[4] = { 0 };
    int rc = libirrep_bridge_spin_project_singlet(2, 6, 6, 6, psi, psi_out);
    ASSERT_EQ_INT(rc, 0);
    double norm2 = 0.0;
    for (int i = 0; i < 4; i++) norm2 += creal(psi_out[i]) * creal(psi_out[i])
                                       + cimag(psi_out[i]) * cimag(psi_out[i]);
    printf("# J=0 projection on triplet |00⟩: ||P|ψ⟩||² = %.3e (expect 0)\n",
           norm2);
    ASSERT_TRUE(norm2 < 1e-10);
}

int main(void) {
    libirrep_bridge_init();
    TEST_RUN(test_singlet_partial_trace_is_maximally_mixed);
    TEST_RUN(test_product_state_partial_trace_is_pure);
    TEST_RUN(test_singlet_input_survives_J_eq_0_projection);
    TEST_RUN(test_triplet_input_vanishes_under_J_eq_0_projection);
    libirrep_bridge_shutdown();
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
