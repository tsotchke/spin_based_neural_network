/*
 * tests/test_nqs_symproj.c
 *
 * Validates the generic finite-group symmetry-projection wrapper and
 * the kagome-specific permutation builders.
 *
 * Tests:
 *  1. Translation-only permutation: every row is a permutation of
 *     0..N-1, identity is in the orbit, and applying every group
 *     element gives an exact orbit closure.
 *  2. p2 (translation × C₂ inversion): same closure properties, |G|
 *     = 2·Lx·Ly, and C₂² ≡ identity on the sublattice-A subset.
 *  3. End-to-end with a base RBM ansatz: ψ_sym(g·s) = ψ_sym(s) to
 *     numerical precision for every g in the orbit (the projector's
 *     defining property).  Tested for both translation-only and p2.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "nqs/nqs_config.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_symproj.h"

static void test_translation_perm_is_proper(void) {
    int Lx = 2, Ly = 2;
    int N = 3 * Lx * Ly;
    int *perm = NULL;
    int G = 0;
    int rc = nqs_kagome_translation_perm(Lx, Ly, &perm, &G);
    ASSERT_EQ_INT(rc, 0);
    ASSERT_EQ_INT(G, Lx * Ly);
    /* Every row maps {0..N-1} bijectively. */
    int *seen = (int *)calloc((size_t)N, sizeof(int));
    for (int g = 0; g < G; g++) {
        memset(seen, 0, (size_t)N * sizeof(int));
        for (int i = 0; i < N; i++) {
            int j = perm[(size_t)g * N + i];
            ASSERT_TRUE(j >= 0 && j < N);
            ASSERT_TRUE(seen[j] == 0);
            seen[j] = 1;
        }
    }
    /* Identity (tx=0, ty=0) is row 0. */
    for (int i = 0; i < N; i++) ASSERT_EQ_INT(perm[i], i);
    free(seen); free(perm);
}

static void test_p2_perm_doubles_translation(void) {
    int Lx = 2, Ly = 2;
    int N = 3 * Lx * Ly;
    int *perm = NULL;
    double *chars = NULL;
    int G = 0;
    int rc = nqs_kagome_p2_perm(Lx, Ly, &perm, &chars, &G);
    ASSERT_EQ_INT(rc, 0);
    ASSERT_EQ_INT(G, 2 * Lx * Ly);
    /* Every row is a valid permutation. */
    int *seen = (int *)calloc((size_t)N, sizeof(int));
    for (int g = 0; g < G; g++) {
        memset(seen, 0, (size_t)N * sizeof(int));
        for (int i = 0; i < N; i++) {
            int j = perm[(size_t)g * N + i];
            ASSERT_TRUE(j >= 0 && j < N);
            ASSERT_TRUE(seen[j] == 0);
            seen[j] = 1;
        }
        ASSERT_NEAR(chars[g], 1.0, 1e-12);
    }
    /* Identity is row 0. */
    for (int i = 0; i < N; i++) ASSERT_EQ_INT(perm[i], i);
    /* C₂ at (tx=ty=0) is row Lx*Ly: applying it twice must give identity
     * on every sublattice-A site (sub=0).  Sub-A sites are indices
     * 3*(cx*Ly + cy) + 0. */
    int row_inv = Lx * Ly;
    for (int cx = 0; cx < Lx; cx++) {
        for (int cy = 0; cy < Ly; cy++) {
            int i = 3 * (cx * Ly + cy);  /* A site */
            int j = perm[(size_t)row_inv * N + i];
            int k = perm[(size_t)row_inv * N + j];
            ASSERT_EQ_INT(k, i);
        }
    }
    free(seen); free(perm); free(chars);
}

/* End-to-end: a real RBM base ansatz, with the symproj wrapper
 * applied, must satisfy ψ_sym(g·s) = ψ_sym(s) for every g. */
static void test_symproj_output_is_invariant_translation(void) {
    int Lx = 2, Ly = 2;
    int N = 3 * Lx * Ly;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_RBM;
    cfg.rbm_hidden_units = 4;
    cfg.rng_seed = 0xA5A5A5A5u;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    ASSERT_TRUE(a != NULL);

    int *perm = NULL;
    int G = 0;
    ASSERT_EQ_INT(nqs_kagome_translation_perm(Lx, Ly, &perm, &G), 0);
    /* All-trivial characters for the trivial irrep (k = 0). */
    double *chars = (double *)malloc((size_t)G * sizeof(double));
    for (int g = 0; g < G; g++) chars[g] = 1.0;

    nqs_symproj_wrapper_t w = {
        .base_log_amp       = nqs_ansatz_log_amp,
        .base_user          = a,
        .num_sites          = N,
        .num_group_elements = G,
        .perm               = perm,
        .characters         = chars,
    };

    /* Pick an arbitrary configuration. */
    int spins[12] = { +1, -1, +1, -1, +1, -1, +1, +1, -1, -1, +1, -1 };
    double base_lp, base_arg;
    nqs_symproj_log_amp(spins, N, &w, &base_lp, &base_arg);

    /* For every group element, apply the permutation and re-evaluate. */
    int *transformed = (int *)malloc((size_t)N * sizeof(int));
    for (int g = 0; g < G; g++) {
        for (int i = 0; i < N; i++) transformed[i] = spins[perm[(size_t)g * N + i]];
        double lp, arg;
        nqs_symproj_log_amp(transformed, N, &w, &lp, &arg);
        /* For the trivial irrep, ψ_sym must be invariant. */
        ASSERT_NEAR(lp, base_lp, 1e-10);
        ASSERT_NEAR(cos(arg), cos(base_arg), 1e-10);
    }

    free(transformed); free(perm); free(chars);
    nqs_ansatz_free(a);
}

static void test_symproj_output_is_invariant_p2(void) {
    int Lx = 2, Ly = 2;
    int N = 3 * Lx * Ly;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_RBM;
    cfg.rbm_hidden_units = 4;
    cfg.rng_seed = 0x5A5A5A5Au;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    ASSERT_TRUE(a != NULL);

    int *perm = NULL;
    double *chars = NULL;
    int G = 0;
    ASSERT_EQ_INT(nqs_kagome_p2_perm(Lx, Ly, &perm, &chars, &G), 0);

    nqs_symproj_wrapper_t w = {
        .base_log_amp       = nqs_ansatz_log_amp,
        .base_user          = a,
        .num_sites          = N,
        .num_group_elements = G,
        .perm               = perm,
        .characters         = chars,
    };

    int spins[12] = { +1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, -1 };
    double base_lp, base_arg;
    nqs_symproj_log_amp(spins, N, &w, &base_lp, &base_arg);

    int *transformed = (int *)malloc((size_t)N * sizeof(int));
    for (int g = 0; g < G; g++) {
        for (int i = 0; i < N; i++) transformed[i] = spins[perm[(size_t)g * N + i]];
        double lp, arg;
        nqs_symproj_log_amp(transformed, N, &w, &lp, &arg);
        ASSERT_NEAR(lp, base_lp, 1e-10);
        ASSERT_NEAR(cos(arg), cos(base_arg), 1e-10);
    }

    free(transformed); free(perm); free(chars);
    nqs_ansatz_free(a);
}

/* p3 (translations × C₃) — 3·L² orbit on an L × L kagome torus. */
static void test_p3_perm_is_proper(void) {
    int L = 3;
    int N = 3 * L * L;
    int *perm = NULL;
    double *chars = NULL;
    int G = 0;
    int rc = nqs_kagome_p3_perm(L, &perm, &chars, &G);
    ASSERT_EQ_INT(rc, 0);
    ASSERT_EQ_INT(G, 3 * L * L);
    int *seen = (int *)calloc((size_t)N, sizeof(int));
    for (int g = 0; g < G; g++) {
        memset(seen, 0, (size_t)N * sizeof(int));
        for (int i = 0; i < N; i++) {
            int j = perm[(size_t)g * N + i];
            ASSERT_TRUE(j >= 0 && j < N);
            ASSERT_TRUE(seen[j] == 0);
            seen[j] = 1;
        }
        ASSERT_NEAR(chars[g], 1.0, 1e-12);
    }
    /* Identity (k = 0, tx = ty = 0) is row 0. */
    for (int i = 0; i < N; i++) ASSERT_EQ_INT(perm[i], i);

    /* C₃³ = identity on every site.  Row L² has (k=1, tx=0, ty=0).
     * Apply C₃ three times via the perm composition g·(g·(g·s))_i. */
    int row_C3 = L * L;
    for (int i = 0; i < N; i++) {
        int j = perm[(size_t)row_C3 * N + i];
        int k = perm[(size_t)row_C3 * N + j];
        int l = perm[(size_t)row_C3 * N + k];
        ASSERT_EQ_INT(l, i);
    }
    free(seen); free(perm); free(chars);
}

static void test_symproj_output_is_invariant_p3(void) {
    int L = 3;
    int N = 3 * L * L;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_RBM;
    cfg.rbm_hidden_units = 6;
    cfg.rng_seed = 0xCAFEFACEu;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    ASSERT_TRUE(a != NULL);

    int *perm = NULL;
    double *chars = NULL;
    int G = 0;
    ASSERT_EQ_INT(nqs_kagome_p3_perm(L, &perm, &chars, &G), 0);

    nqs_symproj_wrapper_t w = {
        .base_log_amp       = nqs_ansatz_log_amp,
        .base_user          = a,
        .num_sites          = N,
        .num_group_elements = G,
        .perm               = perm,
        .characters         = chars,
    };

    int *spins = (int *)malloc((size_t)N * sizeof(int));
    for (int i = 0; i < N; i++) spins[i] = (i % 3 == 0) ? +1 : -1;

    double base_lp, base_arg;
    nqs_symproj_log_amp(spins, N, &w, &base_lp, &base_arg);

    int *transformed = (int *)malloc((size_t)N * sizeof(int));
    for (int g = 0; g < G; g++) {
        for (int i = 0; i < N; i++) transformed[i] = spins[perm[(size_t)g * N + i]];
        double lp, arg;
        nqs_symproj_log_amp(transformed, N, &w, &lp, &arg);
        ASSERT_NEAR(lp, base_lp, 1e-9);
        ASSERT_NEAR(cos(arg), cos(base_arg), 1e-9);
    }

    free(spins); free(transformed); free(perm); free(chars);
    nqs_ansatz_free(a);
}

int main(void) {
    TEST_RUN(test_translation_perm_is_proper);
    TEST_RUN(test_p2_perm_doubles_translation);
    TEST_RUN(test_symproj_output_is_invariant_translation);
    TEST_RUN(test_symproj_output_is_invariant_p2);
    TEST_RUN(test_p3_perm_is_proper);
    TEST_RUN(test_symproj_output_is_invariant_p3);
    TEST_SUMMARY();
}
