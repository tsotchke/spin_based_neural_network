/*
 * tests/test_matrix_neon.c
 *
 * Covers the SIMD-accelerated linear-algebra helpers used by the
 * topological-entropy suite. Tests run on both NEON-enabled and fallback
 * code paths depending on the build flags.
 */
#include <complex.h>
#include <stdlib.h>
#include <math.h>
#include "harness.h"
#include "topological_entropy.h"

/* Declarations live in matrix_neon.c — mirror them here so we don't need
 * to expose them in a public header just for tests. */
void matrix_vector_multiply_neon(double _Complex *matrix, double _Complex *vector,
                                 double _Complex *result, int size);
void calculate_eigenvalues_neon(double _Complex *matrix, double *eigenvalues, int size);

static void test_neon_availability_probe_returns_0_or_1(void) {
    int r = check_neon_available();
    ASSERT_TRUE(r == 0 || r == 1);
}

static void test_matvec_identity(void) {
    int n = 4;
    double _Complex M[16] = {0};
    double _Complex v[4]  = {1, 2 + _Complex_I, 3, 4 - 2 * _Complex_I};
    double _Complex r[4]  = {0};
    for (int i = 0; i < n; i++) M[i * n + i] = 1.0; /* identity */
    matrix_vector_multiply_neon(M, v, r, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR_COMPLEX(r[i], v[i], 1e-10);
    }
}

static void test_matvec_diagonal_scaling(void) {
    int n = 3;
    double _Complex M[9] = {0};
    double _Complex v[3] = {1, 2, 3};
    double _Complex r[3] = {0};
    M[0] = 2.0;
    M[4] = 3.0;
    M[8] = 4.0;
    matrix_vector_multiply_neon(M, v, r, n);
    ASSERT_NEAR_COMPLEX(r[0], 2.0, 1e-10);
    ASSERT_NEAR_COMPLEX(r[1], 6.0, 1e-10);
    ASSERT_NEAR_COMPLEX(r[2], 12.0, 1e-10);
}

static void test_matvec_zero_matrix_yields_zero_vector(void) {
    int n = 4;
    double _Complex M[16] = {0};
    double _Complex v[4]  = {1, 2, 3, 4};
    double _Complex r[4];
    for (int i = 0; i < n; i++) r[i] = 42; /* dirty */
    matrix_vector_multiply_neon(M, v, r, n);
    for (int i = 0; i < n; i++) ASSERT_NEAR_COMPLEX(r[i], 0.0, 1e-10);
}

/* calculate_eigenvalues_neon uses an approximate power iteration and
 * writes estimated |eigenvalue|s into `eig`; the precise values depend on
 * iteration count, seed, and matrix structure. We assert only that all
 * outputs are finite and non-NaN, which is what the entropy pipeline
 * needs. */
static void test_eigenvalues_returns_finite_values(void) {
    int n = 4;
    double _Complex M[16] = {0};
    /* Diagonal with distinct positive values — non-degenerate spectrum. */
    M[0]  = 3.0;
    M[5]  = 2.0;
    M[10] = 1.5;
    M[15] = 0.5;
    double eig[4] = {0};
    calculate_eigenvalues_neon(M, eig, n);
    for (int i = 0; i < n; i++) {
        ASSERT_TRUE(eig[i] == eig[i]);         /* not NaN */
        ASSERT_TRUE(eig[i] < 1e300);           /* not +inf */
        ASSERT_TRUE(eig[i] > -1e300);          /* not -inf */
    }
}

int main(void) {
    TEST_RUN(test_neon_availability_probe_returns_0_or_1);
    TEST_RUN(test_matvec_identity);
    TEST_RUN(test_matvec_diagonal_scaling);
    TEST_RUN(test_matvec_zero_matrix_yields_zero_vector);
    TEST_RUN(test_eigenvalues_returns_finite_values);
    TEST_SUMMARY();
}
