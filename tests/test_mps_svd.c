/*
 * tests/test_mps_svd.c
 *
 * Correctness of the Jacobi SVD used by DMRG sweeps.
 *   - Reconstructs A = U Σ V^T to machine precision.
 *   - Recovers known singular values on simple diagonal matrices.
 *   - Produces an orthonormal U and a matrix-orthogonal V^T.
 *   - Handles rank-deficient inputs gracefully (some s[i] = 0).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "mps/svd.h"
static void mat_mul_diag_vt(const double *U, const double *s, const double *Vt,
                            int m, int n, double *out) {
    /* out[i,j] = Σ_k U[i,k] · s[k] · Vt[k,j]     shape m × n */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double v = 0.0;
            for (int k = 0; k < n; k++) v += U[i*n + k] * s[k] * Vt[k*n + j];
            out[i*n + j] = v;
        }
    }
}
static void test_reconstruction_identity(void) {
    int m = 4, n = 4;
    double A[16];
    for (int i = 0; i < 16; i++) A[i] = (i + 1) * ((i & 1) ? -1.0 : 1.0);
    double U[16], s[4], Vt[16];
    ASSERT_EQ_INT(svd_jacobi(A, m, n, U, s, Vt, 1e-14), 0);
    double R[16];
    mat_mul_diag_vt(U, s, Vt, m, n, R);
    for (int i = 0; i < 16; i++) ASSERT_NEAR(R[i], A[i], 1e-10);
}
static void test_diagonal_singular_values(void) {
    /* A = diag(5, 3, 1): singular values are 5, 3, 1 in descending
     * order. */
    double A[9] = { 5, 0, 0,
                    0, 3, 0,
                    0, 0, 1 };
    double U[9], s[3], Vt[9];
    ASSERT_EQ_INT(svd_jacobi(A, 3, 3, U, s, Vt, 1e-14), 0);
    ASSERT_NEAR(s[0], 5.0, 1e-10);
    ASSERT_NEAR(s[1], 3.0, 1e-10);
    ASSERT_NEAR(s[2], 1.0, 1e-10);
}
static void test_u_columns_orthonormal(void) {
    int m = 6, n = 4;
    double A[24];
    for (int i = 0; i < 24; i++) A[i] = sin(0.3 * i) + 0.1 * i;
    double U[24], s[4], Vt[16];
    ASSERT_EQ_INT(svd_jacobi(A, m, n, U, s, Vt, 1e-14), 0);
    for (int p = 0; p < n; p++) {
        for (int q = p; q < n; q++) {
            double dot = 0.0;
            for (int i = 0; i < m; i++) dot += U[i*n + p] * U[i*n + q];
            double want = (p == q) ? 1.0 : 0.0;
            ASSERT_NEAR(dot, want, 1e-10);
        }
    }
}
static void test_vt_rows_orthonormal(void) {
    int m = 5, n = 4;
    double A[20];
    for (int i = 0; i < 20; i++) A[i] = cos(0.2 * i) + 0.05 * i * i;
    double U[20], s[4], Vt[16];
    svd_jacobi(A, m, n, U, s, Vt, 1e-14);
    for (int p = 0; p < n; p++) {
        for (int q = p; q < n; q++) {
            double dot = 0.0;
            for (int k = 0; k < n; k++) dot += Vt[p*n + k] * Vt[q*n + k];
            double want = (p == q) ? 1.0 : 0.0;
            ASSERT_NEAR(dot, want, 1e-10);
        }
    }
}
static void test_reconstruction_random(void) {
    int m = 8, n = 5;
    double A[40];
    unsigned long long rng = 0xABCDEF;
    for (int i = 0; i < m*n; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        A[i] = ((double)(rng >> 11) / 9007199254740992.0) - 0.5;
    }
    double U[40], s[5], Vt[25];
    svd_jacobi(A, m, n, U, s, Vt, 1e-14);
    double R[40];
    mat_mul_diag_vt(U, s, Vt, m, n, R);
    for (int i = 0; i < m*n; i++) ASSERT_NEAR(R[i], A[i], 1e-9);
}
static void test_rank_deficient(void) {
    /* A rank-1 matrix: all rows are scalar multiples of the first. */
    int m = 4, n = 3;
    double A[12];
    double row0[3] = {1.0, 2.0, -1.5};
    double mult[4] = {1.0, -0.5, 2.0, 3.0};
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) A[i*n + j] = mult[i] * row0[j];
    double U[12], s[3], Vt[9];
    svd_jacobi(A, m, n, U, s, Vt, 1e-14);
    /* Only s[0] should be non-zero; s[1], s[2] ≈ 0. */
    ASSERT_TRUE(s[0] > 1.0);
    ASSERT_NEAR(s[1], 0.0, 1e-9);
    ASSERT_NEAR(s[2], 0.0, 1e-9);
    /* And the reconstruction still matches. */
    double R[12];
    mat_mul_diag_vt(U, s, Vt, m, n, R);
    for (int i = 0; i < 12; i++) ASSERT_NEAR(R[i], A[i], 1e-9);
}
int main(void) {
    TEST_RUN(test_reconstruction_identity);
    TEST_RUN(test_diagonal_singular_values);
    TEST_RUN(test_u_columns_orthonormal);
    TEST_RUN(test_vt_rows_orthonormal);
    TEST_RUN(test_reconstruction_random);
    TEST_RUN(test_rank_deficient);
    TEST_SUMMARY();
}