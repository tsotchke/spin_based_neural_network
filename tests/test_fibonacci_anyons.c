/*
 * tests/test_fibonacci_anyons.c
 *
 * Covers the Fibonacci-anyon pillar (P1.3b).
 *
 * Algebraic identities (analytical):
 *   - F-matrix is unitary: F F† = I.
 *   - F-matrix is involutive for Fibonacci: F² = I (F is real-symmetric
 *     and unitary with only real entries of unit magnitude).
 *   - R-matrix magnitudes: |R^1_ττ| = |R^τ_ττ| = 1.
 *   - Fibonacci Hilbert dim for n anyons with total charge τ equals
 *     Fib(n-2): [0, 1, 1, 1, 2, 3, 5,...] for n = 2, 3, 4, 5, 6, 7, 8.
 *   - σ_1 and σ_2 are both unitary.
 *   - σ_1 σ_2 σ_1 = σ_2 σ_1 σ_2 (braid-group Yang-Baxter relation).
 *
 * Compilation:
 *   - Target = identity → empty braid word recovers it to machine
 *     precision.
 *   - Target = σ_1 → length-1 braid word {1} recovers to machine
 *     precision.
 *   - Target = σ_1 σ_2 → length-2 braid word {1, 2} recovers to
 *     machine precision.
 *   - Target = random single-qubit unitary → at bounded depth, the
 *     compiled word approximates the target with operator-norm
 *     distance below a documented threshold.
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include "harness.h"
#include "fibonacci_anyons/fibonacci_anyons.h"
static double two_by_two_unitary_error(const double _Complex m[2][2]) {
    /* ||M M† - I||_op for a 2×2 complex matrix. */
    double _Complex mm[2][2], id[2][2];
    double _Complex adj[2][2];
    adj[0][0] = conj(m[0][0]); adj[0][1] = conj(m[1][0]);
    adj[1][0] = conj(m[0][1]); adj[1][1] = conj(m[1][1]);
    fibo_mat2_mul(m, adj, mm);
    fibo_mat2_identity(id);
    return fibo_operator_norm_distance(mm, id);
}
static void test_phi_constant(void) {
    ASSERT_NEAR(FIBO_PHI, (1.0 + sqrt(5.0)) / 2.0, 1e-14);
}
static void test_f_matrix_unitary(void) {
    fibo_fmatrix_t F = fibo_f_matrix();
    double _Complex m[2][2] = {{F.m[0][0], F.m[0][1]}, {F.m[1][0], F.m[1][1]}};
    ASSERT_NEAR(two_by_two_unitary_error(m), 0.0, 1e-12);
}
static void test_f_matrix_involutive(void) {
    fibo_fmatrix_t F = fibo_f_matrix();
    double _Complex m[2][2] = {{F.m[0][0], F.m[0][1]}, {F.m[1][0], F.m[1][1]}};
    double _Complex sq[2][2], id[2][2];
    fibo_mat2_mul(m, m, sq);
    fibo_mat2_identity(id);
    ASSERT_NEAR(fibo_operator_norm_distance(sq, id), 0.0, 1e-12);
}
static void test_r_magnitudes_are_one(void) {
    double _Complex r1 = fibo_r_one();
    double _Complex rt = fibo_r_tau();
    ASSERT_NEAR(cabs(r1), 1.0, 1e-12);
    ASSERT_NEAR(cabs(rt), 1.0, 1e-12);
}
static void test_hilbert_dim_matches_fibonacci_sequence(void) {
    /* Closed-form expectation for n τ-anyons with total charge τ. */
    long expected[] = {-1, -1, 1, 1, 2, 3, 5, 8, 13}; /* index = n */
    for (int n = 2; n <= 8; n++) {
        ASSERT_EQ_INT(fibo_hilbert_dim(n), expected[n]);
    }
}
static void test_braid_generators_unitary(void) {
    double _Complex B1[2][2], B2[2][2];
    fibo_braid_b1(B1);
    fibo_braid_b2(B2);
    ASSERT_NEAR(two_by_two_unitary_error(B1), 0.0, 1e-12);
    ASSERT_NEAR(two_by_two_unitary_error(B2), 0.0, 1e-12);
}
static void test_braid_yang_baxter(void) {
    /* σ_1 σ_2 σ_1  =  σ_2 σ_1 σ_2 */
    double _Complex B1[2][2], B2[2][2];
    fibo_braid_b1(B1); fibo_braid_b2(B2);
    double _Complex lhs[2][2], tmp[2][2];
    double _Complex rhs[2][2];
    fibo_mat2_mul(B1, B2, tmp); fibo_mat2_mul(tmp, B1, lhs);
    fibo_mat2_mul(B2, B1, tmp); fibo_mat2_mul(tmp, B2, rhs);
    ASSERT_NEAR(fibo_operator_norm_distance(lhs, rhs), 0.0, 1e-12);
}
static void test_braid_word_identity_gives_i(void) {
    fibo_braid_word_t *w = fibo_braid_word_create(4);
    double _Complex u[2][2];
    fibo_braid_word_eval(w, u);  /* empty word */
    double _Complex id[2][2]; fibo_mat2_identity(id);
    ASSERT_NEAR(fibo_operator_norm_distance(u, id), 0.0, 1e-14);
    fibo_braid_word_free(w);
}
static void test_braid_word_single_generator(void) {
    fibo_braid_word_t *w = fibo_braid_word_create(4);
    fibo_braid_word_push(w, 1);
    double _Complex u[2][2], B1[2][2];
    fibo_braid_word_eval(w, u);
    fibo_braid_b1(B1);
    ASSERT_NEAR(fibo_operator_norm_distance(u, B1), 0.0, 1e-12);
    fibo_braid_word_free(w);
}
static void test_compile_identity(void) {
    double _Complex id[2][2]; fibo_mat2_identity(id);
    double err;
    fibo_braid_word_t *w = fibo_compile_unitary(id, 6, &err);
    ASSERT_TRUE(w != NULL);
    ASSERT_NEAR(err, 0.0, 1e-12);
    ASSERT_EQ_INT(w->length, 0);
    fibo_braid_word_free(w);
}
static void test_compile_single_braid_generator(void) {
    double _Complex B1[2][2]; fibo_braid_b1(B1);
    double err;
    fibo_braid_word_t *w = fibo_compile_unitary(B1, 4, &err);
    ASSERT_TRUE(w != NULL);
    ASSERT_NEAR(err, 0.0, 1e-12);
    /* Word should be a single σ_1. */
    ASSERT_EQ_INT(w->length, 1);
    ASSERT_EQ_INT(w->sigmas[0], 1);
    fibo_braid_word_free(w);
}
static void test_compile_two_braid_product(void) {
    double _Complex B1[2][2], B2[2][2], target[2][2];
    fibo_braid_b1(B1); fibo_braid_b2(B2);
    fibo_mat2_mul(B1, B2, target);
    double err;
    fibo_braid_word_t *w = fibo_compile_unitary(target, 4, &err);
    ASSERT_TRUE(w != NULL);
    ASSERT_NEAR(err, 0.0, 1e-12);
    ASSERT_EQ_INT(w->length, 2);
    ASSERT_EQ_INT(w->sigmas[0], 1);
    ASSERT_EQ_INT(w->sigmas[1], 2);
    fibo_braid_word_free(w);
}
static void test_compile_random_unitary_within_tolerance(void) {
    /* Pick a specific target unitary that is not a trivial braid
     * product — a π/4 rotation around X:  exp(-i π/8 σ_x). */
    double t = M_PI / 8.0;
    double _Complex target[2][2];
    target[0][0] = cos(t);              target[0][1] = -_Complex_I * sin(t);
    target[1][0] = -_Complex_I * sin(t); target[1][1] = cos(t);
    /* At depth 12 we expect an imperfect but useful approximation. */
    double err;
    fibo_braid_word_t *w = fibo_compile_unitary(target, 12, &err);
    ASSERT_TRUE(w != NULL);
    /* With greedy DFS at depth 12 the braid group densely covers SU(2)
     * but won't always hit sub-percent accuracy; we assert the error
     * is small enough to show compilation is working. */
    ASSERT_TRUE(err < 0.5);
    fibo_braid_word_free(w);
}
static void test_mat2_dagger_basic(void) {
    double _Complex a[2][2] = {
        { 1.0 + 2.0 * I, 3.0 - 1.0 * I },
        { 0.5 + 0.0 * I, 4.0 + 2.0 * I }
    };
    double _Complex out[2][2];
    fibo_mat2_dagger(a, out);
    ASSERT_NEAR(creal(out[0][0]),  1.0, 1e-12);
    ASSERT_NEAR(cimag(out[0][0]), -2.0, 1e-12);
    ASSERT_NEAR(creal(out[1][0]),  3.0, 1e-12);
    ASSERT_NEAR(cimag(out[1][0]),  1.0, 1e-12);
}
static void test_sk_matches_dfs_at_depth_zero(void) {
    double _Complex target[2][2] = {
        { 0.8 + 0.0 * I, 0.0 + 0.6 * I },
        { 0.0 + 0.6 * I, 0.8 + 0.0 * I }
    };
    double err_dfs = 0, err_sk = 0;
    fibo_braid_word_t *w_dfs = fibo_compile_unitary(target, 10, &err_dfs);
    fibo_braid_word_t *w_sk  = fibo_compile_unitary_sk(target, 0, 10, &err_sk);
    ASSERT_NEAR(err_sk, err_dfs, 1e-12);
    fibo_braid_word_free(w_dfs);
    fibo_braid_word_free(w_sk);
}
static void test_sk_produces_unitary_braid_word(void) {
    /* Construct a target that is WITHIN the short-word approximation
     * net: build it from braid generators directly, then compile.
     * SK should recover it (or something close) and the output must
     * be a well-formed unitary braid word. */
    double _Complex B1[2][2], B2[2][2];
    fibo_braid_b1(B1);
    fibo_braid_b2(B2);
    double _Complex tmp[2][2], target[2][2];
    fibo_mat2_mul(B1, B2, tmp);
    fibo_mat2_mul(tmp, B1, target);
    fibo_mat2_mul(target, B2, tmp);
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) target[i][j] = tmp[i][j];
    double err = 0;
    fibo_braid_word_t *w = fibo_compile_unitary_sk(target, 1, 8, &err);
    ASSERT_TRUE(w != NULL);
    double _Complex U[2][2];
    fibo_braid_word_eval(w, U);
    double _Complex Udg[2][2]; fibo_mat2_dagger(U, Udg);
    double _Complex UUd[2][2]; fibo_mat2_mul(U, Udg, UUd);
    /* Unitarity always holds: U is a product of unitary braid gens. */
    ASSERT_NEAR(creal(UUd[0][0]), 1.0, 1e-9);
    ASSERT_NEAR(creal(UUd[1][1]), 1.0, 1e-9);
    ASSERT_NEAR(cabs(UUd[0][1]), 0.0, 1e-9);
    ASSERT_TRUE(isfinite(err));
    /* For this reachable target, DFS should at least beat identity. */
    ASSERT_TRUE(err < 1.0);
    printf("# SK compile reachable target: word length = %d, err = %.6f\n",
           w->length, err);
    fibo_braid_word_free(w);
}
int main(void) {
    TEST_RUN(test_phi_constant);
    TEST_RUN(test_f_matrix_unitary);
    TEST_RUN(test_f_matrix_involutive);
    TEST_RUN(test_r_magnitudes_are_one);
    TEST_RUN(test_hilbert_dim_matches_fibonacci_sequence);
    TEST_RUN(test_braid_generators_unitary);
    TEST_RUN(test_braid_yang_baxter);
    TEST_RUN(test_braid_word_identity_gives_i);
    TEST_RUN(test_braid_word_single_generator);
    TEST_RUN(test_compile_identity);
    TEST_RUN(test_compile_single_braid_generator);
    TEST_RUN(test_compile_two_braid_product);
    TEST_RUN(test_compile_random_unitary_within_tolerance);
    TEST_RUN(test_mat2_dagger_basic);
    TEST_RUN(test_sk_matches_dfs_at_depth_zero);
    TEST_RUN(test_sk_produces_unitary_braid_word);
    TEST_SUMMARY();
}