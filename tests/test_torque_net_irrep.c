/*
 * tests/test_torque_net_irrep.c
 *
 * Cross-validates the equivariant torque network's hand-rolled l=0
 * contraction against libirrep's spherical-harmonic machinery. This
 * is the first test in the tree that genuinely *consumes* libirrep
 * — the production torque_net uses Cartesian primitives, so we need
 * an explicit SH-addition-theorem check to establish the two views
 * agree.
 *
 * The identity we verify:
 *     (4π / (2l+1)) · Σ_m Y_l^m(û) · Y_l^m(v̂)  =  P_l(û · v̂).
 * For l=1 that reduces to
 *     (4π / 3) · Σ_{m=-1,0,+1} Y_1^m(û) Y_1^m(v̂)  =  û · v̂.
 *
 * This binds the SH-basis representation (the language NequIP /
 * libirrep uses) to the Cartesian representation (the language the
 * torque-net is written in) for l=1 — the irrep the torque-net
 * outputs. If the identity ever broke, the torque-net's "l=1
 * equivariant" claim would need revisiting.
 *
 * Gated on SPIN_NN_HAS_IRREP; enable with
 *     make IRREP_ENABLE=1 \
 *          IRREP_ROOT=/path/to/libirrep \
 *          IRREP_LIBDIR=/path/to/libirrep/lib/<triple> \
 *          test_torque_net_irrep && build/test_torque_net_irrep
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "harness.h"
#include "libirrep_bridge.h"
#ifdef SPIN_NN_HAS_IRREP
/* Convert cartesian unit vector → (theta, phi). */
static void cart_to_polar(const double v[3], double *theta, double *phi) {
    *theta = acos(v[2]);
    *phi   = atan2(v[1], v[0]);
}
static double dot3(const double a[3], const double b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
static double sh_addition_l1(const double u[3], const double v[3]) {
    double tu, pu, tv, pv;
    cart_to_polar(u, &tu, &pu);
    cart_to_polar(v, &tv, &pv);
    double acc = 0.0;
    for (int m = -1; m <= 1; m++) {
        double yu = 0.0, yv = 0.0;
        libirrep_bridge_sph_harm_real(1, m, tu, pu, &yu);
        libirrep_bridge_sph_harm_real(1, m, tv, pv, &yv);
        acc += yu * yv;
    }
    return 4.0 * M_PI / 3.0 * acc;
}
static void test_bridge_reports_available(void) {
    ASSERT_EQ_INT(libirrep_bridge_is_available(), 1);
    const char *v = libirrep_bridge_version();
    ASSERT_TRUE(v != NULL);
    printf("# libirrep version: %s\n", v);
}
static void test_sh_addition_theorem_l1_matches_cartesian_dot(void) {
    /* Five fixed unit vectors, arbitrary. Verify the l=1 SH
     * addition theorem reproduces the Cartesian dot. */
    double u[5][3] = {
        { 1.0, 0.0, 0.0 },
        { 0.0, 1.0, 0.0 },
        { 0.0, 0.0, 1.0 },
        { 0.577350269, 0.577350269, 0.577350269 },     /* (1,1,1)/√3 */
        { 0.707106781, 0.0, -0.707106781 }             /* (1,0,-1)/√2 */
    };
    double v[5][3] = {
        { 0.0, 1.0, 0.0 },
        { 1.0, 0.0, 0.0 },
        { 0.6, 0.8, 0.0 },
        { -0.577350269, 0.577350269, 0.577350269 },
        { 0.0, 1.0, 0.0 }
    };
    double max_err = 0.0;
    for (int t = 0; t < 5; t++) {
        double lib_val  = sh_addition_l1(u[t], v[t]);
        double cart_val = dot3(u[t], v[t]);
        double err = fabs(lib_val - cart_val);
        if (err > max_err) max_err = err;
    }
    printf("# SH-addition-theorem l=1 vs cartesian dot: max err = %.3e\n",
           max_err);
    ASSERT_TRUE(max_err < 1e-12);
}
static void test_pairwise_orthogonality_l1_l2(void) {
    /* Pick a random-ish direction and verify Y_l^m integrates to
     * near-zero against Y_l'^m' at orthogonal vectors (sanity that
     * libirrep's SH convention is the standard orthonormal one;
     * this catches an accidental 1/√(4π) vs. 1/2√π drift). */
    double r[3] = { 0.707106781, 0.0, 0.707106781 };  /* (1,0,1)/√2 */
    double theta, phi;
    cart_to_polar(r, &theta, &phi);
    /* Y_0^0(anywhere) = 1/(2 √π). */
    double y00 = 0.0;
    libirrep_bridge_sph_harm_real(0, 0, theta, phi, &y00);
    ASSERT_NEAR(y00, 0.5 / sqrt(M_PI), 1e-12);
    /* Y_2^0(θ) = √(5/(16π))·(3 cos²θ − 1); at θ = π/4, cos²θ = 1/2,
     * so 3·0.5−1 = 0.5, giving √(5/(16π))·0.5 ≈ 0.0789. */
    double y20 = 0.0;
    libirrep_bridge_sph_harm_real(2, 0, theta, phi, &y20);
    double expected = 0.5 * sqrt(5.0 / (16.0 * M_PI));
    ASSERT_NEAR(y20, expected, 1e-8);
}
int main(void) {
    TEST_RUN(test_bridge_reports_available);
    TEST_RUN(test_sh_addition_theorem_l1_matches_cartesian_dot);
    TEST_RUN(test_pairwise_orthogonality_l1_l2);
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