/*
 * tests/test_llg.c
 *
 * Covers the Landau-Lifshitz-Gilbert integrator.
 *
 * Physical sanity:
 *   - A single spin in a static field B precesses around B with
 *     frequency ω = γ|B|, preserving |m| = 1.
 *   - With Gilbert damping α > 0, the spin spirals in and aligns
 *     with B: m → B̂ at long times.
 *   - Zero field gives zero dynamics.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "llg/llg.h"
/* Constant field along z. */
typedef struct { double Bx, By, Bz; } const_b_t;
static void const_field(const double *m, double *b, long n, void *u) {
    (void)m;
    const_b_t *cb = (const_b_t *)u;
    for (long i = 0; i < n; i++) {
        b[3*i  ] = cb->Bx;
        b[3*i+1] = cb->By;
        b[3*i+2] = cb->Bz;
    }
}
static void test_cross_product(void) {
    double a[3] = {1, 0, 0};
    double b[3] = {0, 1, 0};
    double c[3];
    llg_cross3(a, b, c);
    ASSERT_NEAR(c[0], 0.0, 1e-12);
    ASSERT_NEAR(c[1], 0.0, 1e-12);
    ASSERT_NEAR(c[2], 1.0, 1e-12);
}
static void test_renormalize_restores_unit_norm(void) {
    double m[6] = {3.0, 4.0, 0.0, 0.0, 0.0, 5.0};
    llg_renormalize(m, 2);
    double n0 = sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);
    double n1 = sqrt(m[3]*m[3] + m[4]*m[4] + m[5]*m[5]);
    ASSERT_NEAR(n0, 1.0, 1e-12);
    ASSERT_NEAR(n1, 1.0, 1e-12);
}
static void test_zero_field_preserves_m(void) {
    const_b_t cb = {0, 0, 0};
    llg_config_t cfg = llg_config_defaults();
    cfg.dt = 1e-12;
    cfg.field_fn = const_field;
    cfg.field_user_data = &cb;
    double m[3] = {0.6, 0.8, 0.0};
    for (int i = 0; i < 100; i++) llg_rk4_step(&cfg, m, 1);
    ASSERT_NEAR(m[0], 0.6, 1e-8);
    ASSERT_NEAR(m[1], 0.8, 1e-8);
    ASSERT_NEAR(m[2], 0.0, 1e-8);
}
static void test_precession_preserves_norm(void) {
    /* Spin starts in +x, field along +z, no damping.
     * |m| should stay 1 throughout. */
    const_b_t cb = {0, 0, 1.0};
    llg_config_t cfg = llg_config_defaults();
    cfg.gamma = 1.0;
    cfg.alpha = 0.0;
    cfg.dt = 1e-3;
    cfg.field_fn = const_field;
    cfg.field_user_data = &cb;
    double m[3] = {1.0, 0.0, 0.0};
    for (int i = 0; i < 500; i++) {
        llg_rk4_step(&cfg, m, 1);
        double n = sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);
        ASSERT_NEAR(n, 1.0, 1e-6);
    }
}
static void test_larmor_frequency(void) {
    /* ṁ = -γ (m × B) with m(0) = (1,0,0) and B = (0,0,1), γ = 1.
     * Analytic solution: m(t) = (cos t, sin t, 0). At t = π/2 the
     * spin lies on +y; at t = π it lies on -x; at t = 2π it returns
     * to the start. A trivially-passing test (e.g. m is unchanged)
     * would fail this because the start/end y must rotate through
     * ±1 while m_z stays zero. */
    const_b_t cb = {0.0, 0.0, 1.0};
    llg_config_t cfg = llg_config_defaults();
    cfg.gamma = 1.0;
    cfg.alpha = 0.0;
    cfg.dt    = 1e-4;
    cfg.field_fn        = const_field;
    cfg.field_user_data = &cb;
    double m[3] = {1.0, 0.0, 0.0};
    /* Advance to t = π/2 (≈ 15708 steps of dt = 1e-4). */
    int steps_q = (int)(1.5707963267948966 / cfg.dt);
    for (int i = 0; i < steps_q; i++) llg_rk4_step(&cfg, m, 1);
    ASSERT_NEAR(m[0], 0.0, 1e-3);
    ASSERT_NEAR(m[1], 1.0, 1e-3);
    ASSERT_NEAR(m[2], 0.0, 1e-3);
    /* Advance to t = π. */
    for (int i = 0; i < steps_q; i++) llg_rk4_step(&cfg, m, 1);
    ASSERT_NEAR(m[0], -1.0, 1e-3);
    ASSERT_NEAR(m[1],  0.0, 1e-3);
    ASSERT_NEAR(m[2],  0.0, 1e-3);
}
static void test_larmor_frequency_scales_with_gamma(void) {
    /* Doubling γ doubles the precession rate: under γ = 2 the spin
     * reaches -x after t = π/2 instead of π. */
    const_b_t cb = {0.0, 0.0, 1.0};
    llg_config_t cfg = llg_config_defaults();
    cfg.gamma = 2.0;
    cfg.alpha = 0.0;
    cfg.dt    = 1e-4;
    cfg.field_fn        = const_field;
    cfg.field_user_data = &cb;
    double m[3] = {1.0, 0.0, 0.0};
    int steps_q = (int)(1.5707963267948966 / cfg.dt);
    for (int i = 0; i < steps_q; i++) llg_rk4_step(&cfg, m, 1);
    ASSERT_NEAR(m[0], -1.0, 2e-3);
    ASSERT_NEAR(m[1],  0.0, 2e-3);
}
static void test_damping_aligns_with_field(void) {
    /* Spin starts in +x, field along +z with Gilbert damping α = 0.2.
     * Long time: m should align with field direction. */
    const_b_t cb = {0, 0, 1.0};
    llg_config_t cfg = llg_config_defaults();
    cfg.gamma = 1.0;
    cfg.alpha = 0.2;
    cfg.dt = 0.05;
    cfg.field_fn = const_field;
    cfg.field_user_data = &cb;
    double m[3] = {1.0, 0.0, 0.0};
    for (int i = 0; i < 2000; i++) llg_rk4_step(&cfg, m, 1);
    ASSERT_NEAR(m[2], 1.0, 5e-2);
}
static void test_heun_and_rk4_agree_for_short_times(void) {
    const_b_t cb = {0.0, 1.0, 0.0};
    llg_config_t cfg_rk = llg_config_defaults();
    cfg_rk.gamma = 1.0;
    cfg_rk.alpha = 0.0;
    cfg_rk.dt = 1e-3;
    cfg_rk.field_fn = const_field;
    cfg_rk.field_user_data = &cb;
    double m1[3] = {1.0, 0.0, 0.0};
    double m2[3] = {1.0, 0.0, 0.0};
    for (int i = 0; i < 100; i++) llg_rk4_step(&cfg_rk, m1, 1);
    for (int i = 0; i < 100; i++) llg_heun_step(&cfg_rk, m2, 1);
    /* After 100 steps of dt=1e-3 both should be very close. */
    ASSERT_NEAR(m1[0], m2[0], 1e-3);
    ASSERT_NEAR(m1[1], m2[1], 1e-3);
    ASSERT_NEAR(m1[2], m2[2], 1e-3);
}
int main(void) {
    TEST_RUN(test_cross_product);
    TEST_RUN(test_renormalize_restores_unit_norm);
    TEST_RUN(test_zero_field_preserves_m);
    TEST_RUN(test_precession_preserves_norm);
    TEST_RUN(test_larmor_frequency);
    TEST_RUN(test_larmor_frequency_scales_with_gamma);
    TEST_RUN(test_damping_aligns_with_field);
    TEST_RUN(test_heun_and_rk4_agree_for_short_times);
    TEST_SUMMARY();
}