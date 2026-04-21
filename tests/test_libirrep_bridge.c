/*
 * tests/test_libirrep_bridge.c
 *
 * Covers the libirrep bridge in both modes. Under the default build
 * (no -DSPIN_NN_HAS_IRREP), the bridge reports unavailable and every
 * computational entry point returns IRREP_BRIDGE_EDISABLED. Under
 * `make IRREP_ENABLE=1 IRREP_ROOT=...`, the bridge forwards to the
 * real library and should return OK with numerically-correct values
 * for a handful of known identities.
 */
#include <math.h>
#include "harness.h"
#include "libirrep_bridge.h"
#ifdef SPIN_NN_HAS_IRREP
/* ---- live-path tests (library linked in) ----------------------- */
static void test_live_is_available(void) {
    ASSERT_EQ_INT(libirrep_bridge_is_available(), 1);
    const char *v = libirrep_bridge_version();
    ASSERT_TRUE(v != NULL);
    ASSERT_TRUE(v[0] != '\0');
}
static void test_live_refcounted_lifecycle(void) {
    ASSERT_EQ_INT(libirrep_bridge_init(),     IRREP_BRIDGE_OK);
    ASSERT_EQ_INT(libirrep_bridge_init(),     IRREP_BRIDGE_OK); /* refcount */
    ASSERT_EQ_INT(libirrep_bridge_shutdown(), IRREP_BRIDGE_OK);
    ASSERT_EQ_INT(libirrep_bridge_shutdown(), IRREP_BRIDGE_OK);
}
static void test_live_sph_harm_Y00_is_constant(void) {
    /* Y_0^0 = 1 / (2 sqrt(π)) ≈ 0.28209479 everywhere. */
    double y = 0.0;
    ASSERT_EQ_INT(libirrep_bridge_sph_harm_real(0, 0, 1.23, 0.45, &y),
                  IRREP_BRIDGE_OK);
    ASSERT_NEAR(y, 0.5 / sqrt(M_PI), 1e-10);
}
static void test_live_cg_unit_element(void) {
    /* <0 0; 0 0 | 0 0> = 1 */
    double c = 0.0;
    ASSERT_EQ_INT(libirrep_bridge_clebsch_gordan(0, 0, 0, 0, 0, 0, &c),
                  IRREP_BRIDGE_OK);
    ASSERT_NEAR(c, 1.0, 1e-12);
}
static void test_live_wigner_d_identity_at_zero(void) {
    /* d^j_{m'm}(0) = δ_{m'm}. */
    double d_same = 0.0, d_diff = 0.0;
    ASSERT_EQ_INT(libirrep_bridge_wigner_d_small(1,  0,  0, 0.0, &d_same),
                  IRREP_BRIDGE_OK);
    ASSERT_EQ_INT(libirrep_bridge_wigner_d_small(1,  1, -1, 0.0, &d_diff),
                  IRREP_BRIDGE_OK);
    ASSERT_NEAR(d_same, 1.0, 1e-12);
    ASSERT_NEAR(d_diff, 0.0, 1e-12);
}
static void test_live_null_args_still_earg(void) {
    ASSERT_EQ_INT(libirrep_bridge_sph_harm_real(0, 0, 0, 0, NULL),
                  IRREP_BRIDGE_EARG);
    ASSERT_EQ_INT(libirrep_bridge_clebsch_gordan(0, 0, 0, 0, 0, 0, NULL),
                  IRREP_BRIDGE_EARG);
    ASSERT_EQ_INT(libirrep_bridge_wigner_d_small(0, 0, 0, 0, NULL),
                  IRREP_BRIDGE_EARG);
}
int main(void) {
    TEST_RUN(test_live_is_available);
    TEST_RUN(test_live_refcounted_lifecycle);
    TEST_RUN(test_live_sph_harm_Y00_is_constant);
    TEST_RUN(test_live_cg_unit_element);
    TEST_RUN(test_live_wigner_d_identity_at_zero);
    TEST_RUN(test_live_null_args_still_earg);
    TEST_SUMMARY();
}
#else /* !SPIN_NN_HAS_IRREP — the default v0.4 build */
/* ---- disabled-path tests (default) ----------------------------- */
static void test_disabled_mode_flags(void) {
    ASSERT_EQ_INT(libirrep_bridge_is_available(), 0);
    ASSERT_EQ_INT(libirrep_bridge_init(),     IRREP_BRIDGE_EDISABLED);
    ASSERT_EQ_INT(libirrep_bridge_shutdown(), IRREP_BRIDGE_EDISABLED);
    ASSERT_TRUE(libirrep_bridge_version() == NULL);
}
static void test_sph_harm_disabled_returns_edisabled(void) {
    double y;
    ASSERT_EQ_INT(libirrep_bridge_sph_harm_real(2, 1, 1.0, 0.5, &y),
                  IRREP_BRIDGE_EDISABLED);
}
static void test_cg_disabled_returns_edisabled(void) {
    double c;
    ASSERT_EQ_INT(libirrep_bridge_clebsch_gordan(1, 0, 1, 0, 2, 0, &c),
                  IRREP_BRIDGE_EDISABLED);
}
static void test_wigner_d_disabled_returns_edisabled(void) {
    double d;
    ASSERT_EQ_INT(libirrep_bridge_wigner_d_small(1, 0, 0, 1.0, &d),
                  IRREP_BRIDGE_EDISABLED);
}
static void test_null_args_return_earg(void) {
    ASSERT_EQ_INT(libirrep_bridge_sph_harm_real(0, 0, 0, 0, NULL),
                  IRREP_BRIDGE_EARG);
    ASSERT_EQ_INT(libirrep_bridge_clebsch_gordan(0, 0, 0, 0, 0, 0, NULL),
                  IRREP_BRIDGE_EARG);
    ASSERT_EQ_INT(libirrep_bridge_wigner_d_small(0, 0, 0, 0, NULL),
                  IRREP_BRIDGE_EARG);
}
int main(void) {
    TEST_RUN(test_disabled_mode_flags);
    TEST_RUN(test_sph_harm_disabled_returns_edisabled);
    TEST_RUN(test_cg_disabled_returns_edisabled);
    TEST_RUN(test_wigner_d_disabled_returns_edisabled);
    TEST_RUN(test_null_args_return_earg);
    TEST_SUMMARY();
}
#endif /* SPIN_NN_HAS_IRREP */