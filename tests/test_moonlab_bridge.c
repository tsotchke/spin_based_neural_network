/*
 * tests/test_moonlab_bridge.c
 *
 * Bridge into ~/Desktop/quantum_simulator (libquantumsim, "Moonlab").
 * Default build exercises the dormant path; moonlab-enabled build
 * runs a distance-3 surface-code round trip through moonlab's MWPM
 * decoder as ground truth for the joint-decoder program.
 */
#include <math.h>
#include "harness.h"
#include "moonlab_bridge.h"
#ifdef SPIN_NN_HAS_MOONLAB
static void test_live_bridge_reports_available(void) {
    ASSERT_EQ_INT(moonlab_bridge_is_available(), 1);
    const char *v = moonlab_bridge_version();
    ASSERT_TRUE(v != NULL);
    ASSERT_TRUE(v[0] != '\0');
}
static void test_live_surface_code_roundtrip_returns_ok(void) {
    int le = -1;
    int rc = moonlab_bridge_surface_code_roundtrip(3, 0.01, &le);
    ASSERT_EQ_INT(rc, MOONLAB_BRIDGE_OK);
    /* le is 0 or 1 — both are legal outcomes of a single round. */
    ASSERT_TRUE(le == 0 || le == 1);
}
static void test_live_logical_error_rate_monotone_in_p(void) {
    /* Higher physical error rate → higher residual-syndrome rate at
     * fixed distance. 200 trials is modest but enough to resolve the
     * monotonic trend. */
    double pL_lo = 0.0, pL_hi = 0.0;
    ASSERT_EQ_INT(moonlab_bridge_surface_code_logical_error_rate(3, 0.005, 200, 1u, &pL_lo),
                  MOONLAB_BRIDGE_OK);
    ASSERT_EQ_INT(moonlab_bridge_surface_code_logical_error_rate(3, 0.10,  200, 2u, &pL_hi),
                  MOONLAB_BRIDGE_OK);
    ASSERT_TRUE(isfinite(pL_lo));
    ASSERT_TRUE(isfinite(pL_hi));
    ASSERT_TRUE(pL_hi >= pL_lo);
}
static void test_live_rejects_invalid_distance(void) {
    int le = 0;
    int rc = moonlab_bridge_surface_code_roundtrip(2, 0.01, &le);   /* even */
    ASSERT_EQ_INT(rc, MOONLAB_BRIDGE_EARG);
    rc = moonlab_bridge_surface_code_roundtrip(1, 0.01, &le);        /* < 3 */
    ASSERT_EQ_INT(rc, MOONLAB_BRIDGE_EARG);
}
int main(void) {
    TEST_RUN(test_live_bridge_reports_available);
    TEST_RUN(test_live_surface_code_roundtrip_returns_ok);
    TEST_RUN(test_live_logical_error_rate_monotone_in_p);
    TEST_RUN(test_live_rejects_invalid_distance);
    TEST_SUMMARY();
}
#else
static void test_disabled_reports_unavailable(void) {
    ASSERT_EQ_INT(moonlab_bridge_is_available(), 0);
    ASSERT_TRUE(moonlab_bridge_version() == NULL);
}
static void test_disabled_roundtrip_returns_edisabled(void) {
    int le = 0;
    int rc = moonlab_bridge_surface_code_roundtrip(3, 0.01, &le);
    ASSERT_EQ_INT(rc, MOONLAB_BRIDGE_EDISABLED);
    ASSERT_EQ_INT(le, -1);
}
static void test_disabled_logical_rate_returns_edisabled(void) {
    double pL = 0.0;
    int rc = moonlab_bridge_surface_code_logical_error_rate(3, 0.01, 10, 1u, &pL);
    ASSERT_EQ_INT(rc, MOONLAB_BRIDGE_EDISABLED);
}
static void test_disabled_null_args_return_earg(void) {
    ASSERT_EQ_INT(moonlab_bridge_surface_code_roundtrip(3, 0.01, NULL),
                  MOONLAB_BRIDGE_EARG);
    ASSERT_EQ_INT(moonlab_bridge_surface_code_logical_error_rate(3, 0.01, 10, 1u, NULL),
                  MOONLAB_BRIDGE_EARG);
}
int main(void) {
    TEST_RUN(test_disabled_reports_unavailable);
    TEST_RUN(test_disabled_roundtrip_returns_edisabled);
    TEST_RUN(test_disabled_logical_rate_returns_edisabled);
    TEST_RUN(test_disabled_null_args_return_earg);
    TEST_SUMMARY();
}
#endif