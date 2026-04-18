/*
 * tests/test_eshkol_bridge.c
 *
 * Covers the lazy Eshkol-FFI bridge in its disabled mode (v0.4 default).
 * With Eshkol absent, every entry point must fail with a clean status
 * code and leave the bridge in a consistent state.
 */
#include "harness.h"
#include "eshkol_bridge.h"

static void test_is_available_matches_build_flag(void) {
    /* v0.4 builds without SPIN_NN_HAS_ESHKOL, so the bridge reports
     * unavailable. If a future build defines it, this test will need
     * updating. */
    ASSERT_EQ_INT(eshkol_bridge_is_available(), 0);
}

static void test_disabled_init_shutdown_errors(void) {
    ASSERT_EQ_INT(eshkol_bridge_init(),     ESHKOL_BRIDGE_EDISABLED);
    ASSERT_EQ_INT(eshkol_bridge_shutdown(), ESHKOL_BRIDGE_EDISABLED);
}

static void test_disabled_load_script_errors(void) {
    ASSERT_EQ_INT(eshkol_bridge_load_script("nonexistent.esk"),
                  ESHKOL_BRIDGE_EDISABLED);
}

static void test_disabled_eval_double_errors(void) {
    double result = 42.0;
    ASSERT_EQ_INT(eshkol_bridge_eval_double("(+ 1 2)", &result),
                  ESHKOL_BRIDGE_EDISABLED);
}

static void test_bad_args_return_earg(void) {
    double result;
    ASSERT_EQ_INT(eshkol_bridge_load_script(NULL), ESHKOL_BRIDGE_EARG);
    ASSERT_EQ_INT(eshkol_bridge_eval_double(NULL, &result), ESHKOL_BRIDGE_EARG);
    ASSERT_EQ_INT(eshkol_bridge_eval_double("(+ 1 2)", NULL), ESHKOL_BRIDGE_EARG);
}

int main(void) {
    TEST_RUN(test_is_available_matches_build_flag);
    TEST_RUN(test_disabled_init_shutdown_errors);
    TEST_RUN(test_disabled_load_script_errors);
    TEST_RUN(test_disabled_eval_double_errors);
    TEST_RUN(test_bad_args_return_earg);
    TEST_SUMMARY();
}
