/*
 * tests/test_qllm_bridge.c
 *
 * Dormant-path tests for the semiclassical_qllm bridge.
 */
#include "harness.h"
#include "qllm_bridge.h"
#ifdef SPIN_NN_HAS_QLLM
static void test_live_placeholder(void) {
    ASSERT_EQ_INT(qllm_bridge_is_available(), 1);
}
int main(void) { TEST_RUN(test_live_placeholder); TEST_SUMMARY(); }
#else
static void test_disabled_reports_unavailable(void) {
    ASSERT_EQ_INT(qllm_bridge_is_available(), 0);
    ASSERT_TRUE(qllm_bridge_version() == NULL);
}
static void test_disabled_lifecycle(void) {
    ASSERT_EQ_INT(qllm_bridge_init(), QLLM_BRIDGE_EDISABLED);
    ASSERT_EQ_INT(qllm_bridge_shutdown(), QLLM_BRIDGE_EDISABLED);
}
static void test_disabled_model_load(void) {
    qllm_model_t *m = (qllm_model_t *)0xDEADBEEF;
    ASSERT_EQ_INT(qllm_bridge_model_load("foo.safetensors", &m),
                  QLLM_BRIDGE_EDISABLED);
    ASSERT_TRUE(m == NULL);
    ASSERT_EQ_INT(qllm_bridge_model_load(NULL, &m), QLLM_BRIDGE_EARG);
    ASSERT_EQ_INT(qllm_bridge_model_load("foo", NULL), QLLM_BRIDGE_EARG);
}
static void test_disabled_decode(void) {
    double in[4] = {0.1, 0.2, 0.3, 0.4};
    double out[4] = {0};
    ASSERT_EQ_INT(qllm_bridge_decode_syndrome(NULL, in, 1, out),
                  QLLM_BRIDGE_EDISABLED);
    ASSERT_EQ_INT(qllm_bridge_decode_syndrome(NULL, NULL, 1, out),
                  QLLM_BRIDGE_EARG);
}
static void test_disabled_nqs_logpsi(void) {
    int spins[4] = {+1, -1, +1, -1};
    double lp = 999.0, ph = 999.0;
    ASSERT_EQ_INT(qllm_bridge_nqs_logpsi(NULL, spins, 4, &lp, &ph),
                  QLLM_BRIDGE_EDISABLED);
    ASSERT_TRUE(lp == 0.0 && ph == 0.0);
}
int main(void) {
    TEST_RUN(test_disabled_reports_unavailable);
    TEST_RUN(test_disabled_lifecycle);
    TEST_RUN(test_disabled_model_load);
    TEST_RUN(test_disabled_decode);
    TEST_RUN(test_disabled_nqs_logpsi);
    TEST_SUMMARY();
}
#endif