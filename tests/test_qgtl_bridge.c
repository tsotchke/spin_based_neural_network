/*
 * tests/test_qgtl_bridge.c
 *
 * Dormant-path tests for the QGTL bridge. Live-path tests follow
 * when QGTL 1.0 stabilises.
 */
#include "harness.h"
#include "qgtl_bridge.h"
#ifdef SPIN_NN_HAS_QGTL
static void test_live_placeholder(void) {
    ASSERT_EQ_INT(qgtl_bridge_is_available(), 1);
}
int main(void) {
    TEST_RUN(test_live_placeholder);
    TEST_SUMMARY();
}
#else
static void test_disabled_reports_unavailable(void) {
    ASSERT_EQ_INT(qgtl_bridge_is_available(), 0);
    ASSERT_TRUE(qgtl_bridge_version() == NULL);
}
static void test_disabled_lifecycle(void) {
    ASSERT_EQ_INT(qgtl_bridge_init(), QGTL_BRIDGE_EDISABLED);
    ASSERT_EQ_INT(qgtl_bridge_shutdown(), QGTL_BRIDGE_EDISABLED);
}
static void test_disabled_compute_qgt(void) {
    double grads[8] = { 0 };
    double G[4] = { 0 };
    ASSERT_EQ_INT(qgtl_bridge_compute_qgt(grads, 2, 2, G), QGTL_BRIDGE_EDISABLED);
    ASSERT_EQ_INT(qgtl_bridge_compute_qgt(NULL, 2, 2, G), QGTL_BRIDGE_EARG);
    ASSERT_EQ_INT(qgtl_bridge_compute_qgt(grads, 0, 2, G), QGTL_BRIDGE_EARG);
}
static void test_disabled_device_backend(void) {
    ASSERT_TRUE(qgtl_bridge_device_backend_open("IBM") == NULL);
    qgtl_bridge_device_backend_close(NULL);   /* no-op, safe */
    double p = 0;
    ASSERT_EQ_INT(qgtl_bridge_device_single_qubit_p(NULL, &p),
                  QGTL_BRIDGE_EDISABLED);
    ASSERT_TRUE(p == -1.0);
    ASSERT_EQ_INT(qgtl_bridge_device_single_qubit_p(NULL, NULL),
                  QGTL_BRIDGE_EARG);
}
int main(void) {
    TEST_RUN(test_disabled_reports_unavailable);
    TEST_RUN(test_disabled_lifecycle);
    TEST_RUN(test_disabled_compute_qgt);
    TEST_RUN(test_disabled_device_backend);
    TEST_SUMMARY();
}
#endif