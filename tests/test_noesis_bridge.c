/*
 * tests/test_noesis_bridge.c
 *
 * Dormant-path tests for the Noesis bridge. Live-path tests follow
 * once the Eshkol-FFI wiring for the proof-trace classifier lands.
 */
#include "harness.h"
#include "noesis_bridge.h"
#ifdef SPIN_NN_HAS_NOESIS
/* Live-path test stubs — populated in a follow-up commit alongside
 * the Noesis workspace init + Eshkol-FFI call. */
static void test_live_bridge_placeholder(void) {
    /* Placeholder until the workspace is wired. */
    ASSERT_EQ_INT(noesis_bridge_is_available(), 1);
}
int main(void) {
    TEST_RUN(test_live_bridge_placeholder);
    TEST_SUMMARY();
}
#else /* !SPIN_NN_HAS_NOESIS */
static void test_disabled_reports_unavailable(void) {
    ASSERT_EQ_INT(noesis_bridge_is_available(), 0);
    ASSERT_TRUE(noesis_bridge_version() == NULL);
}
static void test_disabled_lifecycle(void) {
    ASSERT_EQ_INT(noesis_bridge_init(), NOESIS_BRIDGE_EDISABLED);
    ASSERT_EQ_INT(noesis_bridge_shutdown(), NOESIS_BRIDGE_EDISABLED);
}
static void test_disabled_open_decision_returns_zero(void) {
    noesis_trajectory_snapshot_t snap = {
.sweep_index = 50,
.beta_current = 2.0,
.energy_current = -12.5,
.energy_best_so_far = -12.5,
.stagnation_count = 5,
.windows_opened_so_far = 0,
.feedbacks_applied = 0,
.last_window_outcome = 0.0
    };
    int decision = 999;
    double confidence = 999.0;
    int rc = noesis_bridge_should_open_window(&snap, &decision, &confidence);
    ASSERT_EQ_INT(rc, NOESIS_BRIDGE_EDISABLED);
    ASSERT_EQ_INT(decision, 0);                      /* safe fallback */
    ASSERT_TRUE(confidence == 0.0);
}
static void test_null_args_return_earg(void) {
    int d = 0;
    double c = 0;
    ASSERT_EQ_INT(noesis_bridge_should_open_window(NULL, &d, &c),
                  NOESIS_BRIDGE_EARG);
    noesis_trajectory_snapshot_t snap = {0};
    ASSERT_EQ_INT(noesis_bridge_should_open_window(&snap, NULL, &c),
                  NOESIS_BRIDGE_EARG);
}
static void test_proof_trace_disabled(void) {
    ASSERT_TRUE(noesis_bridge_last_proof_trace() == NULL);
    noesis_proof_trace_free(NULL);                   /* no-op, safe */
}
int main(void) {
    TEST_RUN(test_disabled_reports_unavailable);
    TEST_RUN(test_disabled_lifecycle);
    TEST_RUN(test_disabled_open_decision_returns_zero);
    TEST_RUN(test_null_args_return_earg);
    TEST_RUN(test_proof_trace_disabled);
    TEST_SUMMARY();
}
#endif