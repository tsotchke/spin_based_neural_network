/*
 * src/noesis_bridge.c
 *
 * Dormant wrapper around Noesis (github.com/tsotchke/noesis). When
 * built without SPIN_NN_HAS_NOESIS, every entry point returns
 * NOESIS_BRIDGE_EDISABLED. Live implementation (forthcoming): the
 * bridge spins up a Noesis cognitive workspace via the Eshkol FFI,
 * evaluates a trained proof-trace classifier on the trajectory
 * snapshot, and returns the decision + confidence + retained proof
 * DAG. See eshkol_bridge.c for the Eshkol-FFI idiom this builds on.
 */
#include <stdio.h>
#include <stdlib.h>
#include "noesis_bridge.h"

#ifdef SPIN_NN_HAS_NOESIS
/* Real bridge intentionally gated — enable when Eshkol + Noesis link
 * paths are wired through. The stub below stays as the production
 * fallback. */
static int g_refcount = 0;
#endif

int noesis_bridge_is_available(void) {
#ifdef SPIN_NN_HAS_NOESIS
    return 1;
#else
    return 0;
#endif
}

const char *noesis_bridge_version(void) {
#ifdef SPIN_NN_HAS_NOESIS
    return "1.0.0";   /* pinned at the Noesis release linked against */
#else
    return NULL;
#endif
}

int noesis_bridge_init(void) {
#ifdef SPIN_NN_HAS_NOESIS
    g_refcount++;
    return NOESIS_BRIDGE_OK;
#else
    return NOESIS_BRIDGE_EDISABLED;
#endif
}

int noesis_bridge_shutdown(void) {
#ifdef SPIN_NN_HAS_NOESIS
    if (g_refcount <= 0) return NOESIS_BRIDGE_ENOT_READY;
    g_refcount--;
    return NOESIS_BRIDGE_OK;
#else
    return NOESIS_BRIDGE_EDISABLED;
#endif
}

int noesis_bridge_should_open_window(const noesis_trajectory_snapshot_t *snap,
                                      int *out_decision,
                                      double *out_confidence) {
    if (!snap || !out_decision) return NOESIS_BRIDGE_EARG;
    *out_decision = 0;
    if (out_confidence) *out_confidence = 0.0;
#ifdef SPIN_NN_HAS_NOESIS
    /* Live bridge: call into noesis's proof-trace classifier. Not
     * implemented in this commit — returns the deterministic fallback
     * until the Eshkol-FFI wiring lands. */
    return NOESIS_BRIDGE_EDISABLED;
#else
    return NOESIS_BRIDGE_EDISABLED;
#endif
}

noesis_proof_trace_t *noesis_bridge_last_proof_trace(void) {
    return NULL;
}

void noesis_proof_trace_free(noesis_proof_trace_t *t) {
    (void)t;
}
