/*
 * include/moonlab_bridge.h
 *
 * Lazy bridge to the Moonlab quantum simulator (libquantumsim).
 * Moonlab supplies a reference dense state-vector + surface-code +
 * Fibonacci-anyon stack that this framework cross-validates its own
 * topological-QC pieces against — the ground-truth anchor for the
 * learned-decoder / joint-training program (research-plan
 * §neuromorphic-QEC).
 *
 * Gated behind -DSPIN_NN_HAS_MOONLAB=1 so the default build remains
 * dependency-free. Enable with
 *     make MOONLAB_ENABLE=1 MOONLAB_ROOT=/path/to/quantum_simulator
 */
#ifndef MOONLAB_BRIDGE_H
#define MOONLAB_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MOONLAB_BRIDGE_OK         =  0,
    MOONLAB_BRIDGE_EDISABLED  = -1,
    MOONLAB_BRIDGE_ENOT_READY = -2,
    MOONLAB_BRIDGE_ELIB       = -3,
    MOONLAB_BRIDGE_EARG       = -4
} moonlab_bridge_status_t;

/* 1 iff the bridge was built with -DSPIN_NN_HAS_MOONLAB. */
int moonlab_bridge_is_available(void);

/* Build-time moonlab version string, or NULL if disabled. */
const char *moonlab_bridge_version(void);

/* ---- surface-code round-trip ---------------------------------------
 * Run a single noise + MWPM-decode round on a Moonlab distance-d
 * surface code with depolarizing physical error rate p. Returns 1 if
 * decoding succeeded (no residual logical error), 0 otherwise. On
 * EDISABLED or library failure, writes -1 to *out_logical_error. */
int moonlab_bridge_surface_code_roundtrip(int distance, double p,
                                          int *out_logical_error);

/* ---- batched logical error rate ------------------------------------
 * Empirical p_L from `num_trials` Monte-Carlo rounds at distance d,
 * physical error rate p_phys. The result is the canonical ground-
 * truth reference curve against which learned decoders are scored. */
int moonlab_bridge_surface_code_logical_error_rate(int distance,
                                                    double p_phys,
                                                    int num_trials,
                                                    unsigned rng_seed,
                                                    double *out_p_logical);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_BRIDGE_H */
