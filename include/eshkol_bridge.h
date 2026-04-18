/*
 * include/eshkol_bridge.h
 *
 * Thin bridge from spin_based_neural_network to the Eshkol Scheme runtime.
 *
 * The bridge is *lazy*: eshkol_bridge_init() is only called when a training
 * driver actually needs Eshkol (v0.5 pillars). v0.4 foundation code compiles
 * without Eshkol present; calls that require the runtime fail with
 * ESHKOL_BRIDGE_EDISABLED.
 *
 * The runtime is process-wide and single-threaded (Eshkol's own constraint);
 * the bridge refcounts init/shutdown across callers.
 */
#ifndef ESHKOL_BRIDGE_H
#define ESHKOL_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ESHKOL_BRIDGE_OK           =  0,
    ESHKOL_BRIDGE_EDISABLED    = -1, /* built without SPIN_NN_HAS_ESHKOL */
    ESHKOL_BRIDGE_ENOT_READY   = -2, /* eshkol_bridge_init() not called  */
    ESHKOL_BRIDGE_ELIB         = -3, /* eshkol call returned error       */
    ESHKOL_BRIDGE_EARG         = -4  /* invalid argument                 */
} eshkol_bridge_status_t;

/* Refcounted init/shutdown. Safe to call multiple times from the same thread. */
int eshkol_bridge_init(void);
int eshkol_bridge_shutdown(void);

/* Whether the bridge has a usable Eshkol runtime behind it. */
int eshkol_bridge_is_available(void);

/* Load and evaluate a .esk file under the current context. Returns
 * ESHKOL_BRIDGE_OK on success. The file typically registers C kernels via
 * (extern ...) declarations and sets up the training tape. */
int eshkol_bridge_load_script(const char *path);

/* Evaluate a short Eshkol expression and return a numeric result.
 * Intended for smoke tests and simple probes — real training code should
 * use script files. */
int eshkol_bridge_eval_double(const char *source, double *result);

/* Last error message reported by the bridge or by Eshkol. NULL if none. */
const char *eshkol_bridge_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_BRIDGE_H */
