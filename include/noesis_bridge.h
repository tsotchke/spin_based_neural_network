/*
 * include/noesis_bridge.h
 *
 * Lazy bridge to Noesis, the neuro-symbolic cognitive architecture
 * built on Eshkol (~/Desktop/noesis, github.com/tsotchke/noesis).
 * Primary use case in this framework: a *learned-classifier open
 * policy* for the THQCP coupling scheduler, where noesis's
 * proof-tree + calibrated-uncertainty machinery decides whether a
 * quantum window should open at a given anneal step. This is the
 * concrete realisation of Patent Claim 4 of the THQCP provisional.
 *
 * Gated behind -DSPIN_NN_HAS_NOESIS=1. When built without it, every
 * entry point returns NOESIS_BRIDGE_EDISABLED and the caller is
 * expected to fall back to the deterministic policies
 * (PERIODIC / STAGNATION / NEVER).
 *
 * Unlike the other bridges, the Noesis bridge does not produce a
 * numeric result directly — it returns a *decision* plus a
 * *proof-trace handle* that the caller can dereference for audit /
 * training-signal purposes. Initially the handle is opaque; in a
 * future revision it exposes the proof DAG so the THQCP scheduler
 * can be trained end-to-end with noesis on the same Eshkol AD tape.
 */
#ifndef NOESIS_BRIDGE_H
#define NOESIS_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NOESIS_BRIDGE_OK         =  0,
    NOESIS_BRIDGE_EDISABLED  = -1,
    NOESIS_BRIDGE_ENOT_READY = -2,
    NOESIS_BRIDGE_ELIB       = -3,
    NOESIS_BRIDGE_EARG       = -4
} noesis_bridge_status_t;

/* --- availability + versioning ------------------------------------- */

int         noesis_bridge_is_available(void);
const char *noesis_bridge_version(void);    /* NULL when disabled */

/* --- lifecycle ----------------------------------------------------- */

/* Refcounted init/shutdown. When built with SPIN_NN_HAS_NOESIS, the
 * first init call spins up the Noesis cognitive workspace and returns
 * NOESIS_BRIDGE_OK. Subsequent calls increment a refcount; shutdown
 * decrements, with final shutdown tearing down the workspace. */
int noesis_bridge_init(void);
int noesis_bridge_shutdown(void);

/* --- trajectory snapshot (input to the open-policy decision) -------- */

/* Features the THQCP coupling scheduler exposes to the noesis
 * classifier. Simple numeric vector — noesis handles symbolic /
 * probabilistic reasoning internally. */
typedef struct {
    int    sweep_index;             /* current classical sweep number      */
    double beta_current;            /* β(t) at this sweep                  */
    double energy_current;          /* H_p(s) at this sweep                */
    double energy_best_so_far;      /* min H_p(s) seen on this trajectory  */
    int    stagnation_count;        /* consecutive sweeps with no flip     */
    int    windows_opened_so_far;   /* running count of PHASE_QUANTUM      */
    int    feedbacks_applied;       /* running count of PHASE_FEEDBACK     */
    double last_window_outcome;     /* ±1 if applicable, 0 if no window yet */
} noesis_trajectory_snapshot_t;

/* --- open-policy decision ------------------------------------------ */

/* Asks the noesis classifier whether a quantum window should open at
 * the current trajectory state. When enabled, consults a trained
 * neural-symbolic classifier with calibrated uncertainty and returns
 * 1 if the window should open, 0 if not. When disabled, sets
 * *out_decision = 0 and returns NOESIS_BRIDGE_EDISABLED. */
int noesis_bridge_should_open_window(const noesis_trajectory_snapshot_t *snap,
                                      int *out_decision,
                                      double *out_confidence);

/* --- proof-trace audit (future revision) ---------------------------- */

/* Opaque handle to the proof DAG explaining the last decision. */
typedef struct noesis_proof_trace noesis_proof_trace_t;

/* Fetch the proof-trace for the most recent decision. Caller owns the
 * handle and must free it. Returns NULL if no decision has been made
 * or if noesis is disabled. */
noesis_proof_trace_t *noesis_bridge_last_proof_trace(void);
void                  noesis_proof_trace_free(noesis_proof_trace_t *t);

#ifdef __cplusplus
}
#endif

#endif /* NOESIS_BRIDGE_H */
