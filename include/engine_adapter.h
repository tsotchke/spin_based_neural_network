/*
 * include/engine_adapter.h
 *
 * Single-point-of-contact bridge between spin_based_neural_network and an
 * external neural-network / tensor / reasoning engine. The adapter is
 * engine-neutral: any engine that exposes refcounted init/shutdown +
 * lattice tensor build can plug in via its own translation unit.
 *
 * Planned backends (pick one, per-release):
 *   - Eshkol-native NN engine (working title "eshkol-transformers",
 *     built on https://github.com/tsotchke/eshkol; v0.6+ target)
 *   - Noesis reasoning engine (in development, not yet publicly
 *     released)
 *
 * v0.4 ships the adapter as a dormant stub: every function returns
 * ENGINE_ADAPTER_EDISABLED unless the tree is compiled with
 * -DSPIN_NN_HAS_ENGINE=1 AND the build supplies an engine-specific
 * object that defines the real init/shutdown/version/flatten hooks.
 */
#ifndef ENGINE_ADAPTER_H
#define ENGINE_ADAPTER_H

#include <stddef.h>

#include "ising_model.h"
#include "kitaev_model.h"
#include "spin_models.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ENGINE_ADAPTER_OK         =  0,
    ENGINE_ADAPTER_EDISABLED  = -1, /* built without SPIN_NN_HAS_ENGINE    */
    ENGINE_ADAPTER_ENOT_READY = -2, /* engine_adapter_init() not called    */
    ENGINE_ADAPTER_ELIB       = -3, /* engine-specific call returned error */
    ENGINE_ADAPTER_EARG       = -4  /* invalid argument                    */
} engine_adapter_status_t;

/*
 * Refcounted init/shutdown. Safe to call multiple times; each successful
 * init must be paired with a shutdown. Returns ENGINE_ADAPTER_OK on success.
 */
int engine_adapter_init(void);
int engine_adapter_shutdown(void);

/* Cheap probe — returns 1 iff the build included an engine implementation. */
int engine_adapter_is_available(void);

/* Engine-reported runtime version string. Static pointer; do not free.
 * Returns NULL when the adapter is disabled. */
const char *engine_adapter_engine_version(void);

/* Build-time-baked version tag, fed via -DSPIN_NN_ENGINE_VERSION='"..."'.
 * Always returns non-NULL (possibly "unknown"). */
const char *engine_adapter_build_version(void);

/*
 * Flatten a 3D spin lattice into a contiguous float32 buffer, row-major
 * (x outer, y middle, z inner). Returns the expected element count if
 * `out == NULL`, or the element count actually written on success.
 * Negative return values are engine_adapter_status_t.
 */
long engine_adapter_flatten_ising (const IsingLattice  *l, float *out, size_t out_capacity);
long engine_adapter_flatten_kitaev(const KitaevLattice *l, float *out, size_t out_capacity);
long engine_adapter_flatten_spin  (const SpinLattice   *l, float *out, size_t out_capacity);

#ifdef __cplusplus
}
#endif

#endif /* ENGINE_ADAPTER_H */
