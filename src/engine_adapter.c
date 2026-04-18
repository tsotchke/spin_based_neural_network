/*
 * src/engine_adapter.c
 *
 * Engine-neutral bridge to an external NN / tensor / reasoning engine.
 *
 * v0.4 scope: dormant scaffolding. The flatteners are live and have unit
 * tests; init/shutdown/version return "disabled" unless SPIN_NN_HAS_ENGINE
 * is defined and an engine-specific object is linked in. The engine-specific
 * hooks (weak symbols below) default to no-ops; a linked implementation
 * overrides them.
 *
 * When v0.5+ picks an engine (an Eshkol-native NN engine is the
 * primary planned target; Noesis is a candidate once released), the
 * engine bundles a .c file that:
 *   - #includes its own headers
 *   - defines engine_backend_init(), engine_backend_shutdown(),
 *     engine_backend_version() as strong symbols that replace the stubs
 */
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "engine_adapter.h"

#ifndef SPIN_NN_ENGINE_VERSION
#define SPIN_NN_ENGINE_VERSION "unknown"
#endif

/* ---- engine-specific hooks -------------------------------------------
 * These are "weak" in spirit: a linked engine implementation overrides
 * them. In v0.4 there is no implementation, so the defaults here mean
 * engine_adapter_init() returns ENGINE_ADAPTER_OK (nothing to do) but
 * engine_adapter_is_available() reports 0 since SPIN_NN_HAS_ENGINE is
 * undefined.
 */
#ifdef SPIN_NN_HAS_ENGINE
/* Engine implementation must provide these: */
extern int         engine_backend_init(void);
extern int         engine_backend_shutdown(void);
extern const char *engine_backend_version(void);

static int g_refcount = 0;
#endif

int engine_adapter_is_available(void) {
#ifdef SPIN_NN_HAS_ENGINE
    return 1;
#else
    return 0;
#endif
}

const char *engine_adapter_build_version(void) {
    return SPIN_NN_ENGINE_VERSION;
}

const char *engine_adapter_engine_version(void) {
#ifdef SPIN_NN_HAS_ENGINE
    return engine_backend_version();
#else
    return NULL;
#endif
}

int engine_adapter_init(void) {
#ifdef SPIN_NN_HAS_ENGINE
    if (g_refcount == 0) {
        int rc = engine_backend_init();
        if (rc != 0) {
            fprintf(stderr,
                    "engine_adapter: engine_backend_init failed (rc=%d)\n", rc);
            return ENGINE_ADAPTER_ELIB;
        }
    }
    g_refcount++;
    return ENGINE_ADAPTER_OK;
#else
    return ENGINE_ADAPTER_EDISABLED;
#endif
}

int engine_adapter_shutdown(void) {
#ifdef SPIN_NN_HAS_ENGINE
    if (g_refcount <= 0) return ENGINE_ADAPTER_ENOT_READY;
    g_refcount--;
    if (g_refcount == 0) {
        (void)engine_backend_shutdown();
    }
    return ENGINE_ADAPTER_OK;
#else
    return ENGINE_ADAPTER_EDISABLED;
#endif
}

/* ---- flatteners -------------------------------------------------------
 * Live in v0.4; used by the test suite regardless of engine availability.
 * Row-major (x outer, y middle, z inner). Returns required count if
 * `out == NULL`, else number of floats written. Negative = status enum.
 */
long engine_adapter_flatten_ising(const IsingLattice *l, float *out, size_t out_capacity) {
    if (!l) return ENGINE_ADAPTER_EARG;
    long need = (long)l->size_x * l->size_y * l->size_z;
    if (!out) return need;
    if ((long)out_capacity < need) return ENGINE_ADAPTER_EARG;
    for (int x = 0; x < l->size_x; x++) {
        for (int y = 0; y < l->size_y; y++) {
            for (int z = 0; z < l->size_z; z++) {
                *out++ = (float)l->spins[x][y][z];
            }
        }
    }
    return need;
}

long engine_adapter_flatten_kitaev(const KitaevLattice *l, float *out, size_t out_capacity) {
    if (!l) return ENGINE_ADAPTER_EARG;
    long need = (long)l->size_x * l->size_y * l->size_z;
    if (!out) return need;
    if ((long)out_capacity < need) return ENGINE_ADAPTER_EARG;
    for (int x = 0; x < l->size_x; x++) {
        for (int y = 0; y < l->size_y; y++) {
            for (int z = 0; z < l->size_z; z++) {
                *out++ = (float)l->spins[x][y][z];
            }
        }
    }
    return need;
}

long engine_adapter_flatten_spin(const SpinLattice *l, float *out, size_t out_capacity) {
    if (!l) return ENGINE_ADAPTER_EARG;
    long need = (long)l->size_x * l->size_y * l->size_z * 3;
    if (!out) return need;
    if ((long)out_capacity < need) return ENGINE_ADAPTER_EARG;
    for (int x = 0; x < l->size_x; x++) {
        for (int y = 0; y < l->size_y; y++) {
            for (int z = 0; z < l->size_z; z++) {
                Spin s = l->spins[x][y][z];
                *out++ = (float)s.sx;
                *out++ = (float)s.sy;
                *out++ = (float)s.sz;
            }
        }
    }
    return need;
}
