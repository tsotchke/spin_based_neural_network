/*
 * src/eshkol_bridge.c
 *
 * Lazy wrapper over the Eshkol FFI. Compiles and links without Eshkol
 * present; all entry points return ESHKOL_BRIDGE_EDISABLED in that case.
 *
 * When built with -DSPIN_NN_HAS_ESHKOL=1, this is the only translation unit
 * that includes <eshkol/eshkol_ffi.h>, so the link boundary lives in one
 * place (same pattern as src/engine_adapter.c).
 */
#include <stdio.h>
#include <string.h>

#include "eshkol_bridge.h"

#ifdef SPIN_NN_HAS_ESHKOL
#include <eshkol/eshkol_ffi.h>
static eshkol_ffi_context_t *g_ctx = NULL;
#endif

#ifdef SPIN_NN_HAS_ESHKOL
static int g_refcount = 0;
#endif
static char g_last_error[256] = {0};

static void set_error(const char *msg) {
    if (!msg) { g_last_error[0] = '\0'; return; }
    size_t n = strlen(msg);
    if (n >= sizeof(g_last_error)) n = sizeof(g_last_error) - 1;
    memcpy(g_last_error, msg, n);
    g_last_error[n] = '\0';
}

int eshkol_bridge_is_available(void) {
#ifdef SPIN_NN_HAS_ESHKOL
    return 1;
#else
    return 0;
#endif
}

int eshkol_bridge_init(void) {
#ifdef SPIN_NN_HAS_ESHKOL
    if (g_refcount == 0) {
        g_ctx = eshkol_ffi_init();
        if (!g_ctx) {
            set_error(eshkol_ffi_last_error() ? eshkol_ffi_last_error()
                                              : "eshkol_ffi_init returned NULL");
            return ESHKOL_BRIDGE_ELIB;
        }
    }
    g_refcount++;
    return ESHKOL_BRIDGE_OK;
#else
    set_error("spin_based_neural_network built without SPIN_NN_HAS_ESHKOL");
    return ESHKOL_BRIDGE_EDISABLED;
#endif
}

int eshkol_bridge_shutdown(void) {
#ifdef SPIN_NN_HAS_ESHKOL
    if (g_refcount <= 0) return ESHKOL_BRIDGE_ENOT_READY;
    g_refcount--;
    if (g_refcount == 0 && g_ctx) {
        eshkol_ffi_shutdown(g_ctx);
        g_ctx = NULL;
    }
    return ESHKOL_BRIDGE_OK;
#else
    return ESHKOL_BRIDGE_EDISABLED;
#endif
}

int eshkol_bridge_load_script(const char *path) {
    if (!path) return ESHKOL_BRIDGE_EARG;
#ifdef SPIN_NN_HAS_ESHKOL
    if (g_refcount <= 0 || !g_ctx) return ESHKOL_BRIDGE_ENOT_READY;
    int rc = eshkol_ffi_eval_file(g_ctx, path);
    if (rc != 0) {
        set_error(eshkol_ffi_last_error() ? eshkol_ffi_last_error()
                                          : "eshkol_ffi_eval_file failed");
        return ESHKOL_BRIDGE_ELIB;
    }
    return ESHKOL_BRIDGE_OK;
#else
    (void)path;
    return ESHKOL_BRIDGE_EDISABLED;
#endif
}

int eshkol_bridge_eval_double(const char *source, double *result) {
    if (!source || !result) return ESHKOL_BRIDGE_EARG;
#ifdef SPIN_NN_HAS_ESHKOL
    if (g_refcount <= 0 || !g_ctx) return ESHKOL_BRIDGE_ENOT_READY;
    int rc = eshkol_ffi_eval_double(g_ctx, source, result);
    if (rc != 0) {
        set_error(eshkol_ffi_last_error() ? eshkol_ffi_last_error()
                                          : "eshkol_ffi_eval_double failed");
        return ESHKOL_BRIDGE_ELIB;
    }
    return ESHKOL_BRIDGE_OK;
#else
    (void)source;
    (void)result;
    return ESHKOL_BRIDGE_EDISABLED;
#endif
}

const char *eshkol_bridge_last_error(void) {
    return g_last_error[0] ? g_last_error : NULL;
}
