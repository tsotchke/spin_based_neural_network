/*
 * src/qgtl_bridge.c
 *
 * Dormant wrapper around the Quantum Geometric Tensor Library
 * (github.com/tsotchke/quantum_geometric_tensor). Implementation
 * forthcoming when the QGTL 1.0 public ABI stabilises; this scaffold
 * keeps the API shape stable so downstream code can compile against
 * it regardless.
 */
#include <stdio.h>
#include <stdlib.h>
#include "qgtl_bridge.h"

#ifdef SPIN_NN_HAS_QGTL
static int g_refcount = 0;
#endif

int qgtl_bridge_is_available(void) {
#ifdef SPIN_NN_HAS_QGTL
    return 1;
#else
    return 0;
#endif
}

const char *qgtl_bridge_version(void) {
#ifdef SPIN_NN_HAS_QGTL
    return "0.777";
#else
    return NULL;
#endif
}

int qgtl_bridge_init(void) {
#ifdef SPIN_NN_HAS_QGTL
    g_refcount++;
    return QGTL_BRIDGE_OK;
#else
    return QGTL_BRIDGE_EDISABLED;
#endif
}

int qgtl_bridge_shutdown(void) {
#ifdef SPIN_NN_HAS_QGTL
    if (g_refcount <= 0) return QGTL_BRIDGE_ENOT_READY;
    g_refcount--;
    return QGTL_BRIDGE_OK;
#else
    return QGTL_BRIDGE_EDISABLED;
#endif
}

int qgtl_bridge_compute_qgt(const double *batch_grads,
                             int batch_size, int num_params,
                             double *out_G) {
    if (!batch_grads || !out_G || batch_size <= 0 || num_params <= 0)
        return QGTL_BRIDGE_EARG;
#ifdef SPIN_NN_HAS_QGTL
    /* Live path: call QGTL's hierarchical-tensor QGT. Not implemented
     * in this commit — returns EDISABLED until the linkage lands. */
    (void)batch_grads; (void)batch_size; (void)num_params; (void)out_G;
    return QGTL_BRIDGE_EDISABLED;
#else
    (void)batch_grads; (void)batch_size; (void)num_params; (void)out_G;
    return QGTL_BRIDGE_EDISABLED;
#endif
}

qgtl_device_backend_t *qgtl_bridge_device_backend_open(const char *backend_name) {
    (void)backend_name;
    return NULL;
}

void qgtl_bridge_device_backend_close(qgtl_device_backend_t *b) {
    (void)b;
}

int qgtl_bridge_device_single_qubit_p(qgtl_device_backend_t *b, double *out_p) {
    (void)b;
    if (!out_p) return QGTL_BRIDGE_EARG;
    *out_p = -1.0;
    return QGTL_BRIDGE_EDISABLED;
}
