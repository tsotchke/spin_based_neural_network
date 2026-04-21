/*
 * src/qllm_bridge.c
 *
 * Dormant wrapper around semiclassical_qllm. Live implementation
 * lands when the SafeTensors loader + Riemannian optimiser path is
 * wired through the existing eshkol_bridge AD tape. Until then every
 * entry point returns QLLM_BRIDGE_EDISABLED and the learned-decoder /
 * transformer-NQS code paths fall back to the baseline (MWPM /
 * complex-RBM respectively).
 */
#include <stdio.h>
#include <stdlib.h>
#include "qllm_bridge.h"

#ifdef SPIN_NN_HAS_QLLM
static int g_refcount = 0;
#endif

int qllm_bridge_is_available(void) {
#ifdef SPIN_NN_HAS_QLLM
    return 1;
#else
    return 0;
#endif
}

const char *qllm_bridge_version(void) {
#ifdef SPIN_NN_HAS_QLLM
    return "0.1.0";
#else
    return NULL;
#endif
}

int qllm_bridge_init(void) {
#ifdef SPIN_NN_HAS_QLLM
    g_refcount++;
    return QLLM_BRIDGE_OK;
#else
    return QLLM_BRIDGE_EDISABLED;
#endif
}

int qllm_bridge_shutdown(void) {
#ifdef SPIN_NN_HAS_QLLM
    if (g_refcount <= 0) return QLLM_BRIDGE_ENOT_READY;
    g_refcount--;
    return QLLM_BRIDGE_OK;
#else
    return QLLM_BRIDGE_EDISABLED;
#endif
}

int qllm_bridge_model_load(const char *model_path, qllm_model_t **out) {
    if (!model_path || !out) return QLLM_BRIDGE_EARG;
    *out = NULL;
    return QLLM_BRIDGE_EDISABLED;
}

int qllm_bridge_model_free(qllm_model_t *m) {
    (void)m;
    return QLLM_BRIDGE_OK;
}

int qllm_bridge_model_input_dim(const qllm_model_t *m, int *out_dim) {
    if (!out_dim) return QLLM_BRIDGE_EARG;
    (void)m; *out_dim = 0;
    return QLLM_BRIDGE_EDISABLED;
}

int qllm_bridge_model_output_dim(const qllm_model_t *m, int *out_dim) {
    if (!out_dim) return QLLM_BRIDGE_EARG;
    (void)m; *out_dim = 0;
    return QLLM_BRIDGE_EDISABLED;
}

int qllm_bridge_decode_syndrome(qllm_model_t *m,
                                 const double *input_tokens,
                                 int seq_len,
                                 double *output_correction) {
    if (!input_tokens || !output_correction || seq_len <= 0)
        return QLLM_BRIDGE_EARG;
    (void)m;
    return QLLM_BRIDGE_EDISABLED;
}

int qllm_bridge_nqs_logpsi(qllm_model_t *m,
                            const int *spins, int num_sites,
                            double *out_log, double *out_phase) {
    if (!spins || !out_log || !out_phase || num_sites <= 0)
        return QLLM_BRIDGE_EARG;
    (void)m;
    *out_log = 0.0;
    *out_phase = 0.0;
    return QLLM_BRIDGE_EDISABLED;
}
