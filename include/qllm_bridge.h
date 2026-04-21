/*
 * include/qllm_bridge.h
 *
 * Lazy bridge to semiclassical_qllm, the mixed-curvature product-
 * manifold LLM inference + training engine at
 * github.com/tsotchke/semiclassical_qllm.
 *
 * Primary use in this framework: a transformer- or Mamba-based
 * learned QEC decoder (pillar P1.3) and a transformer-based NQS
 * ansatz (pillar P1.1.a — ViT-NQS). Both workloads benefit from
 * semiclassical_qllm's geodesic/heat-kernel attention, KAN FFN
 * layers, flash/paged/hull KV caches, and Riemannian optimisers.
 *
 * Gated behind -DSPIN_NN_HAS_QLLM=1. When disabled every entry
 * point returns QLLM_BRIDGE_EDISABLED; the framework's learned-
 * decoder path falls back to the MWPM baseline.
 *
 * Related: eshkol_bridge handles the autograd tape; qllm_bridge
 * handles the forward/backward neural inference. Both cooperate
 * through semiclassical_qllm's eshkol_ffi.h handle API.
 */
#ifndef QLLM_BRIDGE_H
#define QLLM_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    QLLM_BRIDGE_OK         =  0,
    QLLM_BRIDGE_EDISABLED  = -1,
    QLLM_BRIDGE_ENOT_READY = -2,
    QLLM_BRIDGE_ELIB       = -3,
    QLLM_BRIDGE_EARG       = -4
} qllm_bridge_status_t;

/* --- availability + lifecycle -------------------------------------- */

int         qllm_bridge_is_available(void);
const char *qllm_bridge_version(void);

int qllm_bridge_init(void);
int qllm_bridge_shutdown(void);

/* --- model handle -------------------------------------------------- */

/* Opaque handle to a loaded semiclassical_qllm model (transformer,
 * Mamba hybrid, KAN, or GRR). Load from SafeTensors / HuggingFace /
 * GeoRefine-compressed format. */
typedef struct qllm_model qllm_model_t;

/* Load a model from disk. Supports SafeTensors + HuggingFace
 * config.json + GeoRefine-compressed files. */
int qllm_bridge_model_load(const char *model_path, qllm_model_t **out);
int qllm_bridge_model_free(qllm_model_t *m);

/* Query model metadata. */
int qllm_bridge_model_input_dim(const qllm_model_t *m, int *out_dim);
int qllm_bridge_model_output_dim(const qllm_model_t *m, int *out_dim);

/* --- learned QEC decoder path (pillar P1.3) ----------------------- */

/* Forward pass: syndrome tokens → correction Pauli string.
 *   input_tokens    — length seq_len × input_dim, flattened.
 *   output_correction — length output_dim (Pauli-string probabilities).
 * When disabled, writes zeros to out and returns EDISABLED. */
int qllm_bridge_decode_syndrome(qllm_model_t *m,
                                 const double *input_tokens,
                                 int seq_len,
                                 double *output_correction);

/* --- transformer-NQS ansatz path (pillar P1.1.a) ------------------ */

/* Forward pass: spin configuration → log-amplitude + phase.
 *   spins      — length N (±1).
 *   out_log    — log|ψ|.
 *   out_phase  — arg ψ. */
int qllm_bridge_nqs_logpsi(qllm_model_t *m,
                            const int *spins, int num_sites,
                            double *out_log, double *out_phase);

#ifdef __cplusplus
}
#endif

#endif /* QLLM_BRIDGE_H */
