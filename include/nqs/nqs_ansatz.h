/*
 * include/nqs/nqs_ansatz.h
 *
 * Wavefunction ansatz: a parametric neural network that maps a spin
 * configuration s ∈ {±1}^N to log|ψ(s)| + i·arg ψ(s).
 *
 * v0.4 ships the legacy-MLP ansatz: a simple fully-connected network
 * producing a single scalar (treated as log |ψ|; the phase is taken
 * as 0). The more expressive ViT / factored-ViT / autoregressive
 * ansätze in nqs_config_t lands once the external NN engine is wired
 * in (see architecture_v0.4.md §P1.1).
 *
 * Ansatz lifetime is bound to an `nqs_ansatz_t` handle which keeps
 * the underlying network + scratch buffers. The ansatz exposes both
 * an amplitude query (used by samplers) and a parameter-space
 * Jacobian vector product helper (used by stochastic reconfiguration).
 */
#ifndef NQS_ANSATZ_H
#define NQS_ANSATZ_H

#include <stddef.h>
#include "nqs_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nqs_ansatz nqs_ansatz_t;

/* Construct an ansatz sized for a `num_sites` spin configuration.
 * Backed by whichever concrete implementation `cfg->ansatz` selects;
 * returns NULL if the requested ansatz is not available in the
 * current build (e.g. NQS_ANSATZ_VIT without the external engine). */
nqs_ansatz_t *nqs_ansatz_create(const nqs_config_t *cfg, int num_sites);

void nqs_ansatz_free(nqs_ansatz_t *a);

/* Number of variational parameters. Used to size QGT workspace. */
long nqs_ansatz_num_params(const nqs_ansatz_t *a);

/* Log-amplitude query, compatible with nqs_log_amp_fn_t. Wrap this
 * call when passing an ansatz to an nqs_sampler_t. */
void nqs_ansatz_log_amp(const int *spins, int num_sites,
                        void *ansatz_user,
                        double *out_log_abs,
                        double *out_arg);

/* Gradient of log ψ with respect to the parameters, evaluated at
 * `spins`. Writes `num_params` doubles into `out_grad`. Returns 0 on
 * success.
 *
 * For complex ansätze this returns only the REAL part of
 * ∂ log ψ / ∂θ_k (which equals ∂ log|ψ| / ∂θ_k for real θ). That is
 * sufficient for the real-projected SR path used by
 * `nqs_sr_step_custom`; holomorphic SR needs the full complex
 * gradient — use `nqs_ansatz_logpsi_gradient_complex` below. */
int nqs_ansatz_logpsi_gradient(nqs_ansatz_t *a,
                               const int *spins, int num_sites,
                               double *out_grad);

/* Holomorphic gradient: writes the real and imaginary parts of
 * ∂ log ψ / ∂θ_k (for k = 0 .. num_params - 1). For purely real
 * ansätze, `out_im` is filled with zeros. Buffers must have length
 * `num_params` each. Returns 0 on success. */
int nqs_ansatz_logpsi_gradient_complex(nqs_ansatz_t *a,
                                        const int *spins, int num_sites,
                                        double *out_re, double *out_im);

/* Returns 1 if the ansatz kind carries a non-trivial complex phase
 * (e.g. NQS_ANSATZ_COMPLEX_RBM), 0 otherwise. Used by the optimizer
 * to decide whether the holomorphic SR path is needed. */
int nqs_ansatz_is_complex(const nqs_ansatz_t *a);

/* Apply a parameter update in-place: params ← params + step * delta.
 * `delta` has length nqs_ansatz_num_params(a). Returns 0 on success. */
int nqs_ansatz_apply_update(nqs_ansatz_t *a,
                            const double *delta, double step);

/* Raw access to the parameter buffer (length nqs_ansatz_num_params(a)).
 * For integrators that need to snapshot / restore parameter state
 * (e.g. Heun / RK multi-stage tVMC). Returns NULL if `a` is NULL. */
double *nqs_ansatz_params_raw(nqs_ansatz_t *a);

#ifdef __cplusplus
}
#endif

#endif /* NQS_ANSATZ_H */
