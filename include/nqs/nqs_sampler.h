/*
 * include/nqs/nqs_sampler.h
 *
 * Metropolis-Hastings sampler over spin configurations for NQS. Each
 * sample is drawn from |ψ(s)|^2 using single-flip or cluster proposals.
 * Thread-safe relative to other samplers (each carries its own RNG
 * state and configuration buffer); the ansatz it queries must itself
 * be safe to call concurrently.
 */
#ifndef NQS_SAMPLER_H
#define NQS_SAMPLER_H

#include <stddef.h>
#include "nqs_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle owning the sampler's RNG state, current configuration,
 * and proposal buffers. Callers get amplitude queries via a callback
 * supplied at init time. */
typedef struct nqs_sampler nqs_sampler_t;

/* Log-amplitude callback. Returns log |ψ(s)| + i·arg ψ(s) packed as a
 * double-pair (real and imaginary parts). Implementations must be
 * pure functions of `spins` — stateful evaluation is the caller's
 * responsibility.
 *
 *   spins:   length num_sites, values ±1
 *   user:    opaque context (e.g. an nqs_ansatz handle)
 *   out_re:  log |ψ|
 *   out_im:  arg ψ (in radians)
 */
typedef void (*nqs_log_amp_fn_t)(const int *spins, int num_sites,
                                 void *user,
                                 double *out_re, double *out_im);

/* Allocate + initialise a sampler over `num_sites` spins. The initial
 * configuration is sampled uniformly from {+1,-1}^N using `rng_seed`.
 * Returns NULL on allocation failure. */
nqs_sampler_t *nqs_sampler_create(int num_sites,
                                  const nqs_config_t *cfg,
                                  nqs_log_amp_fn_t log_amp,
                                  void *log_amp_user);

void nqs_sampler_free(nqs_sampler_t *s);

/* Run `cfg->num_thermalize` warm-up steps to let the chain mix. Call
 * once before the first batch. */
void nqs_sampler_thermalize(nqs_sampler_t *s);

/* Produce one Metropolis sample: returns a pointer to the current
 * configuration after `cfg->num_decorrelate` moves. The pointer is
 * owned by the sampler and is valid until the next call. */
const int *nqs_sampler_next(nqs_sampler_t *s);

/* Bulk-sample `n` configurations into `out[n * num_sites]`. Caller
 * owns `out`. Equivalent to calling `nqs_sampler_next` in a loop and
 * `memcpy`-ing each result. Returns 0 on success. */
int nqs_sampler_batch(nqs_sampler_t *s, int n, int *out);

/* Statistics: acceptance ratio in [0, 1] across the sampler's lifetime. */
double nqs_sampler_acceptance_ratio(const nqs_sampler_t *s);

/* Expose the current configuration without advancing the chain. */
const int *nqs_sampler_current(const nqs_sampler_t *s);

#ifdef __cplusplus
}
#endif

#endif /* NQS_SAMPLER_H */
