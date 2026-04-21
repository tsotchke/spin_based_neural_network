/*
 * include/flow_matching/flow_matching.h
 *
 * Discrete flow-matching sampler scaffold for v0.5 pillar P1.4. v0.4
 * ships a reference continuous-time Markov chain (CTMC) implementation
 * that interpolates between a trivial source distribution (uniform ±1
 * spins) and a target Ising configuration via a user-supplied rate
 * schedule. This is the interface the learned score / rate network
 * will satisfy in v0.5.
 *
 * The CTMC is formulated as in Campbell et al. 2024 / Gat et al. 2024:
 * at each of `num_steps` discrete times, a per-site flip rate
 *     λ_i(t) = (1 - t) / (t + ε)
 * drives each spin toward its target value. An Euler step samples
 *     P(flip at i) = 1 - exp(-λ_i(t) · dt · [s_i ≠ target_i])
 * so that as t → 1 the chain converges to the conditioned target.
 * Unconditional sampling uses an unconditional schedule (see below).
 */
#ifndef FLOW_MATCHING_H
#define FLOW_MATCHING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int num_sites;          /* 1D list of Ising spins ±1                 */
    int num_steps;          /* number of CTMC Euler steps                */
    double integrated_rate;  /* total ∫₀¹ λ dt. Sets relaxation strength. */
    unsigned long long seed; /* xorshift64 seed                           */
} flow_matching_config_t;

static inline flow_matching_config_t flow_matching_config_defaults(void) {
    flow_matching_config_t c;
    c.num_sites        = 16;
    c.num_steps        = 128;
    c.integrated_rate  = 2.0;
    c.seed             = 0xC0FFEE;
    return c;
}

/* Sample a spin trajectory conditioned on `target`. On return, `out`
 * holds the spins at t=1. Requires num_sites == cfg->num_sites and
 * target[i] ∈ {+1, -1}. Returns 0 on success. */
int flow_matching_sample_conditional(const flow_matching_config_t *cfg,
                                     const int *target, int *out);

/* Draw an unconditional sample by running the reverse-time CTMC from
 * a uniform source to a product-of-Bernoulli target induced by
 * `bias[i]` ∈ [-1, 1], i.e. P(s_i = +1) = (1 + bias[i]) / 2. */
int flow_matching_sample_unconditional(const flow_matching_config_t *cfg,
                                       const double *bias, int *out);

/* Per-site rate variant. Instead of a single global rate (c =
 * cfg->integrated_rate), each site has its own rate c_i that can be
 * chosen to achieve a prescribed target magnetisation at t = 1. */
int flow_matching_sample_biased_rates(const flow_matching_config_t *cfg,
                                       const double *bias,
                                       const double *per_site_rates,
                                       int *out);

/* Closed-form optimal per-site rates for reaching a specified mean
 * magnetisation `m_target[i]` ∈ (-|bias[i]|, |bias[i]|) at t=1, given
 * per-site bias (which sets the stationary distribution). The two-
 * state CTMC relation m(1) = bias · (1 − e^{−c}) inverts to
 *     c = -log(1 - m_target / bias)
 * For m_target with the same sign as bias and |m_target| <= |bias|.
 * Writes c_i into `out_rates` (length cfg->num_sites). */
int flow_matching_fit_rates_to_magnetisation(const flow_matching_config_t *cfg,
                                              const double *bias,
                                              const double *m_target,
                                              double *out_rates);

#ifdef __cplusplus
}
#endif

#endif /* FLOW_MATCHING_H */
