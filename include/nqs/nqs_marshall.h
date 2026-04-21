/*
 * include/nqs/nqs_marshall.h
 *
 * Marshall sign rule wrapper for real-valued NQS ansätze on bipartite
 * lattices. Writing ψ_physical(s) = (-1)^{n_↓,A(s)} ψ_RBM(s), where
 * n_↓,A counts the down spins on sublattice A (even (x+y)), turns the
 * off-diagonal matrix elements of the Heisenberg antiferromagnet into
 * negative numbers — a stoquastic rotation that a strictly-positive
 * ansatz like the real RBM can represent.
 *
 * Usage:
 *
 *   nqs_marshall_wrapper_t w = {
 *       .base_log_amp = nqs_ansatz_log_amp,
 *       .base_user    = ansatz,
 *       .size_x       = Lx,
 *       .size_y       = Ly,
 *   };
 *   nqs_sampler_t *s = nqs_sampler_create(N, &cfg,
 *                                          nqs_marshall_log_amp, &w);
 *   // local-energy kernel: also pass nqs_marshall_log_amp.
 *
 * The wrapper leaves |ψ|² unchanged, so the Metropolis distribution is
 * unaffected; only the phase output flips on odd-parity configurations.
 */
#ifndef NQS_MARSHALL_H
#define NQS_MARSHALL_H

#include "nqs/nqs_sampler.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    nqs_log_amp_fn_t base_log_amp;
    void            *base_user;
    int              size_x;
    int              size_y;
} nqs_marshall_wrapper_t;

/* Marshall sign for a single configuration. Returns 0 for even-parity
 * (keep sign), 1 for odd-parity (flip sign). Sublattice A is
 * (x + y) even. */
int nqs_marshall_parity(const int *spins, int size_x, int size_y);

/* log_amp callback: delegates to the base ansatz for the magnitude,
 * then overrides the phase with the Marshall sign. */
void nqs_marshall_log_amp(const int *spins, int num_sites,
                           void *user,
                           double *out_log_abs,
                           double *out_arg);

#ifdef __cplusplus
}
#endif

#endif /* NQS_MARSHALL_H */
