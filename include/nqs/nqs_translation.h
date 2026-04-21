/*
 * include/nqs/nqs_translation.h
 *
 * Momentum-zero translation-symmetry projector for NQS ansätze on
 * 1D chains and 2D square lattices with periodic boundary conditions.
 *
 * The symmetrised wavefunction is
 *     ψ_sym(s) = (1/√|G|) · Σ_τ  χ(τ) · ψ_base(T^τ s)
 * with χ(τ) = +1 for k = 0 (the trivial irrep). Other momenta plug
 * in with χ(τ) = exp(i k · τ), but that needs a complex ansatz and
 * is parked for v0.6 alongside complex RBM.
 *
 * Combining with Marshall / other wrappers: the wrapper is just
 * another log_amp callback, so stacking them is free — wire the
 * Marshall wrapper as the `base_log_amp` below.
 */
#ifndef NQS_TRANSLATION_H
#define NQS_TRANSLATION_H

#include "nqs/nqs_sampler.h"
#include "nqs/nqs_ansatz.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    nqs_log_amp_fn_t base_log_amp;
    void            *base_user;
    int              size_x;
    int              size_y;   /* set to 1 for 1D chains */
} nqs_translation_wrapper_t;

/* log_amp callback for the k = 0 translation-projected wavefunction.
 * Evaluates the base log_amp at every cyclic shift, combines with
 * log-sum-exp (phase-aware so it composes with Marshall), and
 * returns log|ψ_sym(s)| and arg(ψ_sym(s)). */
void nqs_translation_log_amp(const int *spins, int num_sites,
                              void *user,
                              double *out_log_abs,
                              double *out_arg);

/* Gradient callback for the k = 0 translation-projected wavefunction.
 *   ∂ log ψ_sym(s) / ∂θ = Σ_τ w_τ(s) · ∂ log ψ_base(T^τ s) / ∂θ
 * with w_τ = ψ_base(T^τ s) / Σ_τ' ψ_base(T^τ' s) (real-valued). When
 * the base ansatz is the raw RBM, this is a straight weighted sum of
 * rbm_gradient(T^τ s). When the base_log_amp is itself wrapped (e.g.
 * Marshall), the gradient wraps through: Marshall is a sign-only
 * transformation so ∂ log|ψ_marshall| / ∂θ = ∂ log|ψ_base| / ∂θ. */
int nqs_translation_gradient(void *grad_user,
                              nqs_ansatz_t *ansatz,
                              const int *spins, int num_sites,
                              double *out_grad);

#ifdef __cplusplus
}
#endif

#endif /* NQS_TRANSLATION_H */
