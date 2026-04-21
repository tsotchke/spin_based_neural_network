/*
 * include/thermodynamic/rbm_cd.h
 *
 * Restricted Boltzmann machine (RBM) trained via contrastive-divergence
 * (CD-k) on ±1 / {0,1} patterns — the generative-modelling half of the
 * P2.9 thermodynamic-computing pillar (the associative-memory half
 * lives in thermodynamic/hopfield.h).
 *
 * A CD-trained RBM is the canonical spin-substrate generative model:
 * both sampling and gradient estimation run through block-Gibbs
 * transitions of the same Markov chain a p-bit / MTJ hardware array
 * would execute in silicon.
 *
 * Convention: visible v_i ∈ {0, 1}, hidden h_j ∈ {0, 1}. Energy
 *     E(v, h) = -a^T v - b^T h - v^T W h
 * with marginal P(v) ∝ Σ_h exp(-E(v,h)). The ±1 flavour is obtained
 * by substituting 2v - 1.
 */
#ifndef RBM_CD_H
#define RBM_CD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int     num_visible;       /* N */
    int     num_hidden;        /* M */
    double *W;                 /* N×M weight matrix, row-major (v-major) */
    double *a;                 /* length N visible biases              */
    double *b;                 /* length M hidden biases               */
    unsigned long long rng;    /* xorshift64 state — auto-seeded       */
} rbm_cd_t;

rbm_cd_t *rbm_cd_create(int num_visible, int num_hidden,
                         double weight_scale,
                         unsigned long long seed);
void       rbm_cd_free(rbm_cd_t *rbm);

/* Gibbs up-step: sample h ~ P(h|v). `v` and `h` are {0,1} arrays. */
void rbm_cd_sample_h_given_v(rbm_cd_t *rbm, const int *v, int *h);

/* Gibbs down-step: sample v ~ P(v|h). */
void rbm_cd_sample_v_given_h(rbm_cd_t *rbm, const int *h, int *v);

/* Deterministic mean-field up-step: h_j ← σ(b_j + Σ_i v_i W_ij). */
void rbm_cd_mean_h_given_v(const rbm_cd_t *rbm,
                            const int *v, double *mean_h);

/* Deterministic mean-field down-step. */
void rbm_cd_mean_v_given_h(const rbm_cd_t *rbm,
                            const int *h, double *mean_v);

/* Run one contrastive-divergence k-step on a single binary pattern.
 * Internal: starts at v^data, runs k block-Gibbs steps to obtain v',
 * then gradient
 *     ΔW_ij = η [ v^data_i · ⟨h|v^data⟩_j − v'_i · ⟨h|v'⟩_j ]
 *     Δa_i  = η [ v^data_i − v'_i ]
 *     Δb_j  = η [ ⟨h|v^data⟩_j − ⟨h|v'⟩_j ]
 * The "⟨h|v⟩" here is the mean-field activation (not a sample), which
 * is the standard variance-reduced form of CD-k. */
int rbm_cd_train_step(rbm_cd_t *rbm,
                       const int *v_data,
                       int k_gibbs,
                       double learning_rate);

/* Minibatch wrapper: loops rbm_cd_train_step over `num_patterns` rows
 * of `v_data_batch` (row-major, num_patterns × num_visible). */
int rbm_cd_train_batch(rbm_cd_t *rbm,
                        const int *v_data_batch, int num_patterns,
                        int k_gibbs, double learning_rate);

/* Sample `num_samples` visible configurations by running long-run
 * Gibbs from a random start. `burn_in` Gibbs sweeps discarded first,
 * then sample every `thin` sweep. Writes num_samples × N into
 * `out_samples` (row-major, binary). */
int rbm_cd_sample(rbm_cd_t *rbm,
                   int *out_samples, int num_samples,
                   int burn_in, int thin);

/* Free energy F(v) = -a^T v - Σ_j softplus(b_j + Σ_i v_i W_ij).
 * Lower F(v) ↔ higher model probability. */
double rbm_cd_free_energy(const rbm_cd_t *rbm, const int *v);

#ifdef __cplusplus
}
#endif

#endif /* RBM_CD_H */
