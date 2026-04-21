/*
 * include/thermodynamic/hopfield.h
 *
 * Hopfield associative memory as a thermodynamic-computing demonstrator
 * on the spin substrate (pillar P2.9 in architecture_v0.4.md).
 *
 * Stores K bipolar (±1) patterns ξ^μ ∈ {±1}^N via Hebbian learning
 *     J_ij = (1/N) Σ_μ ξ^μ_i ξ^μ_j     (i ≠ j),   J_ii = 0.
 * Each stored pattern is an attractor of the zero-temperature
 * deterministic dynamics s_i ← sign(Σ_j J_ij s_j); at finite T the
 * network samples from the Boltzmann distribution P(s) ∝ exp(+β s^T J s).
 *
 * This is not trained via equilibrium propagation — Hebbian storage is
 * closed-form — but the recall loop uses the same thermodynamic update
 * rule (Glauber Metropolis at β = 1/T) that an energy-based neural
 * substrate would run in hardware.
 */
#ifndef HOPFIELD_H
#define HOPFIELD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int     num_spins;         /* N */
    double *J;                 /* N × N symmetric weight matrix */
    double *h;                 /* optional local field (length N)  */
} hopfield_t;

/* Allocate an N-spin network with zero weights. Returns NULL on OOM. */
hopfield_t *hopfield_create(int num_spins);
void        hopfield_free  (hopfield_t *net);

/* Hebbian storage of K patterns (K × N matrix, row-major, entries ±1).
 * Overwrites any existing weights; clears self-interactions. Returns 0
 * on success, -1 on argument errors. */
int hopfield_store_patterns(hopfield_t *net,
                             const int *patterns, int num_patterns);

/* Synchronous sign update (zero-temperature). One sweep flips each
 * spin to sign(Σ_j J_ij s_j + h_i). Returns the number of spins that
 * changed this sweep. Use to check fixed-point stability. */
int hopfield_sync_update(hopfield_t *net, int *spins);

/* Glauber / Metropolis sweep at inverse temperature β. `rng_state` is
 * the caller-owned xorshift64 state (0 maps to 1). One full sweep
 * visits every site once in shuffled order. */
void hopfield_metropolis_sweep(const hopfield_t *net,
                                int *spins, double beta,
                                unsigned long long *rng_state);

/* Overlap ⟨ξ^μ, s⟩ / N with a reference pattern (length N). Returns
 * a number in [-1, +1]. */
double hopfield_overlap(const hopfield_t *net,
                         const int *pattern, const int *spins);

/* Associative recall: run `num_sweeps` zero-T sync updates from the
 * given initial state. Returns 1 if convergence to a stable fixed
 * point was reached, 0 otherwise. */
int hopfield_recall(hopfield_t *net, int *spins, int num_sweeps);

/* Energy E(s) = -½ s^T J s - h^T s. Used for Metropolis acceptance
 * and for convergence diagnostics. */
double hopfield_energy(const hopfield_t *net, const int *spins);

#ifdef __cplusplus
}
#endif

#endif /* HOPFIELD_H */
