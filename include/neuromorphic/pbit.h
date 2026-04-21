/*
 * include/neuromorphic/pbit.h
 *
 * Probabilistic-bit (p-bit) Ising machine. A p-bit is a noisy Ising
 * spin whose state ±1 is sampled from
 *
 *     P(s_i = +1) = σ(β h_i)   where σ(x) = 1 / (1 + e^{-x})
 *
 * and h_i = Σ_j J_ij s_j + b_i is the local field combining pairwise
 * couplings J and per-site bias b (units absorbed into β).
 *
 * Asynchronous updates (one site at a time in random order) converge
 * to the Boltzmann distribution P(s) ∝ exp(β · (½ s^T J s + b^T s)).
 * This is the canonical model for hardware Ising machines: spintronic
 * p-bits (Nat. Electronics 2025), stochastic neurons, quantum-inspired
 * annealers. Applications include MaxCut, Boolean SAT, and any other
 * QUBO.
 *
 * Pillar P2.4 in the architecture plan.
 */
#ifndef NEUROMORPHIC_PBIT_H
#define NEUROMORPHIC_PBIT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int      n;                /* number of p-bits */
    const double *J;           /* n × n coupling matrix, row-major;
                                  caller owns storage, must outlive net */
    const double *b;           /* length-n bias vector (may be NULL) */
    double   beta;             /* inverse temperature */
    unsigned long long rng;    /* xorshift64 state */
} pbit_net_t;

/* Initialise with supplied coupling matrix. Caller owns J and b. */
void pbit_net_init(pbit_net_t *net, int n,
                    const double *J, const double *b,
                    double beta, unsigned long long seed);

/* Perform `num_sweeps` full asynchronous sweeps over all n p-bits.
 * Each sweep visits every site once in random order. Returns 0. */
int pbit_sweep(pbit_net_t *net, int *spins, int num_sweeps);

/* Compute Ising energy
 *     E(s) = -½ s^T J s - b^T s
 * This convention makes E the quantity MINIMISED by simulated
 * annealing (β → ∞ concentrates probability on the minimum). */
double pbit_ising_energy(const pbit_net_t *net, const int *spins);

/* Build a MaxCut coupling matrix from an edge list. For unweighted
 * MaxCut, assign J_ij = -w_ij for each edge (so that minimising the
 * Ising energy = maximising the cut weight). The output `J` is the
 * n × n symmetric matrix; `b` can be NULL. Returns 0 on success. */
int pbit_maxcut_couplings(int n, int num_edges,
                           const int *edges_u, const int *edges_v,
                           const double *weights,
                           double *out_J);

/* Run simulated-annealing p-bit sweeps with a geometric β schedule
 * from beta_start to beta_end over num_sweeps. Returns the Ising
 * energy of the final configuration and (optionally) the final
 * spin configuration. */
int pbit_anneal(pbit_net_t *net, int *spins,
                 double beta_start, double beta_end, int num_sweeps,
                 double *out_final_energy);

#ifdef __cplusplus
}
#endif

#endif /* NEUROMORPHIC_PBIT_H */
