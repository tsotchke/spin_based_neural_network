/*
 * include/mps/dmrg.h
 *
 * Full 2-site DMRG driver for the 1D XXZ chain. This is the pillar
 * P2.2 piece: an MPS whose bond dimension is grown sweep-by-sweep via
 * SVD truncation, and whose two-site effective Hamiltonian is solved
 * by matrix-free Lanczos (the same module dmrg.c shares with
 * mps_ground_state_dense).
 *
 * The XXZ Hamiltonian is
 *     H = Σ_{i=0}^{N-2}  J_x (S^x_i S^x_{i+1} + S^y_i S^y_{i+1})
 *                        + J_z  S^z_i S^z_{i+1}
 * Standard Heisenberg is J_x = J_z = J; Ising limit is J_x = 0.
 *
 * Bond indices: the MPS tensor at site i has shape (D_{i}, 2, D_{i+1})
 * with D_0 = D_N = 1 and D_i ≤ D_max growing and shrinking during
 * sweeps. The canonical form alternates left- and right-orthogonal
 * tensors across the sweep direction; convergence is declared when
 * the per-sweep energy drop falls below `sweep_tol`.
 */
#ifndef MPS_DMRG_H
#define MPS_DMRG_H

#include <stddef.h>
#include "mps/mps.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double final_energy;
    int    sweeps_performed;
    int    converged;        /* 1 if Δ energy < sweep_tol between sweeps */
    double largest_truncated_sv;  /* max s[D_max] seen over the run */
} mps_dmrg_result_t;

/* Run 2-site DMRG on the 1D XXZ chain described by `cfg`. The MPS
 * ansatz is kept internally; only the ground-state energy (and
 * summary stats) are returned. Up to `cfg->num_sweeps` full sweeps
 * (left → right + right → left) are performed.
 *
 * Returns 0 on success. */
int mps_dmrg_xxz(const mps_config_t *cfg, mps_dmrg_result_t *out);

/* As mps_dmrg_xxz but also materialises the ground-state vector in
 * the computational basis. Works only for N ≤ 20 (dim ≤ 2^20). The
 * caller is responsible for freeing *out_psi on success. *out_dim
 * receives 2^N on success. */
int mps_dmrg_xxz_with_state(const mps_config_t *cfg,
                             mps_dmrg_result_t *out,
                             double **out_psi, long *out_dim);


#ifdef __cplusplus
}
#endif

#endif /* MPS_DMRG_H */
