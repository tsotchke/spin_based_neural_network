/*
 * include/nqs/nqs_lanczos.h
 *
 * NQS → exact state vector → Lanczos post-processing (pillar P2.6).
 *
 * Given a trained `nqs_ansatz_t` over an N-site lattice with N small
 * enough that 2^N fits in memory, this module:
 *
 *   (1) materialises the exact wavefunction amplitudes ψ(s) = exp(log ψ(s))
 *       for every computational-basis state s ∈ {0, ..., 2^N - 1};
 *   (2) evaluates the deterministic variational energy
 *           ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩
 *       on a dense Hamiltonian — no Monte-Carlo noise;
 *   (3) runs Lanczos starting from the NQS state vector and returns the
 *       refined ground-state energy and eigenvector.
 *
 * The point of (3) is that the Krylov subspace built on top of a
 * decent-quality variational state is extremely well-conditioned for
 * finding the ground state — typically a handful of iterations drops
 * the energy error by 1-2 orders of magnitude. This is the P2.6 trick
 * described in architecture_v0.4.md.
 *
 * Hamiltonian representation here is the TFIM:
 *     H = -J Σ_<ij> σ^z_i σ^z_j - Γ Σ_i σ^x_i
 * matching the kernel in nqs_gradient.c. Additional Hamiltonians land
 * as the NQS pillar itself grows.
 */
#ifndef NQS_LANCZOS_H
#define NQS_LANCZOS_H

#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_sampler.h"      /* for nqs_log_amp_fn_t */
#include "mps/lanczos.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Allocate and fill `out_psi` with the exact basis-state amplitudes of
 * the NQS ansatz on an (Lx × Ly) lattice. Returns 0 on success. The
 * returned vector is normalised to ||ψ|| = 1. Caller frees `out_psi`. */
int nqs_materialise_state(nqs_ansatz_t *a, int Lx, int Ly,
                          double **out_psi, long *out_dim);

/* Deterministic variational energy of an NQS ansatz against the TFIM
 * Hamiltonian with couplings (J, Γ) and open boundary conditions. */
int nqs_exact_energy_tfim(nqs_ansatz_t *a, int Lx, int Ly,
                           double J, double Gamma, double *out_energy);

/* Lanczos post-processing: starts from the NQS state vector and runs
 * `max_iters` Krylov steps on H_TFIM, returning the refined ground-
 * state energy and (optionally) eigenvector. Returns 0 on success. */
int nqs_lanczos_refine_tfim(nqs_ansatz_t *a, int Lx, int Ly,
                             double J, double Gamma,
                             int max_iters, double tol,
                             double *out_eigenvalue,
                             double *out_eigenvector,
                             lanczos_result_t *out_result);

/* Heisenberg XXZ (defaults to isotropic when Jz = J):
 *     H = J Σ_<ij> (S^x_i S^x_j + S^y_i S^y_j) + Jz Σ_<ij> S^z_i S^z_j
 * Open boundary conditions on an (Lx × Ly) lattice. */
int nqs_exact_energy_heisenberg(nqs_ansatz_t *a, int Lx, int Ly,
                                 double J, double Jz, double *out_energy);

int nqs_lanczos_refine_heisenberg(nqs_ansatz_t *a, int Lx, int Ly,
                                   double J, double Jz,
                                   int max_iters, double tol,
                                   double *out_eigenvalue,
                                   double *out_eigenvector,
                                   lanczos_result_t *out_result);

/* Variant of nqs_materialise_state that takes an explicit log_amp
 * callback (so Marshall / translation wrappers feed in). */
int nqs_materialise_state_with_cb(nqs_log_amp_fn_t log_amp, void *user,
                                   int Lx, int Ly,
                                   double **out_psi, long *out_dim);

#ifdef __cplusplus
}
#endif

#endif /* NQS_LANCZOS_H */
