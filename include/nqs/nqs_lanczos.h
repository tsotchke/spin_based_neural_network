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

/* Kagome Heisenberg S=½ on a (Lx_cells × Ly_cells) cluster with three
 * sublattices per unit cell (N = 3·Lx_cells·Ly_cells sites, 24 bonds
 * at the 2×2 PBC point).
 *
 *   H = J Σ_<ij> S_i · S_j
 *
 * Off-diagonal (S^+S^-/S^-S^+) matrix elements are real, so even for
 * complex-amplitude ansätze (e.g. the cRBM trained by holomorphic SR)
 * the Hermitian H decouples into two independent real eigenproblems
 * on Re(ψ) and Im(ψ) with the same spectrum. The Lanczos refiner
 * therefore seeds from Re(ψ) = |ψ|·cos(arg ψ) — this picks up the
 * Marshall-like sign structure observed on trained kagome cRBMs (see
 * `nqs_compute_kagome_bond_phase`). */
int nqs_exact_energy_kagome_heisenberg(nqs_ansatz_t *a,
                                        int Lx_cells, int Ly_cells,
                                        double J, int pbc,
                                        double *out_energy);

/* Callback variant of nqs_exact_energy_kagome_heisenberg.  Same dense
 * ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ computation, but the wavefunction comes from an
 * arbitrary log-amplitude callback (e.g. nqs_symproj_log_amp wrapping
 * a trained ansatz).  Used to read off the Monte-Carlo-noise-free
 * variational energy of the projected wavefunction before running
 * Lanczos refinement on top. */
int nqs_exact_energy_kagome_heisenberg_with_cb(nqs_log_amp_fn_t log_amp,
                                                 void *user,
                                                 int Lx_cells, int Ly_cells,
                                                 double J, int pbc,
                                                 double *out_energy);

int nqs_lanczos_refine_kagome_heisenberg(nqs_ansatz_t *a,
                                          int Lx_cells, int Ly_cells,
                                          double J, int pbc,
                                          int max_iters, double tol,
                                          double *out_eigenvalue,
                                          double *out_eigenvector,
                                          lanczos_result_t *out_result);

/* Callback variant: seeds Lanczos from an arbitrary log-amplitude
 * source (e.g. a sector-projection wrapper around a trained ansatz).
 * Identical semantics to the ansatz-based variant otherwise — same
 * Hamiltonian, same dim = 2^N, same Krylov solver. The point is to
 * let the projected wavefunction ψ_sym(s), not the bare base ansatz
 * ψ_base(s), serve as the Krylov seed when training has been done in
 * a non-trivial irrep sector. */
int nqs_lanczos_refine_kagome_heisenberg_with_cb(nqs_log_amp_fn_t log_amp,
                                                  void *user,
                                                  int Lx_cells, int Ly_cells,
                                                  double J, int pbc,
                                                  int max_iters, double tol,
                                                  double *out_eigenvalue,
                                                  double *out_eigenvector,
                                                  lanczos_result_t *out_result);

/* k-lowest-eigenvalue variant for kagome Heisenberg. Returns the k
 * smallest Ritz values of H seeded from the trained ansatz's Re(ψ).
 * The gap E_1 − E_0 lands naturally in `out_eigenvalues[1] -
 * out_eigenvalues[0]` once both have converged. */
int nqs_lanczos_k_lowest_kagome_heisenberg(nqs_ansatz_t *a,
                                            int Lx_cells, int Ly_cells,
                                            double J, int pbc,
                                            int max_iters, int k,
                                            double *out_eigenvalues,
                                            lanczos_result_t *out_result);

/* Callback variant of k-lowest.  Lanczos's Krylov subspace
 * commutes with H, so when seeded from a sector-projected ψ_sym (e.g.
 * via nqs_symproj_log_amp on a trained ansatz) the returned k-lowest
 * eigenvalues are the lowest k of H restricted to that sector.  This
 * is the natural way to extract the low-energy spectrum within a
 * named irrep sector without needing a sector-resolved sparse
 * Hamiltonian. */
int nqs_lanczos_k_lowest_kagome_heisenberg_with_cb(nqs_log_amp_fn_t log_amp,
                                                    void *user,
                                                    int Lx_cells, int Ly_cells,
                                                    double J, int pbc,
                                                    int max_iters, int k,
                                                    double *out_eigenvalues,
                                                    lanczos_result_t *out_result);

/* Variant of nqs_materialise_state that takes an explicit log_amp
 * callback (so Marshall / translation wrappers feed in). */
int nqs_materialise_state_with_cb(nqs_log_amp_fn_t log_amp, void *user,
                                   int Lx, int Ly,
                                   double **out_psi, long *out_dim);

/* Variant of nqs_materialise_state_with_cb for lattices whose site
 * count is not size_x · size_y (kagome has 3 sublattices per cell, so
 * N = 3·Lx·Ly). Required for kagome Lanczos refinement of the
 * complex-RBM trained by the holomorphic SR + kagome Heisenberg
 * kernel. Real-valued output: stores Re(ψ) = |ψ|·cos(arg ψ), which
 * is the Marshall-aligned seed for the subsequent Krylov iteration. */
int nqs_materialise_state_with_cb_N(nqs_log_amp_fn_t log_amp, void *user,
                                     int N,
                                     double **out_psi, long *out_dim);

#ifdef __cplusplus
}
#endif

#endif /* NQS_LANCZOS_H */
