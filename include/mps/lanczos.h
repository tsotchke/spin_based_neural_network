/*
 * include/mps/lanczos.h
 *
 * Matrix-free Lanczos eigensolver. Given a callback that computes
 * H·v for any vector v, finds the smallest (algebraic) eigenvalue and
 * a corresponding eigenvector of H.
 *
 * This is the shared substrate for:
 *   - DMRG ground-state sweeps (P2.2): each two-site eigenproblem is
 *     solved via Lanczos on the effective Hamiltonian.
 *   - NQS Lanczos post-processing (P2.6): the Krylov subspace is built
 *     from the trained wavefunction to squeeze another order of
 *     magnitude of energy accuracy out of it.
 *   - Exact-diagonalisation reference computations for testing.
 *
 * The implementation uses classical three-term recurrence with full
 * re-orthogonalisation against all previous Krylov vectors. For
 * dim ≤ 10^6 and krylov_dim ≤ 200 this is memory-friendly and stable.
 */
#ifndef MPS_LANCZOS_H
#define MPS_LANCZOS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Caller-supplied H·v product:
 *   in:  input vector of length `dim`
 *   out: output vector H·in, length `dim`
 *   user_data: opaque pointer passed from the Lanczos driver
 */
typedef void (*lanczos_matvec_fn_t)(const double *in, double *out,
                                    long dim, void *user_data);

typedef struct {
    double eigenvalue;       /* smallest eigenvalue found */
    int    iterations;       /* Krylov steps actually used */
    int    converged;        /* 1 if the residual norm dropped below tol */
    double residual_norm;    /* ||H v - lambda v|| of the returned vector */
} lanczos_result_t;

/* Solve for the smallest eigenvalue of H (symmetric / Hermitian real-valued).
 *
 *   matvec          — user-supplied matrix-free H·v callback
 *   user_data       — forwarded to matvec
 *   dim             — Hilbert-space dimension
 *   max_iters       — cap on Krylov-subspace dimension (≤ dim)
 *   tol             — residual-norm target for convergence
 *   out_eigenvector — length-dim buffer receiving the eigenvector (may be NULL)
 *   out             — filled with the run's summary statistics
 *
 * Returns 0 on success. */
int lanczos_smallest(lanczos_matvec_fn_t matvec, void *user_data,
                     long dim,
                     int max_iters, double tol,
                     double *out_eigenvector,
                     lanczos_result_t *out);

/* Same, but seeds the Krylov subspace from `initial_vector` (length
 * dim). If NULL, falls back to the deterministic xorshift init. The
 * vector is normalised in-place. This matters for DMRG warm-starts
 * where the previous-iteration ground state is a much better seed
 * than random. */
int lanczos_smallest_with_init(lanczos_matvec_fn_t matvec, void *user_data,
                                long dim,
                                int max_iters, double tol,
                                const double *initial_vector,
                                double *out_eigenvector,
                                lanczos_result_t *out);

/* Multi-eigenvalue variant — extract the k smallest Ritz values after
 * running a full Krylov subspace of dimension max_iters. The rank-1
 * residual-norm early exit is disabled here because convergence of
 * *all* k smallest eigenvalues (particularly the gap E_1 − E_0) needs
 * a larger basis than convergence of just E_0.
 *
 *   initial_vector  — seed (may be NULL)
 *   k               — number of smallest eigenvalues to extract
 *   out_eigenvalues — caller-owned length-k array, filled in
 *                     ascending order on success
 *   out             — summary (iterations = Krylov dim actually built;
 *                     eigenvalue / residual_norm refer to the smallest
 *                     Ritz value and its bound)
 *
 * For gap estimation on small Hilbert spaces (dim ≤ 10⁴) set
 * max_iters = min(dim, 200) and k = 2–5. Returns 0 on success. */
int lanczos_k_smallest_with_init(lanczos_matvec_fn_t matvec, void *user_data,
                                  long dim,
                                  int max_iters,
                                  const double *initial_vector,
                                  int k,
                                  double *out_eigenvalues,
                                  lanczos_result_t *out);

#ifdef __cplusplus
}
#endif

#endif /* MPS_LANCZOS_H */
