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

/* Optional projection callback applied to each Krylov vector after the
 * matvec + reorthogonalisation step.  Used to enforce sector restriction
 * when H is sector-preserving but numerical leakage at machine precision
 * gets amplified by the power-method-like Lanczos dynamics into the
 * dominant eigenvector outside the sector.  In particular, for symmetry-
 * projected ED on small clusters, without this projection step the Krylov
 * basis drifts into the global ground state regardless of seed sector. */
typedef void (*lanczos_project_fn_t)(double *vec, long dim, void *user);

/* Sector-projected k-smallest Lanczos: same as
 * lanczos_k_smallest_with_init but applies `project(w, dim, project_user)`
 * to the Krylov vector after each step.  Pass `project = NULL` to recover
 * the un-projected behaviour exactly. */
int lanczos_k_smallest_projected(lanczos_matvec_fn_t matvec, void *user_data,
                                  long dim,
                                  int max_iters,
                                  const double *initial_vector,
                                  int k,
                                  lanczos_project_fn_t project, void *project_user,
                                  double *out_eigenvalues,
                                  lanczos_result_t *out);

/* Sector-projected single-eigenvalue Lanczos. */
int lanczos_smallest_projected(lanczos_matvec_fn_t matvec, void *user_data,
                                long dim,
                                int max_iters, double tol,
                                const double *initial_vector,
                                lanczos_project_fn_t project, void *project_user,
                                double *out_eigenvector,
                                lanczos_result_t *out);

/* Memory-lean variant: 3-term recurrence with NO Krylov-basis storage —
 * only three vectors live at a time (current, previous, work).  In-loop
 * sector projection at every step kills machine-precision leakage.
 *
 * Trade-offs vs lanczos_smallest_projected:
 *   - O(3·dim) memory instead of O(max_iters·dim).  Required at large
 *     dim where full reorth would need TB-scale RAM.
 *   - No eigenvector reconstruction.  Returns lowest Ritz value only.
 *   - 3-term recurrence loses orthogonality at machine precision over
 *     long Krylov runs; "ghost" copies of converged eigenvalues
 *     appear in the spectrum.  E_0 is robust; sub-extremal Ritz values
 *     are not.
 *
 * Convergence: Ritz-value stability over consecutive iterations
 * (|λ_k − λ_{k-1}| < tol after k > 5 warm-up iters). */
int lanczos_smallest_projected_lean(lanczos_matvec_fn_t matvec, void *user_data,
                                     long dim,
                                     int max_iters, double tol,
                                     const double *initial_vector,
                                     lanczos_project_fn_t project, void *project_user,
                                     double *out_eigenvalue,
                                     lanczos_result_t *out);

/* Run Lanczos starting from `seed` and return the tridiagonal
 * coefficients α[k], β[k] for k = 0..K-1 plus the seed norm.  Does
 * not extract eigenvalues — caller uses the (α, β) pair directly to
 * evaluate the spectral function via continued fraction:
 *
 *     ⟨φ | (ω − H + iη)^{−1} | φ⟩ = ‖φ‖²
 *         / (ω + iη − α₀ − β₁² / (ω + iη − α₁ − β₂² / …))
 *
 * Used by dynamic-correlator computations (S(q, ω), spectral
 * functions, Green's functions).  Full reorthogonalisation against
 * the entire Krylov basis is enabled, which doubles memory but
 * keeps the (α, β) pairs accurate against numerical drift over
 * hundreds of iterations.
 *
 * On entry: alpha and beta are caller-allocated, length ≥ max_iters.
 * On return: out_K = number of Krylov steps actually run (≤ max_iters);
 *            out_seed_norm = ‖seed‖ for the b₀ in the continued
 *            fraction.  May be zero if seed is null. */
int lanczos_continued_fraction(lanczos_matvec_fn_t matvec, void *user_data,
                                long dim,
                                int max_iters,
                                const double *seed,
                                double *alpha, double *beta,
                                int *out_K,
                                double *out_seed_norm);

/* Evaluate the continued fraction
 *     ‖φ‖² / (z − α₀ − β₁² / (z − α₁ − β₂² / …))
 * at z = ω + iη.  K is the number of (α, β) pairs from the Lanczos
 * run; seed_norm = ‖φ‖.  Returns the complex value via real/imag
 * outputs.  Standard Stieltjes-fraction recursion from inside-out. */
void lanczos_cf_evaluate(int K, const double *alpha, const double *beta,
                          double seed_norm,
                          double omega, double eta,
                          double *out_re, double *out_im);

#ifdef __cplusplus
}
#endif

#endif /* MPS_LANCZOS_H */
