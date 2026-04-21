/*
 * include/mps/svd.h
 *
 * Dense singular value decomposition via one-sided Jacobi rotations.
 * Target regime: m, n ≤ ~256, the sizes that appear when splitting a
 * two-site tensor in a DMRG sweep with bond dimension ≤ 64. Pure C,
 * no BLAS/LAPACK dependency.
 *
 *     A = U · Σ · V^T      A ∈ R^{m×n}, m ≥ n
 *
 * U is m×n, Σ is an n-vector of descending non-negative singular
 * values, V is n×n. All buffers are row-major and caller-allocated.
 *
 * The algorithm sweeps over (i,j) pairs (i < j < n) applying Givens
 * rotations that zero the off-diagonal dot product a_i·a_j in the
 * *working matrix* U ← U · R. Convergence detection: sum of squared
 * off-diagonals < tol × trace. Typical iteration count: O(log(1/tol))
 * sweeps, which is 4-6 for tol = 1e-12 at double precision.
 */
#ifndef MPS_SVD_H
#define MPS_SVD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compute A = U Σ V^T by one-sided Jacobi on the columns of A.
 *   a_in: m × n matrix, row-major. Unchanged on return.
 *   U:    m × n output (caller buffer of size m·n).
 *   s:    length-n output (singular values, descending).
 *   Vt:   n × n output (V transposed, row-major).
 *   tol:  off-diagonal stopping tolerance relative to ||A||_F².
 * Returns 0 on success, -1 on argument error.
 */
int svd_jacobi(const double *a_in, int m, int n,
               double *U, double *s, double *Vt,
               double tol);

#ifdef __cplusplus
}
#endif

#endif /* MPS_SVD_H */
