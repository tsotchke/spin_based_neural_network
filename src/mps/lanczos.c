/*
 * src/mps/lanczos.c
 *
 * Matrix-free Lanczos with full re-orthogonalisation. The algorithm:
 *
 *   1. v_0 = normalised random vector (or user-supplied).
 *   2. Build the tridiagonal T_k ∈ R^{k×k} by the three-term recurrence
 *          α_j = <v_j, H v_j>
 *          w_j = H v_j - α_j v_j - β_j v_{j-1}
 *          β_{j+1} = ||w_j||
 *          v_{j+1} = w_j / β_{j+1}
 *      with full re-orthogonalisation of w_j against {v_0, ..., v_j}
 *      to suppress Ritz-vector drift.
 *   3. Diagonalise T_k via the QL algorithm (simple N^2 routine below);
 *      extract the smallest Ritz value.
 *   4. Iterate until ||H v - λ v|| < tol or k reaches max_iters.
 *
 * The Ritz vector is reconstructed at the end from the stored
 * Lanczos basis and the T eigenvector.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "mps/lanczos.h"

static double vec_dot(const double *a, const double *b, long n) {
    double s = 0.0;
    for (long i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static double vec_norm(const double *a, long n) {
    return sqrt(vec_dot(a, a, n));
}

static void vec_axpy(double *y, double alpha, const double *x, long n) {
    for (long i = 0; i < n; i++) y[i] += alpha * x[i];
}

static void vec_scale(double *y, double alpha, long n) {
    for (long i = 0; i < n; i++) y[i] *= alpha;
}

/* Simple tridiagonal QL algorithm, in-place. Writes eigenvalues into d
 * and, when z != NULL, transforms the supplied accumulator matrix z
 * (column-major, length n*n) accordingly. */
static void tridiag_ql(double *d, double *e, int n, double *z) {
    for (int l = 0; l < n; l++) {
        int iter = 0;
        int m;
        do {
            for (m = l; m < n - 1; m++) {
                double dd = fabs(d[m]) + fabs(d[m + 1]);
                if (fabs(e[m]) + dd == dd) break;
            }
            if (m != l) {
                if (iter++ == 60) return;  /* give up */
                double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                double r = sqrt(g * g + 1.0);
                g = d[m] - d[l] + e[l] / (g + (g >= 0 ? r : -r));
                double s = 1.0, c = 1.0, p = 0.0;
                int i;
                for (i = m - 1; i >= l; i--) {
                    double f = s * e[i];
                    double b = c * e[i];
                    if (fabs(f) >= fabs(g)) {
                        c = g / f;
                        r = sqrt(c * c + 1.0);
                        e[i + 1] = f * r;
                        s = 1.0 / r;
                        c *= s;
                    } else {
                        s = f / g;
                        r = sqrt(s * s + 1.0);
                        e[i + 1] = g * r;
                        c = 1.0 / r;
                        s *= c;
                    }
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i + 1] = g + p;
                    g = c * r - b;
                    if (z) {
                        for (int k = 0; k < n; k++) {
                            double z_ki   = z[(size_t)k * (size_t)n + i];
                            double z_kip1 = z[(size_t)k * (size_t)n + i + 1];
                            z[(size_t)k * (size_t)n + i + 1] = s * z_ki + c * z_kip1;
                            z[(size_t)k * (size_t)n + i    ] = c * z_ki - s * z_kip1;
                        }
                    }
                }
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }
}

int lanczos_smallest_with_init(lanczos_matvec_fn_t matvec, void *user_data,
                                long dim,
                                int max_iters, double tol,
                                const double *initial_vector,
                                double *out_eigenvector,
                                lanczos_result_t *out);

int lanczos_smallest(lanczos_matvec_fn_t matvec, void *user_data,
                     long dim,
                     int max_iters, double tol,
                     double *out_eigenvector,
                     lanczos_result_t *out) {
    return lanczos_smallest_with_init(matvec, user_data, dim,
                                       max_iters, tol, NULL,
                                       out_eigenvector, out);
}

int lanczos_smallest_with_init(lanczos_matvec_fn_t matvec, void *user_data,
                                long dim,
                                int max_iters, double tol,
                                const double *initial_vector,
                                double *out_eigenvector,
                                lanczos_result_t *out) {
    if (!matvec || dim <= 0 || max_iters <= 0) return -1;
    if (out) {
        out->eigenvalue = 0.0;
        out->iterations = 0;
        out->converged = 0;
        out->residual_norm = 0.0;
    }
    if (max_iters > dim) max_iters = (int)dim;

    /* Store the full Lanczos basis (max_iters × dim) for full reorth
     * and Ritz-vector reconstruction. */
    double *V = calloc((size_t)max_iters * (size_t)dim, sizeof(double));
    double *alpha = calloc((size_t)max_iters, sizeof(double));
    double *beta  = calloc((size_t)(max_iters + 1), sizeof(double));
    double *w     = calloc((size_t)dim, sizeof(double));
    if (!V || !alpha || !beta || !w) {
        free(V); free(alpha); free(beta); free(w); return -1;
    }

    /* Initial vector: caller-supplied or deterministic pseudo-random.
     * Uniform starts can lie in a subspace orthogonal to the smallest
     * eigenvector for Hamiltonians with high symmetry; in DMRG the
     * previous-iteration 2-site ground state is a far better seed. */
    if (initial_vector) {
        memcpy(V, initial_vector, (size_t)dim * sizeof(double));
    } else {
        unsigned long long rng = 0xA5A5A5A5A5A5A5A5ULL ^ (unsigned long long)dim;
        for (long i = 0; i < dim; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            double u = (double)(rng >> 11) / 9007199254740992.0;
            V[i] = u - 0.5;
        }
    }
    double nrm = vec_norm(V, dim);
    if (nrm < 1e-14) {
        /* Caller gave a degenerate init — fall back to deterministic. */
        unsigned long long rng = 0xBEEFBABEULL ^ (unsigned long long)dim;
        for (long i = 0; i < dim; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            double u = (double)(rng >> 11) / 9007199254740992.0;
            V[i] = u - 0.5;
        }
        nrm = vec_norm(V, dim);
    }
    if (nrm > 0) for (long i = 0; i < dim; i++) V[i] /= nrm;

    double lambda = 0.0;
    int k;
    double resid_norm = 0.0;

    /* Working copies for the tridiagonal solve, regenerated each step. */
    double *d = malloc((size_t)max_iters * sizeof(double));
    double *e = malloc((size_t)max_iters * sizeof(double));
    double *Z = malloc((size_t)max_iters * (size_t)max_iters * sizeof(double));
    if (!d || !e || !Z) {
        free(V); free(alpha); free(beta); free(w);
        free(d); free(e); free(Z);
        return -1;
    }

    for (k = 0; k < max_iters; k++) {
        double *v_k = &V[(size_t)k * (size_t)dim];
        matvec(v_k, w, dim, user_data);

        /* α_k = <v_k, H v_k> */
        alpha[k] = vec_dot(v_k, w, dim);

        /* w ← w - α_k v_k - β_k v_{k-1}  */
        vec_axpy(w, -alpha[k], v_k, dim);
        if (k > 0) {
            double *v_prev = &V[(size_t)(k - 1) * (size_t)dim];
            vec_axpy(w, -beta[k], v_prev, dim);
        }

        /* Full re-orthogonalisation. */
        for (int j = 0; j <= k; j++) {
            double *v_j = &V[(size_t)j * (size_t)dim];
            double p = vec_dot(v_j, w, dim);
            vec_axpy(w, -p, v_j, dim);
        }

        beta[k + 1] = vec_norm(w, dim);
        if (beta[k + 1] < 1e-14) { k++; break; }

        /* Build the current tridiagonal and extract the smallest Ritz. */
        int K = k + 1;
        for (int i = 0; i < K; i++) d[i] = alpha[i];
        for (int i = 0; i < K - 1; i++) e[i] = beta[i + 1];
        e[K - 1] = 0.0;
        /* Identity in Z. */
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++)
                Z[(size_t)i * (size_t)K + (size_t)j] = (i == j) ? 1.0 : 0.0;
        tridiag_ql(d, e, K, Z);

        /* Find smallest eigenvalue. */
        int idx_min = 0;
        for (int i = 1; i < K; i++) if (d[i] < d[idx_min]) idx_min = i;
        double lam_new = d[idx_min];

        /* Residual norm estimate for convergence: |β_{k+1}| · |z[K-1, idx_min]|
         * — standard Lanczos residual bound. */
        double z_last = Z[(size_t)(K - 1) * (size_t)K + (size_t)idx_min];
        resid_norm = fabs(beta[k + 1] * z_last);

        if (resid_norm < tol && k >= 1) {
            lambda = lam_new;
            k = K;
            if (out_eigenvector) {
                /* Reconstruct Ritz vector  v = Σ_j Z[j, idx_min] V_j. */
                memset(out_eigenvector, 0, (size_t)dim * sizeof(double));
                for (int j = 0; j < K; j++) {
                    double coeff = Z[(size_t)j * (size_t)K + (size_t)idx_min];
                    double *v_j = &V[(size_t)j * (size_t)dim];
                    vec_axpy(out_eigenvector, coeff, v_j, dim);
                }
                /* Renormalise. */
                double n2 = vec_norm(out_eigenvector, dim);
                if (n2 > 0) vec_scale(out_eigenvector, 1.0 / n2, dim);
            }
            if (out) {
                out->eigenvalue = lambda;
                out->iterations = K;
                out->converged = 1;
                out->residual_norm = resid_norm;
            }
            free(V); free(alpha); free(beta); free(w);
            free(d); free(e); free(Z);
            return 0;
        }
        lambda = lam_new;

        /* v_{k+1} = w / β_{k+1}  */
        if (k + 1 < max_iters) {
            double *v_next = &V[(size_t)(k + 1) * (size_t)dim];
            for (long i = 0; i < dim; i++) v_next[i] = w[i] / beta[k + 1];
        }
    }

    /* Not converged — recompute the best Ritz estimate from the full
     * K-dim tridiagonal and (optionally) materialise the eigenvector.
     * This runs unconditionally: the eigenvalue must reflect the full
     * Lanczos basis built during the loop, not the possibly-stale
     * intermediate Ritz value from the last in-loop step (which can
     * be wrong if the last step did an early β=0 break). */
    {
        int K = k;
        if (K < 1) K = 1;
        for (int i = 0; i < K; i++) d[i] = alpha[i];
        for (int i = 0; i < K - 1; i++) e[i] = beta[i + 1];
        e[K - 1] = 0.0;
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++)
                Z[(size_t)i * (size_t)K + (size_t)j] = (i == j) ? 1.0 : 0.0;
        tridiag_ql(d, e, K, Z);
        int idx_min = 0;
        for (int i = 1; i < K; i++) if (d[i] < d[idx_min]) idx_min = i;
        lambda = d[idx_min];
        if (out_eigenvector) {
            memset(out_eigenvector, 0, (size_t)dim * sizeof(double));
            for (int j = 0; j < K; j++) {
                double coeff = Z[(size_t)j * (size_t)K + (size_t)idx_min];
                double *v_j = &V[(size_t)j * (size_t)dim];
                vec_axpy(out_eigenvector, coeff, v_j, dim);
            }
            double n2 = vec_norm(out_eigenvector, dim);
            if (n2 > 0) vec_scale(out_eigenvector, 1.0 / n2, dim);
        }
    }
    if (out) {
        out->eigenvalue = lambda;
        out->iterations = k;
        out->converged = 0;
        out->residual_norm = resid_norm;
    }
    free(V); free(alpha); free(beta); free(w);
    free(d); free(e); free(Z);
    return 0;
}

/* k-smallest eigenvalue variant. No early exit: runs a full max_iters
 * steps so all k Ritz values are well-converged. Implementation is a
 * trimmed clone of lanczos_smallest_with_init with rank-k post-sort. */
int lanczos_k_smallest_with_init(lanczos_matvec_fn_t matvec, void *user_data,
                                  long dim,
                                  int max_iters,
                                  const double *initial_vector,
                                  int k,
                                  double *out_eigenvalues,
                                  lanczos_result_t *out) {
    if (!matvec || dim <= 0 || max_iters <= 0 || k <= 0 || !out_eigenvalues)
        return -1;
    if (max_iters > dim) max_iters = (int)dim;
    if (k > max_iters)   k = max_iters;
    if (out) {
        out->eigenvalue = 0.0;
        out->iterations = 0;
        out->converged = 0;
        out->residual_norm = 0.0;
    }

    double *V     = calloc((size_t)max_iters * (size_t)dim, sizeof(double));
    double *alpha = calloc((size_t)max_iters, sizeof(double));
    double *beta  = calloc((size_t)(max_iters + 1), sizeof(double));
    double *w     = calloc((size_t)dim, sizeof(double));
    if (!V || !alpha || !beta || !w) {
        free(V); free(alpha); free(beta); free(w); return -1;
    }

    if (initial_vector) {
        memcpy(V, initial_vector, (size_t)dim * sizeof(double));
    } else {
        unsigned long long rng = 0xA5A5A5A5A5A5A5A5ULL ^ (unsigned long long)dim;
        for (long i = 0; i < dim; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            double u = (double)(rng >> 11) / 9007199254740992.0;
            V[i] = u - 0.5;
        }
    }
    double nrm = vec_norm(V, dim);
    if (nrm < 1e-14) {
        unsigned long long rng = 0xBEEFBABEULL ^ (unsigned long long)dim;
        for (long i = 0; i < dim; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            double u = (double)(rng >> 11) / 9007199254740992.0;
            V[i] = u - 0.5;
        }
        nrm = vec_norm(V, dim);
    }
    if (nrm > 0) for (long i = 0; i < dim; i++) V[i] /= nrm;

    int kb;
    for (kb = 0; kb < max_iters; kb++) {
        double *v_k = &V[(size_t)kb * (size_t)dim];
        matvec(v_k, w, dim, user_data);
        alpha[kb] = vec_dot(v_k, w, dim);
        vec_axpy(w, -alpha[kb], v_k, dim);
        if (kb > 0) {
            double *v_prev = &V[(size_t)(kb - 1) * (size_t)dim];
            vec_axpy(w, -beta[kb], v_prev, dim);
        }
        for (int j = 0; j <= kb; j++) {
            double *v_j = &V[(size_t)j * (size_t)dim];
            double p = vec_dot(v_j, w, dim);
            vec_axpy(w, -p, v_j, dim);
        }
        beta[kb + 1] = vec_norm(w, dim);
        if (beta[kb + 1] < 1e-14) { kb++; break; }
        if (kb + 1 < max_iters) {
            double *v_next = &V[(size_t)(kb + 1) * (size_t)dim];
            for (long i = 0; i < dim; i++) v_next[i] = w[i] / beta[kb + 1];
        }
    }

    int K = kb;
    if (K < 1) K = 1;
    if (k > K) k = K;

    double *d = malloc((size_t)K * sizeof(double));
    double *e = malloc((size_t)K * sizeof(double));
    double *Z = malloc((size_t)K * (size_t)K * sizeof(double));
    if (!d || !e || !Z) {
        free(V); free(alpha); free(beta); free(w);
        free(d); free(e); free(Z); return -1;
    }
    for (int i = 0; i < K; i++) d[i] = alpha[i];
    for (int i = 0; i < K - 1; i++) e[i] = beta[i + 1];
    e[K - 1] = 0.0;
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++)
            Z[(size_t)i * (size_t)K + (size_t)j] = (i == j) ? 1.0 : 0.0;
    tridiag_ql(d, e, K, Z);

    /* Sort-ascending (simple selection; K ≤ max_iters ≤ few hundred). */
    for (int i = 0; i < K - 1; i++) {
        int imin = i;
        for (int j = i + 1; j < K; j++) if (d[j] < d[imin]) imin = j;
        if (imin != i) { double t = d[i]; d[i] = d[imin]; d[imin] = t; }
    }
    for (int i = 0; i < k; i++) out_eigenvalues[i] = d[i];

    if (out) {
        out->eigenvalue    = d[0];
        out->iterations    = K;
        /* Converged flag: use the k-th eigenvalue's residual proxy
         * (|β_K| × last component of its Ritz vector). Since we
         * re-sorted Z got shuffled — recompute from the unsorted
         * diagonalisation would be cleaner; this is a loose estimate. */
        out->converged     = (beta[K] < 1e-10) ? 1 : 0;
        out->residual_norm = beta[K];
    }

    free(V); free(alpha); free(beta); free(w);
    free(d); free(e); free(Z);
    return 0;
}

/* Sector-projected k-smallest Lanczos.  Identical to
 * lanczos_k_smallest_with_init except for an optional
 * `project(w, dim, user)` call inserted after the full
 * reorthogonalisation step in each Krylov iteration.
 *
 * Why this is needed: when seeking eigenvalues of H restricted to a
 * sector α (with H sector-preserving), naive Lanczos seeded by a
 * sector-pure vector should stay in α exactly.  In practice the
 * matvec carries machine-precision (~10⁻¹⁴) leakage out of α, and
 * the power-method-like Lanczos dynamics amplify the leak to the
 * global ground state over O(log dim · log eigen-ratio) iterations,
 * swamping the in-sector signal.  Projecting after each step pushes
 * the leak back to zero.  Pass `project = NULL` for stock behaviour. */
int lanczos_k_smallest_projected(lanczos_matvec_fn_t matvec, void *user_data,
                                  long dim,
                                  int max_iters,
                                  const double *initial_vector,
                                  int k,
                                  lanczos_project_fn_t project, void *project_user,
                                  double *out_eigenvalues,
                                  lanczos_result_t *out) {
    if (!matvec || dim <= 0 || max_iters <= 0 || k <= 0 || !out_eigenvalues)
        return -1;
    if (max_iters > dim) max_iters = (int)dim;
    if (k > max_iters)   k = max_iters;
    if (out) {
        out->eigenvalue = 0.0;
        out->iterations = 0;
        out->converged = 0;
        out->residual_norm = 0.0;
    }

    double *V     = calloc((size_t)max_iters * (size_t)dim, sizeof(double));
    double *alpha = calloc((size_t)max_iters, sizeof(double));
    double *beta  = calloc((size_t)(max_iters + 1), sizeof(double));
    double *w     = calloc((size_t)dim, sizeof(double));
    if (!V || !alpha || !beta || !w) {
        free(V); free(alpha); free(beta); free(w); return -1;
    }

    if (initial_vector) {
        memcpy(V, initial_vector, (size_t)dim * sizeof(double));
    } else {
        unsigned long long rng = 0xA5A5A5A5A5A5A5A5ULL ^ (unsigned long long)dim;
        for (long i = 0; i < dim; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            double u = (double)(rng >> 11) / 9007199254740992.0;
            V[i] = u - 0.5;
        }
        if (project) project(V, dim, project_user);
    }
    double nrm = vec_norm(V, dim);
    if (nrm < 1e-14) {
        unsigned long long rng = 0xBEEFBABEULL ^ (unsigned long long)dim;
        for (long i = 0; i < dim; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            double u = (double)(rng >> 11) / 9007199254740992.0;
            V[i] = u - 0.5;
        }
        if (project) project(V, dim, project_user);
        nrm = vec_norm(V, dim);
    }
    if (nrm > 0) for (long i = 0; i < dim; i++) V[i] /= nrm;

    int kb;
    for (kb = 0; kb < max_iters; kb++) {
        double *v_k = &V[(size_t)kb * (size_t)dim];
        matvec(v_k, w, dim, user_data);
        alpha[kb] = vec_dot(v_k, w, dim);
        vec_axpy(w, -alpha[kb], v_k, dim);
        if (kb > 0) {
            double *v_prev = &V[(size_t)(kb - 1) * (size_t)dim];
            vec_axpy(w, -beta[kb], v_prev, dim);
        }
        for (int j = 0; j <= kb; j++) {
            double *v_j = &V[(size_t)j * (size_t)dim];
            double p = vec_dot(v_j, w, dim);
            vec_axpy(w, -p, v_j, dim);
        }
        if (project) project(w, dim, project_user);
        beta[kb + 1] = vec_norm(w, dim);
        if (beta[kb + 1] < 1e-14) { kb++; break; }
        if (kb + 1 < max_iters) {
            double *v_next = &V[(size_t)(kb + 1) * (size_t)dim];
            for (long i = 0; i < dim; i++) v_next[i] = w[i] / beta[kb + 1];
        }
    }

    int K = kb;
    if (K < 1) K = 1;
    if (k > K) k = K;

    double *d = malloc((size_t)K * sizeof(double));
    double *e = malloc((size_t)K * sizeof(double));
    double *Z = malloc((size_t)K * (size_t)K * sizeof(double));
    if (!d || !e || !Z) {
        free(V); free(alpha); free(beta); free(w);
        free(d); free(e); free(Z); return -1;
    }
    for (int i = 0; i < K; i++) d[i] = alpha[i];
    for (int i = 0; i < K - 1; i++) e[i] = beta[i + 1];
    e[K - 1] = 0.0;
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++)
            Z[(size_t)i * (size_t)K + (size_t)j] = (i == j) ? 1.0 : 0.0;
    tridiag_ql(d, e, K, Z);

    for (int i = 0; i < K - 1; i++) {
        int imin = i;
        for (int j = i + 1; j < K; j++) if (d[j] < d[imin]) imin = j;
        if (imin != i) { double t = d[i]; d[i] = d[imin]; d[imin] = t; }
    }
    for (int i = 0; i < k; i++) out_eigenvalues[i] = d[i];

    if (out) {
        out->eigenvalue    = d[0];
        out->iterations    = K;
        out->converged     = (beta[K] < 1e-10) ? 1 : 0;
        out->residual_norm = beta[K];
    }

    free(V); free(alpha); free(beta); free(w);
    free(d); free(e); free(Z);
    return 0;
}

/* Sector-projected single-eigenvalue Lanczos.  Convergence-driven
 * variant of the projected k=1 case. */
int lanczos_smallest_projected(lanczos_matvec_fn_t matvec, void *user_data,
                                long dim,
                                int max_iters, double tol,
                                const double *initial_vector,
                                lanczos_project_fn_t project, void *project_user,
                                double *out_eigenvector,
                                lanczos_result_t *out) {
    if (!matvec || dim <= 0 || max_iters <= 0) return -1;
    if (out) {
        out->eigenvalue = 0.0;
        out->iterations = 0;
        out->converged = 0;
        out->residual_norm = 0.0;
    }
    if (max_iters > dim) max_iters = (int)dim;

    double *V = calloc((size_t)max_iters * (size_t)dim, sizeof(double));
    double *alpha = calloc((size_t)max_iters, sizeof(double));
    double *beta  = calloc((size_t)(max_iters + 1), sizeof(double));
    double *w     = calloc((size_t)dim, sizeof(double));
    if (!V || !alpha || !beta || !w) {
        free(V); free(alpha); free(beta); free(w); return -1;
    }

    if (initial_vector) {
        memcpy(V, initial_vector, (size_t)dim * sizeof(double));
    } else {
        unsigned long long rng = 0xA5A5A5A5A5A5A5A5ULL ^ (unsigned long long)dim;
        for (long i = 0; i < dim; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            double u = (double)(rng >> 11) / 9007199254740992.0;
            V[i] = u - 0.5;
        }
        if (project) project(V, dim, project_user);
    }
    double nrm = vec_norm(V, dim);
    if (nrm < 1e-14) {
        unsigned long long rng = 0xBEEFBABEULL ^ (unsigned long long)dim;
        for (long i = 0; i < dim; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            double u = (double)(rng >> 11) / 9007199254740992.0;
            V[i] = u - 0.5;
        }
        if (project) project(V, dim, project_user);
        nrm = vec_norm(V, dim);
    }
    if (nrm > 0) for (long i = 0; i < dim; i++) V[i] /= nrm;

    double lambda = 0.0;
    int k;
    double resid_norm = 0.0;

    double *d = malloc((size_t)max_iters * sizeof(double));
    double *e = malloc((size_t)max_iters * sizeof(double));
    double *Z = malloc((size_t)max_iters * (size_t)max_iters * sizeof(double));
    if (!d || !e || !Z) {
        free(V); free(alpha); free(beta); free(w);
        free(d); free(e); free(Z); return -1;
    }

    for (k = 0; k < max_iters; k++) {
        double *v_k = &V[(size_t)k * (size_t)dim];
        matvec(v_k, w, dim, user_data);
        alpha[k] = vec_dot(v_k, w, dim);
        vec_axpy(w, -alpha[k], v_k, dim);
        if (k > 0) {
            double *v_prev = &V[(size_t)(k - 1) * (size_t)dim];
            vec_axpy(w, -beta[k], v_prev, dim);
        }
        for (int j = 0; j <= k; j++) {
            double *v_j = &V[(size_t)j * (size_t)dim];
            double p = vec_dot(v_j, w, dim);
            vec_axpy(w, -p, v_j, dim);
        }
        if (project) project(w, dim, project_user);
        beta[k + 1] = vec_norm(w, dim);
        if (beta[k + 1] < 1e-14) { k++; break; }

        int K = k + 1;
        for (int i = 0; i < K; i++) d[i] = alpha[i];
        for (int i = 0; i < K - 1; i++) e[i] = beta[i + 1];
        e[K - 1] = 0.0;
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++)
                Z[(size_t)i * (size_t)K + (size_t)j] = (i == j) ? 1.0 : 0.0;
        tridiag_ql(d, e, K, Z);

        int idx_min = 0;
        for (int i = 1; i < K; i++) if (d[i] < d[idx_min]) idx_min = i;
        double lam_new = d[idx_min];
        double z_last = Z[(size_t)(K - 1) * (size_t)K + (size_t)idx_min];
        resid_norm = fabs(beta[k + 1] * z_last);

        if (resid_norm < tol && k >= 1) {
            lambda = lam_new;
            int Kf = K;
            if (out_eigenvector) {
                memset(out_eigenvector, 0, (size_t)dim * sizeof(double));
                for (int j = 0; j < Kf; j++) {
                    double coeff = Z[(size_t)j * (size_t)Kf + (size_t)idx_min];
                    double *v_j = &V[(size_t)j * (size_t)dim];
                    vec_axpy(out_eigenvector, coeff, v_j, dim);
                }
                double n2 = vec_norm(out_eigenvector, dim);
                if (n2 > 0) vec_scale(out_eigenvector, 1.0 / n2, dim);
            }
            if (out) {
                out->eigenvalue = lambda;
                out->iterations = Kf;
                out->converged = 1;
                out->residual_norm = resid_norm;
            }
            free(V); free(alpha); free(beta); free(w);
            free(d); free(e); free(Z);
            return 0;
        }
        lambda = lam_new;

        if (k + 1 < max_iters) {
            double *v_next = &V[(size_t)(k + 1) * (size_t)dim];
            for (long i = 0; i < dim; i++) v_next[i] = w[i] / beta[k + 1];
        }
    }

    /* Not converged — recompute Ritz from full basis. */
    int K = k;
    if (K < 1) {
        free(V); free(alpha); free(beta); free(w); free(d); free(e); free(Z);
        return -1;
    }
    for (int i = 0; i < K; i++) d[i] = alpha[i];
    for (int i = 0; i < K - 1; i++) e[i] = beta[i + 1];
    e[K - 1] = 0.0;
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++)
            Z[(size_t)i * (size_t)K + (size_t)j] = (i == j) ? 1.0 : 0.0;
    tridiag_ql(d, e, K, Z);
    int idx_min = 0;
    for (int i = 1; i < K; i++) if (d[i] < d[idx_min]) idx_min = i;
    lambda = d[idx_min];
    if (out_eigenvector) {
        memset(out_eigenvector, 0, (size_t)dim * sizeof(double));
        for (int j = 0; j < K; j++) {
            double coeff = Z[(size_t)j * (size_t)K + (size_t)idx_min];
            double *v_j = &V[(size_t)j * (size_t)dim];
            vec_axpy(out_eigenvector, coeff, v_j, dim);
        }
        double n2 = vec_norm(out_eigenvector, dim);
        if (n2 > 0) vec_scale(out_eigenvector, 1.0 / n2, dim);
    }
    if (out) {
        out->eigenvalue = lambda;
        out->iterations = K;
        out->converged = 0;
        out->residual_norm = resid_norm;
    }
    free(V); free(alpha); free(beta); free(w);
    free(d); free(e); free(Z);
    return 0;
}

/* ===========================================================================
 *  Continued-fraction Lanczos for spectral functions / dynamic correlators
 * =========================================================================*/

int lanczos_continued_fraction(lanczos_matvec_fn_t matvec, void *user_data,
                                long dim,
                                int max_iters,
                                const double *seed,
                                double *alpha, double *beta,
                                int *out_K,
                                double *out_seed_norm) {
    if (!matvec || dim <= 0 || max_iters <= 0 || !seed ||
        !alpha || !beta || !out_K)
        return -1;
    if (max_iters > dim) max_iters = (int)dim;

    double seed_norm = vec_norm(seed, dim);
    if (out_seed_norm) *out_seed_norm = seed_norm;
    if (seed_norm < 1e-300) {
        *out_K = 0;
        return 0;
    }

    /* Full reorthogonalisation: keep all Krylov vectors. */
    double *V = calloc((size_t)max_iters * (size_t)dim, sizeof(double));
    double *w = calloc((size_t)dim, sizeof(double));
    if (!V || !w) { free(V); free(w); return -1; }

    /* v_0 = seed / ||seed|| */
    double inv = 1.0 / seed_norm;
    for (long i = 0; i < dim; i++) V[i] = seed[i] * inv;
    beta[0] = 0.0;          /* β_0 unused in continued fraction */

    int k;
    for (k = 0; k < max_iters; k++) {
        double *v_k = &V[(size_t)k * (size_t)dim];
        matvec(v_k, w, dim, user_data);
        alpha[k] = vec_dot(v_k, w, dim);
        vec_axpy(w, -alpha[k], v_k, dim);
        if (k > 0) {
            double *v_prev = &V[(size_t)(k - 1) * (size_t)dim];
            vec_axpy(w, -beta[k], v_prev, dim);
        }
        /* Full reorth */
        for (int j = 0; j <= k; j++) {
            double *v_j = &V[(size_t)j * (size_t)dim];
            double p = vec_dot(v_j, w, dim);
            vec_axpy(w, -p, v_j, dim);
        }
        double bn = vec_norm(w, dim);
        if (k + 1 < max_iters) beta[k + 1] = bn;
        if (bn < 1e-14) { k++; break; }
        if (k + 1 < max_iters) {
            double *v_next = &V[(size_t)(k + 1) * (size_t)dim];
            for (long i = 0; i < dim; i++) v_next[i] = w[i] / bn;
        }
    }
    *out_K = k;

    free(V); free(w);
    return 0;
}

void lanczos_cf_evaluate(int K, const double *alpha, const double *beta,
                          double seed_norm,
                          double omega, double eta,
                          double *out_re, double *out_im) {
    if (K <= 0 || !alpha || !beta || !out_re || !out_im) {
        if (out_re) *out_re = 0.0;
        if (out_im) *out_im = 0.0;
        return;
    }
    /* Build the continued fraction inside-out.
     *   d_K = z − α_{K-1}                                (innermost)
     *   d_k = z − α_{k-1} − β_k² / d_{k+1}                for k = K-1..1
     *   result = ‖φ‖² / d_1                              (outermost)
     * with z = ω + iη (complex). */
    double zr = omega, zi = eta;
    double dr = zr - alpha[K - 1];
    double di = zi;
    for (int k = K - 1; k >= 1; k--) {
        /* term = β_k² / d  where d = (dr, di) is the previously-built
         * fraction tail. */
        double bk2 = beta[k] * beta[k];
        double mag2 = dr * dr + di * di;
        double tr = bk2 * dr / mag2;
        double ti = -bk2 * di / mag2;     /* β_k²/d, complex inverse */
        dr = (zr - alpha[k - 1]) - tr;
        di = zi - ti;
    }
    /* result = seed_norm² / (dr + i di) */
    double mag2 = dr * dr + di * di;
    *out_re =  seed_norm * seed_norm * dr / mag2;
    *out_im = -seed_norm * seed_norm * di / mag2;
}

/* ===========================================================================
 *  Memory-lean projecting Lanczos for large-dim sector ED.
 *
 *  Three-term recurrence with NO Krylov-basis storage — only three
 *  vectors live at a time (v_curr, v_prev, w).  In-loop sector
 *  projection at every step kills machine-precision sector leakage
 *  (the same fix as lanczos_smallest_projected, but without the
 *  O(max_iters · dim) memory cost of full reorthogonalisation).
 *
 *  Trade-off: no eigenvector reconstruction (would need V[]); does
 *  not produce sub-extremal Ritz values reliably (3-term recurrence
 *  loses orthogonality and develops "ghost" copies).  Just E_0.
 *
 *  Use case: kagome 3×3 PBC (N=27, dim=2^27, 1 GB / vector).  Full-
 *  reorth Lanczos at 300 iters needs 300 GB; this needs ~3 GB.
 * =========================================================================*/

int lanczos_smallest_projected_lean(lanczos_matvec_fn_t matvec, void *user_data,
                                     long dim,
                                     int max_iters, double tol,
                                     const double *initial_vector,
                                     lanczos_project_fn_t project, void *project_user,
                                     double *out_eigenvalue,
                                     lanczos_result_t *out) {
    if (!matvec || dim <= 0 || max_iters <= 0) return -1;
    if (out) {
        out->eigenvalue = 0.0;
        out->iterations = 0;
        out->converged = 0;
        out->residual_norm = 0.0;
    }
    if (max_iters > dim) max_iters = (int)dim;

    double *v_curr = calloc((size_t)dim, sizeof(double));
    double *v_prev = calloc((size_t)dim, sizeof(double));
    double *w      = calloc((size_t)dim, sizeof(double));
    double *alpha  = calloc((size_t)max_iters, sizeof(double));
    double *beta   = calloc((size_t)(max_iters + 1), sizeof(double));
    double *d      = malloc((size_t)max_iters * sizeof(double));
    double *e      = malloc((size_t)max_iters * sizeof(double));
    if (!v_curr || !v_prev || !w || !alpha || !beta || !d || !e) {
        free(v_curr); free(v_prev); free(w);
        free(alpha); free(beta); free(d); free(e);
        return -1;
    }

    /* Initial vector: caller-supplied or deterministic random. */
    if (initial_vector) {
        memcpy(v_curr, initial_vector, (size_t)dim * sizeof(double));
    } else {
        unsigned long long rng = 0xA5A5A5A5A5A5A5A5ULL ^ (unsigned long long)dim;
        for (long i = 0; i < dim; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            double u = (double)(rng >> 11) / 9007199254740992.0;
            v_curr[i] = u - 0.5;
        }
    }
    if (project) project(v_curr, dim, project_user);
    double n = vec_norm(v_curr, dim);
    if (n < 1e-14) {
        /* Project killed everything — try a different deterministic seed. */
        unsigned long long rng = 0xBEEFBABEULL ^ (unsigned long long)dim;
        for (long i = 0; i < dim; i++) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            double u = (double)(rng >> 11) / 9007199254740992.0;
            v_curr[i] = u - 0.5;
        }
        if (project) project(v_curr, dim, project_user);
        n = vec_norm(v_curr, dim);
    }
    if (n > 0) for (long i = 0; i < dim; i++) v_curr[i] /= n;

    double lambda = 0.0;
    double prev_lambda = 0.0;
    int k;
    int converged = 0;

    for (k = 0; k < max_iters; k++) {
        /* w = H v_curr */
        matvec(v_curr, w, dim, user_data);

        alpha[k] = vec_dot(v_curr, w, dim);

        /* w ← w − α_k v_curr − β_k v_prev */
        vec_axpy(w, -alpha[k], v_curr, dim);
        if (k > 0) vec_axpy(w, -beta[k], v_prev, dim);

        /* In-loop sector projection — kills the leak. */
        if (project) project(w, dim, project_user);

        beta[k + 1] = vec_norm(w, dim);
        if (beta[k + 1] < 1e-14) { k++; break; }

        /* Diagonalise the current tridiagonal to extract Ritz value. */
        int K = k + 1;
        for (int i = 0; i < K; i++) d[i] = alpha[i];
        for (int i = 0; i < K - 1; i++) e[i] = beta[i + 1];
        e[K - 1] = 0.0;
        tridiag_ql(d, e, K, NULL);
        int idx = 0;
        for (int i = 1; i < K; i++) if (d[i] < d[idx]) idx = i;
        prev_lambda = lambda;
        lambda = d[idx];

        /* Convergence based on Ritz-value stability (no eigenvector
         * residual since we don't have it). */
        if (k > 5 && fabs(lambda - prev_lambda) < tol) {
            converged = 1;
            k++;
            break;
        }

        /* Slide vectors:  v_prev ← v_curr;  v_curr ← w / β; */
        memcpy(v_prev, v_curr, (size_t)dim * sizeof(double));
        double inv_b = 1.0 / beta[k + 1];
        for (long i = 0; i < dim; i++) v_curr[i] = w[i] * inv_b;
    }

    if (out_eigenvalue) *out_eigenvalue = lambda;
    if (out) {
        out->eigenvalue    = lambda;
        out->iterations    = k;
        out->converged     = converged;
        out->residual_norm = (k > 0) ? fabs(lambda - prev_lambda) : 0.0;
    }
    free(v_curr); free(v_prev); free(w); free(alpha); free(beta); free(d); free(e);
    return 0;
}
