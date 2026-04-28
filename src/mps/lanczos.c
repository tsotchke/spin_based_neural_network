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
