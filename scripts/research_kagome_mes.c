/*
 * scripts/research_kagome_mes.c
 *
 * Empirical extraction of the lattice modular S matrix on the kagome
 * AFM Heisenberg ground-state manifold via the Zhang-Grover-Vishwanath
 * 2012 Minimum-Entropy-State protocol.
 *
 * Methodology:
 *   1.  Load 4 sector ground-state eigvecs |ψ_α⟩ (α ∈ {A_1,A_2,B_1,B_2}).
 *   2.  Define two bipartitions of the torus:
 *         - Cut X: subsystem A = sites in the strip cy = 0 (cycle along y)
 *         - Cut Y: subsystem A = sites in the strip cx = 0 (cycle along x)
 *       Each cut surrounds a non-contractible cycle.
 *   3.  For each cut and each unit vector α ∈ S^{2(K-1)+1} (K=4 sector
 *       states; for real ψ → S^3 in ℝ^4), compute the bipartite von
 *       Neumann entropy of |ψ(α)⟩ = Σ_a α_a |ψ_a⟩.
 *   4.  Search for the K minima of S — these are the K MES (minimum-
 *       entropy states) for that cut.
 *   5.  The unitary U_x→y mapping MES_X to MES_Y is the empirical
 *       lattice modular S in the MES basis.
 *
 * Symbolic prediction (KagomeZ2.{wl,py}):
 *   S_{Z_2 TC} = (1/2) · [[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]]
 * which is (1/2)·Hadamard_4.  Empirical |U_x→y| should match this up
 * to permutations and gauge phases.
 *
 * Build:  make IRREP_ENABLE=1 OPENMP=1 research_kagome_mes
 * Run:    env OMP_NUM_THREADS=14 ./build/research_kagome_mes
 *
 * NOTE: requires libirrep for partial_trace + hermitian eigvals.
 *       Caveat at finite N=27: 4 sector ground states are NOT exactly
 *       degenerate (E spread ~0.18 J); MES search is over the
 *       symmetry-projected sector basis and the resulting lattice S
 *       matrix is approximate — its closeness to the symbolic
 *       (1/2)·Hadamard_4 is the testable prediction.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <sys/time.h>

#include "libirrep_bridge.h"
#include <irrep/irrep.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double wall_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

/* Maximum supported number of sector states.  Runtime K ≤ MAX_K is
 * read from argv (see main).  MAX_K caps the fixed-size scratch arrays
 * in the sample_t struct and the alpha vector.  Bumping it to 12 would
 * cover most realistic anyon-model GS manifolds without recompiling. */
#define MAX_K 8
/* For backwards compat the legacy code referred to a fixed `K`; we keep
 * the symbol but resolve it to a runtime variable inside main(). */

static int load_eigvec_real(const char *path, int *out_N, int *out_ir,
                              double *out_E0, double **out_psi, long *out_dim) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int header[4]; double meta[2];
    if (fread(header, sizeof(int), 4, f) != 4) { fclose(f); return -1; }
    if (fread(meta, sizeof(double), 2, f) != 2) { fclose(f); return -1; }
    int N = header[0], ir = header[2];
    long dim = (long)meta[1];
    if (dim != (1L << N)) { fclose(f); return -1; }
    double *psi = malloc((size_t)dim * sizeof(double));
    if (!psi) { fclose(f); return -1; }
    if ((long)fread(psi, sizeof(double), (size_t)dim, f) != dim) {
        free(psi); fclose(f); return -1;
    }
    fclose(f);
    *out_N = N; *out_ir = ir; *out_E0 = meta[0];
    *out_psi = psi; *out_dim = dim;
    return 0;
}

/* Real-input partial trace, OpenMP-parallel over the β bucket loop.
 * Avoids the 2 GB complex copy of psi.  ψ is real, so ρ_A is real-symmetric
 * and we accumulate only the upper triangle, then mirror to full rho_c
 * (complex output for libirrep eigvalsh).
 *
 * Per thread we allocate a private rho_part[dimA*dimA] of doubles
 * (= 0.5M·8 B = 4 MB at nA=9).  14 threads → 56 MB scratch.  No global
 * locks needed; final reduction is serial over thread arrays.
 */
static int partial_trace_real_omp(const double *psi_real, int N, long dim,
                                   const int *sites_A, int nA,
                                   double _Complex *rho_c) {
    (void)dim;
    int local_dim = 2;
    int in_A[32] = {0};
    for (int k = 0; k < nA; k++) in_A[sites_A[k]] = 1;
    int nB = N - nA;
    int sites_B[32]; { int bi = 0;
        for (int s = 0; s < N; s++) if (!in_A[s]) sites_B[bi++] = s;
    }
    long dA = 1L << nA;
    long dB = 1L << nB;
    long weight[32];
    for (int s = 0; s < N; s++) weight[s] = 1L << s;

    int nthreads = 1;
    #ifdef _OPENMP
    #pragma omp parallel
    { if (omp_get_thread_num() == 0) nthreads = omp_get_num_threads(); }
    #endif

    double *rho_thr = calloc((size_t)nthreads * dA * dA, sizeof(double));
    if (!rho_thr) return -1;

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        double *rho_local = rho_thr + (size_t)tid * dA * dA;
        double *v = malloc((size_t)dA * sizeof(double));
        if (v) {
            #ifdef _OPENMP
            #pragma omp for schedule(static)
            #endif
            for (long beta = 0; beta < dB; beta++) {
                long b_remaining = beta;
                int b_digits[32] = {0};
                for (int k = 0; k < nB; k++) {
                    b_digits[k] = (int)(b_remaining % local_dim);
                    b_remaining /= local_dim;
                }
                long i_base = 0;
                for (int k = 0; k < nB; k++)
                    i_base += (long)b_digits[k] * weight[sites_B[k]];

                for (long alpha = 0; alpha < dA; alpha++) {
                    long a_remaining = alpha;
                    long offset = 0;
                    for (int k = 0; k < nA; k++) {
                        int d = (int)(a_remaining % local_dim);
                        a_remaining /= local_dim;
                        offset += (long)d * weight[sites_A[k]];
                    }
                    v[alpha] = psi_real[i_base + offset];
                }

                /* Accumulate upper triangle: rho_local[a,c] += v[a]*v[c]. */
                for (long a = 0; a < dA; a++) {
                    double va = v[a];
                    double *row = rho_local + a * dA;
                    for (long c = a; c < dA; c++) row[c] += va * v[c];
                }
            }
            free(v);
        }
    }

    /* Reduce per-thread upper triangles into a single double matrix,
     * then symmetrise and copy into the complex output rho_c. */
    double *rho = calloc((size_t)dA * dA, sizeof(double));
    if (!rho) { free(rho_thr); return -1; }
    for (int t = 0; t < nthreads; t++) {
        double *src = rho_thr + (size_t)t * dA * dA;
        for (long a = 0; a < dA; a++)
            for (long c = a; c < dA; c++)
                rho[a*dA + c] += src[a*dA + c];
    }
    for (long a = 0; a < dA; a++)
        for (long c = a; c < dA; c++) {
            double v = rho[a*dA + c];
            rho_c[a*dA + c] = v + 0.0*I;
            if (c != a) rho_c[c*dA + a] = v + 0.0*I;
        }
    free(rho); free(rho_thr);
    return 0;
}

/* Compute bipartite von Neumann entropy S of ψ_real (length dim) on the
 * subsystem given by sites_A (size nA). */
static int entropy_of_real_state(const double *psi_real, int N, long dim,
                                  const int *sites_A, int nA, double *out_S) {
    long dimA = 1L << nA;
    (void)dim;
    double _Complex *rho = malloc((size_t)dimA * dimA * sizeof(double _Complex));
    if (!rho) return -1;
    int rc = partial_trace_real_omp(psi_real, N, dim, sites_A, nA, rho);
    if (rc != 0) { free(rho); return rc; }
    double *eigs = malloc((size_t)dimA * sizeof(double));
    if (!eigs) { free(rho); return -1; }
    if (irrep_hermitian_eigvals((int)dimA, rho, eigs) != IRREP_OK) {
        free(rho); free(eigs); return -1;
    }
    double S = 0.0;
    for (long j = 0; j < dimA; j++) {
        double l = eigs[j];
        if (l > 1e-15) S -= l * log(l);
    }
    *out_S = S;
    free(rho); free(eigs);
    return 0;
}

/* Form the linear combination ψ(α) = Σ_a α_a ψ_a, real. */
static void form_combination(const double * const *psi_a, const double *alpha,
                               int K_states, long dim, double *out) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (long s = 0; s < dim; s++) {
        double v = 0.0;
        for (int a = 0; a < K_states; a++) v += alpha[a] * psi_a[a][s];
        out[s] = v;
    }
    /* Normalise */
    double norm2 = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:norm2) schedule(static)
    #endif
    for (long s = 0; s < dim; s++) norm2 += out[s] * out[s];
    if (norm2 > 0.0) {
        double inv = 1.0 / sqrt(norm2);
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (long s = 0; s < dim; s++) out[s] *= inv;
    }
}

/* Generate a coarse grid of unit K-vectors on S^{K-1} ⊂ ℝ^K via
 * spherical coordinates with K-1 angles:
 *   α_0     = cos(θ_1)
 *   α_1     = sin(θ_1) cos(θ_2)
 *   α_k     = sin(θ_1)...sin(θ_k) cos(θ_{k+1})    (1 ≤ k ≤ K-2)
 *   α_{K-1} = sin(θ_1)...sin(θ_{K-2}) sin(θ_{K-1})
 * The first K-2 angles run over [0, π] (polar), the last over [0, 2π)
 * (azimuthal).  Half-redundancy from antipodal symmetry α ↔ -α (same
 * |ψ|, same RDM) is kept because phases matter for the basis-change
 * matrix output.
 *
 * The `n_t[]` array is of length K-1.
 */
static long grid_size_kgen(int K_states, const int *n_t) {
    long s = 1;
    for (int k = 0; k < K_states - 1; k++) s *= n_t[k];
    return s;
}

static void grid_to_alpha_kgen(long i, int K_states, const int *n_t,
                                 double *alpha) {
    int idx[MAX_K];
    long rem = i;
    for (int k = K_states - 2; k >= 0; k--) {
        idx[k] = (int)(rem % n_t[k]);
        rem /= n_t[k];
    }
    /* Build angles. */
    double theta[MAX_K];
    for (int k = 0; k < K_states - 2; k++)
        theta[k] = M_PI * (idx[k] + 0.5) / n_t[k];
    /* Last angle is azimuthal in [0, 2π). */
    theta[K_states - 2] = 2.0 * M_PI * idx[K_states - 2] / n_t[K_states - 2];
    /* Spherical → Cartesian. */
    double s = 1.0;
    for (int k = 0; k < K_states - 1; k++) {
        alpha[k] = s * cos(theta[k]);
        s *= sin(theta[k]);
    }
    alpha[K_states - 1] = s;
}

/* ---- Geometry: extract the kagome strip bipartition. ------------ */
/* For an L×L kagome PBC cluster, sites are indexed by (cx, cy, sub):
 *     site(cx, cy, sub) = 3 * (cx*L + cy) + sub        sub ∈ {0,1,2}.
 * Cut along y-cycle (i.e. wrapping in y direction): A = sites with
 * cy ∈ {0, ..., strip-1}.  Strip width 1 → 3L sites per A. */
static int build_strip_y(int L, int strip_w, int *sites_A) {
    int nA = 0;
    for (int cx = 0; cx < L; cx++)
    for (int cy = 0; cy < strip_w; cy++)
    for (int sub = 0; sub < 3; sub++) {
        sites_A[nA++] = 3 * (cx*L + cy) + sub;
    }
    /* Sort ascending. */
    for (int i = 0; i < nA-1; i++) for (int j = i+1; j < nA; j++)
        if (sites_A[j] < sites_A[i]) { int t = sites_A[i]; sites_A[i]=sites_A[j]; sites_A[j]=t; }
    return nA;
}

static int build_strip_x(int L, int strip_w, int *sites_A) {
    int nA = 0;
    for (int cx = 0; cx < strip_w; cx++)
    for (int cy = 0; cy < L; cy++)
    for (int sub = 0; sub < 3; sub++) {
        sites_A[nA++] = 3 * (cx*L + cy) + sub;
    }
    for (int i = 0; i < nA-1; i++) for (int j = i+1; j < nA; j++)
        if (sites_A[j] < sites_A[i]) { int t = sites_A[i]; sites_A[i]=sites_A[j]; sites_A[j]=t; }
    return nA;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s L K psi_path_1 ... psi_path_K [n_t_1 ... n_t_{K-1}]\n",
                argv[0]);
        fprintf(stderr,
                "  K = number of sector states (2 ≤ K ≤ %d).  K-1 grid args\n"
                "  follow the eigvec paths; each defaults to 6 if omitted.\n"
                "  Backwards-compat: if argv[2] is a path (not a small int),\n"
                "  K=4 is assumed and the legacy CLI is honoured.\n",
                MAX_K);
        return 1;
    }
    int L = atoi(argv[1]);
    /* Detect legacy K=4 invocation: if argv[2] looks like a path (has '/'
     * or contains a '.' likely a file extension), fall back to K=4. */
    int K_states;
    int first_path_idx;
    {
        const char *a2 = argv[2];
        int looks_like_path = 0;
        for (const char *c = a2; *c; c++)
            if (*c == '/' || *c == '.') { looks_like_path = 1; break; }
        if (looks_like_path) {
            K_states = 4;
            first_path_idx = 2;
        } else {
            K_states = atoi(argv[2]);
            first_path_idx = 3;
        }
    }
    if (K_states < 2 || K_states > MAX_K) {
        fprintf(stderr, "FAIL: K=%d out of range [2, %d]\n", K_states, MAX_K);
        return 1;
    }
    int needed_argc = first_path_idx + K_states;
    if (argc < needed_argc) {
        fprintf(stderr, "FAIL: K=%d requires %d eigvec paths, got %d\n",
                K_states, K_states, argc - first_path_idx);
        return 1;
    }
    const char *paths[MAX_K];
    for (int a = 0; a < K_states; a++) paths[a] = argv[first_path_idx + a];

    /* Read K-1 grid args; default 6. */
    int n_t[MAX_K];
    int grid_argc_avail = argc - (first_path_idx + K_states);
    for (int k = 0; k < K_states - 1; k++) {
        if (k < grid_argc_avail) n_t[k] = atoi(argv[first_path_idx + K_states + k]);
        else n_t[k] = 6;
        if (n_t[k] < 1) n_t[k] = 1;
    }
    int N = 3 * L * L;
    long dim = 1L << N;
    fprintf(stderr, "# Kagome MES extraction at L=%d, N=%d, dim=%ld, K=%d\n",
            L, N, dim, K_states);
    fprintf(stderr, "# Grid: n_t = [");
    for (int k = 0; k < K_states - 1; k++)
        fprintf(stderr, "%d%s", n_t[k], (k < K_states-2) ? ", " : "");
    fprintf(stderr, "] → %ld α points per cut\n",
            grid_size_kgen(K_states, n_t));

    /* Load eigvecs. */
    double *psi[MAX_K] = {NULL};
    double E0[MAX_K] = {0};
    int Ncheck, ircheck; long dimcheck;
    for (int a = 0; a < K_states; a++) {
        if (load_eigvec_real(paths[a], &Ncheck, &ircheck, &E0[a],
                              &psi[a], &dimcheck) != 0 ||
            Ncheck != N || dimcheck != dim) {
            fprintf(stderr, "FAIL: cannot load %s (or N/dim mismatch)\n", paths[a]);
            return 1;
        }
        fprintf(stderr, "# loaded sector %d: %s  E_0 = %.10f\n", a, paths[a], E0[a]);
    }

    /* Two bipartitions: y-strip (cuts x-cycle) and x-strip (cuts y-cycle) */
    int sites_X[64], sites_Y[64];
    int nA_X = build_strip_y(L, 1, sites_X);  /* y-strip: cy=0 only */
    int nA_Y = build_strip_x(L, 1, sites_Y);  /* x-strip: cx=0 only */
    fprintf(stderr, "# bipartition X: nA = %d sites (cy=0 strip)\n", nA_X);
    fprintf(stderr, "# bipartition Y: nA = %d sites (cx=0 strip)\n", nA_Y);

    /* Working buffer for ψ(α). */
    double *psi_alpha = malloc((size_t)dim * sizeof(double));
    if (!psi_alpha) return 1;

    /* Output JSON. */
    printf("{\n");
    printf("  \"system\": {\"L\": %d, \"N\": %d, \"K_sectors\": %d},\n",
           L, N, K_states);
    printf("  \"sectors_E0\": [");
    for (int a = 0; a < K_states; a++)
        printf("%.10f%s", E0[a], (a < K_states-1) ? ", " : "");
    printf("],\n");
    printf("  \"sectors_paths\": [");
    for (int a = 0; a < K_states; a++)
        printf("\"%s\"%s", paths[a], (a < K_states-1) ? ", " : "");
    printf("],\n");
    printf("  \"grid\": {\"n_t\": [");
    for (int k = 0; k < K_states-1; k++)
        printf("%d%s", n_t[k], (k < K_states-2) ? ", " : "");
    printf("]},\n");
    printf("  \"bipartitions\": {\n");
    printf("    \"X\": {\"nA\": %d, \"sites_A\": [", nA_X);
    for (int i = 0; i < nA_X; i++) printf("%d%s", sites_X[i], (i<nA_X-1)?", ":"");
    printf("]},\n");
    printf("    \"Y\": {\"nA\": %d, \"sites_A\": [", nA_Y);
    for (int i = 0; i < nA_Y; i++) printf("%d%s", sites_Y[i], (i<nA_Y-1)?", ":"");
    printf("]}\n  },\n");

    long n_pts = grid_size_kgen(K_states, n_t);
    double t0 = wall_seconds();

    /* For each cut, scan grid, sort by S, report top-K minima. */
    const char *cut_names[2] = {"X", "Y"};
    int *cut_sites[2] = {sites_X, sites_Y};
    int cut_nA[2] = {nA_X, nA_Y};

    typedef struct { double S; double alpha[MAX_K]; } sample_t;

    sample_t *best_samples[2] = {NULL, NULL};

    for (int cut = 0; cut < 2; cut++) {
        sample_t *samples = malloc((size_t)n_pts * sizeof(sample_t));
        if (!samples) return 1;
        fprintf(stderr, "# cut %s scanning %ld points...\n", cut_names[cut], n_pts);
        double tc0 = wall_seconds();
        for (long g = 0; g < n_pts; g++) {
            double alpha[MAX_K] = {0};
            grid_to_alpha_kgen(g, K_states, n_t, alpha);
            form_combination((const double *const*)psi, alpha, K_states, dim, psi_alpha);
            double S = 0.0;
            int rc = entropy_of_real_state(psi_alpha, N, dim,
                                            cut_sites[cut], cut_nA[cut], &S);
            if (rc != 0) S = 1e6;
            samples[g].S = S;
            for (int a = 0; a < K_states; a++) samples[g].alpha[a] = alpha[a];
            if ((g % 16) == 0) {
                double dt = wall_seconds() - tc0;
                fprintf(stderr, "#   %ld/%ld   last S = %.4f   wall=%.0f s\n",
                        g, n_pts, S, dt);
            }
        }
        fprintf(stderr, "# cut %s scan done in %.1f s\n",
                cut_names[cut], wall_seconds() - tc0);

        /* Sort by S ascending, keep top K. */
        for (long i = 0; i < n_pts-1; i++) for (long j = i+1; j < n_pts; j++) {
            if (samples[j].S < samples[i].S) {
                sample_t t = samples[i]; samples[i] = samples[j]; samples[j] = t;
            }
        }

        printf("  \"cut_%s\": {\n", cut_names[cut]);
        printf("    \"min_S\":  %.6f,\n", samples[0].S);
        printf("    \"max_S\":  %.6f,\n", samples[n_pts-1].S);
        printf("    \"top_8_minima\": [\n");
        long n_top = (n_pts < 8) ? n_pts : 8;
        for (long i = 0; i < n_top; i++) {
            printf("      {\"S\": %.6f, \"alpha\": [", samples[i].S);
            for (int a = 0; a < K_states; a++)
                printf("%.6f%s", samples[i].alpha[a],
                       (a < K_states-1) ? ", " : "");
            printf("]}%s\n", (i < n_top-1) ? "," : "");
        }
        printf("    ]\n  }%s\n", (cut == 0) ? "," : "");

        best_samples[cut] = samples;
    }

    /* Compute the basis-change matrix between the K minima of cut X
     * and cut Y.  This is the empirical lattice modular S matrix in
     * the MES basis (up to gauge phases). */
    printf("  ,\"empirical_modular_S_in_MES_basis\": [\n");
    for (int i = 0; i < K_states; i++) {
        printf("    [");
        for (int j = 0; j < K_states; j++) {
            /* ⟨MES_X^{(i)} | MES_Y^{(j)}⟩ — both real combinations. */
            double overlap = 0.0;
            #ifdef _OPENMP
            #pragma omp parallel for reduction(+:overlap) schedule(static)
            #endif
            for (long s = 0; s < dim; s++) {
                double mes_x = 0.0, mes_y = 0.0;
                for (int a = 0; a < K_states; a++) {
                    mes_x += best_samples[0][i].alpha[a] * psi[a][s];
                    mes_y += best_samples[1][j].alpha[a] * psi[a][s];
                }
                overlap += mes_x * mes_y;
            }
            /* Note: alpha is normalised on the unit sphere, so the
             * combination is normalised iff the ψ_a are orthonormal
             * (they are — sector-projected eigvecs of an Hermitian
             * H are orthogonal across sectors, and Lanczos normalises). */
            printf("%.6f%s", overlap, (j < K_states-1) ? ", " : "");
        }
        printf("]%s\n", (i < K_states-1) ? "," : "");
    }
    printf("  ],\n");
    if (K_states == 4) {
        printf("  \"symbolic_Z2_TC_modular_S\": [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5], [0.5, -0.5, -0.5, 0.5]],\n");
    }
    printf("  \"interpretation\": \"Empirical lattice modular S = ⟨MES_X | MES_Y⟩.  For Z_2 TC at K=4 this should equal (1/2)·Hadamard_4 up to permutations and ±1 phases.  At K>4 there is no canonical comparison (Z_2 TC has only 4 anyon types); use the matrix singular values + structure as the empirical observable.\",\n");
    printf("  \"total_wall_s\": %.1f\n", wall_seconds() - t0);
    printf("}\n");

    for (int a = 0; a < K_states; a++) free(psi[a]);
    for (int cut = 0; cut < 2; cut++) free(best_samples[cut]);
    free(psi_alpha);
    return 0;
}
