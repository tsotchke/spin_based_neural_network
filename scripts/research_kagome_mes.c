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

static double wall_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

#define K 4   /* number of sector states */

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

/* Compute bipartite von Neumann entropy S of ψ_real (length dim) on the
 * subsystem given by sites_A (size nA).  ψ is real-valued; converted
 * to complex on the fly.  Returns 0 on success and writes into *out_S. */
static int entropy_of_real_state(const double *psi_real, int N, long dim,
                                  const int *sites_A, int nA, double *out_S) {
    long dimA = 1L << nA;
    double _Complex *psi_c = malloc((size_t)dim * sizeof(double _Complex));
    if (!psi_c) return -1;
    for (long s = 0; s < dim; s++) psi_c[s] = psi_real[s] + 0.0 * I;
    double _Complex *rho = malloc((size_t)dimA * dimA * sizeof(double _Complex));
    if (!rho) { free(psi_c); return -1; }
    int rc = libirrep_bridge_partial_trace_spin_half(N, psi_c, sites_A, nA, rho);
    if (rc != 0) { free(psi_c); free(rho); return rc; }
    double *eigs = malloc((size_t)dimA * sizeof(double));
    if (!eigs) { free(psi_c); free(rho); return -1; }
    if (irrep_hermitian_eigvals((int)dimA, rho, eigs) != IRREP_OK) {
        free(psi_c); free(rho); free(eigs); return -1;
    }
    double S = 0.0;
    for (long j = 0; j < dimA; j++) {
        double l = eigs[j];
        if (l > 1e-15) S -= l * log(l);
    }
    *out_S = S;
    free(psi_c); free(rho); free(eigs);
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

/* Generate a coarse grid of unit 4-vectors in ℝ^4 via spherical coords:
 *   α_0 = cos(θ_1)
 *   α_1 = sin(θ_1) cos(θ_2)
 *   α_2 = sin(θ_1) sin(θ_2) cos(θ_3)
 *   α_3 = sin(θ_1) sin(θ_2) sin(θ_3)
 * with θ_1, θ_2 ∈ [0, π], θ_3 ∈ [0, 2π).
 * Half-redundancy from antipodal symmetry α ↔ -α (same |ψ|, same RDM)
 * we keep redundancy because phases matter for the basis-change matrix. */
static long grid_size(int n_t1, int n_t2, int n_t3) {
    return (long)n_t1 * n_t2 * n_t3;
}

static void grid_to_alpha(int i, int n_t1, int n_t2, int n_t3, double alpha[K]) {
    int it1 = i / (n_t2 * n_t3);
    int it23 = i % (n_t2 * n_t3);
    int it2 = it23 / n_t3;
    int it3 = it23 % n_t3;
    double t1 = M_PI * (it1 + 0.5) / n_t1;
    double t2 = M_PI * (it2 + 0.5) / n_t2;
    double t3 = 2.0 * M_PI * it3 / n_t3;
    alpha[0] = cos(t1);
    alpha[1] = sin(t1) * cos(t2);
    alpha[2] = sin(t1) * sin(t2) * cos(t3);
    alpha[3] = sin(t1) * sin(t2) * sin(t3);
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
        fprintf(stderr, "usage: %s L psi_path_A1 psi_path_A2 psi_path_B1 psi_path_B2 [n_t1=8 [n_t2=8 [n_t3=8]]]\n", argv[0]);
        fprintf(stderr, "  Loads the 4 sector eigvecs and runs MES search on x- and y-cycle bipartitions.\n");
        return 1;
    }
    int L = atoi(argv[1]);
    const char *paths[K] = {argv[2], argv[3], argv[4], argv[5]};
    int n_t1 = (argc > 6) ? atoi(argv[6]) : 8;
    int n_t2 = (argc > 7) ? atoi(argv[7]) : 8;
    int n_t3 = (argc > 8) ? atoi(argv[8]) : 8;
    int N = 3 * L * L;
    long dim = 1L << N;
    fprintf(stderr, "# Kagome MES extraction at L=%d, N=%d, dim=%ld\n", L, N, dim);
    fprintf(stderr, "# Grid: n_t1=%d n_t2=%d n_t3=%d → %ld α points per cut\n",
            n_t1, n_t2, n_t3, grid_size(n_t1, n_t2, n_t3));

    /* Load eigvecs. */
    double *psi[K] = {NULL};
    double E0[K] = {0};
    int Ncheck, ircheck; long dimcheck;
    for (int a = 0; a < K; a++) {
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
    printf("  \"system\": {\"L\": %d, \"N\": %d, \"K_sectors\": %d},\n", L, N, K);
    printf("  \"sectors_E0\": [%.10f, %.10f, %.10f, %.10f],\n",
           E0[0], E0[1], E0[2], E0[3]);
    printf("  \"grid\": {\"n_t1\": %d, \"n_t2\": %d, \"n_t3\": %d},\n",
           n_t1, n_t2, n_t3);
    printf("  \"bipartitions\": {\n");
    printf("    \"X\": {\"nA\": %d, \"sites_A\": [", nA_X);
    for (int i = 0; i < nA_X; i++) printf("%d%s", sites_X[i], (i<nA_X-1)?", ":"");
    printf("]},\n");
    printf("    \"Y\": {\"nA\": %d, \"sites_A\": [", nA_Y);
    for (int i = 0; i < nA_Y; i++) printf("%d%s", sites_Y[i], (i<nA_Y-1)?", ":"");
    printf("]}\n  },\n");

    long n_pts = grid_size(n_t1, n_t2, n_t3);
    double t0 = wall_seconds();

    /* For each cut, scan grid, sort by S, report top-K minima. */
    const char *cut_names[2] = {"X", "Y"};
    int *cut_sites[2] = {sites_X, sites_Y};
    int cut_nA[2] = {nA_X, nA_Y};

    typedef struct { double S; double alpha[K]; } sample_t;

    sample_t *best_samples[2] = {NULL, NULL};

    for (int cut = 0; cut < 2; cut++) {
        sample_t *samples = malloc((size_t)n_pts * sizeof(sample_t));
        if (!samples) return 1;
        fprintf(stderr, "# cut %s scanning %ld points...\n", cut_names[cut], n_pts);
        double tc0 = wall_seconds();
        for (long g = 0; g < n_pts; g++) {
            double alpha[K];
            grid_to_alpha((int)g, n_t1, n_t2, n_t3, alpha);
            form_combination((const double *const*)psi, alpha, K, dim, psi_alpha);
            double S = 0.0;
            int rc = entropy_of_real_state(psi_alpha, N, dim,
                                            cut_sites[cut], cut_nA[cut], &S);
            if (rc != 0) S = 1e6;
            samples[g].S = S;
            for (int a = 0; a < K; a++) samples[g].alpha[a] = alpha[a];
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
            printf("      {\"S\": %.6f, \"alpha\": [%.6f, %.6f, %.6f, %.6f]}%s\n",
                   samples[i].S,
                   samples[i].alpha[0], samples[i].alpha[1],
                   samples[i].alpha[2], samples[i].alpha[3],
                   (i < n_top-1) ? "," : "");
        }
        printf("    ]\n  }%s\n", (cut == 0) ? "," : "");

        best_samples[cut] = samples;
    }

    /* Compute the basis-change matrix between the K minima of cut X
     * and cut Y.  This is the empirical lattice modular S matrix in
     * the MES basis (up to gauge phases). */
    printf("  ,\"empirical_modular_S_in_MES_basis\": [\n");
    for (int i = 0; i < K; i++) {
        printf("    [");
        for (int j = 0; j < K; j++) {
            /* ⟨MES_X^{(i)} | MES_Y^{(j)}⟩ — both real combinations. */
            double overlap = 0.0;
            #ifdef _OPENMP
            #pragma omp parallel for reduction(+:overlap) schedule(static)
            #endif
            for (long s = 0; s < dim; s++) {
                double mes_x = 0.0, mes_y = 0.0;
                for (int a = 0; a < K; a++) {
                    mes_x += best_samples[0][i].alpha[a] * psi[a][s];
                    mes_y += best_samples[1][j].alpha[a] * psi[a][s];
                }
                overlap += mes_x * mes_y;
            }
            /* Note: alpha is normalised on the unit sphere, so the
             * combination is normalised iff the ψ_a are orthonormal
             * (they are — sector-projected eigvecs of an Hermitian
             * H are orthogonal across sectors, and Lanczos normalises). */
            printf("%.6f%s", overlap, (j < K-1) ? ", " : "");
        }
        printf("]%s\n", (i < K-1) ? "," : "");
    }
    printf("  ],\n");
    printf("  \"symbolic_Z2_TC_modular_S\": [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5], [0.5, -0.5, -0.5, 0.5]],\n");
    printf("  \"interpretation\": \"Empirical lattice modular S = ⟨MES_X | MES_Y⟩.  For Z_2 TC, this should equal (1/2)·Hadamard_4 up to permutations and ±1 phases.  Caveat: at L=3 PBC the 4 sector states are not exactly degenerate (E spread ~0.18 J), so the MES analysis is approximate.\",\n");
    printf("  \"total_wall_s\": %.1f\n", wall_seconds() - t0);
    printf("}\n");

    for (int a = 0; a < K; a++) free(psi[a]);
    for (int cut = 0; cut < 2; cut++) free(best_samples[cut]);
    free(psi_alpha);
    return 0;
}
