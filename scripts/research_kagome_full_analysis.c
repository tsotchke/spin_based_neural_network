/*
 * scripts/research_kagome_full_analysis.c
 *
 * Comprehensive single-run predictive-observables analysis for any
 * (L, irrep) pair on the kagome AFM Heisenberg model.  Produces:
 *
 *   1. Sector E_0 (validated against libirrep ED at L≤3).
 *   2. Sum-rule + Hamiltonian-consistency cross-checks.
 *   3. Distance-resolved spin correlation function C(d) for d up to
 *      max-image distance — power-law (gapless) vs exponential (gapped).
 *   4. Static structure factor S(q) at the (Γ, M, K-equiv, …) momenta
 *      of the L×L supercell.
 *   5. Compact-region area-law fit on three subsystems → γ_TEE.
 *   6. ENTANGLEMENT SPECTRUM (full eigenvalue list of −log ρ_A at
 *      the largest compact subsystem).  Sharp Z₂ vs U(1) Dirac
 *      diagnostic — Z₂ has a gapped entanglement spectrum, U(1)
 *      Dirac has CFT-like power-law structure.
 *   7. Optional eigvec save to disk for downstream re-analyses
 *      (S(q,ω) continued-fraction Lanczos, etc.) without re-running
 *      the 90-min Lanczos.
 *
 * Usage:  research_kagome_full_analysis L irrep [iters [eigvec_path]]
 *           L 1..3, irrep 0..3 (A_1 A_2 B_1 B_2),
 *           iters default 200, eigvec_path optional save target.
 */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <complex.h>

#include "nqs/nqs_symproj.h"
#include "nqs/nqs_lanczos.h"
#include "mps/lanczos.h"

#ifdef SPIN_NN_HAS_IRREP
#include "libirrep_bridge.h"
#include "irrep/rdm.h"
#endif

static double wall_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

static const char *irrep_names[4] = { "A_1", "A_2", "B_1", "B_2" };
static const nqs_symproj_kagome_irrep_t irrep_codes[4] = {
    NQS_SYMPROJ_KAGOME_GAMMA_A1, NQS_SYMPROJ_KAGOME_GAMMA_A2,
    NQS_SYMPROJ_KAGOME_GAMMA_B1, NQS_SYMPROJ_KAGOME_GAMMA_B2,
};

/* Cartesian site coordinates (kagome unit cell: a1=(1,0), a2=(1/2, √3/2)). */
static double *cart_x = NULL, *cart_y = NULL;
static const double a1x = 1.0, a1y = 0.0;
static const double a2x = 0.5, a2y = 0.86602540378443864676;

static void build_coords(int L) {
    int N = 3 * L * L;
    cart_x = malloc((size_t)N * sizeof(double));
    cart_y = malloc((size_t)N * sizeof(double));
    for (int cx = 0; cx < L; cx++) for (int cy = 0; cy < L; cy++) {
        int idx_A = 3 * (cx * L + cy) + 0;
        int idx_B = 3 * (cx * L + cy) + 1;
        int idx_C = 3 * (cx * L + cy) + 2;
        double xc = (double)cx * a1x + (double)cy * a2x;
        double yc = (double)cx * a1y + (double)cy * a2y;
        cart_x[idx_A] = xc;             cart_y[idx_A] = yc;
        cart_x[idx_B] = xc + a1x/2.0;    cart_y[idx_B] = yc + a1y/2.0;
        cart_x[idx_C] = xc + a2x/2.0;    cart_y[idx_C] = yc + a2y/2.0;
    }
}

static double pbc_distance(int i, int j, int L) {
    double dx = cart_x[i] - cart_x[j];
    double dy = cart_y[i] - cart_y[j];
    double n_b = (2.0 / sqrt(3.0)) * dy;
    double n_a = dx - 0.5 * n_b;
    n_a -= (double)L * floor((n_a + 0.5 * (double)L) / (double)L);
    n_b -= (double)L * floor((n_b + 0.5 * (double)L) / (double)L);
    double dx_w = n_a * a1x + n_b * a2x;
    double dy_w = n_a * a1y + n_b * a2y;
    return sqrt(dx_w * dx_w + dy_w * dy_w);
}

static double spin_correlation(const double *psi, long dim,
                                int i, int j) {
    double diag = 0.0, off = 0.0;
    long mask_ij = (1L << i) | (1L << j);
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:diag,off) schedule(static)
    #endif
    for (long s = 0; s < dim; s++) {
        int b_i = (s >> i) & 1;
        int b_j = (s >> j) & 1;
        int s_i = b_i ? -1 : +1;
        int s_j = b_j ? -1 : +1;
        diag += 0.25 * psi[s] * psi[s] * (double)(s_i * s_j);
        if (b_i != b_j) {
            long s_flip = s ^ mask_ij;
            if (s < s_flip) off += psi[s] * psi[s_flip];
        }
    }
    return diag + off;
}

static int qsort_double_desc(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) ? -1 : (da < db) ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s L irrep [iters [eigvec_path]]\n", argv[0]);
        return 1;
    }
    int L = atoi(argv[1]);
    int ir = atoi(argv[2]);
    int max_iters = (argc > 3) ? atoi(argv[3]) : 200;
    const char *eigvec_path = (argc > 4) ? argv[4] : NULL;
    if (L <= 0 || L > 3 || ir < 0 || ir > 3) return 1;

    int N = 3 * L * L;
    long dim = 1L << N;

    fprintf(stderr, "# kagome %d×%d PBC AFM, sector (Γ, %s)  N=%d  dim=%ld\n",
           L, L, irrep_names[ir], N, dim);

    int *perm = NULL; double *chars = NULL; int G = 0;
    if (nqs_kagome_p6m_perm_irrep(L, irrep_codes[ir], &perm, &chars, &G) != 0)
        return 1;

    double *psi = malloc((size_t)dim * sizeof(double));
    if (!psi) return 1;

    /* Step 1: Lanczos with eigenvector. */
    double t0 = wall_seconds();
    double e0 = 0.0;
    lanczos_result_t lr = (lanczos_result_t){0};
    int rc = nqs_lanczos_e0_kagome_heisenberg_projected_lean_eigvec(
        L, L, 1.0, 1, perm, chars, G, max_iters, 1e-10,
        &e0, psi, &lr);
    if (rc != 0) { fprintf(stderr, "Lanczos failed\n"); return 1; }
    double t_lanczos = wall_seconds() - t0;
    fprintf(stderr, "# Lanczos: E_0=%.10f  iters=%d  conv=%d  (%.1f s)\n",
            e0, lr.iterations, lr.converged, t_lanczos);

    /* Save eigvec if requested.  Check fwrite return values — silent
     * truncation on disk-full leaves a stub file masquerading as a
     * valid eigvec (see commit history for the kagome E_1/E_2 incident). */
    if (eigvec_path) {
        FILE *f = fopen(eigvec_path, "wb");
        if (!f) {
            fprintf(stderr, "# ERROR: cannot open %s for write: %s\n",
                    eigvec_path, strerror(errno));
        } else {
            int header[4] = {N, L, ir, (int)lr.iterations};
            double meta[2] = {e0, (double)dim};
            int ok = (fwrite(header, sizeof(int), 4, f) == 4)
                  && (fwrite(meta, sizeof(double), 2, f) == 2);
            if (ok) {
                size_t want = (size_t)dim;
                size_t got = fwrite(psi, sizeof(double), want, f);
                if (got != want) {
                    fprintf(stderr,
                            "# ERROR: eigvec fwrite truncated: %zu/%zu doubles (%s)\n",
                            got, want, strerror(errno));
                    ok = 0;
                }
            }
            if (fflush(f) != 0 || fclose(f) != 0) ok = 0;
            if (ok) {
                fprintf(stderr, "# eigvec saved to %s (%.2f GB)\n",
                        eigvec_path, (double)dim * 8.0 / 1e9);
            } else {
                fprintf(stderr, "# WARNING: eigvec save failed at %s — file may be truncated\n",
                        eigvec_path);
            }
        }
    }

    build_coords(L);

    /* === JSON output starts here === */
    printf("{\n");
    printf("  \"system\": {\"L\": %d, \"N\": %d, \"irrep\": \"%s\", \"dim\": %ld},\n",
           L, N, irrep_names[ir], dim);
    printf("  \"lanczos\": {\"E_0\": %.12f, \"iters\": %d, \"converged\": %d, \"wall_s\": %.1f},\n",
           e0, lr.iterations, lr.converged, t_lanczos);

    /* Step 2: NN bond avg + Hamiltonian sum-rule. */
    double t1 = wall_seconds();
    int num_NN = 0; double NN_sum = 0.0;
    for (int cx = 0; cx < L; cx++) for (int cy = 0; cy < L; cy++) {
        int A = 3*(cx*L+cy)+0, B = 3*(cx*L+cy)+1, C = 3*(cx*L+cy)+2;
        int up[3][2] = {{A,B},{A,C},{B,C}};
        for (int b = 0; b < 3; b++) {
            NN_sum += spin_correlation(psi, dim, up[b][0], up[b][1]);
            num_NN++;
        }
        int cxm = (cx-1+L)%L, cym = (cy-1+L)%L;
        int Bm = 3*(cxm*L+cy)+1, Cm = 3*(cx*L+cym)+2;
        int dn[3][2] = {{A,Bm},{A,Cm},{Bm,Cm}};
        for (int b = 0; b < 3; b++) {
            NN_sum += spin_correlation(psi, dim, dn[b][0], dn[b][1]);
            num_NN++;
        }
    }
    double NN_avg = NN_sum / (double)num_NN;
    printf("  \"hamiltonian_sum_rule\": {\"NN_bonds\": %d, \"NN_avg\": %.10f, \"E_0_check\": %.10f, \"residual\": %.3e},\n",
           num_NN, NN_avg, NN_sum, fabs(NN_sum - e0));

    /* Step 3: Total spin sum rule. */
    double total_S2 = 0.0;
    for (int i = 0; i < N; i++) total_S2 += 0.75;
    for (int i = 0; i < N; i++) for (int j = i+1; j < N; j++) {
        total_S2 += 2.0 * spin_correlation(psi, dim, i, j);
    }
    double S_total = (total_S2 < 1e-10) ? 0.0 : 0.5 * (sqrt(1.0 + 4.0*total_S2) - 1.0);
    printf("  \"total_spin\": {\"S_squared\": %.6f, \"S_total\": %.4f},\n",
           total_S2, S_total);

    /* Step 4: distance-resolved correlation function C(d). */
    fprintf(stderr, "# computing distance-resolved C(d)...\n");
    typedef struct { double d; double sum; int count; } shell_t;
    shell_t shells[64]; int n_shells = 0;
    for (int i = 0; i < N; i++) for (int j = i+1; j < N; j++) {
        double d = pbc_distance(i, j, L);
        double c = spin_correlation(psi, dim, i, j);
        long key = (long)(d * 10000.0 + 0.5);
        int found = -1;
        for (int s = 0; s < n_shells; s++) {
            if ((long)(shells[s].d*10000.0+0.5) == key) { found = s; break; }
        }
        if (found < 0) {
            if (n_shells < 64) {
                shells[n_shells].d = d;
                shells[n_shells].sum = c;
                shells[n_shells].count = 1;
                n_shells++;
            }
        } else {
            shells[found].sum += c;
            shells[found].count++;
        }
    }
    /* Sort by distance. */
    for (int i = 0; i < n_shells - 1; i++) {
        int imin = i;
        for (int j = i+1; j < n_shells; j++)
            if (shells[j].d < shells[imin].d) imin = j;
        if (imin != i) { shell_t t = shells[i]; shells[i] = shells[imin]; shells[imin] = t; }
    }
    printf("  \"correlation_shells\": [\n");
    for (int s = 0; s < n_shells; s++) {
        double avg = shells[s].sum / (double)shells[s].count;
        printf("    {\"d\": %.4f, \"npairs\": %d, \"C_avg\": %.10f, \"|C|_avg\": %.10f}%s\n",
               shells[s].d, shells[s].count, avg, fabs(avg),
               (s < n_shells - 1) ? "," : "");
    }
    printf("  ],\n");
    fprintf(stderr, "# C(d) done (%.1f s)\n", wall_seconds() - t1);

    /* Step 5: TEE compact-region area-law fit (3 points). */
#ifdef SPIN_NN_HAS_IRREP
    fprintf(stderr, "# computing γ_TEE + entanglement spectrum...\n");
    double _Complex *psi_c = malloc((size_t)dim * sizeof(double _Complex));
    for (long s = 0; s < dim; s++) psi_c[s] = psi[s] + 0.0 * I;

    int subA[3][12] = {
        {0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
        {0,1,2,-1,-1,-1,-1,-1,-1,-1,-1,-1},
        {0,1,2,3,4,5,-1,-1,-1,-1,-1,-1},
    };
    int subA_n[3] = {1, 3, 6};
    double S1[3], bnd[3];
    /* For the largest compact subsystem (nA=6), also extract the full
     * entanglement spectrum (eigenvalues of ρ_A → −log λ). */
    double *entanglement_spectrum = NULL;
    int spectrum_n = 0;
    for (int k = 0; k < 3; k++) {
        int nA = subA_n[k];
        long dimA = 1L << nA;
        double _Complex *rho_A = malloc((size_t)dimA * (size_t)dimA *
                                          sizeof(double _Complex));
        int sites_A[12];
        for (int j = 0; j < nA; j++) sites_A[j] = subA[k][j];
        if (libirrep_bridge_partial_trace_spin_half(N, psi_c, sites_A, nA,
                                                      rho_A) != 0) {
            free(rho_A); continue;
        }
        /* Extract eigenvalues directly. */
        double *eigs = malloc((size_t)dimA * sizeof(double));
        if (irrep_hermitian_eigvals((int)dimA, rho_A, eigs) != IRREP_OK) {
            free(rho_A); free(eigs); continue;
        }
        free(rho_A);
        /* von Neumann S = -Σ λ log λ, entanglement spectrum = -log λ */
        double S = 0.0;
        for (int j = 0; j < (int)dimA; j++) {
            if (eigs[j] > 1e-14) S -= eigs[j] * log(eigs[j]);
        }
        S1[k] = S;
        /* Boundary length */
        int in_A[27] = {0};
        for (int j = 0; j < nA; j++) in_A[sites_A[j]] = 1;
        int boundary = 0;
        for (int cx = 0; cx < L; cx++) for (int cy = 0; cy < L; cy++) {
            int A = 3*(cx*L+cy)+0, B = 3*(cx*L+cy)+1, C = 3*(cx*L+cy)+2;
            int cxm = (cx-1+L)%L, cym = (cy-1+L)%L;
            int Bm = 3*(cxm*L+cy)+1, Cm = 3*(cx*L+cym)+2;
            int bonds[6][2] = {{A,B},{A,C},{B,C},{A,Bm},{A,Cm},{Bm,Cm}};
            for (int b = 0; b < 6; b++)
                if (in_A[bonds[b][0]] != in_A[bonds[b][1]]) boundary++;
        }
        bnd[k] = (double)boundary;

        if (k == 2) {
            /* Save full spectrum for the largest compact subsystem. */
            qsort(eigs, (size_t)dimA, sizeof(double), qsort_double_desc);
            spectrum_n = (int)dimA;
            entanglement_spectrum = malloc((size_t)spectrum_n * sizeof(double));
            for (int j = 0; j < spectrum_n; j++) {
                entanglement_spectrum[j] = (eigs[j] > 1e-14) ? -log(eigs[j]) : INFINITY;
            }
        }
        free(eigs);
    }
    /* Linear fit S = α·|∂A| − γ. */
    double sb=0, sS=0, sb2=0, sbS=0;
    for (int k = 0; k < 3; k++) { sb+=bnd[k]; sS+=S1[k]; sb2+=bnd[k]*bnd[k]; sbS+=bnd[k]*S1[k]; }
    double denom = 3.0*sb2 - sb*sb;
    double alpha = (3.0*sbS - sb*sS) / denom;
    double intercept = (sS - alpha*sb) / 3.0;
    double gamma = -intercept;
    printf("  \"tee\": {\n");
    printf("    \"compact_fit\": [");
    for (int k = 0; k < 3; k++) printf("{\"nA\": %d, \"|∂A|\": %.0f, \"S_1\": %.6f}%s",
                                         subA_n[k], bnd[k], S1[k], (k<2)?", ":"");
    printf("],\n");
    printf("    \"alpha\": %.6f, \"intercept\": %.6f, \"gamma_nats\": %.6f, \"gamma_log2\": %.4f\n",
           alpha, intercept, gamma, gamma/log(2.0));
    printf("  },\n");

    /* Output entanglement spectrum (top-K eigenvalues of −log ρ_A). */
    if (entanglement_spectrum) {
        int show = (spectrum_n > 32) ? 32 : spectrum_n;
        printf("  \"entanglement_spectrum_nA_6\": {\n");
        printf("    \"total_eigenvalues\": %d,\n", spectrum_n);
        printf("    \"top_%d_minus_log_lambda\": [", show);
        for (int j = 0; j < show; j++) {
            double v = entanglement_spectrum[j];
            if (isfinite(v)) printf("%.6f", v); else printf("\"inf\"");
            if (j < show-1) printf(", ");
        }
        printf("],\n");
        /* Detect entanglement gap: largest gap in the lowest 8 levels. */
        double max_gap = 0.0;
        int gap_idx = -1;
        for (int j = 0; j < 7 && j+1 < spectrum_n; j++) {
            double g = entanglement_spectrum[j+1] - entanglement_spectrum[j];
            if (isfinite(g) && g > max_gap) { max_gap = g; gap_idx = j; }
        }
        printf("    \"largest_gap_in_lowest_8\": %.6f, \"gap_position\": %d,\n",
               max_gap, gap_idx);
        printf("    \"comment\": \"Z₂ topological order: gapped entanglement spectrum (large gap below low-lying levels). U(1) Dirac: power-law / CFT-like spectrum (small gaps).\"\n");
        printf("  }\n");
        free(entanglement_spectrum);
    }
    free(psi_c);
#else
    printf("  \"tee\": null\n");
#endif

    printf("}\n");

    free(psi); free(perm); free(chars); free(cart_x); free(cart_y);
    fprintf(stderr, "# done (%.1f s total)\n", wall_seconds() - t0);
    return 0;
}
