/*
 * scripts/research_kagome_eigvec_post.c
 *
 * Post-processor: load a saved eigenvector from disk and compute
 * additional observables WITHOUT re-running the 90-min Lanczos.
 *
 *   1. Static structure factor S(q) at all inequivalent momenta of
 *      the L×L supercell.  Tests whether kagome 3×3 PBC has Bragg-
 *      peak-like structure at K (consistent with 120° AFM) or
 *      diffuse signal (consistent with spin liquid).
 *   2. Half-cluster entanglement entropy S(nA = N/2) — sharper
 *      cleavage diagnostic than compact subsystems.
 *   3. Multiple-Renyi-α spectrum (α ∈ {1, 2, 3, 4, ∞}) on the largest
 *      compact subsystem, reusing the pre-computed eigenvalues.
 *
 * Usage:  research_kagome_eigvec_post eigvec_path
 * Header format (per save):  4×int + 2×double + 2^N × double.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

#ifdef SPIN_NN_HAS_IRREP
#include "libirrep_bridge.h"
#include "irrep/rdm.h"
#endif

static const double a1x = 1.0, a1y = 0.0;
static const double a2x = 0.5, a2y = 0.86602540378443864676;

static double *cart_x = NULL, *cart_y = NULL;
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

/* C_{ij} = ⟨S_i·S_j⟩ on real ψ. */
static double spin_correlation(const double *psi, long dim,
                                int i, int j) {
    double diag = 0.0, off = 0.0;
    long mask = (1L << i) | (1L << j);
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
            long s_flip = s ^ mask;
            if (s < s_flip) off += psi[s] * psi[s_flip];
        }
    }
    return diag + off;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s eigvec_path\n", argv[0]); return 1; }

    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    int header[4]; double meta[2];
    fread(header, sizeof(int), 4, f);
    fread(meta, sizeof(double), 2, f);
    int N = header[0], L = header[1], ir = header[2];
    long dim = (long)meta[1];
    if (dim != (1L << N)) { fclose(f); fprintf(stderr, "dim mismatch\n"); return 1; }
    double *psi = malloc((size_t)dim * sizeof(double));
    if (!psi) { fclose(f); return 1; }
    if ((long)fread(psi, sizeof(double), (size_t)dim, f) != dim) {
        fclose(f); free(psi);
        fprintf(stderr, "short read\n"); return 1;
    }
    fclose(f);
    fprintf(stderr, "# loaded eigvec: N=%d, L=%d, ir=%d, E_0=%.10f\n",
            N, L, ir, meta[0]);
    build_coords(L);

    /* Build the full 12 / 27 × 12 / 27 correlation matrix C[i][j]. */
    double C[27][27];
    fprintf(stderr, "# computing all C_{ij}...\n");
    for (int i = 0; i < N; i++) for (int j = i; j < N; j++) {
        double c = (i == j) ? 0.75 : spin_correlation(psi, dim, i, j);
        C[i][j] = c; C[j][i] = c;
    }

    /* Output JSON. */
    printf("{\n");
    printf("  \"system\": {\"L\": %d, \"N\": %d, \"irrep\": %d, \"E_0\": %.10f},\n",
           L, N, ir, meta[0]);

    /* Static structure factor S(q) at all inequivalent supercell momenta:
     * q = (m1·b1 + m2·b2) / L, mi ∈ {0..L-1}.  L² distinct points.
     * b1 = 2π(1, -1/√3), b2 = 2π(0, 2/√3). */
    double bx1 = 2.0*M_PI; double by1 = -2.0*M_PI / sqrt(3.0);
    double bx2 = 0.0;     double by2 =  4.0*M_PI / sqrt(3.0);
    printf("  \"structure_factor\": [\n");
    int first = 1;
    for (int m1 = 0; m1 < L; m1++) for (int m2 = 0; m2 < L; m2++) {
        double qx = ((double)m1 / L) * bx1 + ((double)m2 / L) * bx2;
        double qy = ((double)m1 / L) * by1 + ((double)m2 / L) * by2;
        double Sq_re = 0, Sq_im = 0;
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            double phase = qx * (cart_x[i] - cart_x[j]) +
                            qy * (cart_y[i] - cart_y[j]);
            Sq_re += cos(phase) * C[i][j];
            Sq_im += sin(phase) * C[i][j];
        }
        Sq_re /= (double)N;
        Sq_im /= (double)N;
        if (!first) printf(",\n");
        printf("    {\"m1\": %d, \"m2\": %d, \"qx\": %.4f, \"qy\": %.4f, \"S_re\": %.6f, \"S_im\": %.6f}",
               m1, m2, qx, qy, Sq_re, Sq_im);
        first = 0;
    }
    printf("\n  ],\n");

#ifdef SPIN_NN_HAS_IRREP
    /* Multiple Renyi at largest compact subsystem nA=6.  Reuses the
     * eigenvalue spectrum from a single partial-trace call. */
    int nA = 6;
    long dimA = 1L << nA;
    double _Complex *psi_c = malloc((size_t)dim * sizeof(double _Complex));
    for (long s = 0; s < dim; s++) psi_c[s] = psi[s] + 0.0 * I;
    double _Complex *rho = malloc((size_t)dimA*dimA*sizeof(double _Complex));
    int sites_A[6] = {0,1,2,3,4,5};
    if (libirrep_bridge_partial_trace_spin_half(N, psi_c, sites_A, nA, rho) == 0) {
        double *eigs = malloc((size_t)dimA * sizeof(double));
        if (irrep_hermitian_eigvals((int)dimA, rho, eigs) == IRREP_OK) {
            double S_alphas[5] = {0};
            double alphas[5] = {1.0, 2.0, 3.0, 4.0, 0.0};   /* α=1, 2, 3, 4, ∞ */
            double sum_logsum = 0.0;     /* for α = 1, von Neumann */
            for (int j = 0; j < (int)dimA; j++) {
                double l = eigs[j];
                if (l <= 1e-15) continue;
                sum_logsum -= l * log(l);
                S_alphas[1] += l*l;
                S_alphas[2] += l*l*l;
                S_alphas[3] += l*l*l*l;
            }
            S_alphas[0] = sum_logsum;
            S_alphas[1] = -log(S_alphas[1]);
            S_alphas[2] = 0.5 * -log(S_alphas[2]);
            S_alphas[3] = (1.0/3.0) * -log(S_alphas[3]);
            /* α=∞: -log(λ_max) */
            double lam_max = eigs[0];
            for (int j = 1; j < (int)dimA; j++) if (eigs[j] > lam_max) lam_max = eigs[j];
            S_alphas[4] = -log(lam_max);

            printf("  \"renyi_spectrum_nA_6\": {\n");
            printf("    \"S_1_vN\": %.6f,\n", S_alphas[0]);
            printf("    \"S_2\":    %.6f,\n", S_alphas[1]);
            printf("    \"S_3\":    %.6f,\n", S_alphas[2]);
            printf("    \"S_4\":    %.6f,\n", S_alphas[3]);
            printf("    \"S_inf\":  %.6f\n",   S_alphas[4]);
            printf("  }\n");
        }
        free(eigs);
    }
    free(rho); free(psi_c);
#endif

    printf("}\n");
    free(psi); free(cart_x); free(cart_y);
    return 0;
}
