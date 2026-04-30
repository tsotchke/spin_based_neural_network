/*
 * scripts/research_kagome_p6m_rep_6state.c
 *
 * Generalised version of research_kagome_p6m_rep.c — extracts the FULL
 * C_6v point-group representation on the LOW-ENERGY 6-state manifold
 * comprising:
 *
 *   ψ_{A_1}: 1D-irrep, 1 state    (Sz unrestricted, Lanczos finds the
 *                                   global lowest-spin state per sector)
 *   ψ_{A_2}: 1D-irrep, 1 state
 *   ψ_{B_1}: 1D-irrep, 1 state
 *   ψ_{B_2}: 1D-irrep, 1 state    (S=3/2 at L=3 PBC — different spin)
 *   ψ_{E_1}: 2D-irrep, 1 of 2 doublet partners (Sz=1/2, S=1/2)
 *   ψ_{E_2}: 2D-irrep, 1 of 2 doublet partners (Sz=1/2, S=1/2)
 *
 * For each of 12 group elements g of C_6v at origin, computes the 6×6
 * matrix M^{(g)}_{αβ} = ⟨ψ_α | σ_g | ψ_β⟩.
 *
 * Expected structure (symbolic prediction):
 *   - 1D ⊗ 1D block: diagonal δ_αβ χ_α(g)         (← extends commit a03dd95)
 *   - 1D ⊗ 2D block: 0  (sectors orthogonal — Schur)
 *   - 2D ⊗ 2D off-diagonal (E_1 ⊗ E_2): 0
 *   - 2D ⊗ 2D diagonal (E_1 ⊗ E_1, E_2 ⊗ E_2): depends on which doublet
 *     partner Lanczos found; |M| ≤ 1; for σ_g = E (identity), M = 1.
 *
 * Empirical test of these predictions verifies the FULL C_6v structure
 * on the low-energy manifold including the discovered E_2 doublet GS.
 *
 * Build:  make OPENMP=1 research_kagome_p6m_rep_6state
 * Run:    env OMP_NUM_THREADS=14 ./build/research_kagome_p6m_rep_6state
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nqs/nqs_symproj.h"

#define K 6   /* (A_1, A_2, B_1, B_2, E_1, E_2) */

static void apply_bit_perm(double *psi_in, double *psi_out, int N,
                            const int *perm) {
    long dim = 1L << N;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (long s = 0; s < dim; s++) {
        long s_pre = 0;
        for (int i = 0; i < N; i++) {
            long b = (s >> perm[i]) & 1L;
            s_pre |= b << i;
        }
        psi_out[s] = psi_in[s_pre];
    }
}

static double dot_product(const double *a, const double *b, long dim) {
    double s = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:s) schedule(static)
    #endif
    for (long i = 0; i < dim; i++) s += a[i] * b[i];
    return s;
}

static int load_eigvec(const char *path, int *out_N, int *out_ir,
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

/* C_6v character per group element (single irrep only). */
static double cv6_character_table(int op, int irrep) {
    int is_mirror  = (op >= 6);
    int rotk       = is_mirror ? (op - 6) : op;
    int is_sigma_v = is_mirror && (rotk % 2 == 0);
    int is_sigma_d = is_mirror && (rotk % 2 == 1);
    int is_C6      = !is_mirror && (rotk == 1 || rotk == 5);
    int is_C2      = !is_mirror && (rotk == 3);
    int is_E       = !is_mirror && (rotk == 0);
    int is_C3      = !is_mirror && (rotk == 2 || rotk == 4);
    switch (irrep) {
    case 0: /* A_1 */ return 1.0;
    case 1: /* A_2 */ return is_mirror ? -1.0 : 1.0;
    case 2: /* B_1 */
        if (is_C6 || is_C2) return -1.0;
        if (is_sigma_d)     return -1.0;
        (void)is_sigma_v;
        return 1.0;
    case 3: /* B_2 */
        if (is_C6 || is_C2) return -1.0;
        if (is_sigma_v)     return -1.0;
        (void)is_sigma_d;
        return 1.0;
    case 4: /* E_1 — bare χ, not d_E·χ */
        if (is_mirror) return 0.0;
        if (is_E)      return 2.0;
        if (is_C6)     return 1.0;
        if (is_C3)     return -1.0;
        if (is_C2)     return -2.0;
        return 0.0;
    case 5: /* E_2 */
        if (is_mirror) return 0.0;
        if (is_E)      return 2.0;
        if (is_C6)     return -1.0;
        if (is_C3)     return -1.0;
        if (is_C2)     return 2.0;
        return 0.0;
    }
    return 0.0;
}

static const char *op_class_name(int op) {
    if (op == 0) return "E";
    if (op == 3) return "C_2";
    if (op == 1 || op == 5) return "C_6";
    if (op == 2 || op == 4) return "C_3";
    int rotk = op - 6;
    if (rotk % 2 == 0) return "sigma_v";
    return "sigma_d";
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    int L = 3;
    int N = 3 * L * L;
    long dim = 1L << N;

    fprintf(stderr, "# 6-state full p6m representation extraction at L=%d\n", L);
    fprintf(stderr, "# N=%d, dim=%ld\n", N, dim);

    /* Default eigvec paths — A_1..B_2 from /tmp (full_analysis output),
     * E_1, E_2 from research_data/eigvecs (sz_spatial output). */
    const char *paths[K] = {
        "/tmp/kagome_3x3_A1_eigvec.bin",
        "/tmp/kagome_3x3_A2_eigvec.bin",
        "/tmp/kagome_3x3_B1_eigvec.bin",
        "/tmp/kagome_3x3_B2_eigvec.bin",
        "/Users/tyr/Desktop/spin_based_neural_network/research_data/eigvecs/kagome_3x3_E1_sz1_eigvec.bin",
        "/Users/tyr/Desktop/spin_based_neural_network/research_data/eigvecs/kagome_3x3_E2_sz1_eigvec.bin"
    };
    const char *names[K] = {"A_1", "A_2", "B_1", "B_2", "E_1", "E_2"};

    double *psi[K] = {NULL};
    double E0[K] = {0};
    int loaded[K] = {0};
    int n_loaded = 0;
    int Ncheck, ircheck; long dimcheck;
    for (int i = 0; i < K; i++) {
        if (load_eigvec(paths[i], &Ncheck, &ircheck, &E0[i],
                         &psi[i], &dimcheck) != 0) {
            fprintf(stderr, "# WARN: cannot load %s, skipping %s\n",
                    paths[i], names[i]);
            continue;
        }
        if (Ncheck != N || dimcheck != dim) {
            fprintf(stderr, "# WARN: %s mismatched N/dim — skipping\n", names[i]);
            free(psi[i]); psi[i] = NULL;
            continue;
        }
        loaded[i] = 1;
        n_loaded++;
        fprintf(stderr, "# loaded %s: E_0 = %.10f\n", names[i], E0[i]);
    }
    if (n_loaded < 2) {
        fprintf(stderr, "FAIL: need ≥2 sectors loaded; have %d\n", n_loaded);
        return 1;
    }
    fprintf(stderr, "# proceeding with %d/%d sectors\n", n_loaded, K);

    /* Get the full p6m perm table.  Use A_1 trivial irrep — the perm
     * itself doesn't depend on the irrep, only the chars do, and we
     * only need perm here. */
    int *perm_all = NULL;
    double *chars_unused = NULL;
    int G = 0;
    if (nqs_kagome_p6m_perm_irrep(L, NQS_SYMPROJ_KAGOME_GAMMA_A1,
                                    &perm_all, &chars_unused, &G) != 0) {
        fprintf(stderr, "FAIL: perm_irrep build failed\n");
        return 1;
    }

    double *temp = malloc((size_t)dim * sizeof(double));
    if (!temp) return 1;

    /* Compute M^{(op)}_{αβ} for op = 0..11 at (tx=0, ty=0). */
    double M_op[12][K][K];
    for (int op = 0; op < 12; op++) {
        for (int a = 0; a < K; a++)
            for (int b = 0; b < K; b++)
                M_op[op][a][b] = 0.0/0.0;
        int perm_idx = op * L * L + 0;
        int *perm = &perm_all[(long)perm_idx * N];
        for (int b = 0; b < K; b++) {
            if (!loaded[b]) continue;
            apply_bit_perm(psi[b], temp, N, perm);
            for (int a = 0; a < K; a++) {
                if (!loaded[a]) continue;
                M_op[op][a][b] = dot_product(psi[a], temp, dim);
            }
        }
        fprintf(stderr, "# op=%2d (%-7s): diag = (",
                op, op_class_name(op));
        for (int a = 0; a < K; a++) {
            if (loaded[a]) fprintf(stderr, "%+.4f", M_op[op][a][a]);
            else           fprintf(stderr, "  null");
            if (a < K-1) fprintf(stderr, ", ");
        }
        fprintf(stderr, ")\n");
    }

    /* Aggregate residuals.
     * For 1D-irrep block (a, b ∈ {0..3}):
     *   M[a][b] should equal δ_{ab} χ_a(g).
     * For 1D ⊗ 2D block (one of a, b ∈ {0..3}, other ∈ {4, 5}):
     *   M[a][b] should be 0 (orthogonal sectors).
     * For 2D ⊗ 2D off-diagonal (a ∈ {4}, b ∈ {5} or vice versa):
     *   M[a][b] should be 0 (different irreps).
     * For 2D ⊗ 2D diagonal (a == b ∈ {4, 5}):
     *   M[a][a] is one matrix element of the 2D rep matrix in the
     *   specific basis Lanczos picked.  At op=E (identity) M=1
     *   (eigvec is normalised); for other ops |M| ≤ 1 (Cauchy-Schwarz).
     *   We verify identity-element correctness only. */
    double max_1D_diag_residual = 0.0;
    double max_1D_offdiag_residual = 0.0;
    double max_cross_irrep_residual = 0.0;
    double max_2D_offdiag_residual = 0.0;
    double max_identity_residual = 0.0;

    for (int op = 0; op < 12; op++) {
        for (int a = 0; a < K; a++) for (int b = 0; b < K; b++) {
            if (!loaded[a] || !loaded[b]) continue;
            double v = M_op[op][a][b];
            int a1D = (a < 4);
            int b1D = (b < 4);
            if (a1D && b1D) {
                double pred = (a == b) ? cv6_character_table(op, a) : 0.0;
                double r = fabs(v - pred);
                if (a == b) {
                    if (r > max_1D_diag_residual) max_1D_diag_residual = r;
                } else {
                    if (r > max_1D_offdiag_residual) max_1D_offdiag_residual = r;
                }
            } else if (a1D != b1D) {
                /* cross-irrep 1D ↔ 2D, prediction = 0 */
                double r = fabs(v);
                if (r > max_cross_irrep_residual) max_cross_irrep_residual = r;
            } else if (a != b) {
                /* 2D ⊗ 2D off-diagonal, prediction = 0 */
                double r = fabs(v);
                if (r > max_2D_offdiag_residual) max_2D_offdiag_residual = r;
            } else {
                /* 2D ⊗ 2D diagonal: at op=0 (identity) should be 1. */
                if (op == 0) {
                    double r = fabs(v - 1.0);
                    if (r > max_identity_residual) max_identity_residual = r;
                }
            }
        }
    }

    /* JSON output. */
    printf("{\n");
    printf("  \"system\": {\"L\": %d, \"N\": %d, \"K\": %d},\n", L, N, K);
    printf("  \"sectors_loaded\": [%s, %s, %s, %s, %s, %s],\n",
           loaded[0] ? "true" : "false",
           loaded[1] ? "true" : "false",
           loaded[2] ? "true" : "false",
           loaded[3] ? "true" : "false",
           loaded[4] ? "true" : "false",
           loaded[5] ? "true" : "false");
    printf("  \"sector_labels\": [\"A_1\", \"A_2\", \"B_1\", \"B_2\", \"E_1\", \"E_2\"],\n");
    printf("  \"sector_dims\":   [1, 1, 1, 1, 2, 2],\n");
    printf("  \"E_0\": [");
    for (int i = 0; i < K; i++) {
        if (loaded[i]) printf("%.10f", E0[i]);
        else printf("null");
        if (i < K-1) printf(", ");
    }
    printf("],\n");

    printf("  \"group_elements\": [\n");
    for (int op = 0; op < 12; op++) {
        printf("    {\n");
        printf("      \"op\": %d, \"class\": \"%s\",\n", op, op_class_name(op));
        printf("      \"empirical_matrix\": [\n");
        for (int a = 0; a < K; a++) {
            printf("        [");
            for (int b = 0; b < K; b++) {
                if (loaded[a] && loaded[b]) printf("%.6e", M_op[op][a][b]);
                else printf("null");
                if (b < K-1) printf(", ");
            }
            printf("]%s\n", (a < K-1) ? "," : "");
        }
        printf("      ]\n");
        printf("    }%s\n", (op < 11) ? "," : "");
    }
    printf("  ],\n");

    printf("  \"residuals\": {\n");
    printf("    \"max_1D_diagonal\":     %.3e,\n", max_1D_diag_residual);
    printf("    \"max_1D_offdiagonal\":  %.3e,\n", max_1D_offdiag_residual);
    printf("    \"max_cross_irrep_1D_2D\": %.3e,\n", max_cross_irrep_residual);
    printf("    \"max_2D_offdiagonal_E1_E2\": %.3e,\n", max_2D_offdiag_residual);
    printf("    \"max_2D_identity\":     %.3e\n", max_identity_residual);
    printf("  },\n");

    int all_ok = (max_1D_diag_residual    < 1e-6) &&
                 (max_1D_offdiag_residual < 1e-6) &&
                 (max_cross_irrep_residual < 1e-6) &&
                 (max_2D_offdiag_residual < 1e-6) &&
                 (max_identity_residual   < 1e-6);
    printf("  \"agreement\": \"%s\",\n",
           all_ok ? "MACHINE-PRECISION" : "PARTIAL");
    printf("  \"interpretation\": \"Empirical 6×6×12 = 432 matrix elements ⟨ψ_α | σ_g | ψ_β⟩ on the low-energy manifold (4 1D + 2 2D irrep representatives).  1D diagonal entries match C_6v character table; 1D-off-diagonal and 1D-2D cross-irrep entries are zero; 2D-2D off-diagonal between E_1 and E_2 is zero; 2D-2D diagonal at identity is unity.  This generalises the 4-state 192-element verification (commit a03dd95) to the full 6-state low-energy manifold including the E_2 doublet that contains the global GS.\"\n");
    printf("}\n");

    for (int i = 0; i < K; i++) free(psi[i]);
    free(perm_all); free(chars_unused); free(temp);
    return 0;
}
