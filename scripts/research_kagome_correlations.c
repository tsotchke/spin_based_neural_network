/*
 * scripts/research_kagome_correlations.c
 *
 * Static spin-spin correlations ⟨S_i·S_j⟩ and structure factor S(q) on
 * the kagome 2×2 PBC AFM ground state.  Pipeline:
 *
 *   1. Random complex RBM → P_{Γ,B_1} → projecting Lanczos → ψ_0
 *      (matches libirrep ED to ~1e-11)
 *   2. Compute the 12×12 correlation matrix
 *        C_{ij} = ⟨ψ_0 | S_i·S_j | ψ_0⟩
 *      directly on the materialised real wavefunction.
 *   3. Group correlations by inter-site distance |r_i − r_j|.
 *   4. Compute the static structure factor at the 4 inequivalent
 *      momentum points of the 2×2 PBC cluster:
 *        S(q) = (1/N) Σ_{ij} e^{i q · (r_i − r_j)} C_{ij}
 *      Momenta: Γ = (0,0), M_x = (π,0), M_y = (0,π), K-equiv (π,π).
 *
 * Physics interpretation:
 *   - kagome 120° AFM (Néel-ordered candidate): Bragg peak at K-point.
 *   - kagome QSL (spin liquid, herbertsmithite): no sharp peaks; broad
 *     diffuse signal.
 *   - Numbers: |C(d)| should decay with d; the rate distinguishes
 *     algebraic (QSL with gapless modes) from exponential (gapped QSL).
 *
 * Build: make IRREP_ENABLE=1 research_kagome_correlations
 * Run:   ./build/research_kagome_correlations
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nqs/nqs_config.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_symproj.h"
#include "nqs/nqs_lanczos.h"
#include "mps/lanczos.h"

/* Kagome 2×2 PBC site coordinates (cell_x, cell_y, sublattice).
 * Using the convention from kg_site() in nqs_lanczos.c:
 *     site = 3 * (cx * Ly + cy) + sub        with sub ∈ {0,1,2} = {A,B,C}
 *
 * Cartesian position of (cx, cy, sub) in the kagome basis:
 *   a1 = (1, 0)
 *   a2 = (1/2, √3/2)
 *   sublattice offsets: A → (0, 0); B → a1/2; C → a2/2.
 *
 * Two-cell PBC: shortest-image distance evaluated explicitly. */
static double cart_x[12], cart_y[12];
static const double sqrt3_over_2 = 0.86602540378443864676;
static const double a1x = 1.0, a1y = 0.0;
static const double a2x = 0.5, a2y = 0.86602540378443864676;

static void kagome_2x2_build_coords(void) {
    int Lx = 2, Ly = 2;
    for (int cx = 0; cx < Lx; cx++) for (int cy = 0; cy < Ly; cy++) {
        int idx_A = 3 * (cx * Ly + cy) + 0;
        int idx_B = 3 * (cx * Ly + cy) + 1;
        int idx_C = 3 * (cx * Ly + cy) + 2;
        double xc = (double)cx * a1x + (double)cy * a2x;
        double yc = (double)cx * a1y + (double)cy * a2y;
        cart_x[idx_A] = xc;            cart_y[idx_A] = yc;
        cart_x[idx_B] = xc + a1x/2.0;   cart_y[idx_B] = yc + a1y/2.0;
        cart_x[idx_C] = xc + a2x/2.0;   cart_y[idx_C] = yc + a2y/2.0;
    }
    (void)sqrt3_over_2;
}

/* Shortest-image distance on the 2×2 supercell defined by lattice
 * vectors L_x = 2 a1, L_y = 2 a2.  Compute Δr_x, Δr_y in (a1, a2)
 * basis, fold each to [-1, 1], convert back to Cartesian, take norm. */
static double pbc_distance(int i, int j) {
    /* Inverse of (a1, a2) basis matrix.  a1 = (1, 0), a2 = (0.5, √3/2).
     * Δ in cartesian → coefficients (n_a, n_b) such that
     * Δ = n_a a1 + n_b a2.  Solve: Δ_x = n_a + 0.5 n_b, Δ_y = (√3/2) n_b.
     * ⇒ n_b = (2/√3) Δ_y; n_a = Δ_x − 0.5 n_b. */
    double dx = cart_x[i] - cart_x[j];
    double dy = cart_y[i] - cart_y[j];
    double n_b = (2.0 / sqrt(3.0)) * dy;
    double n_a = dx - 0.5 * n_b;
    /* Fold to [-1, 1] supercell (Lx = Ly = 2 cells). */
    n_a -= 2.0 * floor((n_a + 1.0) / 2.0);
    n_b -= 2.0 * floor((n_b + 1.0) / 2.0);
    /* Reconstruct cartesian */
    double dx_w = n_a * a1x + n_b * a2x;
    double dy_w = n_a * a1y + n_b * a2y;
    return sqrt(dx_w * dx_w + dy_w * dy_w);
}

/* Compute C_{ij} = ⟨S_i·S_j⟩ for a real-valued normalized state psi.
 *
 *   C_{ij} = ⟨S^z_i S^z_j⟩ + ⟨(S^+_i S^-_j + S^-_i S^+_j)/2⟩
 *          = diag_term + offdiag_term
 *
 *   diag_term = (1/4) Σ_s ψ(s)² · s_i · s_j
 *             where s_k = +1 if bit k of s is 0, else -1.
 *
 *   offdiag_term = Σ_{s : s_i ≠ s_j} ψ(s) · ψ(s ⊕ {i,j})  (real ψ)
 *
 * The (1/2) prefactor in (S^+S^- + S^-S^+) cancels with the
 * single-direction enumeration of pairs.  See full derivation in
 * the commit message accompanying this file. */
static double spin_correlation(const double *psi, long dim,
                                int N, int i, int j) {
    double diag = 0.0, off = 0.0;
    long mask_ij = (1L << i) | (1L << j);
    for (long s = 0; s < dim; s++) {
        int b_i = (s >> i) & 1;          /* 0 ↔ up,  1 ↔ down */
        int b_j = (s >> j) & 1;
        int s_i = b_i ? -1 : +1;
        int s_j = b_j ? -1 : +1;
        diag += 0.25 * psi[s] * psi[s] * (double)(s_i * s_j);
        if (b_i != b_j) {
            long s_flip = s ^ mask_ij;
            /* Contribute once per unordered {s, s_flip} pair: enforce
             * canonical s < s_flip. */
            if (s < s_flip) off += psi[s] * psi[s_flip];
        }
    }
    /* The off-diagonal sum was over unordered pairs, but each ordered
     * pair (s, s_flip) appears in both orders in the operator action.
     * The factor of (1/2) in the operator definition is cancelled by
     * the unordered counting; check by comparing to ⟨ψ|H|ψ⟩ pieces. */
    return diag + off;
    (void)N;
}

int main(void) {
    int L = 2;
    int N = 3 * L * L;          /* 12 */
    long dim = 1L << N;

    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz           = NQS_ANSATZ_COMPLEX_RBM;
    cfg.rbm_hidden_units = 16;
    cfg.rbm_init_scale   = 0.05;
    cfg.hamiltonian      = NQS_HAM_KAGOME_HEISENBERG;
    cfg.j_coupling       = 1.0;
    cfg.kagome_pbc       = 1;
    cfg.rng_seed         = 0xB1B1B1B1u;

    printf("# kagome 2×2 PBC AFM Heisenberg static spin correlations\n");
    printf("#   ⟨S_i·S_j⟩ on the libirrep-ED-quality (Γ, B_1) ground state\n");
    printf("\n");

    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    if (!a) return 1;

    int *perm = NULL;
    double *chars = NULL;
    int G = 0;
    if (nqs_kagome_p6m_perm_irrep(L, NQS_SYMPROJ_KAGOME_GAMMA_B1,
                                    &perm, &chars, &G) != 0) {
        fprintf(stderr, "perm build failed\n"); return 1;
    }
    nqs_symproj_wrapper_t wrap = {
        .base_log_amp       = nqs_ansatz_log_amp,
        .base_user          = a,
        .num_sites          = N,
        .num_group_elements = G,
        .perm               = perm,
        .characters         = chars,
    };

    double *psi = malloc((size_t)dim * sizeof(double));
    lanczos_result_t lr = (lanczos_result_t){0};
    double e0 = 0.0;
    int rc = nqs_lanczos_refine_kagome_heisenberg_projected(
        nqs_symproj_log_amp, &wrap, L, L, cfg.j_coupling, cfg.kagome_pbc,
        perm, chars, G, 300, 1e-12, &e0, psi, &lr);
    if (rc != 0) { fprintf(stderr, "Lanczos failed (%d)\n", rc); return 1; }
    printf("# E_0 = %.10f  Δ vs ED = %.2e\n", e0, fabs(e0 - (-5.4448752170)));
    printf("# ψ_0 norm = 1 (Lanczos-normalised)\n\n");

    kagome_2x2_build_coords();

    /* Build the full 12×12 correlation matrix. */
    double C[12][12];
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
        if (i == j) C[i][j] = 0.75;     /* ⟨S_i² ⟩ = S(S+1) = 3/4 */
        else C[i][j] = spin_correlation(psi, dim, N, i, j);
    }

    /* Group correlations by shortest-image distance (round to 4 decimals
     * to identify equivalence classes). */
    typedef struct { double dist; double sum; int count; } shell_t;
    shell_t shells[64];
    int n_shells = 0;
    for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) {
        double d = pbc_distance(i, j);
        long key = (long)(d * 10000.0 + 0.5);   /* rounded */
        int found = -1;
        for (int s = 0; s < n_shells; s++) {
            if ((long)(shells[s].dist * 10000.0 + 0.5) == key) { found = s; break; }
        }
        if (found < 0) {
            shells[n_shells].dist  = d;
            shells[n_shells].sum   = C[i][j];
            shells[n_shells].count = 1;
            n_shells++;
        } else {
            shells[found].sum += C[i][j];
            shells[found].count++;
        }
    }
    /* Sort shells by distance, ascending. */
    for (int i = 0; i < n_shells - 1; i++) {
        int imin = i;
        for (int j = i + 1; j < n_shells; j++)
            if (shells[j].dist < shells[imin].dist) imin = j;
        if (imin != i) { shell_t t = shells[i]; shells[i] = shells[imin]; shells[imin] = t; }
    }

    printf("# Spin-spin correlations grouped by shortest-image distance:\n");
    printf("# %4s %5s %14s\n", "d", "#pairs", "⟨S_i·S_j⟩_avg");
    double total_check = 0.0;
    for (int s = 0; s < n_shells; s++) {
        double avg = shells[s].sum / (double)shells[s].count;
        printf("  %5.3f %5d %14.10f\n", shells[s].dist, shells[s].count, avg);
        total_check += 2.0 * shells[s].sum;     /* sum over all unordered pairs */
    }
    /* Sum-rule check: 2·Σ_{i<j} ⟨S_i·S_j⟩ + Σ_i ⟨S_i²⟩ = ⟨(Σ S)²⟩ = S_tot(S_tot+1).
     * For singlet S_tot = 0 → expected sum = 0.  Recover total spin from this. */
    double diag_sum = 0.0;
    for (int i = 0; i < N; i++) diag_sum += C[i][i];   /* = N · 3/4 */
    double S_total_squared = total_check + diag_sum;
    /* S_total_squared = S(S+1).  Solve for S. */
    double S_total = 0.5 * (sqrt(1.0 + 4.0 * S_total_squared) - 1.0);
    printf("\n# Sum-rule check: Σ_{ij} ⟨S_i·S_j⟩ = ⟨(Σ S)²⟩ = S_tot(S_tot+1)\n");
    printf("#   ⟨(Σ S)²⟩ = %.6f → S_total ≈ %.4f (expect 0 for AFM singlet GS)\n",
           S_total_squared, S_total);

    /* Static structure factor S(q) at the 4 inequivalent momentum points
     * of the 2×2 supercell in the (a1*, a2*) reciprocal basis.
     *   q points: (n_a · π, n_b · π) for (n_a, n_b) ∈ {0, 1}².
     * Reciprocal lattice vectors b1, b2 such that b1·a1 = 2π, etc.
     *   b1 = 2π (1, -1/√3), b2 = 2π (0, 2/√3).
     * For our supercell (Lx = 2 cells of a1, Ly = 2 cells of a2),
     * allowed momenta are q = (m1 b1 + m2 b2) / 2 with m_i ∈ {0, 1}.
     */
    double bx1 = 2.0 * M_PI;             double by1 = -2.0 * M_PI / sqrt(3.0);
    double bx2 = 0.0;                    double by2 = 4.0 * M_PI / sqrt(3.0);
    int    momentum_grid[4][2] = { {0,0}, {1,0}, {0,1}, {1,1} };
    const char *q_label[4] = { "Γ (0,0)", "M_a (b1/2)", "M_b (b2/2)", "K-equiv (b1+b2)/2" };

    printf("\n# Static structure factor S(q) = (1/N) Σ_{ij} e^{i q·(r_i − r_j)} ⟨S_i·S_j⟩:\n");
    printf("# %-20s %15s %15s\n", "q", "Re S(q)", "Im S(q)");
    for (int m = 0; m < 4; m++) {
        double qx = 0.5 * ((double)momentum_grid[m][0] * bx1 + (double)momentum_grid[m][1] * bx2);
        double qy = 0.5 * ((double)momentum_grid[m][0] * by1 + (double)momentum_grid[m][1] * by2);
        double Sq_re = 0.0, Sq_im = 0.0;
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            double phase = qx * (cart_x[i] - cart_x[j]) + qy * (cart_y[i] - cart_y[j]);
            Sq_re += cos(phase) * C[i][j];
            Sq_im += sin(phase) * C[i][j];
        }
        Sq_re /= (double)N;
        Sq_im /= (double)N;
        printf("  %-20s %15.8f %15.8f\n", q_label[m], Sq_re, Sq_im);
    }

    /* Print full 12×12 correlation matrix for completeness. */
    printf("\n# Full ⟨S_i·S_j⟩ matrix (lower triangle):\n");
    for (int i = 0; i < N; i++) {
        printf("# row %2d:", i);
        for (int j = 0; j <= i; j++) {
            printf(" %+8.4f", C[i][j]);
        }
        printf("\n");
    }

    free(psi); free(perm); free(chars);
    nqs_ansatz_free(a);
    return 0;
}
