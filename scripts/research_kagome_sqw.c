/*
 * scripts/research_kagome_sqw.c
 *
 * Dynamic structure factor S(q, ω) of the kagome 2×2 PBC AFM ground
 * state via Lanczos continued fraction.  Pipeline:
 *
 *   1. ψ_0 from projecting Lanczos in (Γ, B_1) sector  (E_0 to 1e-11)
 *   2. For each q ∈ {Γ, M_a, M_b, K-equiv}:
 *        |φ_q⟩ = S^z_q |ψ_0⟩    (complex; Re and Im parts each real)
 *      where S^z_q = (1/√N) Σ_i e^{i q · r_i} S^z_i
 *   3. Run Lanczos seeded from Re|φ_q⟩ → α[], β[], ‖Re|φ_q⟩‖
 *      Run Lanczos seeded from Im|φ_q⟩ → α'[], β'[], ‖Im|φ_q⟩‖
 *   4. Evaluate the spectral function via continued fraction at each
 *      ω of an output grid:
 *        S(q, ω) = -(1/π) Im [G_Re(ω + iη) + G_Im(ω + iη)]
 *      where G_X(z) = ‖X⟩‖² / (z − α_X[0] − β_X[1]² / (z − α_X[1] − …))
 *      and the H eigenvalue spectrum is referenced to E_0:
 *        ω here is excitation energy ω = E_n − E_0.
 *
 * For real H + real ψ_0, the cross-terms ⟨Re | (ω−H+iη)^{-1} | Im⟩
 * vanish because (ω−H+iη)^{-1} is symmetric (in the real-symmetric
 * basis), so S(q, ω) cleanly splits into the two contributions above.
 *
 * Output: S(q, ω) at ω ∈ [0, ω_max] for each of the 4 distinct momenta.
 * Predicts inelastic-neutron-scattering line shapes that experiments
 * on kagome materials (herbertsmithite, ZnCu₃(OH)₆Cl₂) directly probe.
 *
 * Build: make IRREP_ENABLE=1 research_kagome_sqw
 * Run:   ./build/research_kagome_sqw
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

/* Cartesian site coordinates and helpers — same convention as
 * scripts/research_kagome_correlations.c. */
static double cart_x[12], cart_y[12];
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
}

/* Apply S^z_q to a real ψ:
 *      [S^z_q ψ](s) = (1/√N) Σ_i e^{i q·r_i} (s_i / 2) ψ(s)       (diag)
 * where s_i ∈ {+1, -1}.  Output split into Re and Im parts. */
static void apply_Szq(const double *psi_in, long dim, int N,
                       double qx, double qy,
                       double *out_re, double *out_im) {
    double inv_sqrtN = 1.0 / sqrt((double)N);
    for (long s = 0; s < dim; s++) {
        double sumR = 0.0, sumI = 0.0;
        for (int i = 0; i < N; i++) {
            int b = (s >> i) & 1;
            double sz = b ? -0.5 : +0.5;
            double phase = qx * cart_x[i] + qy * cart_y[i];
            sumR += sz * cos(phase);
            sumI += sz * sin(phase);
        }
        out_re[s] = inv_sqrtN * sumR * psi_in[s];
        out_im[s] = inv_sqrtN * sumI * psi_in[s];
    }
}

/* Forward declaration of internal kagome H matvec — we'll hook into
 * the existing one via a wrapper-friendly entry. */
typedef struct {
    int Lx_cells, Ly_cells, N;
    double J;
    int pbc;
} kagome_ctx_t;

static double kagome_diag(long state, const kagome_ctx_t *ctx) {
    double e = 0.0;
    int Lx = ctx->Lx_cells, Ly = ctx->Ly_cells;
    double J = ctx->J;
    for (int cx = 0; cx < Lx; cx++) for (int cy = 0; cy < Ly; cy++) {
        int A = 3 * (cx * Ly + cy) + 0;
        int B = 3 * (cx * Ly + cy) + 1;
        int C = 3 * (cx * Ly + cy) + 2;
        int sA = ((state >> A) & 1) ? -1 : +1;
        int sB = ((state >> B) & 1) ? -1 : +1;
        int sC = ((state >> C) & 1) ? -1 : +1;
        e += 0.25 * J * (double)(sA*sB + sA*sC + sB*sC);
        int cxm = (cx - 1 + Lx) % Lx;
        int cym = (cy - 1 + Ly) % Ly;
        int Bm = 3 * (cxm * Ly + cy) + 1;
        int Cm = 3 * (cx * Ly + cym) + 2;
        int sBm = ((state >> Bm) & 1) ? -1 : +1;
        int sCm = ((state >> Cm) & 1) ? -1 : +1;
        e += 0.25 * J * (double)(sA*sBm + sA*sCm + sBm*sCm);
    }
    return e;
}

static void kagome_matvec(const double *in, double *out, long dim, void *ud) {
    kagome_ctx_t *ctx = (kagome_ctx_t *)ud;
    int Lx = ctx->Lx_cells, Ly = ctx->Ly_cells;
    double J = ctx->J;
    int pbc = ctx->pbc;
    for (long s = 0; s < dim; s++) {
        double y = kagome_diag(s, ctx) * in[s];
        for (int cx = 0; cx < Lx; cx++) for (int cy = 0; cy < Ly; cy++) {
            int A = 3 * (cx * Ly + cy) + 0;
            int B = 3 * (cx * Ly + cy) + 1;
            int C = 3 * (cx * Ly + cy) + 2;
            int up_bonds[3][2] = {{A,B}, {A,C}, {B,C}};
            for (int b = 0; b < 3; b++) {
                int u = up_bonds[b][0], v = up_bonds[b][1];
                int su = ((s >> u) & 1) ? -1 : +1;
                int sv = ((s >> v) & 1) ? -1 : +1;
                if (su != sv) y += 0.5 * J * in[s ^ (1L << u) ^ (1L << v)];
            }
            int cxm = (cx - 1 + Lx) % Lx;
            int cym = (cy - 1 + Ly) % Ly;
            int Bm = 3 * (cxm * Ly + cy) + 1;
            int Cm = 3 * (cx * Ly + cym) + 2;
            int dn_bonds[3][2] = {{A,Bm}, {A,Cm}, {Bm,Cm}};
            for (int b = 0; b < 3; b++) {
                int u = dn_bonds[b][0], v = dn_bonds[b][1];
                int su = ((s >> u) & 1) ? -1 : +1;
                int sv = ((s >> v) & 1) ? -1 : +1;
                if (su != sv) y += 0.5 * J * in[s ^ (1L << u) ^ (1L << v)];
            }
        }
        out[s] = y;
    }
    (void)pbc;
}

int main(void) {
    int L = 2;
    int N = 3 * L * L;             /* 12 */
    long dim = 1L << N;

    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz           = NQS_ANSATZ_COMPLEX_RBM;
    cfg.rbm_hidden_units = 16;
    cfg.rbm_init_scale   = 0.05;
    cfg.hamiltonian      = NQS_HAM_KAGOME_HEISENBERG;
    cfg.j_coupling       = 1.0;
    cfg.kagome_pbc       = 1;
    cfg.rng_seed         = 0xB1B1B1B1u;

    printf("# kagome 2×2 PBC AFM Heisenberg dynamic structure factor S(q, ω)\n");
    printf("# pipeline: random RBM → P_{Γ,B_1} → projecting Lanczos → ψ_0\n");
    printf("#           |φ_q⟩ = S^z_q |ψ_0⟩  (Re, Im parts independently)\n");
    printf("#           continued-fraction Lanczos from Re|φ_q⟩ and Im|φ_q⟩\n");
    printf("#           S(q, ω) = -(1/π) Im[G_Re(ω+iη) + G_Im(ω+iη)]\n");
    printf("\n");

    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    if (!a) return 1;

    int *perm = NULL;
    double *chars = NULL;
    int G = 0;
    if (nqs_kagome_p6m_perm_irrep(L, NQS_SYMPROJ_KAGOME_GAMMA_B1,
                                    &perm, &chars, &G) != 0) return 1;
    nqs_symproj_wrapper_t wrap = {
        .base_log_amp       = nqs_ansatz_log_amp,
        .base_user          = a,
        .num_sites          = N,
        .num_group_elements = G,
        .perm               = perm,
        .characters         = chars,
    };

    double *psi0 = malloc((size_t)dim * sizeof(double));
    lanczos_result_t lr = (lanczos_result_t){0};
    double e0 = 0.0;
    int rc = nqs_lanczos_refine_kagome_heisenberg_projected(
        nqs_symproj_log_amp, &wrap, L, L, cfg.j_coupling, cfg.kagome_pbc,
        perm, chars, G, 300, 1e-12, &e0, psi0, &lr);
    if (rc != 0) { fprintf(stderr, "Lanczos failed\n"); return 1; }
    printf("# E_0 = %.10f  Δ vs ED = %.2e\n\n", e0, fabs(e0 - (-5.4448752170)));

    kagome_2x2_build_coords();

    /* Reciprocal lattice vectors */
    double bx1 = 2.0 * M_PI;             double by1 = -2.0 * M_PI / sqrt(3.0);
    double bx2 = 0.0;                    double by2 = 4.0 * M_PI / sqrt(3.0);
    int momentum_grid[4][2] = { {0,0}, {1,0}, {0,1}, {1,1} };
    const char *q_label[4] = { "Γ", "M_a", "M_b", "K-eq" };

    kagome_ctx_t ctx = { .Lx_cells = L, .Ly_cells = L, .N = N,
                          .J = cfg.j_coupling, .pbc = cfg.kagome_pbc };

    /* ω grid: 0 to 8 J in 81 steps; broadening η = 0.05 J. */
    int n_omega = 81;
    double omega_max = 8.0;
    double eta = 0.05;
    int max_iters = 200;

    double *alpha_re = malloc((size_t)max_iters * sizeof(double));
    double *beta_re  = malloc((size_t)(max_iters + 1) * sizeof(double));
    double *alpha_im = malloc((size_t)max_iters * sizeof(double));
    double *beta_im  = malloc((size_t)(max_iters + 1) * sizeof(double));
    double *phi_re   = malloc((size_t)dim * sizeof(double));
    double *phi_im   = malloc((size_t)dim * sizeof(double));

    /* Output: a 4×n_omega matrix S(q, ω) printed as a table. */
    double *S_grid = calloc((size_t)4 * (size_t)n_omega, sizeof(double));

    for (int m = 0; m < 4; m++) {
        double qx = 0.5 * ((double)momentum_grid[m][0] * bx1 +
                            (double)momentum_grid[m][1] * bx2);
        double qy = 0.5 * ((double)momentum_grid[m][0] * by1 +
                            (double)momentum_grid[m][1] * by2);

        /* Build |φ_q⟩ = S^z_q |ψ_0⟩ */
        apply_Szq(psi0, dim, N, qx, qy, phi_re, phi_im);

        /* Continued-fraction Lanczos seeded from Re|φ_q⟩ */
        int Kr = 0; double norm_re = 0.0;
        rc = lanczos_continued_fraction(kagome_matvec, &ctx, dim,
                                         max_iters, phi_re,
                                         alpha_re, beta_re, &Kr, &norm_re);
        if (rc != 0) { fprintf(stderr, "CF Lanczos Re failed\n"); return 1; }

        /* Continued-fraction Lanczos seeded from Im|φ_q⟩ */
        int Ki = 0; double norm_im = 0.0;
        rc = lanczos_continued_fraction(kagome_matvec, &ctx, dim,
                                         max_iters, phi_im,
                                         alpha_im, beta_im, &Ki, &norm_im);
        if (rc != 0) { fprintf(stderr, "CF Lanczos Im failed\n"); return 1; }

        /* Sum rule check (frequency integral): ∫ S(q,ω) dω =
         * ⟨ψ_0|S^z_{-q} S^z_q|ψ_0⟩ = ‖φ_re‖² + ‖φ_im‖²  for real-q */
        double sum_rule = norm_re * norm_re + norm_im * norm_im;

        printf("# q = %s  (qx, qy) = (%.4f, %.4f)\n", q_label[m], qx, qy);
        printf("#   ‖Re|φ_q⟩‖² = %.6f, ‖Im|φ_q⟩‖² = %.6f, sum = %.6f\n",
               norm_re*norm_re, norm_im*norm_im, sum_rule);
        printf("#   K_re = %d, K_im = %d Krylov iters\n", Kr, Ki);

        /* Evaluate spectral function on ω-grid.
         * ω measured RELATIVE to E_0: shifted Hamiltonian (H − E_0). */
        for (int io = 0; io < n_omega; io++) {
            double omega = (double)io * omega_max / (n_omega - 1);
            double Gr_re = 0, Gr_im = 0;
            double Gi_re = 0, Gi_im = 0;
            if (Kr > 0) lanczos_cf_evaluate(Kr, alpha_re, beta_re, norm_re,
                                             omega + e0, eta, &Gr_re, &Gr_im);
            if (Ki > 0) lanczos_cf_evaluate(Ki, alpha_im, beta_im, norm_im,
                                             omega + e0, eta, &Gi_re, &Gi_im);
            /* S(q, ω) = -(1/π) Im G(ω + iη)  with the H − E_0 shift
             * built in by passing z = ω + E_0 + i η (no, we want
             * z = (E_0 + ω) + iη to land at H eigenvalue E_0 + ω = E_n).
             * Actually S(q, ω) = sum_n |⟨n|S^z_q|0⟩|² δ(ω − (E_n − E_0))
             * which from the CF representation (without shift) is
             * -(1/π) Im[G_full(ω + E_0 + iη)] · 1 (consistent). */
            double Sqw = -(1.0 / M_PI) * (Gr_im + Gi_im);
            S_grid[m * n_omega + io] = Sqw;
        }
    }

    printf("\n# S(q, ω) table — columns: ω, S(Γ,ω), S(M_a,ω), S(M_b,ω), S(K-eq,ω)\n");
    printf("# η = %.3f J, %d ω-points spanning [0, %.1f] J\n", eta, n_omega, omega_max);
    for (int io = 0; io < n_omega; io++) {
        double omega = (double)io * omega_max / (n_omega - 1);
        printf("  %7.4f  %12.6e  %12.6e  %12.6e  %12.6e\n",
               omega,
               S_grid[0 * n_omega + io], S_grid[1 * n_omega + io],
               S_grid[2 * n_omega + io], S_grid[3 * n_omega + io]);
    }

    /* Identify peaks: simple local maxima above a threshold. */
    printf("\n# Spectral peaks (Lorentzian-broadened delta functions):\n");
    printf("# %-8s %12s %12s\n", "q", "ω_peak (J)", "S_peak");
    for (int m = 0; m < 4; m++) {
        for (int io = 1; io < n_omega - 1; io++) {
            double s_prev = S_grid[m * n_omega + (io - 1)];
            double s_here = S_grid[m * n_omega + io];
            double s_next = S_grid[m * n_omega + (io + 1)];
            if (s_here > s_prev && s_here > s_next && s_here > 0.05) {
                double omega = (double)io * omega_max / (n_omega - 1);
                printf("  %-8s %12.4f %12.6f\n", q_label[m], omega, s_here);
            }
        }
    }

    free(S_grid); free(alpha_re); free(beta_re); free(alpha_im); free(beta_im);
    free(phi_re); free(phi_im); free(psi0); free(perm); free(chars);
    nqs_ansatz_free(a);
    return 0;
}
