/*
 * scripts/research_kagome_tee.c
 *
 * Topological entanglement entropy of the kagome AFM Heisenberg
 * 2×2 PBC ground state.  Pipeline:
 *
 *   1. Random complex RBM (untrained) → P_{Γ,B_1} projection
 *   2. Lanczos with in-loop sector projection → ED-quality ψ_0
 *      (matches libirrep sector ED to ~1e-11 — see test_nqs_sector_lanczos)
 *   3. For each subsystem geometry A (varying size and shape):
 *        ρ_A = Tr_B |ψ_0⟩⟨ψ_0|
 *        S_1(A) = -Tr ρ_A log ρ_A          (von Neumann)
 *        S_2(A) = -log Tr ρ_A^2             (Renyi α=2)
 *   4. Linear fit S(A) = α·|∂A| − γ + ...
 *      γ is the topological entanglement entropy.  Z₂ spin liquid:
 *      γ = log 2 ≈ 0.6931.  Trivial / antiferromagnet: γ = 0.
 *
 * The N=12 PBC cluster is small enough that exact ψ_0 is fast (~0.5 s)
 * but boundary-length resolution is limited — we report the full
 * S(|A|, |∂A|) table plus the fit.  Larger clusters (N=27 PBC, N=48
 * via NQS variational floor) extend the boundary range; this is the
 * methodological baseline.
 *
 * Build:  make IRREP_ENABLE=1 research_kagome_tee
 * Run:    ./build/research_kagome_tee
 *
 * References:
 *   Kitaev & Preskill, PRL 96, 110404 (2006) — topological entropy
 *   Levin & Wen, PRL 96, 110405 (2006)        — same formula, dual derivation
 *   Depenbrock-McCulloch-Schollwöck, PRL 109, 067201 (2012) — γ ≈ log 2
 *     for kagome AFM via DMRG on cylinders.  Open question whether
 *     small-cluster PBC clusters reproduce this.
 */
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nqs/nqs_config.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_symproj.h"
#include "nqs/nqs_lanczos.h"
#include "mps/lanczos.h"

#ifdef SPIN_NN_HAS_IRREP
#include "libirrep_bridge.h"
#endif

/* Count the number of bonds with one endpoint in A and one in B
 * (the boundary length |∂A|) on the kagome 2×2 PBC cluster.  Bonds:
 * 6 up-triangles + 6 down-triangles × 3 bonds each = 24, but
 * dedup by considering each bond once. */
static int kagome_2x2_pbc_boundary_length(const int *in_A, int N) {
    /* Encode bonds explicitly for the L=2 PBC kagome cluster.  Site
     * indexing matches kg_site(cx, cy, sub, Ly) = 3*(cx*Ly + cy) + sub
     * with sub ∈ {0,1,2} = {A, B, C} sublattices. */
    (void)N;
    /* Up-triangle bonds within each cell: (A,B), (A,C), (B,C). */
    /* Down-triangle bonds:                                    *
     *   (A_(cx,cy),     B_(cx-1,cy))                          *
     *   (A_(cx,cy),     C_(cx,cy-1))                          *
     *   (B_(cx-1,cy),   C_(cx,cy-1))                          */
    int Lx = 2, Ly = 2;
    int boundary = 0;
    for (int cx = 0; cx < Lx; cx++) for (int cy = 0; cy < Ly; cy++) {
        int A = 3 * (cx * Ly + cy) + 0;
        int B = 3 * (cx * Ly + cy) + 1;
        int C = 3 * (cx * Ly + cy) + 2;
        int cxm = (cx - 1 + Lx) % Lx;
        int cym = (cy - 1 + Ly) % Ly;
        int Bm = 3 * (cxm * Ly + cy) + 1;
        int Cm = 3 * (cx * Ly + cym) + 2;
        int bonds[6][2] = {
            {A,B},  {A,C},  {B,C},
            {A,Bm}, {A,Cm}, {Bm,Cm}
        };
        for (int b = 0; b < 6; b++) {
            int u = bonds[b][0], v = bonds[b][1];
            if (in_A[u] != in_A[v]) boundary++;
        }
    }
    /* Each bond counted once because the down-triangle bonds attach
     * to a unique up-triangle's A corner. */
    return boundary;
}

static void print_subsystem_indicator(const int *in_A, int N) {
    printf("    sites_A = {");
    int first = 1;
    for (int i = 0; i < N; i++) {
        if (in_A[i]) {
            if (!first) printf(",");
            printf("%d", i);
            first = 0;
        }
    }
    printf("}");
}

#ifdef SPIN_NN_HAS_IRREP

static int compute_entropies(const double _Complex *psi, int N,
                              const int *in_A,
                              double *out_S1, double *out_S2) {
    /* Build sites_A list */
    int sites_A[12];
    int nA = 0;
    for (int i = 0; i < N; i++) if (in_A[i]) sites_A[nA++] = i;
    long dimA = 1L << nA;
    double _Complex *rho = malloc((size_t)dimA * (size_t)dimA *
                                   sizeof(double _Complex));
    if (!rho) return -1;
    int rc = libirrep_bridge_partial_trace_spin_half(N, psi, sites_A, nA, rho);
    if (rc != 0) { free(rho); return rc; }
    /* von Neumann (destroys rho) */
    double _Complex *rho_copy = malloc((size_t)dimA * (size_t)dimA *
                                         sizeof(double _Complex));
    memcpy(rho_copy, rho, (size_t)dimA * (size_t)dimA *
                            sizeof(double _Complex));
    /* libirrep_bridge_entropy_* takes the *matrix dimension* (= 2^nA for
     * spin-1/2), not the number of sites — the bridge docstring is
     * inconsistent with the underlying libirrep call.  test_libirrep_rdm
     * passes n=2 for a single-qubit ρ confirming this. */
    rc = libirrep_bridge_entropy_vonneumann(rho_copy, (int)dimA, out_S1);
    free(rho_copy);
    if (rc != 0) { free(rho); return rc; }
    rc = libirrep_bridge_entropy_renyi(rho, (int)dimA, 2.0, out_S2);
    free(rho);
    return rc;
}

#endif

int main(void) {
#ifndef SPIN_NN_HAS_IRREP
    fprintf(stderr,
            "research_kagome_tee: requires SPIN_NN_HAS_IRREP.  "
            "Build with `make IRREP_ENABLE=1 research_kagome_tee`.\n");
    return 1;
#else
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

    printf("# kagome 2×2 PBC AFM Heisenberg topological entanglement entropy\n");
    printf("# pipeline: random RBM → P_{Γ,B_1} → projecting Lanczos → ψ_0\n");
    printf("\n");

    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    if (!a) { fprintf(stderr, "ansatz_create failed\n"); return 1; }

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

    /* Lanczos refine: get the ψ_0 vector, sector-projected. */
    double *psi_real = malloc((size_t)dim * sizeof(double));
    if (!psi_real) return 1;
    lanczos_result_t lr = (lanczos_result_t){0};
    double e0 = 0.0;
    int rc = nqs_lanczos_refine_kagome_heisenberg_projected(
        nqs_symproj_log_amp, &wrap, L, L, cfg.j_coupling, cfg.kagome_pbc,
        perm, chars, G, 300, 1e-12, &e0, psi_real, &lr);
    if (rc != 0) { fprintf(stderr, "Lanczos failed (%d)\n", rc); return 1; }
    printf("# E_0 = %.10f  (libirrep ED: -5.4448752170)  Δ = %.2e\n",
           e0, fabs(e0 - (-5.4448752170)));
    printf("# Lanczos iters = %d\n\n", lr.iterations);

    /* Convert real → complex (imag = 0).  ψ_0 is normalized. */
    double _Complex *psi = malloc((size_t)dim * sizeof(double _Complex));
    for (long s = 0; s < dim; s++) psi[s] = psi_real[s] + 0.0 * I;

    /* Subsystem geometries to scan: enumerate by site count nA from 1
     * to 6 (half the cluster), and for each nA pick a few candidate
     * shapes by choosing site indices.  We use a "first nA sites" rule
     * for reproducibility, plus a few hand-chosen "compact" shapes. */
    printf("# subsystem-size scan (sites_A picked by first-nA rule):\n");
    printf("# %3s %4s %12s %12s\n", "nA", "|∂A|", "S_1 (vN)", "S_2 (Renyi)");

    /* Collect (boundary_length, S_1) pairs for the area-law fit. */
    int    fit_n = 0;
    double fit_b[24];   /* boundary length */
    double fit_S[24];   /* von Neumann entropy */

    int in_A[12];
    /* First-nA shape: keep sites 0..nA-1 in A. */
    for (int nA = 1; nA <= 6; nA++) {
        for (int i = 0; i < N; i++) in_A[i] = (i < nA) ? 1 : 0;
        int boundary = kagome_2x2_pbc_boundary_length(in_A, N);
        double S1 = 0.0, S2 = 0.0;
        rc = compute_entropies(psi, N, in_A, &S1, &S2);
        if (rc != 0) { fprintf(stderr, "entropy failed at nA=%d\n", nA); continue; }
        printf("  %3d %4d %12.6f %12.6f", nA, boundary, S1, S2);
        print_subsystem_indicator(in_A, N);
        printf("\n");
        if (fit_n < 24) {
            fit_b[fit_n] = (double)boundary;
            fit_S[fit_n] = S1;
            fit_n++;
        }
    }
    printf("\n");

    /* Hand-chosen compact shapes to add boundary diversity. */
    printf("# hand-chosen compact subsystems:\n");
    printf("# %3s %4s %12s %12s\n", "nA", "|∂A|", "S_1 (vN)", "S_2 (Renyi)");

    int compact_shapes[][12] = {
        /* one full up-triangle (cell 0,0): A=0, B=1, C=2 */
        {1,1,1,0,0,0,0,0,0,0,0,0},
        /* two adjacent up-triangles: cell (0,0) + cell (1,0) */
        {1,1,1,1,1,1,0,0,0,0,0,0},
        /* one row of cells (cell (0,0) + (0,1)): sites 0..5 same as nA=6 above */
        /* skip — same as above */
        /* one diagonal: cells (0,0) and (1,1), 6 sites */
        {1,1,1,0,0,0,0,0,0,1,1,1},
        /* alternating pattern (3 of 12 sites) */
        {1,0,0,1,0,0,1,0,0,1,0,0},
    };
    const char *compact_names[] = {
        "1 up-triangle (cell 00)",
        "2 adjacent triangles (cells 00+10)",
        "diagonal pair (cells 00+11)",
        "all A-sublattice sites",
    };
    int n_compact = sizeof(compact_shapes) / sizeof(compact_shapes[0]);
    for (int s = 0; s < n_compact; s++) {
        int nA = 0;
        for (int i = 0; i < N; i++) nA += compact_shapes[s][i];
        int boundary = kagome_2x2_pbc_boundary_length(compact_shapes[s], N);
        double S1 = 0.0, S2 = 0.0;
        rc = compute_entropies(psi, N, compact_shapes[s], &S1, &S2);
        if (rc != 0) continue;
        printf("  %3d %4d %12.6f %12.6f  %s\n",
               nA, boundary, S1, S2, compact_names[s]);
        if (fit_n < 24) {
            fit_b[fit_n] = (double)boundary;
            fit_S[fit_n] = S1;
            fit_n++;
        }
    }
    printf("\n");

    /* Two-track linear fit S = α · |∂A| − γ.  Compact partitions
     * (single sites + triangles + connected blocks of adjacent
     * triangles) sit on the area-law branch.  Non-compact partitions
     * (e.g. all-A-sublattice scattering across cells) have different
     * boundary structure and corner contributions, so we report them
     * separately.
     *
     * Compact partitions for L=2 PBC kagome: nA ∈ {1, 3, 6} with the
     * specific compact shapes (single site, 1 triangle, 2-adjacent-
     * triangles).  These have boundary lengths 4, 6, 8 respectively. */
    double bf[3] = {4, 6, 8};
    double Sf[3];
    /* Pick out the matching rows from our fit_b / fit_S arrays. */
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < fit_n; i++) {
            if ((int)fit_b[i] == (int)bf[j]) { Sf[j] = fit_S[i]; break; }
        }
    }
    double sum_b = 0, sum_S = 0, sum_b2 = 0, sum_bS = 0;
    for (int i = 0; i < 3; i++) {
        sum_b  += bf[i];
        sum_S  += Sf[i];
        sum_b2 += bf[i] * bf[i];
        sum_bS += bf[i] * Sf[i];
    }
    double n_d = 3.0;
    double denom = n_d * sum_b2 - sum_b * sum_b;
    double alpha_compact = (n_d * sum_bS - sum_b * sum_S) / denom;
    double intercept_compact = (sum_S - alpha_compact * sum_b) / n_d;
    double gamma_compact = -intercept_compact;

    printf("# area-law fit S_1(|∂A|) on COMPACT partitions only:\n");
    printf("#   |∂A|=4  (1 site):       S_1 = %.6f\n", Sf[0]);
    printf("#   |∂A|=6  (1 triangle):    S_1 = %.6f\n", Sf[1]);
    printf("#   |∂A|=8  (2 adj triang.): S_1 = %.6f\n", Sf[2]);
    printf("#   slope α (entropy/bond): %.6f\n", alpha_compact);
    printf("#   intercept              : %.6f\n", intercept_compact);
    printf("#   ⇒ γ_TEE (compact)       : %.6f nats  (%.3f × log 2)\n",
           gamma_compact, gamma_compact / log(2.0));
    printf("#\n");
    printf("# reference values:\n");
    printf("#   Z₂ spin liquid (Depenbrock-McCulloch-Schollwöck '12 on cylinder):\n");
    printf("#                                                    γ = log 2 ≈ 0.6931 nats\n");
    printf("#   Trivial / no topological order:                   γ = 0\n");
    printf("#\n");
    printf("# WARNING: 2×2 PBC has very limited boundary diversity (only 3 compact\n");
    printf("# partitions giving |∂A| ∈ {4, 6, 8}).  Finite-size corrections at this\n");
    printf("# cluster can be O(1/|A|) ∼ O(0.5).  γ at this resolution is suggestive\n");
    printf("# (sign + magnitude vs log 2) but not converged.  Rigorous extraction\n");
    printf("# requires N=27 PBC cluster or 4×∞ cylinder DMRG.\n");

    free(psi); free(psi_real); free(perm); free(chars);
    nqs_ansatz_free(a);
    return 0;
#endif
}
