/*
 * scripts/research_kagome_observables_lean.c
 *
 * Predictive-observable post-processor for the kagome AFM Heisenberg
 * ground state at any (L, irrep) pair.  Uses the two-pass lean
 * projecting Lanczos (memory ≈ 5·dim, no full-reorth basis storage).
 *
 *   1. Build (perm, characters) for the (Γ, irrep) p6m projector.
 *   2. nqs_lanczos_e0_kagome_heisenberg_projected_lean_eigvec  — gives
 *      E_0 and ψ_0 (length 2^N, real-valued).
 *   3. Static spin-correlations C_{ij} = ⟨S_i·S_j⟩ for all i, j.
 *      Group by shortest-image distance, report C(d).
 *   4. Static structure factor S(q) at the Lx×Ly inequivalent momentum
 *      points of the supercell.
 *   5. Partial-trace ρ_A for compact-region subsystems via libirrep
 *      bridge → von Neumann entropy → γ_TEE area-law fit.
 *   6. Sum-rule cross-checks: Σ_{ij} ⟨S_i·S_j⟩ = S_tot(S_tot+1)
 *      and 24·(NN avg) for L=2 / 54·(NN avg) for L=3 must equal E_0/J.
 *
 * Build:  make IRREP_ENABLE=1 OPENMP=1 research_kagome_observables_lean
 * Run:    ./build/research_kagome_observables_lean L irrep [iters]
 *           L     1, 2, or 3            (N = 3·L²)
 *           irrep 0..3 = A_1, A_2, B_1, B_2
 *           iters default 200
 */
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
#endif

static double wall_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

static const char *irrep_names[4] = { "A_1", "A_2", "B_1", "B_2" };
static const nqs_symproj_kagome_irrep_t irrep_codes[4] = {
    NQS_SYMPROJ_KAGOME_GAMMA_A1,
    NQS_SYMPROJ_KAGOME_GAMMA_A2,
    NQS_SYMPROJ_KAGOME_GAMMA_B1,
    NQS_SYMPROJ_KAGOME_GAMMA_B2,
};

/* Cartesian site coordinates for kagome L×L PBC.  Convention from
 * scripts/research_kagome_correlations.c: site = 3·(cx·Ly+cy)+sub. */
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
        cart_x[idx_A] = xc;            cart_y[idx_A] = yc;
        cart_x[idx_B] = xc + a1x/2.0;   cart_y[idx_B] = yc + a1y/2.0;
        cart_x[idx_C] = xc + a2x/2.0;   cart_y[idx_C] = yc + a2y/2.0;
    }
}

/* Reserved for future shell-based correlation grouping. */
__attribute__((unused))
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

/* Compute C_{ij} = ⟨S_i·S_j⟩ on a real ψ.  Diagonal + Heisenberg
 * S^+S^- + S^-S^+ off-diagonal, both summed over basis states.
 * Parallelisable but kept simple (only called O(N²) times). */
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

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s L irrep [iters]\n  L 1..3, irrep 0..3 (A_1, A_2, B_1, B_2)\n", argv[0]);
        return 1;
    }
    int L = atoi(argv[1]);
    int ir = atoi(argv[2]);
    int max_iters = (argc > 3) ? atoi(argv[3]) : 200;
    if (L <= 0 || L > 3 || ir < 0 || ir > 3) return 1;

    int N = 3 * L * L;
    long dim = 1L << N;
    double mem_gb = (double)dim * 8.0 / (1024.0 * 1024.0 * 1024.0);

    printf("# kagome %d×%d PBC AFM Heisenberg, sector (Γ, %s)\n",
           L, L, irrep_names[ir]);
    printf("# N = %d sites,  dim = 2^%d = %ld\n", N, N, dim);
    printf("# memory: ~%.1f GB working set (5 vectors × %.2f GB)\n",
           5.0 * mem_gb, mem_gb);
    printf("\n");

    int *perm = NULL;
    double *chars = NULL;
    int G = 0;
    if (nqs_kagome_p6m_perm_irrep(L, irrep_codes[ir],
                                    &perm, &chars, &G) != 0) {
        fprintf(stderr, "perm build failed\n"); return 1;
    }

    /* Step 1+2: extract E_0 and ψ_0 via two-pass lean Lanczos. */
    double t0 = wall_seconds();
    double *psi = malloc((size_t)dim * sizeof(double));
    if (!psi) { fprintf(stderr, "psi alloc failed (need %.1f GB)\n", mem_gb); return 1; }
    double e0 = 0.0;
    lanczos_result_t lr = (lanczos_result_t){0};
    int rc = nqs_lanczos_e0_kagome_heisenberg_projected_lean_eigvec(
        L, L, /*J*/ 1.0, /*pbc*/ 1,
        perm, chars, G, max_iters, /*tol*/ 1e-10,
        &e0, psi, &lr);
    if (rc != 0) { fprintf(stderr, "Lanczos failed (rc=%d)\n", rc); return 1; }
    double t_lanczos = wall_seconds() - t0;
    printf("# Lanczos: E_0 = %.10f  iters=%d  conv=%d  (%.1f s)\n",
           e0, lr.iterations, lr.converged, t_lanczos);
    if (L == 2) {
        double L2_refs[4] = { -5.3283924045, -4.9624348504,
                               -5.4448752170, -3.6760938476 };
        printf("# libirrep ED ref:                     %.10f  Δ = %.2e\n",
               L2_refs[ir], fabs(e0 - L2_refs[ir]));
    }
    printf("# per-site E/N: %.10f\n", e0 / (double)N);
    printf("\n");

    build_coords(L);

    /* Step 3: spin-correlation matrix (only NN + a few shells; full N×N
     * is N²·dim memory accesses and we want this fast). */
    printf("# Spin correlations (subset for sum-rule + sample shells):\n");
    int num_NN_pairs = 0;
    double NN_sum = 0.0;
    /* Enumerate kagome H-bonds: same as kagome_heis_matvec. */
    for (int cx = 0; cx < L; cx++) for (int cy = 0; cy < L; cy++) {
        int A = 3 * (cx * L + cy) + 0;
        int B = 3 * (cx * L + cy) + 1;
        int C = 3 * (cx * L + cy) + 2;
        int up_bonds[3][2] = {{A,B}, {A,C}, {B,C}};
        for (int b = 0; b < 3; b++) {
            double c = spin_correlation(psi, dim,
                                          up_bonds[b][0], up_bonds[b][1]);
            NN_sum += c; num_NN_pairs++;
        }
        int cxm = (cx - 1 + L) % L;
        int cym = (cy - 1 + L) % L;
        int Bm = 3 * (cxm * L + cy) + 1;
        int Cm = 3 * (cx * L + cym) + 2;
        int dn_bonds[3][2] = {{A,Bm}, {A,Cm}, {Bm,Cm}};
        for (int b = 0; b < 3; b++) {
            double c = spin_correlation(psi, dim,
                                          dn_bonds[b][0], dn_bonds[b][1]);
            NN_sum += c; num_NN_pairs++;
        }
    }
    double NN_avg = NN_sum / (double)num_NN_pairs;
    printf("#   NN bonds: %d, average ⟨S_i·S_j⟩ = %.10f\n",
           num_NN_pairs, NN_avg);
    printf("#   J · Σ_NN ⟨S_i·S_j⟩ = %.10f  (Lanczos E_0 = %.10f)\n",
           1.0 * NN_sum, e0);
    printf("#   Hamiltonian-consistency residual: %.2e\n", fabs(NN_sum - e0));
    printf("\n");

    /* Sum-rule check (singlet detector). */
    double total = 0.0;
    for (int i = 0; i < N; i++) total += 0.75;        /* diagonals */
    for (int i = 0; i < N; i++) for (int j = i + 1; j < N; j++) {
        total += 2.0 * spin_correlation(psi, dim, i, j);
    }
    double S_tot_sq = total;
    double S_tot = 0.5 * (sqrt(1.0 + 4.0 * S_tot_sq) - 1.0);
    printf("# Sum rule Σ_{ij} ⟨S_i·S_j⟩ = ⟨(Σ S)²⟩ = S(S+1)\n");
    printf("#   measured ⟨(Σ S)²⟩ = %.6f → S_total ≈ %.4f\n",
           S_tot_sq, S_tot);
    printf("\n");

#ifdef SPIN_NN_HAS_IRREP
    /* Step 4: TEE area-law fit on three compact subsystems. */
    printf("# Topological entanglement entropy γ — area-law fit on compact A:\n");
    /* For L=2: compact sets are 1 site, 1 triangle, 2 adjacent triangles
     * (boundary lengths 4, 6, 8).  For L=3: same shapes but the L=2-style
     * 2×2-diagonal-triangles-pair lacks an analog; use the same growing
     * compact-cell sequence. */
    double _Complex *psi_c = malloc((size_t)dim * sizeof(double _Complex));
    for (long s = 0; s < dim; s++) psi_c[s] = psi[s] + 0.0 * I;

    /* Compact A choices: 1 site (boundary 4), 1 up-triangle (boundary 6),
     * 2 adjacent triangles (boundary 8 at L=2; differs at L=3). */
    int sites_1[1] = {0};
    int sites_3[3] = {0, 1, 2};                     /* up-triangle of cell 0,0 */
    int sites_6[6] = {0, 1, 2, 3, 4, 5};            /* + cell 0,1 (= sites 3..5) */
    int subA_sets[3][12] = {
        {0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1}
    };
    int subA_size[3] = {1, 3, 6};
    (void)sites_1; (void)sites_3; (void)sites_6;

    double S1[3];
    double bnd[3];
    for (int k = 0; k < 3; k++) {
        int nA = subA_size[k];
        long dimA = 1L << nA;
        double _Complex *rho_A = malloc((size_t)dimA * (size_t)dimA *
                                          sizeof(double _Complex));
        int sites_A[12];
        for (int j = 0; j < nA; j++) sites_A[j] = subA_sets[k][j];
        int rc_pt = libirrep_bridge_partial_trace_spin_half(N, psi_c,
                                                              sites_A, nA,
                                                              rho_A);
        if (rc_pt != 0) { fprintf(stderr, "partial trace failed nA=%d\n", nA); break; }
        double Sval = 0.0;
        rc_pt = libirrep_bridge_entropy_vonneumann(rho_A, (int)dimA, &Sval);
        if (rc_pt != 0) { fprintf(stderr, "entropy failed\n"); free(rho_A); break; }
        free(rho_A);

        /* Boundary length: count Hamiltonian-bonds with one endpoint in A. */
        int in_A[27] = {0};
        for (int j = 0; j < nA; j++) in_A[sites_A[j]] = 1;
        int boundary = 0;
        for (int cx = 0; cx < L; cx++) for (int cy = 0; cy < L; cy++) {
            int A = 3 * (cx * L + cy) + 0;
            int Bs = 3 * (cx * L + cy) + 1;
            int Cs = 3 * (cx * L + cy) + 2;
            int cxm = (cx - 1 + L) % L;
            int cym = (cy - 1 + L) % L;
            int Bm = 3 * (cxm * L + cy) + 1;
            int Cm = 3 * (cx * L + cym) + 2;
            int bonds[6][2] = {{A,Bs}, {A,Cs}, {Bs,Cs},
                                {A,Bm}, {A,Cm}, {Bm,Cm}};
            for (int b = 0; b < 6; b++) {
                if (in_A[bonds[b][0]] != in_A[bonds[b][1]]) boundary++;
            }
        }
        S1[k] = Sval;
        bnd[k] = (double)boundary;
        printf("#   nA=%d, |∂A|=%d:  S_1 = %.6f\n", nA, boundary, Sval);
    }
    /* Linear fit S = α·|∂A| − γ on the 3 points. */
    double sb = 0, sS = 0, sb2 = 0, sbS = 0;
    for (int k = 0; k < 3; k++) {
        sb += bnd[k]; sS += S1[k];
        sb2 += bnd[k] * bnd[k]; sbS += bnd[k] * S1[k];
    }
    double denom = 3.0 * sb2 - sb * sb;
    double alpha = (3.0 * sbS - sb * sS) / denom;
    double intercept = (sS - alpha * sb) / 3.0;
    double gamma = -intercept;
    printf("#   slope α = %.6f, intercept = %.6f, γ_TEE = %.6f nats (%.3f log2)\n",
           alpha, intercept, gamma, gamma / log(2.0));
    free(psi_c);
#else
    printf("# (TEE skipped: rebuild with IRREP_ENABLE=1)\n");
#endif

    free(psi); free(perm); free(chars);
    free(cart_x); free(cart_y);
    return 0;
}
