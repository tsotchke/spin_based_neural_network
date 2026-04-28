/*
 * scripts/research_kagome_3x3_e0.c
 *
 * Push to N=27 PBC kagome AFM Heisenberg via memory-lean projecting
 * Lanczos.  Uses the 3-term-recurrence variant with in-loop sector
 * projection — no Krylov-basis storage, no NQS materialise, no
 * eigenvector reconstruction.  Just the (Γ, B_1) sector ground-state
 * energy (and the other 1D irreps if requested).
 *
 * Pipeline (per irrep):
 *   1. Build (perm, characters) for the (Γ, irrep) projector via
 *      nqs_kagome_p6m_perm_irrep at the requested L.
 *   2. Deterministic random vector of length 2^N → in-place sector
 *      projection.
 *   3. lanczos_smallest_projected_lean: 3-term recurrence with the
 *      sector projector applied to each Krylov vector after the matvec.
 *      Convergence on Ritz-value stability.
 *   4. Print E_0(α) for each irrep α.
 *
 * Memory at L=3 (N=27): three vectors × 2^27 × 8 bytes = 3.07 GB
 * (Lanczos working set), plus 1.07 GB scratch for the projector's
 * out-buffer.  Total ~4.5 GB.  Fits comfortably in M2 Ultra
 * unified memory (64-192 GB depending on config).
 *
 * Build: make research_kagome_3x3_e0
 * Run:   ./build/research_kagome_3x3_e0 [L [iters]]
 *          L     default 3   (cluster-size param: N = 3·L²)
 *          iters default 200
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nqs/nqs_symproj.h"
#include "nqs/nqs_lanczos.h"
#include "mps/lanczos.h"

static const char *irrep_names[4] = { "A_1", "A_2", "B_1", "B_2" };
static const nqs_symproj_kagome_irrep_t irrep_codes[4] = {
    NQS_SYMPROJ_KAGOME_GAMMA_A1,
    NQS_SYMPROJ_KAGOME_GAMMA_A2,
    NQS_SYMPROJ_KAGOME_GAMMA_B1,
    NQS_SYMPROJ_KAGOME_GAMMA_B2,
};

int main(int argc, char **argv) {
    int L = (argc > 1) ? atoi(argv[1]) : 3;
    int max_iters = (argc > 2) ? atoi(argv[2]) : 200;
    if (L <= 0 || L > 3) {
        fprintf(stderr, "L must be 1-3 (N=3·L² fits in 2^N ≤ 2^27 working memory at L=3)\n");
        return 1;
    }

    int N = 3 * L * L;
    long dim = 1L << N;
    double mem_gb = (double)dim * 8.0 / (1024.0 * 1024.0 * 1024.0);

    printf("# kagome %d×%d PBC AFM Heisenberg sector ground states\n", L, L);
    printf("# N = %d sites,  dim = 2^%d = %ld states\n", N, N, dim);
    printf("# memory per Lanczos vector: %.2f GB (4 vectors needed = %.2f GB working set)\n",
           mem_gb, 4.0 * mem_gb);
    printf("# Lanczos: lean 3-term recurrence + in-loop p6m projection,\n");
    printf("#          max %d iters, Ritz-value tolerance 1e-9\n",
           max_iters);
    printf("\n");

    /* Run all 4 1D irreps.  At L=2 the GS is in B_1 at -5.4448752170;
     * at L=3 the libirrep ED scan reports B_1 too with Δ ≈ 0.05 J. */
    double e0[4];
    int    iters[4];
    int    converged[4];
    double seconds[4];

    for (int i = 0; i < 4; i++) {
        int *perm = NULL;
        double *chars = NULL;
        int G = 0;
        if (nqs_kagome_p6m_perm_irrep(L, irrep_codes[i],
                                        &perm, &chars, &G) != 0) {
            fprintf(stderr, "perm build failed for %s\n", irrep_names[i]);
            return 1;
        }

        clock_t t0 = clock();
        lanczos_result_t lr = (lanczos_result_t){0};
        double e = 0.0;
        int rc = nqs_lanczos_e0_kagome_heisenberg_projected_lean(
            L, L, /*J*/ 1.0, /*pbc*/ 1,
            perm, chars, G,
            max_iters, /*tol*/ 1e-9,
            &e, &lr);
        seconds[i] = (double)(clock() - t0) / CLOCKS_PER_SEC;
        if (rc != 0) {
            fprintf(stderr, "Lanczos failed for %s (rc=%d)\n", irrep_names[i], rc);
            free(perm); free(chars);
            return 1;
        }
        e0[i] = e;
        iters[i] = lr.iterations;
        converged[i] = lr.converged;

        printf("# (Γ, %s):  E_0 = %.10f  iters = %d  conv = %d  (%.1f s)\n",
               irrep_names[i], e0[i], iters[i], converged[i], seconds[i]);
        free(perm); free(chars);
    }

    /* Identify global GS irrep = the lowest E_0 across the 4 irreps. */
    int gs_idx = 0;
    for (int i = 1; i < 4; i++) if (e0[i] < e0[gs_idx]) gs_idx = i;
    printf("\n");
    printf("# Global ground state: in (Γ, %s) sector at E_0 = %.10f\n",
           irrep_names[gs_idx], e0[gs_idx]);
    printf("# per-site E_0 / N:    %.10f\n", e0[gs_idx] / (double)N);

    /* Spin gap to next-lowest sector (cross-sector excitation). */
    int second_idx = (gs_idx == 0) ? 1 : 0;
    for (int i = 0; i < 4; i++) {
        if (i == gs_idx) continue;
        if (e0[i] < e0[second_idx]) second_idx = i;
    }
    printf("# Lowest-other-sector excitation: in (Γ, %s) at E_1 = %.10f\n",
           irrep_names[second_idx], e0[second_idx]);
    printf("# Cross-sector gap Δ = E_1 − E_0 = %.6f J\n",
           e0[second_idx] - e0[gs_idx]);

    /* Reference values for cross-validation. */
    if (L == 2) {
        printf("\n# Reference (libirrep ED at L=2):\n");
        printf("#   A_1: -5.3283924045\n");
        printf("#   A_2: -4.9624348504\n");
        printf("#   B_1: -5.4448752170 (global GS)\n");
        printf("#   B_2: -3.6760938476\n");
    } else if (L == 3) {
        printf("\n# Reference (libirrep ED at L=3, from research_kagome_irrep_scan):\n");
        printf("#   B_1 reported as global GS with Δ ≈ 0.052 J\n");
    }

    return 0;
}
