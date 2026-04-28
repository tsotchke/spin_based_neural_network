/*
 * scripts/research_kagome_sector_spectrum.c
 *
 * Sector-resolved low-energy spectrum of the kagome 2×2 PBC AFM
 * Heisenberg model via Lanczos seeded from sector-projected random
 * complex-RBM wavefunctions.  Scans all four 1D irreps of the
 * Γ-point little group C_6v of the p6m wallpaper group:
 *
 *     A_1   trivial            (ferromagnetic-friendly sector)
 *     A_2   sign on σ           (anti-symmetric in all six mirrors)
 *     B_1   sign on C_6, σ_d    (ground state lives here on 2×2 PBC)
 *     B_2   sign on C_6, σ_v
 *
 * Hilbert-space dimension at N = 12 sites is 4096; the four 1D
 * sectors plus the two 2D irreps E_1 / E_2 (not yet exposed in
 * nqs_symproj — those are matrix-valued projectors) account for
 * the full space:
 *
 *     dim(A_1) + dim(A_2) + dim(B_1) + dim(B_2) + 2·dim(E_1) + 2·dim(E_2) = 4096
 *
 * The script does NOT train the ansatz: a freshly-initialized random
 * complex RBM has nonzero overlap with every sector ground state with
 * probability 1, and that is all Lanczos needs.  So this run is fast
 * (~5 s per irrep, 4 irreps).
 *
 * Cross-check: sorted union of {A_1, A_2, B_1, B_2} sector spectra
 * should reproduce the 1D-irrep eigenvalues of H_kagome (with the E
 * irreps' degenerate pairs interleaved).  The lowest E_0 = −5.44488
 * confirms the GS lives in B_1 (per the libirrep sector ED scan).
 *
 * Usage:
 *   make IRREP_ENABLE=1 research_kagome_sector_spectrum
 *   ./build/research_kagome_sector_spectrum [k [hidden [iters]]]
 *     k       default 4    (number of lowest eigenvalues per sector)
 *     hidden  default 16
 *     iters   default 0    (no training; project bare random RBM)
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "nqs/nqs_config.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_symproj.h"
#include "nqs/nqs_lanczos.h"
#include "mps/lanczos.h"

static const char *irrep_name(int ir) {
    switch (ir) {
        case NQS_SYMPROJ_KAGOME_GAMMA_A1: return "A_1";
        case NQS_SYMPROJ_KAGOME_GAMMA_A2: return "A_2";
        case NQS_SYMPROJ_KAGOME_GAMMA_B1: return "B_1";
        case NQS_SYMPROJ_KAGOME_GAMMA_B2: return "B_2";
    }
    return "?";
}

int main(int argc, char **argv) {
    int k       = (argc > 1) ? atoi(argv[1]) : 4;
    int hidden  = (argc > 2) ? atoi(argv[2]) : 16;
    int iters   = (argc > 3) ? atoi(argv[3]) : 0;        /* unused for now */
    (void)iters;

    int L = 2;
    int N = 3 * L * L;          /* 12 */

    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz           = NQS_ANSATZ_COMPLEX_RBM;
    cfg.rbm_hidden_units = hidden;
    cfg.rbm_init_scale   = 0.05;
    cfg.hamiltonian      = NQS_HAM_KAGOME_HEISENBERG;
    cfg.j_coupling       = 1.0;
    cfg.kagome_pbc       = 1;
    cfg.rng_seed         = 0xB1B1B1B1u;

    printf("# kagome 2×2 PBC Heisenberg sector spectrum scan\n");
    printf("# ansatz: complex RBM, %d hidden units (untrained — random init)\n",
           hidden);
    printf("# k = %d eigenvalues per sector\n", k);
    printf("\n");

    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    if (!a) { fprintf(stderr, "ansatz_create failed\n"); return 1; }

    /* Aggregate spectrum across all 4 irreps for the cross-check at the
     * end.  k_total ≤ 4·k, allocated for a worst-case k=10. */
    double *all_evals = malloc((size_t)(4 * k) * sizeof(double));
    int    *all_irrep = malloc((size_t)(4 * k) * sizeof(int));
    int     n_total   = 0;

    int irreps[4] = {
        NQS_SYMPROJ_KAGOME_GAMMA_A1,
        NQS_SYMPROJ_KAGOME_GAMMA_A2,
        NQS_SYMPROJ_KAGOME_GAMMA_B1,
        NQS_SYMPROJ_KAGOME_GAMMA_B2,
    };

    for (int i = 0; i < 4; i++) {
        int ir = irreps[i];
        const char *name = irrep_name(ir);

        int *perm = NULL;
        double *chars = NULL;
        int G = 0;
        if (nqs_kagome_p6m_perm_irrep(L, ir, &perm, &chars, &G) != 0) {
            fprintf(stderr, "perm/chars build failed for irrep %s\n", name);
            return 1;
        }
        nqs_symproj_wrapper_t wrap = {
            .base_log_amp       = nqs_ansatz_log_amp,
            .base_user          = a,
            .num_sites          = N,
            .num_group_elements = G,
            .perm               = perm,
            .characters         = chars,
        };

        clock_t t0 = clock();
        double *evals = malloc((size_t)k * sizeof(double));
        lanczos_result_t lr = (lanczos_result_t){0};
        /* Sector-projected Lanczos: feed the (perm, chars) directly so
         * each Krylov step projects back into the sector, defeating
         * machine-precision leak amplification. */
        int rc = nqs_lanczos_k_lowest_kagome_heisenberg_projected(
            nqs_symproj_log_amp, &wrap, L, L, cfg.j_coupling, cfg.kagome_pbc,
            perm, chars, G,
            300, k, evals, &lr);
        double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;

        if (rc != 0) {
            printf("# (Γ, %s):   Lanczos failed (rc=%d)\n", name, rc);
        } else {
            printf("# (Γ, %s):   ", name);
            for (int j = 0; j < k; j++) {
                printf("E_%d=%.8f%s", j, evals[j], (j == k - 1) ? "" : "  ");
                if (n_total < 4 * k) {
                    all_evals[n_total] = evals[j];
                    all_irrep[n_total] = ir;
                    n_total++;
                }
            }
            printf("   (%.2f s)\n", dt);
        }

        free(evals); free(perm); free(chars);
    }
    nqs_ansatz_free(a);

    /* Sort the aggregated eigenvalues and dump the global low-energy
     * window labelled by the irrep each came from. */
    printf("\n");
    printf("# combined low-energy window across A_1 / A_2 / B_1 / B_2:\n");
    /* Simple O(n^2) sort, n ≤ 16. */
    for (int i = 0; i < n_total - 1; i++) {
        int min = i;
        for (int j = i + 1; j < n_total; j++) {
            if (all_evals[j] < all_evals[min]) min = j;
        }
        if (min != i) {
            double tmp = all_evals[i]; all_evals[i] = all_evals[min];
            all_evals[min] = tmp;
            int    ti  = all_irrep[i]; all_irrep[i] = all_irrep[min];
            all_irrep[min] = ti;
        }
    }
    int show = n_total < 12 ? n_total : 12;
    for (int i = 0; i < show; i++) {
        printf("#   %2d   E = %.8f   sector = %s\n", i,
               all_evals[i], irrep_name(all_irrep[i]));
    }

    free(all_evals); free(all_irrep);
    return 0;
}
