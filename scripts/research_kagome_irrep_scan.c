/*
 * scripts/research_kagome_irrep_scan.c
 *
 * Research artefact: scan the four 1D irreps {A_1, A_2, B_1, B_2} of
 * C_6v at Γ on the kagome AFM, identify the ground-state irrep at
 * each cluster size, and report the spin gap Δ = E_1 − E_0.
 *
 * Cluster sizes (sites = 3·L²):
 *   L = 2   N = 12   per-irrep sector dims O(10¹) — homegrown overlap
 *   L = 3   N = 27   per-irrep sector dims O(10⁴) — NEW; was infeasible
 *                    on full-Hilbert-space Lanczos before this commit.
 *   L = 4   N = 48   per-irrep sector dims O(10⁶) — workstation-scale,
 *                    ~minutes per Lanczos run.
 *
 * Outputs the per-irrep spectrum table + the global GS-irrep label
 * + Δ.  This is the data that distinguishes Z₂ spin liquid (finite Δ
 * persistent in L) from Dirac spin liquid (Δ → 0 with L).
 *
 * Build:  make IRREP_ENABLE=1 research_kagome_irrep_scan
 * Run:    ./build/research_kagome_irrep_scan        (L = 2 default)
 *         ./build/research_kagome_irrep_scan 3      (L = 3, N = 27)
 *         ./build/research_kagome_irrep_scan 4      (L = 4, N = 48 — slow)
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef SPIN_NN_HAS_IRREP
#include "nqs/nqs_kspace_ed.h"

static const char *irrep_name(nqs_kspace_irrep_t i) {
    switch (i) {
    case NQS_KSPACE_IRREP_A1: return "A_1";
    case NQS_KSPACE_IRREP_A2: return "A_2";
    case NQS_KSPACE_IRREP_B1: return "B_1";
    case NQS_KSPACE_IRREP_B2: return "B_2";
    }
    return "?";
}

int main(int argc, char **argv) {
    int L = (argc > 1) ? atoi(argv[1]) : 2;
    if (L <= 0) L = 2;
    int max_iters = (argc > 2) ? atoi(argv[2]) : 120;
    int N = 3 * L * L;
    if (N > 27) {
        fprintf(stderr, "WARN: N = %d > 27 — rep_table_build is capped at 27 in this libirrep build.\n", N);
    }

    printf("# kagome AFM Heisenberg, J = 1, L = %d, N = %d\n", L, N);
    printf("# scanning C_6v 1D irreps at Γ via libirrep p6mm sector projection\n");
    printf("# Lanczos: reorthogonalised, %d iterations\n", max_iters);
    printf("\n");

    double e0 = 0, e1 = 0;
    double per_irrep_e0[4] = { 0 };
    nqs_kspace_irrep_t gs_irrep = NQS_KSPACE_IRREP_A1;

    clock_t t0 = clock();
    int rc = nqs_kspace_ed_kagome_scan_gamma_1d_irreps(
        L, /*J*/ 1.0, max_iters, &e0, &e1, &gs_irrep, per_irrep_e0);
    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;

    if (rc != 0) {
        fprintf(stderr, "scan failed (rc=%d)\n", rc);
        return 1;
    }

    printf("# per-irrep lowest eigenvalues (Γ point):\n");
    for (int i = 0; i < 4; i++) {
        nqs_kspace_irrep_t irr = (nqs_kspace_irrep_t)i;
        char tag = (irr == gs_irrep) ? '*' : ' ';
        printf("#   %s : %.10f  %c\n", irrep_name(irr), per_irrep_e0[i], tag);
    }
    printf("\n");
    printf("# global E_0 = %.10f  in (Γ, %s)\n", e0, irrep_name(gs_irrep));
    printf("# global E_1 = %.10f\n", e1);
    printf("# spin gap Δ = E_1 − E_0 = %.6f J\n", e1 - e0);
    printf("# per-site E_0 / N = %.6f J\n", e0 / (double)N);
    printf("\n");
    printf("# elapsed: %.1f s\n", elapsed);
    return 0;
}

#else /* !SPIN_NN_HAS_IRREP */

int main(void) {
    fprintf(stderr,
            "research_kagome_irrep_scan: built without -DSPIN_NN_HAS_IRREP. "
            "Build with `make IRREP_ENABLE=1 research_kagome_irrep_scan`.\n");
    return 1;
}

#endif
