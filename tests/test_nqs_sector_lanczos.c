/*
 * tests/test_nqs_sector_lanczos.c
 *
 * Unit test locking in the sector-resolved spectrum result of the
 * kagome 2×2 PBC AFM Heisenberg model via the in-loop projecting
 * Lanczos pipeline.
 *
 *   pipeline:  random complex RBM  →  P_α projector wrapper
 *                                  →  materialise ψ_sym
 *                                  →  Lanczos with in-loop P_α projection
 *                                  →  per-sector lowest eigenvalue
 *
 * Asserts each E_0 matches the libirrep sector-ED reference (the
 * libirrep_sg_heisenberg_sector_build_at_k construction in
 * nqs_kspace_ed.c) to 7 digits.  Without the in-loop projection,
 * machine-precision sector leakage gets amplified by power-method
 * dynamics and Lanczos returns the global ground state from any seed
 * — this test would fail.  Codifies the fix in commit aaa1518.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "harness.h"
#include "nqs/nqs_config.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_symproj.h"
#include "nqs/nqs_lanczos.h"
#include "mps/lanczos.h"

/* libirrep sector-ED references (from the
 * nqs_kspace_ed_kagome_scan_gamma_1d_irreps L=2 run). */
static const double E0_A1_ref = -5.3283924045;
static const double E0_A2_ref = -4.9624348504;
static const double E0_B1_ref = -5.4448752170;
static const double E0_B2_ref = -3.6760938476;

static void test_kagome_2x2_sector_lanczos_matches_libirrep_ed_impl(void) {
    int L = 2;
    int N = 3 * L * L;        /* = 12 */

    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz           = NQS_ANSATZ_COMPLEX_RBM;
    cfg.rbm_hidden_units = 16;
    cfg.rbm_init_scale   = 0.05;
    cfg.hamiltonian      = NQS_HAM_KAGOME_HEISENBERG;
    cfg.j_coupling       = 1.0;
    cfg.kagome_pbc       = 1;
    cfg.rng_seed         = 0xB1B1B1B1u;

    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    ASSERT_TRUE(a != NULL);

    nqs_symproj_kagome_irrep_t irreps[4] = {
        NQS_SYMPROJ_KAGOME_GAMMA_A1,
        NQS_SYMPROJ_KAGOME_GAMMA_A2,
        NQS_SYMPROJ_KAGOME_GAMMA_B1,
        NQS_SYMPROJ_KAGOME_GAMMA_B2,
    };
    const char *names[4] = {"A_1", "A_2", "B_1", "B_2"};
    double refs[4] = { E0_A1_ref, E0_A2_ref, E0_B1_ref, E0_B2_ref };

    for (int i = 0; i < 4; i++) {
        int *perm = NULL;
        double *chars = NULL;
        int G = 0;
        ASSERT_EQ_INT(nqs_kagome_p6m_perm_irrep(L, irreps[i],
                                                 &perm, &chars, &G), 0);
        nqs_symproj_wrapper_t wrap = {
            .base_log_amp       = nqs_ansatz_log_amp,
            .base_user          = a,
            .num_sites          = N,
            .num_group_elements = G,
            .perm               = perm,
            .characters         = chars,
        };

        double evals[4] = {0};
        lanczos_result_t lr = (lanczos_result_t){0};
        int rc = nqs_lanczos_k_lowest_kagome_heisenberg_projected(
            nqs_symproj_log_amp, &wrap, L, L, cfg.j_coupling, cfg.kagome_pbc,
            perm, chars, G,
            300, 4, evals, &lr);
        ASSERT_EQ_INT(rc, 0);

        printf("# %s: E_0 = %.10f  (libirrep ED %.10f)  Δ = %.2e\n",
               names[i], evals[0], refs[i], fabs(evals[0] - refs[i]));
        ASSERT_TRUE(fabs(evals[0] - refs[i]) < 1e-7);

        free(perm); free(chars);
    }

    nqs_ansatz_free(a);
}

int main(void) {
    TEST_RUN(test_kagome_2x2_sector_lanczos_matches_libirrep_ed_impl);
    TEST_SUMMARY();
}
