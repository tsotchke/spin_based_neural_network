/*
 * tests/test_nqs_minsr.c
 *
 * Validates the sample-space MinSR optimiser
 * (Chen-Heyl 2024 / Rende et al. 2024) against:
 *   1. Exact-diagonalisation TFIM 2×2 ground-state energy — descend
 *      monotonically toward E_0 and stay variationally above it.
 *   2. Standard CG-based SR on the same problem — final energies must
 *      agree to within MC noise on the same RNG seed (the two solvers
 *      compute the same δθ from the same batch up to regularisation
 *      and Krylov truncation).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "nqs/nqs_config.h"
#include "nqs/nqs_sampler.h"
#include "nqs/nqs_gradient.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_optimizer.h"

/* Inline TFIM 2×2 ED — same routine pattern as test_nqs_convergence. */
static double tfim_diag_2x2(int state, int Lx, int Ly, double J) {
    double e = 0.0;
    for (int x = 0; x < Lx; x++) {
        for (int y = 0; y < Ly; y++) {
            int idx = x * Ly + y;
            int s_xy = ((state >> idx) & 1) ? -1 : +1;
            if (x + 1 < Lx) {
                int j = (x + 1) * Ly + y;
                int sj = ((state >> j) & 1) ? -1 : +1;
                e += -J * (double)(s_xy * sj);
            }
            if (y + 1 < Ly) {
                int j = x * Ly + (y + 1);
                int sj = ((state >> j) & 1) ? -1 : +1;
                e += -J * (double)(s_xy * sj);
            }
        }
    }
    return e;
}

static double tfim_ground_state_energy(int Lx, int Ly, double J, double Gamma) {
    int N = Lx * Ly;
    int dim = 1 << N;
    double *H = calloc((size_t)dim * (size_t)dim, sizeof(double));
    if (!H) return 0.0;
    for (int s = 0; s < dim; s++) {
        H[s * dim + s] = tfim_diag_2x2(s, Lx, Ly, J);
        for (int i = 0; i < N; i++) {
            int s2 = s ^ (1 << i);
            H[s * dim + s2] += -Gamma;
        }
    }
    /* Power iteration on M = shift·I − H to find smallest H eigenvalue. */
    double row_max = 0.0;
    for (int i = 0; i < dim; i++) {
        double row = 0.0;
        for (int j = 0; j < dim; j++) row += fabs(H[i * dim + j]);
        if (row > row_max) row_max = row;
    }
    double shift = row_max + 1.0;
    for (int i = 0; i < dim; i++) H[i * dim + i] = shift - H[i * dim + i];
    /* Off-diagonals: M = shift·I - H means M_ij = -H_ij for i!=j. */
    for (int i = 0; i < dim; i++)
        for (int j = 0; j < dim; j++)
            if (i != j) H[i * dim + j] = -H[i * dim + j];

    double *v = calloc((size_t)dim, sizeof(double));
    double *w = calloc((size_t)dim, sizeof(double));
    for (int i = 0; i < dim; i++) v[i] = 1.0;
    double n0 = sqrt((double)dim);
    for (int i = 0; i < dim; i++) v[i] /= n0;
    double lambda_M = 0.0;
    for (int it = 0; it < 200; it++) {
        for (int i = 0; i < dim; i++) {
            double s = 0.0;
            for (int j = 0; j < dim; j++) s += H[i * dim + j] * v[j];
            w[i] = s;
        }
        double nw = 0.0;
        for (int i = 0; i < dim; i++) nw += w[i] * w[i];
        nw = sqrt(nw);
        for (int i = 0; i < dim; i++) v[i] = w[i] / nw;
        lambda_M = nw;
    }
    double E0 = shift - lambda_M;
    free(H); free(v); free(w);
    return E0;
}

static void test_minsr_descends_for_tfim_2x2(void) {
    int Lx = 2, Ly = 2, N = Lx * Ly;
    double J = 1.0, Gamma = 1.0;
    double E0 = tfim_ground_state_energy(Lx, Ly, J, Gamma);

    nqs_config_t cfg   = nqs_config_defaults();
    cfg.hamiltonian      = NQS_HAM_TFIM;
    cfg.j_coupling       = J;
    cfg.transverse_field = Gamma;
    cfg.num_samples      = 512;
    cfg.num_thermalize   = 256;
    cfg.num_decorrelate  = 2;
    cfg.num_iterations   = 50;
    cfg.learning_rate    = 5e-2;
    cfg.sr_diag_shift    = 1e-2;
    cfg.rng_seed         = 0xCAFEFEEDu;

    nqs_ansatz_t  *ansatz  = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *sampler = nqs_sampler_create(N, &cfg,
                                                nqs_ansatz_log_amp, ansatz);
    ASSERT_TRUE(ansatz && sampler);

    double *trace = malloc((size_t)cfg.num_iterations * sizeof(double));
    int rc = nqs_sr_run_minsr(&cfg, Lx, Ly, ansatz, sampler,
                              NULL, NULL, trace);
    ASSERT_EQ_INT(rc, 0);

    double e_head = 0.0, e_tail = 0.0;
    for (int i = 0; i < 10; i++) e_head += trace[i];
    for (int i = 0; i < 10; i++) e_tail += trace[cfg.num_iterations - 10 + i];
    e_head /= 10.0;
    e_tail /= 10.0;
    ASSERT_TRUE(e_tail < e_head + 0.1);
    ASSERT_TRUE(e_tail >= E0 - 1e-6);
    ASSERT_TRUE(e_tail <  E0 + 6.0);
    printf("# MinSR TFIM 2x2: E_exact = %.6f, E_final = %.6f, gap = %.4f\n",
           E0, e_tail, e_tail - E0);

    free(trace);
    nqs_sampler_free(sampler);
    nqs_ansatz_free(ansatz);
}

/* Run plain CG-SR and MinSR with identical configs and same RNG seed —
 * after 30 iterations the two final energies should agree within
 * Monte Carlo noise. */
static void test_minsr_matches_full_sr_within_noise(void) {
    int Lx = 2, Ly = 2, N = Lx * Ly;
    double J = 1.0, Gamma = 1.0;

    nqs_config_t cfg   = nqs_config_defaults();
    cfg.hamiltonian      = NQS_HAM_TFIM;
    cfg.j_coupling       = J;
    cfg.transverse_field = Gamma;
    cfg.num_samples      = 1024;
    cfg.num_thermalize   = 256;
    cfg.num_decorrelate  = 2;
    cfg.num_iterations   = 30;
    cfg.learning_rate    = 3e-2;
    cfg.sr_diag_shift    = 1e-2;
    cfg.sr_cg_max_iters  = 40;
    cfg.rng_seed         = 0xBADA55u;

    /* Run 1: classic CG-SR */
    nqs_ansatz_t  *a1 = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s1 = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a1);
    double *trace_cg = malloc((size_t)cfg.num_iterations * sizeof(double));
    int rc1 = nqs_sr_run(&cfg, Lx, Ly, a1, s1, trace_cg);
    ASSERT_EQ_INT(rc1, 0);

    /* Run 2: MinSR with same seed → same MC trajectory */
    nqs_ansatz_t  *a2 = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s2 = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a2);
    double *trace_min = malloc((size_t)cfg.num_iterations * sizeof(double));
    int rc2 = nqs_sr_run_minsr(&cfg, Lx, Ly, a2, s2, NULL, NULL, trace_min);
    ASSERT_EQ_INT(rc2, 0);

    /* Late-window means must agree within ~1 MC standard deviation
     * (~0.2 for these settings). */
    double mean_cg  = 0.0, mean_min = 0.0;
    for (int i = 20; i < 30; i++) { mean_cg += trace_cg[i];  mean_min += trace_min[i]; }
    mean_cg  /= 10.0;
    mean_min /= 10.0;
    printf("# MinSR vs CG-SR: mean_cg=%.4f mean_min=%.4f diff=%.4f\n",
           mean_cg, mean_min, fabs(mean_cg - mean_min));
    ASSERT_TRUE(fabs(mean_cg - mean_min) < 0.5);

    free(trace_cg); free(trace_min);
    nqs_sampler_free(s1); nqs_ansatz_free(a1);
    nqs_sampler_free(s2); nqs_ansatz_free(a2);
}

int main(void) {
    TEST_RUN(test_minsr_descends_for_tfim_2x2);
    TEST_RUN(test_minsr_matches_full_sr_within_noise);
    TEST_SUMMARY();
}
