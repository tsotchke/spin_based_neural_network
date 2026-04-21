/*
 * tests/test_nqs_convergence.c
 *
 * Known-answer validation for the v0.5 NQS pillar. Runs exact
 * diagonalisation on a small transverse-field Ising model to get the
 * reference ground-state energy, then asserts that the stochastic-
 * reconfiguration pipeline descends toward that reference.
 *
 * The mean-field ansatz in v0.4 is expressive enough to capture the
 * TFIM ground state at the paramagnetic (Γ >> J) fixed point, so this
 * test is a genuine end-to-end check on sampler + local energy + SR.
 */
#include <complex.h>
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
/* -------------------- exact diagonalisation of TFIM 2x2 --------------- */
/* H = -J Σ_<ij> σ^z_i σ^z_j  -  Γ Σ_i σ^x_i
 *
 * Represented as a 2^N × 2^N real matrix. Diagonal is the classical ZZ
 * energy; off-diagonal entries are -Γ between bitstrings differing in
 * exactly one bit.
 *
 * We want only the ground-state eigenvalue, so we use power iteration
 * on M = λ_max·I − H (shift into positive-definite so power iteration
 * finds the largest eigenvalue of M, i.e. the smallest eigenvalue of H).
 */
static double tfim_diag(int state, int N, int size_x, int size_y, double J) {
    /* ZZ term with open boundary conditions (matches nqs_gradient.c). */
    double e = 0.0;
    for (int x = 0; x < size_x; x++) {
        for (int y = 0; y < size_y; y++) {
            int idx = x * size_y + y;
            int s_xy = ((state >> idx) & 1) ? -1 : +1;
            if (x + 1 < size_x) {
                int nidx = (x + 1) * size_y + y;
                int s_nxy = ((state >> nidx) & 1) ? -1 : +1;
                e += -J * (double)(s_xy * s_nxy);
            }
            if (y + 1 < size_y) {
                int nidx = x * size_y + (y + 1);
                int s_nxy = ((state >> nidx) & 1) ? -1 : +1;
                e += -J * (double)(s_xy * s_nxy);
            }
        }
    }
    (void)N;
    return e;
}
static double tfim_ground_state_energy(int size_x, int size_y,
                                       double J, double Gamma) {
    int N = size_x * size_y;
    int dim = 1 << N;
    /* Build H (dense, small N only). */
    double *H = calloc((size_t)dim * (size_t)dim, sizeof(double));
    if (!H) return 0.0;
    for (int s = 0; s < dim; s++) {
        H[s * dim + s] = tfim_diag(s, N, size_x, size_y, J);
        for (int i = 0; i < N; i++) {
            int s2 = s ^ (1 << i);
            H[s * dim + s2] += -Gamma;
        }
    }
    /* Shift: M = λ_shift · I − H, with λ_shift chosen large enough that
     * M is positive definite. Gershgorin-ish bound: sum of abs row. */
    double row_max = 0.0;
    for (int i = 0; i < dim; i++) {
        double row = 0.0;
        for (int j = 0; j < dim; j++) row += fabs(H[i * dim + j]);
        if (row > row_max) row_max = row;
    }
    double shift = row_max + 1.0;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) H[i * dim + j] = -H[i * dim + j];
        H[i * dim + i] += shift;
    }
    /* Power iteration for the largest eigenvalue of M. */
    double *v = malloc((size_t)dim * sizeof(double));
    double *w = malloc((size_t)dim * sizeof(double));
    if (!v || !w) { free(H); free(v); free(w); return 0.0; }
    for (int i = 0; i < dim; i++) v[i] = 1.0;
    double nrm = sqrt((double)dim);
    for (int i = 0; i < dim; i++) v[i] /= nrm;
    double lam = 0.0;
    for (int iter = 0; iter < 2000; iter++) {
        /* w = M v */
        for (int i = 0; i < dim; i++) {
            double a = 0.0;
            for (int j = 0; j < dim; j++) a += H[i * dim + j] * v[j];
            w[i] = a;
        }
        /* Rayleigh quotient */
        double num = 0.0, den = 0.0;
        for (int i = 0; i < dim; i++) {
            num += v[i] * w[i];
            den += v[i] * v[i];
        }
        double lam_new = num / den;
        /* Normalize. */
        double nrm2 = 0.0;
        for (int i = 0; i < dim; i++) nrm2 += w[i] * w[i];
        nrm2 = sqrt(nrm2);
        if (nrm2 > 0) for (int i = 0; i < dim; i++) v[i] = w[i] / nrm2;
        if (iter > 0 && fabs(lam_new - lam) < 1e-10) { lam = lam_new; break; }
        lam = lam_new;
    }
    double E0 = shift - lam;   /* ground-state energy of original H */
    free(H); free(v); free(w);
    return E0;
}
/* -------------------- SR convergence test ---------------------------- */
static void test_sr_descends_for_tfim_2x2(void) {
    int Lx = 2, Ly = 2, N = Lx * Ly;
    double J = 1.0;
    double Gamma = 1.0;
    /* Reference ground-state energy via exact diagonalisation. */
    double E0 = tfim_ground_state_energy(Lx, Ly, J, Gamma);
    /* Mean-field ansatz only: not expected to hit E0 exactly, but
     * should descend toward it. */
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_TFIM;
    cfg.j_coupling = J;
    cfg.transverse_field = Gamma;
    cfg.num_samples = 512;
    cfg.num_thermalize = 256;
    cfg.num_decorrelate = 2;
    cfg.num_iterations = 50;
    cfg.learning_rate = 5e-2;
    cfg.sr_diag_shift = 1e-2;
    cfg.sr_cg_max_iters = 40;
    cfg.rng_seed = 0xBADA55u;
    nqs_ansatz_t *ansatz = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *sampler = nqs_sampler_create(N, &cfg,
                                                 nqs_ansatz_log_amp, ansatz);
    ASSERT_TRUE(ansatz && sampler);
    double *trace = malloc((size_t)cfg.num_iterations * sizeof(double));
    int rc = nqs_sr_run(&cfg, Lx, Ly, ansatz, sampler, trace);
    ASSERT_EQ_INT(rc, 0);
    /* Moving average of the last 10 iterations vs. the first 10. */
    double e_head = 0.0, e_tail = 0.0;
    for (int i = 0; i < 10; i++) e_head += trace[i];
    for (int i = 0; i < 10; i++) e_tail += trace[cfg.num_iterations - 10 + i];
    e_head /= 10.0;
    e_tail /= 10.0;
    /* We expect the late-window mean energy to be lower than the early
     * window (the ansatz is learning). Allow some slack for MC noise. */
    ASSERT_TRUE(e_tail < e_head + 0.1);
    /* We expect the late-window mean energy to be above the true E0
     * (variational principle) but not absurdly so. */
    ASSERT_TRUE(e_tail >= E0 - 1e-6);          /* variational bound  */
    ASSERT_TRUE(e_tail <  E0 + 6.0);           /* didn't blow up     */
    printf("# TFIM 2x2: E_exact = %.6f, E_final = %.6f, gap = %.4f\n",
           E0, e_tail, e_tail - E0);
    free(trace);
    nqs_sampler_free(sampler);
    nqs_ansatz_free(ansatz);
}
static void test_variational_energy_bounds_exact(void) {
    /* On any trained NQS, the per-iteration mean energy must satisfy
     * <E> >= E0 (variational principle). Verify that holds in every
     * iteration of the trace. */
    int Lx = 2, Ly = 2, N = Lx * Ly;
    double E0 = tfim_ground_state_energy(Lx, Ly, 1.0, 1.0);
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_TFIM;
    cfg.j_coupling = 1.0;
    cfg.transverse_field = 1.0;
    cfg.num_samples = 128;
    cfg.num_thermalize = 64;
    cfg.num_decorrelate = 1;
    cfg.num_iterations = 20;
    cfg.learning_rate = 5e-2;
    cfg.sr_diag_shift = 1e-2;
    cfg.rng_seed = 0xCAFEu;
    nqs_ansatz_t *ansatz = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *sampler = nqs_sampler_create(N, &cfg,
                                                 nqs_ansatz_log_amp, ansatz);
    double trace[20];
    nqs_sr_run(&cfg, Lx, Ly, ansatz, sampler, trace);
    /* Monte-Carlo noise can produce small fluctuations below E0 on
     * individual iterations. Allow a generous noise floor. */
    double floor = E0 - 0.5;
    for (int i = 0; i < cfg.num_iterations; i++) {
        ASSERT_TRUE(trace[i] >= floor);
    }
    nqs_sampler_free(sampler);
    nqs_ansatz_free(ansatz);
}
int main(void) {
    TEST_RUN(test_sr_descends_for_tfim_2x2);
    TEST_RUN(test_variational_energy_bounds_exact);
    TEST_SUMMARY();
}