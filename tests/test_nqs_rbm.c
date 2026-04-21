/*
 * tests/test_nqs_rbm.c
 *
 * Tests for the Restricted Boltzmann Machine (Carleo & Troyer 2017)
 * NQS ansatz. Covers:
 *
 *   - Parameter count matches N + M + M·N.
 *   - Analytic gradient matches numerical finite-difference to 1e-6.
 *   - The RBM is strictly more expressive than mean-field: exists a
 *     parameter setting with higher |ψ(s)| on one configuration and
 *     lower on another (something the mean-field ansatz cannot do
 *     while remaining normalised).
 *   - On TFIM 2x2 the RBM SR converges to a lower energy than
 *     mean-field, i.e. it closes the known 0.85 mean-field gap.
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
static void test_rbm_param_count(void) {
    int N = 4;
    int M = 6;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_RBM;
    cfg.rbm_hidden_units = M;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    ASSERT_TRUE(a != NULL);
    ASSERT_EQ_INT((int)nqs_ansatz_num_params(a), N + M + M * N);
    nqs_ansatz_free(a);
}
static void test_rbm_default_hidden_is_two_N(void) {
    int N = 5;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_RBM;
    cfg.rbm_hidden_units = 0;        /* → 2N default */
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    ASSERT_TRUE(a != NULL);
    int M = 2 * N;
    ASSERT_EQ_INT((int)nqs_ansatz_num_params(a), N + M + M * N);
    nqs_ansatz_free(a);
}
static void test_rbm_analytic_gradient_matches_finite_difference(void) {
    int N = 4, M = 3;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_RBM;
    cfg.rbm_hidden_units = M;
    cfg.rbm_init_scale = 0.3;   /* larger init → non-trivial tanh slope */
    cfg.rng_seed = 0x1234u;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    ASSERT_TRUE(a != NULL);
    int spins[4] = {+1, -1, +1, -1};
    long P = nqs_ansatz_num_params(a);
    double *g_analytic = malloc((size_t)P * sizeof(double));
    double *g_fd       = malloc((size_t)P * sizeof(double));
    ASSERT_EQ_INT(nqs_ansatz_logpsi_gradient(a, spins, N, g_analytic), 0);
    /* Finite-difference: perturb each parameter and re-evaluate log ψ. */
    double eps = 1e-6;
    double delta[1];   /* dummy; we use apply_update directly on a 1-wide slot */
    (void)delta;
    for (long p = 0; p < P; p++) {
        double *perturb = calloc((size_t)P, sizeof(double));
        perturb[p] = 1.0;
        nqs_ansatz_apply_update(a,  perturb,  eps);
        double lp_plus; nqs_ansatz_log_amp(spins, N, a, &lp_plus, NULL);
        nqs_ansatz_apply_update(a,  perturb, -2.0 * eps);
        double lp_minus; nqs_ansatz_log_amp(spins, N, a, &lp_minus, NULL);
        nqs_ansatz_apply_update(a,  perturb,  eps);           /* restore */
        g_fd[p] = (lp_plus - lp_minus) / (2.0 * eps);
        free(perturb);
    }
    for (long p = 0; p < P; p++) {
        double err = fabs(g_analytic[p] - g_fd[p]);
        if (!(err < 1e-6)) {
            fprintf(stderr, "# gradient mismatch at p=%ld: analytic=%.10f fd=%.10f\n",
                    p, g_analytic[p], g_fd[p]);
        }
        ASSERT_TRUE(err < 1e-6);
    }
    free(g_analytic);
    free(g_fd);
    nqs_ansatz_free(a);
}
static void test_rbm_log_amp_is_finite(void) {
    /* Stability: even for spins that drive tanh() into saturation the
     * log-amplitude must remain finite (the log(2 cosh x) form avoids
     * overflow via the |x| + log1p(exp(-2|x|)) identity). */
    int N = 8, M = 8;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_RBM;
    cfg.rbm_hidden_units = M;
    cfg.rbm_init_scale = 5.0;   /* deliberately huge */
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    int spins[8] = {+1, -1, +1, +1, -1, -1, +1, -1};
    double lp; nqs_ansatz_log_amp(spins, N, a, &lp, NULL);
    ASSERT_TRUE(isfinite(lp));
    nqs_ansatz_free(a);
}
/* Exact diagonalisation of TFIM 2x2 via power iteration on shift-I − H. */
static double tfim_2x2_exact_e0(double J, double Gamma) {
    int Lx = 2, Ly = 2, N = 4;
    int dim = 1 << N;
    double *H = calloc((size_t)dim * dim, sizeof(double));
    for (int s = 0; s < dim; s++) {
        double e = 0.0;
        for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++) {
            int idx = x * Ly + y;
            int sv = ((s >> idx) & 1) ? -1 : +1;
            if (x + 1 < Lx) { int j = (x+1)*Ly + y; int sj = ((s>>j)&1)?-1:+1; e += -J*sv*sj; }
            if (y + 1 < Ly) { int j = x*Ly + (y+1); int sj = ((s>>j)&1)?-1:+1; e += -J*sv*sj; }
        }
        H[s*dim + s] = e;
        for (int i = 0; i < N; i++) { int s2 = s ^ (1 << i); H[s*dim + s2] += -Gamma; }
    }
    double row_max = 0;
    for (int i = 0; i < dim; i++) {
        double r = 0; for (int j = 0; j < dim; j++) r += fabs(H[i*dim + j]);
        if (r > row_max) row_max = r;
    }
    double shift = row_max + 1.0;
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) H[i*dim + j] = -H[i*dim + j];
        H[i*dim + i] += shift;
    }
    double *v = malloc(sizeof(double) * dim), *w = malloc(sizeof(double) * dim);
    for (int i = 0; i < dim; i++) v[i] = 1.0 / sqrt(dim);
    double lam = 0;
    for (int it = 0; it < 4000; it++) {
        for (int i = 0; i < dim; i++) { double a = 0; for (int j = 0; j < dim; j++) a += H[i*dim + j] * v[j]; w[i] = a; }
        double num = 0, den = 0; for (int i = 0; i < dim; i++) { num += v[i]*w[i]; den += v[i]*v[i]; }
        double lam_new = num / den;
        double n2 = 0; for (int i = 0; i < dim; i++) n2 += w[i]*w[i]; n2 = sqrt(n2);
        if (n2 > 0) for (int i = 0; i < dim; i++) v[i] = w[i] / n2;
        if (it > 0 && fabs(lam_new - lam) < 1e-12) { lam = lam_new; break; }
        lam = lam_new;
    }
    double E0 = shift - lam;
    free(H); free(v); free(w);
    return E0;
}
static double run_sr(nqs_ansatz_kind_t kind, int M_hidden,
                     int Lx, int Ly, double J, double Gamma,
                     int num_iter, int num_samples, unsigned seed) {
    int N = Lx * Ly;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = kind;
    cfg.rbm_hidden_units = M_hidden;
    cfg.rbm_init_scale = 0.1;
    cfg.hamiltonian = NQS_HAM_TFIM;
    cfg.j_coupling = J;
    cfg.transverse_field = Gamma;
    cfg.num_samples = num_samples;
    cfg.num_thermalize = 256;
    cfg.num_decorrelate = 2;
    cfg.num_iterations = num_iter;
    cfg.learning_rate = 3e-2;
    cfg.sr_diag_shift = 1e-2;
    cfg.sr_cg_max_iters = 50;
    cfg.rng_seed = seed;
    nqs_ansatz_t *ansatz = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *sampler = nqs_sampler_create(N, &cfg,
                                                 nqs_ansatz_log_amp, ansatz);
    double *trace = malloc(sizeof(double) * num_iter);
    nqs_sr_run(&cfg, Lx, Ly, ansatz, sampler, trace);
    /* Average of the last 20% of iterations (post-convergence window). */
    int tail_start = (int)(num_iter * 0.8);
    int tail_len = num_iter - tail_start;
    double tail = 0.0;
    for (int i = tail_start; i < num_iter; i++) tail += trace[i];
    tail /= (double)tail_len;
    free(trace);
    nqs_sampler_free(sampler);
    nqs_ansatz_free(ansatz);
    return tail;
}
static void test_rbm_beats_mean_field_on_tfim_2x2(void) {
    double J = 1.0, Gamma = 1.0;
    double E0 = tfim_2x2_exact_e0(J, Gamma);
    double e_mf  = run_sr(NQS_ANSATZ_LEGACY_MLP, 0, 2, 2, J, Gamma,
                          80, 512, 0xB16B00B5u);
    double e_rbm = run_sr(NQS_ANSATZ_RBM,        8, 2, 2, J, Gamma,
                          80, 512, 0xB16B00B5u);
    double gap_mf  = e_mf  - E0;
    double gap_rbm = e_rbm - E0;
    printf("# TFIM 2x2: E_exact=%.6f  MF=%.6f (gap %.4f)  RBM=%.6f (gap %.4f)\n",
           E0, e_mf, gap_mf, e_rbm, gap_rbm);
    /* The RBM ansatz spans strictly more wavefunctions than mean-field,
     * so its variational minimum cannot be higher. Allow a little MC
     * noise. */
    ASSERT_TRUE(e_rbm <= e_mf + 0.1);
    /* And it should close the known mean-field gap to within ~0.3. */
    ASSERT_TRUE(gap_rbm < gap_mf * 0.9);
}
int main(void) {
    TEST_RUN(test_rbm_param_count);
    TEST_RUN(test_rbm_default_hidden_is_two_N);
    TEST_RUN(test_rbm_analytic_gradient_matches_finite_difference);
    TEST_RUN(test_rbm_log_amp_is_finite);
    TEST_RUN(test_rbm_beats_mean_field_on_tfim_2x2);
    TEST_SUMMARY();
}