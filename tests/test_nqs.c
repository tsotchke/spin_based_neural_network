/*
 * tests/test_nqs.c
 *
 * Covers the NQS scaffold introduced for v0.5 pillar P1.1:
 *   - default mean-field ansatz lifecycle
 *   - Metropolis sampler (thermalisation, acceptance, batch API)
 *   - local-energy estimators for TFIM / Heisenberg / J1-J2
 *   - stochastic reconfiguration step + energy convergence
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "nqs/nqs_config.h"
#include "nqs/nqs_sampler.h"
#include "nqs/nqs_gradient.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_optimizer.h"
/* ---------- ansatz lifecycle --------------------------------------- */
static void test_ansatz_defaults_create_and_free(void) {
    nqs_config_t cfg = nqs_config_defaults();
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, 9);
    ASSERT_TRUE(a != NULL);
    ASSERT_EQ_INT(nqs_ansatz_num_params(a), 9);
    nqs_ansatz_free(a);
}
static void test_ansatz_vit_unsupported_in_v04(void) {
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_VIT;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, 9);
    ASSERT_TRUE(a == NULL);  /* ViT path is dormant in v0.4 */
}
static void test_ansatz_gradient_is_spin_vector(void) {
    nqs_config_t cfg = nqs_config_defaults();
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, 4);
    ASSERT_TRUE(a != NULL);
    int spins[4] = {+1, -1, +1, -1};
    double grad[4];
    ASSERT_EQ_INT(nqs_ansatz_logpsi_gradient(a, spins, 4, grad), 0);
    for (int i = 0; i < 4; i++) ASSERT_NEAR(grad[i], (double)spins[i], 1e-12);
    nqs_ansatz_free(a);
}
static void test_ansatz_apply_update_shifts_params(void) {
    nqs_config_t cfg = nqs_config_defaults();
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, 3);
    ASSERT_TRUE(a != NULL);
    int spins[3] = {+1, +1, +1};
    double before_log, before_arg;
    nqs_ansatz_log_amp(spins, 3, a, &before_log, &before_arg);
    double delta[3] = {1.0, 2.0, 3.0};
    ASSERT_EQ_INT(nqs_ansatz_apply_update(a, delta, 0.1), 0);
    double after_log, after_arg;
    nqs_ansatz_log_amp(spins, 3, a, &after_log, &after_arg);
    /* log_amp shifted by Σ spins[i] * step * delta[i] = 0.1*(1+2+3) = 0.6 */
    ASSERT_NEAR(after_log - before_log, 0.6, 1e-12);
    nqs_ansatz_free(a);
}
/* ---------- sampler ------------------------------------------------- */
/* A uniform-amplitude test ansatz: ψ(s) = 1 for all configurations. */
static void uniform_log_amp(const int *s, int n, void *u,
                            double *lr, double *li) {
    (void)s; (void)n; (void)u;
    *lr = 0.0;
    *li = 0.0;
}
static void test_sampler_creates_and_thermalizes(void) {
    nqs_config_t cfg = nqs_config_defaults();
    cfg.num_thermalize = 64;
    cfg.num_decorrelate = 2;
    cfg.rng_seed = 42;
    nqs_sampler_t *s = nqs_sampler_create(16, &cfg, uniform_log_amp, NULL);
    ASSERT_TRUE(s != NULL);
    nqs_sampler_thermalize(s);
    const int *cur = nqs_sampler_current(s);
    ASSERT_TRUE(cur != NULL);
    for (int i = 0; i < 16; i++) ASSERT_TRUE(cur[i] == 1 || cur[i] == -1);
    nqs_sampler_free(s);
}
static void test_sampler_batch_writes_all_pm1(void) {
    nqs_config_t cfg = nqs_config_defaults();
    cfg.rng_seed = 7;
    cfg.num_thermalize = 32;
    cfg.num_decorrelate = 1;
    int N = 8;
    nqs_sampler_t *s = nqs_sampler_create(N, &cfg, uniform_log_amp, NULL);
    ASSERT_TRUE(s != NULL);
    nqs_sampler_thermalize(s);
    int batch_size = 64;
    int *batch = malloc((size_t)batch_size * (size_t)N * sizeof(int));
    ASSERT_TRUE(batch != NULL);
    ASSERT_EQ_INT(nqs_sampler_batch(s, batch_size, batch), 0);
    for (int i = 0; i < batch_size * N; i++) {
        ASSERT_TRUE(batch[i] == 1 || batch[i] == -1);
    }
    free(batch);
    nqs_sampler_free(s);
}
static void test_sampler_acceptance_for_uniform_is_one(void) {
    /* Uniform amplitude → every proposal is accepted. */
    nqs_config_t cfg = nqs_config_defaults();
    cfg.num_thermalize = 128;
    cfg.rng_seed = 3;
    nqs_sampler_t *s = nqs_sampler_create(10, &cfg, uniform_log_amp, NULL);
    ASSERT_TRUE(s != NULL);
    nqs_sampler_thermalize(s);
    ASSERT_NEAR(nqs_sampler_acceptance_ratio(s), 1.0, 1e-12);
    nqs_sampler_free(s);
}
/* ---------- local-energy estimators --------------------------------- */
static void test_tfim_diagonal_all_up_is_minus_J_times_bonds(void) {
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_TFIM;
    cfg.transverse_field = 0.0;   /* isolate the diagonal */
    cfg.j_coupling = 1.0;
    /* 3x3 lattice: open-BC bonds = 2*(3-1)*3 = 12. H = -J·Σ s_i s_{i+1}.
     * For all-up (all +1): local energy = -J × 12 = -12. */
    int spins[9] = {+1,+1,+1, +1,+1,+1, +1,+1,+1};
    double E = nqs_local_energy(&cfg, 3, 3, spins,
                                uniform_log_amp, NULL);
    ASSERT_NEAR(E, -12.0, 1e-12);
}
static void test_tfim_with_transverse_field_is_finite(void) {
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_TFIM;
    cfg.transverse_field = 0.5;
    cfg.j_coupling = 1.0;
    int spins[9] = {+1,-1,+1, -1,+1,-1, +1,-1,+1};
    double E = nqs_local_energy(&cfg, 3, 3, spins,
                                uniform_log_amp, NULL);
    ASSERT_TRUE(E == E); /* not NaN */
    ASSERT_TRUE(E > -1e3 && E < 1e3);
}
static void test_heisenberg_diagonal_antiferro_neel(void) {
    /* Neel state: all antiparallel. Heisenberg diagonal = J/4 × Σ (s_i s_j).
     * On a 3x3 open-BC lattice the 12 bonds all have s_i s_j = -1, so
     * diagonal = J/4 × (-12) = -3. Off-diagonal term also fires: for each
     * antiparallel pair we add 0.5·J · ψ(flipped)/ψ. With uniform ψ, that
     * ratio is 1, so off-diagonal contributes 0.5·1·12 = 6. Total = 3. */
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_HEISENBERG;
    cfg.j_coupling = 1.0;
    int spins[9] = {+1,-1,+1, -1,+1,-1, +1,-1,+1};
    double E = nqs_local_energy(&cfg, 3, 3, spins,
                                uniform_log_amp, NULL);
    ASSERT_NEAR(E, 3.0, 1e-12);
}
static void test_j1_j2_reduces_to_heisenberg_at_j2_zero(void) {
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_J1_J2;
    cfg.j_coupling = 1.0;
    cfg.j2_coupling = 0.0;
    int spins[9] = {+1,-1,+1, -1,+1,-1, +1,-1,+1};
    double E = nqs_local_energy(&cfg, 3, 3, spins,
                                uniform_log_amp, NULL);
    ASSERT_NEAR(E, 3.0, 1e-12);
}
static void test_energy_accumulator_mean_and_variance(void) {
    nqs_energy_accumulator_t acc;
    nqs_energy_accumulator_init(&acc);
    double samples[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    for (int i = 0; i < 5; i++) nqs_energy_accumulator_add(&acc, samples[i]);
    ASSERT_NEAR(nqs_energy_accumulator_mean(&acc), 3.0, 1e-12);
    /* Population variance = Σ(x-μ)² / N = 10 / 5 = 2. */
    ASSERT_NEAR(nqs_energy_accumulator_variance(&acc), 2.0, 1e-12);
}
/* ---------- stochastic reconfiguration -------------------------------- */
static void test_sr_step_runs_without_crash(void) {
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_TFIM;
    cfg.transverse_field = 1.0;
    cfg.num_samples = 64;
    cfg.num_thermalize = 64;
    cfg.num_decorrelate = 1;
    cfg.num_iterations = 1;
    cfg.sr_diag_shift = 1e-2;
    cfg.sr_cg_max_iters = 10;
    int Lx = 2, Ly = 2;
    int N = Lx * Ly;
    nqs_ansatz_t *ansatz = nqs_ansatz_create(&cfg, N);
    ASSERT_TRUE(ansatz != NULL);
    nqs_sampler_t *sampler = nqs_sampler_create(N, &cfg,
                                                 nqs_ansatz_log_amp, ansatz);
    ASSERT_TRUE(sampler != NULL);
    nqs_sampler_thermalize(sampler);
    nqs_sr_step_info_t info;
    int rc = nqs_sr_step(&cfg, Lx, Ly, ansatz, sampler, &info);
    ASSERT_EQ_INT(rc, 0);
    ASSERT_TRUE(info.mean_energy == info.mean_energy);   /* not NaN */
    ASSERT_TRUE(info.acceptance_ratio >= 0.0 && info.acceptance_ratio <= 1.0);
    nqs_sampler_free(sampler);
    nqs_ansatz_free(ansatz);
}
static void test_sr_run_produces_energy_trace(void) {
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_TFIM;
    cfg.transverse_field = 0.5;
    cfg.num_samples = 32;
    cfg.num_thermalize = 32;
    cfg.num_decorrelate = 1;
    cfg.num_iterations = 5;
    cfg.learning_rate = 1e-3;
    cfg.sr_diag_shift = 1e-2;
    int Lx = 2, Ly = 2, N = Lx * Ly;
    nqs_ansatz_t *ansatz = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *sampler = nqs_sampler_create(N, &cfg,
                                                 nqs_ansatz_log_amp, ansatz);
    ASSERT_TRUE(ansatz && sampler);
    double trace[5];
    int rc = nqs_sr_run(&cfg, Lx, Ly, ansatz, sampler, trace);
    ASSERT_EQ_INT(rc, 0);
    for (int i = 0; i < 5; i++) ASSERT_TRUE(trace[i] == trace[i]);
    nqs_sampler_free(sampler);
    nqs_ansatz_free(ansatz);
}
int main(void) {
    TEST_RUN(test_ansatz_defaults_create_and_free);
    TEST_RUN(test_ansatz_vit_unsupported_in_v04);
    TEST_RUN(test_ansatz_gradient_is_spin_vector);
    TEST_RUN(test_ansatz_apply_update_shifts_params);
    TEST_RUN(test_sampler_creates_and_thermalizes);
    TEST_RUN(test_sampler_batch_writes_all_pm1);
    TEST_RUN(test_sampler_acceptance_for_uniform_is_one);
    TEST_RUN(test_tfim_diagonal_all_up_is_minus_J_times_bonds);
    TEST_RUN(test_tfim_with_transverse_field_is_finite);
    TEST_RUN(test_heisenberg_diagonal_antiferro_neel);
    TEST_RUN(test_j1_j2_reduces_to_heisenberg_at_j2_zero);
    TEST_RUN(test_energy_accumulator_mean_and_variance);
    TEST_RUN(test_sr_step_runs_without_crash);
    TEST_RUN(test_sr_run_produces_energy_trace);
    TEST_SUMMARY();
}