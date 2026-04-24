/*
 * tests/test_nqs_excited.c
 *
 * Excited-state stochastic reconfiguration via orthogonal-ansatz
 * penalty. Two scales of test, paired:
 *
 *   (1) Unit-level: on a frozen reference, verify that the penalty
 *       acts only as a gradient pressure — running excited SR with
 *       penalty_mu = 0 must give the same energy trajectory as plain
 *       holomorphic SR (the penalty must *only* enter through the
 *       augmented local-energy expectation).
 *
 *   (2) Physics: on the 2-site Heisenberg AFM (E₀ = -0.75, E₁ = +0.25,
 *       Δ = J = 1), given a pretrained reference approximating the
 *       ground state, excited SR with a large penalty must push the
 *       trained ansatz's energy decisively above the ground state and
 *       toward the triplet gap.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "harness.h"
#include "nqs/nqs_config.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_sampler.h"
#include "nqs/nqs_optimizer.h"

static void populate_cfg_for_2site_heisenberg(nqs_config_t *cfg,
                                               int num_iter, unsigned seed) {
    *cfg = nqs_config_defaults();
    cfg->ansatz           = NQS_ANSATZ_COMPLEX_RBM;
    cfg->rbm_hidden_units = 8;
    cfg->rbm_init_scale   = 0.05;
    cfg->hamiltonian      = NQS_HAM_HEISENBERG;
    cfg->j_coupling       = 1.0;
    cfg->num_samples      = 512;
    cfg->num_thermalize   = 256;
    cfg->num_decorrelate  = 2;
    cfg->num_iterations   = num_iter;
    cfg->learning_rate    = 0.04;
    cfg->sr_diag_shift    = 1e-2;
    cfg->sr_cg_max_iters  = 80;
    cfg->sr_cg_tol        = 1e-7;
    cfg->rng_seed         = seed;
}

static void test_excited_zero_mu_matches_holomorphic(void) {
    /* With penalty_mu = 0 the excited step is mathematically identical
     * to holomorphic SR — the augmented E_loc equals the physical
     * E_loc, and the gradient direction is unchanged. Running both
     * from the same seed must produce energy trajectories that agree
     * within MC noise. Test is on equivalence only; whether either arm
     * reaches the singlet is the subject of test 3. */
    int Lx = 2, Ly = 1, N = Lx * Ly;
    int iters = 200;

    nqs_config_t cfg_h;
    populate_cfg_for_2site_heisenberg(&cfg_h, iters, 0x2222u);
    nqs_ansatz_t  *a_h = nqs_ansatz_create(&cfg_h, N);
    nqs_sampler_t *s_h = nqs_sampler_create(N, &cfg_h, nqs_ansatz_log_amp, a_h);
    ASSERT_TRUE(a_h && s_h);
    double *trace_h = malloc(sizeof(double) * (size_t)iters);
    ASSERT_TRUE(trace_h != NULL);
    int rc_h = nqs_sr_run_holomorphic(&cfg_h, Lx, Ly, a_h, s_h,
                                        nqs_ansatz_log_amp, a_h, trace_h);
    ASSERT_EQ_INT(rc_h, 0);

    /* A separate zero-iteration reference ansatz, used only to keep
     * a valid ref_log_amp pointer alive during the excited run.
     * At mu=0 its contents are irrelevant to the update. */
    nqs_config_t cfg_ref;
    populate_cfg_for_2site_heisenberg(&cfg_ref, 0, 0xEB1u);
    nqs_ansatz_t *a_ref = nqs_ansatz_create(&cfg_ref, N);
    ASSERT_TRUE(a_ref);

    nqs_config_t cfg_e;
    populate_cfg_for_2site_heisenberg(&cfg_e, iters, 0x2222u);
    nqs_ansatz_t  *a_e = nqs_ansatz_create(&cfg_e, N);
    nqs_sampler_t *s_e = nqs_sampler_create(N, &cfg_e, nqs_ansatz_log_amp, a_e);
    ASSERT_TRUE(a_e && s_e);
    double *trace_e = malloc(sizeof(double) * (size_t)iters);
    ASSERT_TRUE(trace_e != NULL);
    int rc_e = nqs_sr_run_excited(&cfg_e, Lx, Ly, a_e, s_e,
                                    nqs_ansatz_log_amp, a_e,
                                    nqs_ansatz_log_amp, a_ref,
                                    0.0 /* mu */, trace_e);
    ASSERT_EQ_INT(rc_e, 0);

    /* Tail-mean agreement — MC noise is the only source of drift. */
    double m_h = 0, m_e = 0;
    for (int i = iters - 20; i < iters; i++) { m_h += trace_h[i]; m_e += trace_e[i]; }
    m_h /= 20.0; m_e /= 20.0;
    printf("# mu=0 equivalence: holomorphic tail=%.4f  excited tail=%.4f\n",
           m_h, m_e);
    ASSERT_TRUE(isfinite(m_h) && isfinite(m_e));
    /* The two paths start from the same seed, take the same sampler
     * steps, and compute the same gradient; tail means must match to
     * tight MC slack. */
    ASSERT_TRUE(fabs(m_h - m_e) < 0.05);

    free(trace_e); free(trace_h);
    nqs_sampler_free(s_e); nqs_ansatz_free(a_e);
    nqs_ansatz_free(a_ref);
    nqs_sampler_free(s_h); nqs_ansatz_free(a_h);
}

static void test_excited_penalty_drives_energy_above_ground_state(void) {
    /* 2-site Heisenberg AFM: E₀ = -0.75, E₁ = +0.25. Gap = 1.
     * Pretrain a reference ansatz deep in the singlet sector with
     * holomorphic SR. Then train a second complex RBM with excited SR
     * + large penalty steering it away from the reference — its
     * converged energy must be substantially above the reference. */
    int Lx = 2, Ly = 1, N = Lx * Ly;

    /* --- Pretrain reference ansatz toward the singlet GS. --- */
    int ref_iters = 400;
    nqs_config_t cfg_ref;
    populate_cfg_for_2site_heisenberg(&cfg_ref, ref_iters, 0x2222u);
    nqs_ansatz_t  *a_ref = nqs_ansatz_create(&cfg_ref, N);
    nqs_sampler_t *s_ref = nqs_sampler_create(N, &cfg_ref,
                                                nqs_ansatz_log_amp, a_ref);
    ASSERT_TRUE(a_ref && s_ref);
    double *trace_ref = malloc(sizeof(double) * (size_t)ref_iters);
    ASSERT_TRUE(trace_ref != NULL);
    int rc_ref = nqs_sr_run_holomorphic(&cfg_ref, Lx, Ly, a_ref, s_ref,
                                          nqs_ansatz_log_amp, a_ref, trace_ref);
    ASSERT_EQ_INT(rc_ref, 0);
    double E_ref = 0.0;
    for (int i = ref_iters - 20; i < ref_iters; i++) E_ref += trace_ref[i];
    E_ref /= 20.0;
    printf("# reference GS: E_ref tail mean = %.4f (E₀=-0.75)\n", E_ref);
    /* 2-site Heisenberg: exact E₀=-0.75. Even a loose convergence
     * (E_ref < -0.4) still leaves a decisive gap to the triplet
     * +0.25. The precise depth is seed-dependent; we only require
     * that the reference is *clearly* in the singlet sector. */
    ASSERT_TRUE(E_ref < -0.4);

    /* Reference is now frozen — its sampler is no longer needed. The
     * ansatz stays alive for ref_log_amp queries during excited SR. */
    nqs_sampler_free(s_ref); s_ref = NULL;

    /* --- Train excited-state ansatz with large penalty. --- */
    int exc_iters = 400;
    nqs_config_t cfg_exc;
    populate_cfg_for_2site_heisenberg(&cfg_exc, exc_iters, 0xEB3u);
    cfg_exc.learning_rate = 0.03;     /* smaller step: penalty adds force */
    cfg_exc.sr_diag_shift  = 5e-3;
    nqs_ansatz_t  *a_exc = nqs_ansatz_create(&cfg_exc, N);
    nqs_sampler_t *s_exc = nqs_sampler_create(N, &cfg_exc,
                                                nqs_ansatz_log_amp, a_exc);
    ASSERT_TRUE(a_exc && s_exc);

    double *trace_exc = malloc(sizeof(double) * (size_t)exc_iters);
    ASSERT_TRUE(trace_exc != NULL);
    int rc_exc = nqs_sr_run_excited(&cfg_exc, Lx, Ly, a_exc, s_exc,
                                      nqs_ansatz_log_amp, a_exc,
                                      nqs_ansatz_log_amp, a_ref,
                                      5.0 /* mu — strong push */,
                                      trace_exc);
    ASSERT_EQ_INT(rc_exc, 0);

    /* Tail mean — the reported energy is the *physical* ⟨H⟩, not the
     * loss. It must land substantially above E_ref. */
    double E_exc = 0.0;
    for (int i = exc_iters - 20; i < exc_iters; i++) E_exc += trace_exc[i];
    E_exc /= 20.0;
    printf("# excited-SR tail mean: E=%.4f  (E_ref=%.4f, E₀=-0.75, E₁=+0.25, gap=1)\n",
           E_exc, E_ref);
    ASSERT_TRUE(isfinite(E_exc));
    /* The penalty must have demonstrably steered away from the
     * reference. Require at least 0.5 units of separation (half the
     * true gap), and land below the variational upper bound of +0.7. */
    ASSERT_TRUE(E_exc > E_ref + 0.5);
    ASSERT_TRUE(E_exc <  0.7);

    free(trace_exc); free(trace_ref);
    nqs_sampler_free(s_exc); nqs_ansatz_free(a_exc);
    nqs_ansatz_free(a_ref);
}

static void test_excited_on_kagome_pipeline_smoke(void) {
    /* Kagome Heisenberg on the 2x2 PBC cluster (N=12). Purely a
     * *wiring* smoke that exercises the excited-SR code path through
     * the multi-sublattice kagome local-energy kernel. No convergence
     * claim: on a freshly-initialised cRBM reference, the penalty
     * gradient can destabilise single SR steps until thermalisation
     * catches up — the research-scale driver
     * (scripts/research_kagome_N12_diagnostics.c) pretrains to
     * convergence first, as does the decisive 2-site test above.
     *
     * Here we only assert that `nqs_sr_run_excited` returns rc=0 on
     * kagome — i.e. the dispatch + kernel + diagnostic piping is
     * intact and no crash occurs. Numerical correctness of the
     * excited-state energy on kagome is covered by the driver run. */
    int Lx = 2, Ly = 2, N = 3 * Lx * Ly;    /* = 12 */
    int iters = 20;

    nqs_config_t cfg_gs = nqs_config_defaults();
    cfg_gs.ansatz           = NQS_ANSATZ_COMPLEX_RBM;
    cfg_gs.rbm_hidden_units = 12;
    cfg_gs.rbm_init_scale   = 0.05;
    cfg_gs.hamiltonian      = NQS_HAM_KAGOME_HEISENBERG;
    cfg_gs.j_coupling       = 1.0;
    cfg_gs.kagome_pbc       = 1;
    cfg_gs.num_samples      = 256;
    cfg_gs.num_thermalize   = 128;
    cfg_gs.num_decorrelate  = 2;
    cfg_gs.num_iterations   = iters;
    cfg_gs.learning_rate    = 0.01;    /* smaller step: avoid runaway */
    cfg_gs.sr_diag_shift    = 1e-2;
    cfg_gs.sr_cg_max_iters  = 40;
    cfg_gs.sr_cg_tol        = 1e-7;
    cfg_gs.rng_seed         = 0xC0AEE1u;
    nqs_ansatz_t  *a_gs = nqs_ansatz_create(&cfg_gs, N);
    nqs_sampler_t *s_gs = nqs_sampler_create(N, &cfg_gs, nqs_ansatz_log_amp, a_gs);
    ASSERT_TRUE(a_gs && s_gs);
    double *trace_gs = malloc(sizeof(double) * (size_t)iters);
    ASSERT_TRUE(trace_gs != NULL);
    int rc_gs = nqs_sr_run_holomorphic(&cfg_gs, Lx, Ly, a_gs, s_gs,
                                         nqs_ansatz_log_amp, a_gs, trace_gs);
    ASSERT_EQ_INT(rc_gs, 0);

    nqs_config_t cfg_exc = cfg_gs;
    cfg_exc.rng_seed = 0xC0AEE2u;
    nqs_ansatz_t  *a_exc = nqs_ansatz_create(&cfg_exc, N);
    nqs_sampler_t *s_exc = nqs_sampler_create(N, &cfg_exc, nqs_ansatz_log_amp, a_exc);
    ASSERT_TRUE(a_exc && s_exc);
    double *trace_exc = malloc(sizeof(double) * (size_t)iters);
    ASSERT_TRUE(trace_exc != NULL);
    int rc_exc = nqs_sr_run_excited(&cfg_exc, Lx, Ly, a_exc, s_exc,
                                      nqs_ansatz_log_amp, a_exc,
                                      nqs_ansatz_log_amp, a_gs,
                                      0.5 /* mu — gentle penalty */,
                                      trace_exc);
    ASSERT_EQ_INT(rc_exc, 0);

    printf("# kagome N=12 excited-SR pipeline smoke: rc=0 (%d iters completed)\n", iters);

    free(trace_exc); free(trace_gs);
    nqs_sampler_free(s_exc); nqs_ansatz_free(a_exc);
    nqs_sampler_free(s_gs); nqs_ansatz_free(a_gs);
}

static void test_excited_rejects_null_ref(void) {
    int Lx = 2, Ly = 1, N = Lx * Ly;
    nqs_config_t cfg;
    populate_cfg_for_2site_heisenberg(&cfg, 10, 0xEB4u);
    nqs_ansatz_t  *a = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a);
    ASSERT_TRUE(a && s);
    int rc = nqs_sr_run_excited(&cfg, Lx, Ly, a, s,
                                  nqs_ansatz_log_amp, a,
                                  NULL, NULL, 1.0, NULL);
    ASSERT_TRUE(rc != 0);
    nqs_sampler_free(s); nqs_ansatz_free(a);
}

int main(void) {
    TEST_RUN(test_excited_zero_mu_matches_holomorphic);
    TEST_RUN(test_excited_rejects_null_ref);
    TEST_RUN(test_excited_penalty_drives_energy_above_ground_state);
    TEST_RUN(test_excited_on_kagome_pipeline_smoke);
    TEST_SUMMARY();
}
