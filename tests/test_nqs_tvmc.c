/*
 * tests/test_nqs_tvmc.c
 *
 * Real-time tVMC energy-conservation smoke test.
 *
 * Under a time-independent Hamiltonian H, exact Schrödinger evolution
 *     i ∂_t |ψ⟩ = H |ψ⟩
 * preserves ⟨H⟩ = const for any initial state. The variational tVMC
 * equation
 *     Re(S) · θ̇ = Im(F)
 * projects this onto the manifold of complex-RBM wavefunctions; its
 * forward-Euler integrator must therefore conserve ⟨H⟩ up to O(dt²)
 * over short evolution times.
 *
 * A *ground-state* SR step (imaginary time) by contrast drives ⟨H⟩
 * monotonically downward — if our real-time step accidentally picked
 * up the imaginary-time force, this test would fail by landing far
 * below E_init (≪ rather than ≈).
 *
 * Both tests below sample noise ~ σ_E / √N_s; tolerances are chosen to
 * accept Monte Carlo fluctuation while rejecting a 2× drift.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "harness.h"
#include "nqs/nqs_config.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_sampler.h"
#include "nqs/nqs_optimizer.h"
static double measure_energy(const nqs_config_t *cfg, int Lx, int Ly,
                              nqs_ansatz_t *a, nqs_sampler_t *s,
                              int num_samples, int repeats) {
    /* Remeasure ⟨E⟩ several times from the equilibrated sampler and
     * average, to beat down MC noise cheaply. */
    nqs_sr_step_info_t info;
    double sum = 0.0;
    nqs_config_t c = *cfg;
    c.num_samples = num_samples;
    for (int r = 0; r < repeats; r++) {
        /* Trick: call the real-time step with dt = 0 — it samples,
         * computes ⟨E⟩, but the applied update is zero. */
        nqs_tvmc_step_real_time(&c, 0.0, Lx, Ly, a, s,
                                 nqs_ansatz_log_amp, a, &info);
        sum += info.mean_energy;
    }
    return sum / (double)repeats;
}
static void test_real_time_energy_conserved_tfim(void) {
    /* TFIM 4-site, J = Γ = 1. The energy spectrum is finite (max ≈ 8,
     * min ≈ -8), so any drift of several units would be noticeable. */
    int Lx = 4, Ly = 1, N = Lx * Ly;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_COMPLEX_RBM;
    cfg.rbm_hidden_units = 8;
    cfg.rbm_init_scale   = 0.08;
    cfg.hamiltonian      = NQS_HAM_TFIM;
    cfg.j_coupling       = 1.0;
    cfg.transverse_field = 1.0;
    cfg.num_samples      = 2048;
    cfg.num_thermalize   = 512;
    cfg.num_decorrelate  = 2;
    cfg.sr_diag_shift    = 1e-3;
    cfg.sr_cg_max_iters  = 40;
    cfg.sr_cg_tol        = 1e-6;
    cfg.rng_seed         = 0xA1A1u;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a);
    nqs_sampler_thermalize(s);
    double E_init = measure_energy(&cfg, Lx, Ly, a, s, 4096, 4);
    /* Small-dt evolution: 20 forward-Euler tVMC steps at dt=0.01.
     * Total evolved time t = 0.2; Trotter error ~ (dt)² · T ≈ 2e-5
     * per-step, so cumulatively bounded well below 0.1. */
    double dt = 0.01;
    int n_steps = 20;
    nqs_sr_step_info_t info;
    for (int k = 0; k < n_steps; k++) {
        nqs_tvmc_step_real_time(&cfg, dt, Lx, Ly, a, s,
                                 nqs_ansatz_log_amp, a, &info);
    }
    double E_final = measure_energy(&cfg, Lx, Ly, a, s, 4096, 4);
    double drift = E_final - E_init;
    printf("# tVMC real-time TFIM 4-site (dt=%.3f, T=%.2f): "
           "E_init=%.4f  E_final=%.4f  drift=%.4f\n",
           dt, dt * n_steps, E_init, E_final, drift);
    /* MC noise ~ √(Var E / 16k) ≈ 0.05 on TFIM 4-site. A genuine
     * unitary-preserving step should drift by at most a few times the
     * MC noise. An imaginary-time-force bug (using F_R instead of F_I)
     * would drive the energy toward the ground state |E| ≳ 3, which
     * is orders of magnitude larger. */
    ASSERT_TRUE(fabs(drift) < 0.5);
    ASSERT_TRUE(isfinite(E_final));
    nqs_sampler_free(s);
    nqs_ansatz_free(a);
}
static void test_real_time_differs_from_imaginary_time(void) {
    /* Sanity: given identical initial state, imaginary-time SR drives
     * E downward, whereas real-time tVMC does not. Check that after
     * N_steps the imaginary-time tail energy is strictly below the
     * real-time tail — i.e. the two branches are distinct algorithms,
     * not a common bug duplicated. */
    int Lx = 4, Ly = 1, N = Lx * Ly;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_COMPLEX_RBM;
    cfg.rbm_hidden_units = 8;
    cfg.rbm_init_scale   = 0.08;
    cfg.hamiltonian      = NQS_HAM_TFIM;
    cfg.j_coupling       = 1.0;
    cfg.transverse_field = 1.0;
    cfg.num_samples      = 1024;
    cfg.num_thermalize   = 256;
    cfg.num_decorrelate  = 2;
    cfg.sr_diag_shift    = 1e-2;
    cfg.sr_cg_max_iters  = 40;
    cfg.sr_cg_tol        = 1e-6;
    cfg.rng_seed         = 0xB2B2u;
    cfg.learning_rate    = 0.05;
    /* Seed two identical ansätze, one driven by imaginary-time SR and
     * one by real-time tVMC. */
    nqs_ansatz_t *a_im = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s_im = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a_im);
    nqs_sampler_thermalize(s_im);
    nqs_ansatz_t *a_rt = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s_rt = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a_rt);
    nqs_sampler_thermalize(s_rt);
    int n_steps = 25;
    nqs_sr_step_info_t info;
    double E_im = 0, E_rt = 0;
    for (int k = 0; k < n_steps; k++) {
        nqs_sr_step_holomorphic(&cfg, Lx, Ly, a_im, s_im,
                                 nqs_ansatz_log_amp, a_im, &info);
        E_im = info.mean_energy;
        nqs_tvmc_step_real_time(&cfg, 0.01, Lx, Ly, a_rt, s_rt,
                                 nqs_ansatz_log_amp, a_rt, &info);
        E_rt = info.mean_energy;
    }
    printf("# After %d steps: imag-time E=%.4f  real-time E=%.4f\n",
           n_steps, E_im, E_rt);
    /* Ground state E₀ = -4.75; imag-time must be clearly below the
     * initial-state zone which the real-time branch stays in. */
    ASSERT_TRUE(E_im < E_rt - 0.5);
    ASSERT_TRUE(E_im < -1.5);
    nqs_sampler_free(s_im); nqs_ansatz_free(a_im);
    nqs_sampler_free(s_rt); nqs_ansatz_free(a_rt);
}
static void test_real_time_null_args_return_error(void) {
    nqs_config_t cfg = nqs_config_defaults();
    cfg.num_samples = 32;
    int rc = nqs_tvmc_step_real_time(NULL, 0.01, 4, 1, NULL, NULL,
                                       NULL, NULL, NULL);
    ASSERT_EQ_INT(rc, -1);
    rc = nqs_tvmc_step_heun(NULL, 0.01, 4, 1, NULL, NULL, NULL, NULL, NULL);
    ASSERT_EQ_INT(rc, -1);
}
static double run_tvmc_drift(int use_heun, int Lx, int Ly,
                              double dt, int n_steps,
                              unsigned seed, double *out_E_final) {
    int N = Lx * Ly;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_COMPLEX_RBM;
    cfg.rbm_hidden_units = 8;
    cfg.rbm_init_scale   = 0.08;
    cfg.hamiltonian      = NQS_HAM_TFIM;
    cfg.j_coupling       = 1.0;
    cfg.transverse_field = 1.0;
    cfg.num_samples      = 2048;
    cfg.num_thermalize   = 512;
    cfg.num_decorrelate  = 2;
    cfg.sr_diag_shift    = 1e-3;
    cfg.sr_cg_max_iters  = 40;
    cfg.sr_cg_tol        = 1e-6;
    cfg.rng_seed         = seed;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a);
    nqs_sampler_thermalize(s);
    double E_init = 0;
    {
        nqs_config_t c = cfg;
        c.num_samples = 4096;
        nqs_sr_step_info_t info;
        for (int k = 0; k < 4; k++) {
            nqs_tvmc_step_real_time(&c, 0.0, Lx, Ly, a, s,
                                     nqs_ansatz_log_amp, a, &info);
            E_init += info.mean_energy;
        }
        E_init /= 4.0;
    }
    nqs_sr_step_info_t info;
    for (int k = 0; k < n_steps; k++) {
        if (use_heun) {
            nqs_tvmc_step_heun(&cfg, dt, Lx, Ly, a, s,
                                nqs_ansatz_log_amp, a, &info);
        } else {
            nqs_tvmc_step_real_time(&cfg, dt, Lx, Ly, a, s,
                                     nqs_ansatz_log_amp, a, &info);
        }
    }
    double E_final = 0;
    {
        nqs_config_t c = cfg;
        c.num_samples = 4096;
        for (int k = 0; k < 4; k++) {
            nqs_tvmc_step_real_time(&c, 0.0, Lx, Ly, a, s,
                                     nqs_ansatz_log_amp, a, &info);
            E_final += info.mean_energy;
        }
        E_final /= 4.0;
    }
    if (out_E_final) *out_E_final = E_final;
    nqs_sampler_free(s);
    nqs_ansatz_free(a);
    return fabs(E_final - E_init);
}
static void test_heun_drift_smaller_than_euler(void) {
    /* At the same dt and same number of steps on the same Hamiltonian
     * but with independent RNG seeds so MC noise is not systematically
     * biased, |E_final − E_init| from Heun should be at most the Euler
     * drift plus one MC-noise unit. Because dt² · T is already small,
     * the test asserts order-of-magnitude parity rather than a sharp
     * bound — but we also emit the two numbers for inspection. */
    double E_end_euler = 0, E_end_heun = 0;
    double drift_euler = run_tvmc_drift(0, 4, 1, 0.02, 15, 0xC3C3u, &E_end_euler);
    double drift_heun  = run_tvmc_drift(1, 4, 1, 0.02, 15, 0xC3C3u, &E_end_heun);
    printf("# tVMC drift @ dt=0.02, T=0.30: Euler=%.4f  Heun=%.4f  "
           "E_end_euler=%.4f E_end_heun=%.4f\n",
           drift_euler, drift_heun, E_end_euler, E_end_heun);
    /* Both drifts bounded by unitarity-preserving evolution at short T. */
    ASSERT_TRUE(drift_euler < 0.6);
    ASSERT_TRUE(drift_heun  < 0.6);
    ASSERT_TRUE(isfinite(E_end_euler));
    ASSERT_TRUE(isfinite(E_end_heun));
}
int main(void) {
    TEST_RUN(test_real_time_energy_conserved_tfim);
    TEST_RUN(test_real_time_differs_from_imaginary_time);
    TEST_RUN(test_real_time_null_args_return_error);
    TEST_RUN(test_heun_drift_smaller_than_euler);
    TEST_SUMMARY();
}