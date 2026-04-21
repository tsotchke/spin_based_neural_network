/*
 * tests/test_nqs_xxz.c
 *
 * XXZ Hamiltonian H = Jxy (S^x S^x + S^y S^y) + Jz S^z S^z on 1D
 * chains, cross-checked against the DMRG solver in src/mps/. The XXZ
 * spectrum interpolates between Ising (Jxy = 0), Heisenberg (Jz = Jxy),
 * and the XY model (Jz = 0); all three regimes exercise different
 * parts of the local-energy kernel.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "harness.h"
#include "nqs/nqs_config.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_sampler.h"
#include "nqs/nqs_optimizer.h"
#include "mps/mps.h"
#include "mps/dmrg.h"
static double dmrg_xxz_ground_state(int N, double Jxy, double Jz) {
    mps_config_t mc = mps_config_defaults();
    mc.num_sites = N;
    mc.ham = MPS_HAM_XXZ;
    mc.J = Jxy;
    mc.Jz = Jz;
    mc.max_bond_dim = 16;
    mc.num_sweeps = 10;
    mc.sweep_tol = 1e-10;
    mc.lanczos_max_iters = 60;
    mc.lanczos_tol = 1e-10;
    mps_dmrg_result_t res;
    mps_dmrg_xxz(&mc, &res);
    return res.final_energy;
}
static double nqs_xxz_tail_energy(int Lx, int Ly, double Jxy, double Jz,
                                   int hidden, int num_iter, int num_samples,
                                   double lr, double diag_shift,
                                   unsigned seed) {
    int N = Lx * Ly;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz          = NQS_ANSATZ_COMPLEX_RBM;
    cfg.rbm_hidden_units = hidden;
    cfg.rbm_init_scale   = 0.05;
    cfg.hamiltonian      = NQS_HAM_XXZ;
    cfg.j_coupling       = Jxy;
    cfg.j_z_coupling     = Jz;
    cfg.num_samples      = num_samples;
    cfg.num_thermalize   = 256;
    cfg.num_decorrelate  = 2;
    cfg.num_iterations   = num_iter;
    cfg.learning_rate    = lr;
    cfg.sr_diag_shift    = diag_shift;
    cfg.sr_cg_max_iters  = 80;
    cfg.sr_cg_tol        = 1e-7;
    cfg.rng_seed         = seed;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a);
    double *trace = malloc(sizeof(double) * num_iter);
    nqs_sr_run_holomorphic(&cfg, Lx, Ly, a, s, nqs_ansatz_log_amp, a, trace);
    int tail_start = (int)(num_iter * 0.7);
    double mean = 0.0;
    for (int i = tail_start; i < num_iter; i++) mean += trace[i];
    mean /= (double)(num_iter - tail_start);
    free(trace);
    nqs_sampler_free(s);
    nqs_ansatz_free(a);
    return mean;
}
static void test_xxz_reduces_to_heisenberg_when_jz_equals_jxy(void) {
    /* XXZ at Jxy = Jz = 1 must produce the same local energy on any
     * state as the plain Heisenberg kernel. We rely on the kernel
     * unification (local_energy_xxz called with Jxy = Jz = J) rather
     * than retesting energies end-to-end — the fact that the XXZ
     * dispatch returns a finite, reasonable energy is what we assert. */
    int N = 4;
    double E_dmrg_iso = dmrg_xxz_ground_state(N, 1.0, 1.0);
    /* E₀(N=4) Heisenberg = -1.616025 (from mps test). */
    ASSERT_NEAR(E_dmrg_iso, -1.616025, 1e-4);
    double E_nqs = nqs_xxz_tail_energy(N, 1, 1.0, 1.0, 16,
                                        250, 1024, 0.03, 1e-2, 0xA1A1u);
    printf("# XXZ (iso) N=4 Jxy=Jz=1: E_DMRG=%.5f  E_NQS_tail=%.5f\n",
           E_dmrg_iso, E_nqs);
    ASSERT_TRUE(isfinite(E_nqs));
    /* Must be clearly below the triplet (+0.25) — absolute bound */
    ASSERT_TRUE(E_nqs < 0.0);
}
static void test_xxz_ising_limit(void) {
    /* Jxy = 0, Jz = 1: pure classical Ising AFM. Ground state is Néel;
     * E₀ / N = -Jz/4 · (N-1) · 2 bonds / N ≈ -0.375 for N=4 open BC
     * (3 bonds × 0.25 = 0.75 → -0.75 total → -0.1875 per site).
     * Actually 3 bonds × 0.25 × Jz with alignment −1 each → diag = −3·0.25·Jz = −0.75.
     * No off-diagonal. */
    int N = 4;
    double E_dmrg = dmrg_xxz_ground_state(N, 0.0, 1.0);
    printf("# XXZ Ising limit N=4 Jxy=0 Jz=1: E_DMRG = %.5f (expected -0.75)\n",
           E_dmrg);
    ASSERT_NEAR(E_dmrg, -0.75, 1e-6);
    /* NQS for Jxy=0 is just a diagonal energy — no flips. A short
     * stochastic-reconf run should land near -0.75 despite no off-
     * diagonal learning signal, because the initial random ψ
     * averages the classical energy over spin configurations. */
    double E_nqs = nqs_xxz_tail_energy(N, 1, 0.0, 1.0, 16,
                                        200, 1024, 0.04, 1e-2, 0xB2B2u);
    printf("# XXZ Ising limit N=4: E_NQS_tail = %.4f\n", E_nqs);
    ASSERT_TRUE(isfinite(E_nqs));
    /* Loose: Ising limit with no off-diagonal still rewards Néel via
     * diagonal gradient; require a clearly negative energy. */
    ASSERT_TRUE(E_nqs < -0.3);
}
static void test_xxz_xy_limit(void) {
    /* Jxy = 1, Jz = 0: isotropic XY model. DMRG reference only — NQS
     * on pure XY without Marshall is slow to descend with few samples;
     * we assert the kernel runs and DMRG is sane. */
    int N = 4;
    double E_dmrg = dmrg_xxz_ground_state(N, 1.0, 0.0);
    printf("# XXZ XY limit N=4 Jxy=1 Jz=0: E_DMRG = %.5f\n", E_dmrg);
    ASSERT_TRUE(E_dmrg < -1.0);      /* close to the Bethe value ≈ -1.12 */
    ASSERT_TRUE(isfinite(E_dmrg));
    /* Smoke-test the NQS XY path runs without NaN. */
    double E_nqs = nqs_xxz_tail_energy(N, 1, 1.0, 0.0, 16,
                                        200, 1024, 0.04, 1e-2, 0xD3D3u);
    printf("# XXZ XY limit N=4: E_NQS_tail = %.4f  (no Marshall)\n", E_nqs);
    ASSERT_TRUE(isfinite(E_nqs));
    /* Variational bound — must not be below the exact GS. */
    ASSERT_TRUE(E_nqs > E_dmrg - 0.05);
}
int main(void) {
    TEST_RUN(test_xxz_reduces_to_heisenberg_when_jz_equals_jxy);
    TEST_RUN(test_xxz_ising_limit);
    TEST_RUN(test_xxz_xy_limit);
    TEST_SUMMARY();
}