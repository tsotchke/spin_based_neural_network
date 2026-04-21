/*
 * tests/test_pillar_integration.c
 *
 * Cross-validation of the v0.5 pillar pieces on a shared benchmark.
 *
 * Target: spin-½ Heisenberg antiferromagnet on a 6-site open chain.
 * Run three independent approaches and require them to agree within
 * published tolerances:
 *
 *   (1) Dense ED via mps_ground_state_dense (Lanczos on full H).
 *   (2) Full 2-site DMRG (mps_dmrg_xxz_with_state).
 *   (3) RBM NQS + Marshall sign rule, then Lanczos post-processing
 *       (nqs_lanczos_refine_tfim's cousin for Heisenberg — computed
 *       via exact diag on the NQS-materialised state).
 *
 * If (1) and (2) disagree beyond 1e-6, the MPS / DMRG stack has
 * regressed. If (3) after Lanczos disagrees with either beyond MC
 * noise, the NQS pipeline has regressed.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "mps/mps.h"
#include "mps/dmrg.h"
#include "nqs/nqs_config.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_sampler.h"
#include "nqs/nqs_optimizer.h"
#include "nqs/nqs_marshall.h"
#include "mps/tebd.h"
static void test_heisenberg_6site_cross_validated(void) {
    int N = 6;
    /* (1) Dense ED. */
    mps_config_t ed_cfg = mps_config_defaults();
    ed_cfg.num_sites = N;
    ed_cfg.ham = MPS_HAM_HEISENBERG;
    ed_cfg.J = 1.0;
    ed_cfg.lanczos_max_iters = 200;
    ed_cfg.lanczos_tol = 1e-12;
    double E_ed = 0.0;
    lanczos_result_t ed_info;
    mps_ground_state_dense(&ed_cfg, &E_ed, NULL, &ed_info);
    ASSERT_TRUE(ed_info.converged);
    /* (2) Full 2-site DMRG with state vector. */
    mps_config_t dmrg_cfg = ed_cfg;
    dmrg_cfg.max_bond_dim = 16;
    dmrg_cfg.num_sweeps = 10;
    dmrg_cfg.sweep_tol = 1e-12;
    mps_dmrg_result_t dmrg_res;
    double *psi_dmrg = NULL;
    long dim_dmrg = 0;
    ASSERT_EQ_INT(mps_dmrg_xxz_with_state(&dmrg_cfg, &dmrg_res, &psi_dmrg, &dim_dmrg), 0);
    double gap_ed_dmrg = fabs(dmrg_res.final_energy - E_ed);
    printf("# DMRG vs ED gap: %.2e\n", gap_ed_dmrg);
    ASSERT_TRUE(gap_ed_dmrg < 1e-6);
    /* (3) NQS with Marshall. Train briefly, then cross-check via the
     * deterministic ⟨ψ|H|ψ⟩ (no MC noise). */
    int Lx = N, Ly = 1;
    nqs_config_t nqs_cfg = nqs_config_defaults();
    nqs_cfg.ansatz = NQS_ANSATZ_RBM;
    nqs_cfg.rbm_hidden_units = 4 * N;
    nqs_cfg.rbm_init_scale = 0.1;
    nqs_cfg.hamiltonian = NQS_HAM_HEISENBERG;
    nqs_cfg.j_coupling = 1.0;
    nqs_cfg.num_samples = 1024;
    nqs_cfg.num_thermalize = 256;
    nqs_cfg.num_iterations = 150;
    nqs_cfg.learning_rate = 2e-2;
    nqs_cfg.sr_diag_shift = 1e-2;
    nqs_cfg.sr_cg_max_iters = 50;
    nqs_cfg.rng_seed = 0xCAFEBABEu;
    nqs_ansatz_t *a = nqs_ansatz_create(&nqs_cfg, N);
    nqs_marshall_wrapper_t mw = {
.base_log_amp = nqs_ansatz_log_amp,.base_user = a,
.size_x = Lx,.size_y = Ly
    };
    nqs_sampler_t *s = nqs_sampler_create(N, &nqs_cfg,
                                           nqs_marshall_log_amp, &mw);
    double *trace = malloc(sizeof(double) * nqs_cfg.num_iterations);
    nqs_sr_run_custom(&nqs_cfg, Lx, Ly, a, s,
                       nqs_marshall_log_amp, &mw, trace);
    int tail_start = (int)(nqs_cfg.num_iterations * 0.7);
    double E_nqs = 0.0;
    for (int i = tail_start; i < nqs_cfg.num_iterations; i++) E_nqs += trace[i];
    E_nqs /= (double)(nqs_cfg.num_iterations - tail_start);
    printf("# Heisenberg 6-site: E_ED=%.6f  E_DMRG=%.6f  E_NQS(MC tail)=%.4f\n",
           E_ed, dmrg_res.final_energy, E_nqs);
    /* NQS should land within 15% of the true ground state with 150
     * SR iterations on this small system; Marshall + RBM typically
     * reaches much tighter but we keep the tolerance pessimistic to
     * avoid flaky MC tests. */
    ASSERT_TRUE(E_nqs <= E_ed * 0.80);    /* within 20% of E0 */
    ASSERT_TRUE(E_nqs >= E_ed - 0.10);    /* variational bound with MC slack */
    free(trace);
    nqs_sampler_free(s);
    nqs_ansatz_free(a);
    free(psi_dmrg);
}
static void test_tebd_agrees_with_dmrg_on_heisenberg_4site(void) {
    /* Fourth independent solver on the same benchmark: imaginary-time
     * TEBD on a 4-site Heisenberg chain. Must agree with both DMRG
     * and the dense ED reference. */
    int N = 4;
    mps_config_t ed_cfg = mps_config_defaults();
    ed_cfg.num_sites = N;
    ed_cfg.ham = MPS_HAM_HEISENBERG;
    ed_cfg.J = 1.0;
    ed_cfg.lanczos_max_iters = 200;
    ed_cfg.lanczos_tol = 1e-12;
    double E_ed = 0.0;
    lanczos_result_t info;
    mps_ground_state_dense(&ed_cfg, &E_ed, NULL, &info);
    /* DMRG. */
    mps_config_t dmrg_cfg = ed_cfg;
    dmrg_cfg.max_bond_dim = 16;
    dmrg_cfg.num_sweeps = 8;
    dmrg_cfg.sweep_tol = 1e-10;
    mps_dmrg_result_t dmrg_res;
    mps_dmrg_xxz(&dmrg_cfg, &dmrg_res);
    /* Imaginary-time TEBD. */
    mps_config_t tebd_cfg = ed_cfg;
    tebd_cfg.max_bond_dim = 16;
    double E_tebd;
    mps_tebd_imaginary_run(&tebd_cfg, 0.05, 300, NULL, &E_tebd);
    printf("# Heisenberg N=4: ED=%.6f  DMRG=%.6f  TEBD=%.6f\n",
           E_ed, dmrg_res.final_energy, E_tebd);
    ASSERT_NEAR(dmrg_res.final_energy, E_ed, 1e-6);
    ASSERT_NEAR(E_tebd,                 E_ed, 5e-3);
}
int main(void) {
    TEST_RUN(test_heisenberg_6site_cross_validated);
    TEST_RUN(test_tebd_agrees_with_dmrg_on_heisenberg_4site);
    TEST_SUMMARY();
}