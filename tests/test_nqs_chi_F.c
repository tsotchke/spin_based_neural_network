/*
 * tests/test_nqs_chi_F.c
 *
 * Sample-based Tr(S) / fidelity-susceptibility diagnostic.
 *
 *     S_{k,l} = ⟨O_k* O_l⟩_{|ψ|²} − ⟨O_k*⟩⟨O_l⟩
 *     Tr(S)   = Σ_k ( ⟨|O_k|²⟩ − |⟨O_k⟩|² )
 *
 * Verifies:
 *   (1) Tr(S) is finite and non-negative on a freshly-initialised
 *       complex RBM on TFIM 2x2.
 *   (2) The optional out_per_param equals Tr(S) / num_params.
 *   (3) Real-amplitude ansatz (legacy MLP) also gives finite,
 *       non-negative Tr(S).
 *   (4) NULL out_per_param is tolerated.
 *   (5) A larger batch gives a trace that is close to a smaller
 *       batch's trace (Monte-Carlo consistency, not exact equality).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "harness.h"
#include "nqs/nqs_config.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_sampler.h"
#include "nqs/nqs_diagnostics.h"

static void test_chi_F_complex_rbm_tfim(void) {
    int Lx = 2, Ly = 2, N = Lx * Ly;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz           = NQS_ANSATZ_COMPLEX_RBM;
    cfg.rbm_hidden_units = 8;
    cfg.rbm_init_scale   = 0.05;
    cfg.hamiltonian      = NQS_HAM_TFIM;
    cfg.j_coupling       = 1.0;
    cfg.transverse_field = 1.0;
    cfg.num_samples      = 1024;
    cfg.num_thermalize   = 256;
    cfg.num_decorrelate  = 2;
    cfg.rng_seed         = 0xC4F0u;
    nqs_ansatz_t  *a = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a);
    ASSERT_TRUE(a && s);

    double trace_S = 0.0, per_param = 0.0;
    int rc = nqs_compute_chi_F(&cfg, Lx, Ly, a, s, &trace_S, &per_param);
    ASSERT_EQ_INT(rc, 0);
    ASSERT_TRUE(isfinite(trace_S));
    ASSERT_TRUE(trace_S >= 0.0);
    ASSERT_TRUE(isfinite(per_param));
    ASSERT_TRUE(per_param >= 0.0);

    long nparams = nqs_ansatz_num_params(a);
    ASSERT_TRUE(nparams > 0);
    ASSERT_NEAR(per_param, trace_S / (double)nparams, 1e-10);

    printf("# TFIM 2x2 complex-RBM: Tr(S)=%.4f  per_param=%.6g  nparams=%ld\n",
           trace_S, per_param, nparams);

    nqs_sampler_free(s);
    nqs_ansatz_free(a);
}

static void test_chi_F_legacy_mlp_real_path(void) {
    int Lx = 2, Ly = 2, N = Lx * Ly;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz           = NQS_ANSATZ_LEGACY_MLP;
    cfg.hamiltonian      = NQS_HAM_TFIM;
    cfg.j_coupling       = 1.0;
    cfg.transverse_field = 1.0;
    cfg.num_samples      = 512;
    cfg.num_thermalize   = 256;
    cfg.num_decorrelate  = 2;
    cfg.rng_seed         = 0xC4F1u;
    nqs_ansatz_t  *a = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a);
    ASSERT_TRUE(a && s);

    double trace_S = 0.0;
    int rc = nqs_compute_chi_F(&cfg, Lx, Ly, a, s, &trace_S, NULL);
    ASSERT_EQ_INT(rc, 0);
    ASSERT_TRUE(isfinite(trace_S));
    ASSERT_TRUE(trace_S >= 0.0);

    printf("# TFIM 2x2 legacy-MLP: Tr(S)=%.4f  (NULL per_param ok)\n", trace_S);

    nqs_sampler_free(s);
    nqs_ansatz_free(a);
}

static void test_chi_F_bad_args(void) {
    /* Basic contract checks. Must not segfault and must return error. */
    double t = 0, p = 0;
    ASSERT_TRUE(nqs_compute_chi_F(NULL, 2, 2, NULL, NULL, &t, &p) != 0);
    ASSERT_TRUE(nqs_compute_chi_F(NULL, 2, 2, NULL, NULL, NULL, NULL) != 0);
}

static void test_chi_F_batch_monte_carlo_consistency(void) {
    /* Two independent samples of Tr(S) at different batch sizes on the
     * same ansatz should agree to within MC slack. Checks the helper
     * is averaging sensibly (not e.g. summing without normalisation). */
    int Lx = 2, Ly = 2, N = Lx * Ly;
    nqs_config_t cfg_small = nqs_config_defaults();
    cfg_small.ansatz           = NQS_ANSATZ_COMPLEX_RBM;
    cfg_small.rbm_hidden_units = 6;
    cfg_small.rbm_init_scale   = 0.05;
    cfg_small.hamiltonian      = NQS_HAM_TFIM;
    cfg_small.j_coupling       = 1.0;
    cfg_small.transverse_field = 1.0;
    cfg_small.num_samples      = 512;
    cfg_small.num_thermalize   = 256;
    cfg_small.num_decorrelate  = 2;
    cfg_small.rng_seed         = 0xC4F2u;

    nqs_config_t cfg_big = cfg_small;
    cfg_big.num_samples = 4096;
    cfg_big.rng_seed    = 0xC4F3u;

    nqs_ansatz_t  *a_small = nqs_ansatz_create(&cfg_small, N);
    nqs_sampler_t *s_small = nqs_sampler_create(N, &cfg_small,
                                                 nqs_ansatz_log_amp, a_small);
    nqs_ansatz_t  *a_big   = nqs_ansatz_create(&cfg_small, N);  /* same init */
    nqs_sampler_t *s_big   = nqs_sampler_create(N, &cfg_big,
                                                 nqs_ansatz_log_amp, a_big);
    ASSERT_TRUE(a_small && s_small && a_big && s_big);

    double t_small = 0, t_big = 0;
    int rc_s = nqs_compute_chi_F(&cfg_small, Lx, Ly, a_small, s_small,
                                  &t_small, NULL);
    int rc_b = nqs_compute_chi_F(&cfg_big,   Lx, Ly, a_big,   s_big,
                                  &t_big,   NULL);
    ASSERT_EQ_INT(rc_s, 0);
    ASSERT_EQ_INT(rc_b, 0);

    /* Both finite, both non-negative, both non-trivial (not a pathological
     * zero). */
    ASSERT_TRUE(isfinite(t_small));
    ASSERT_TRUE(isfinite(t_big));
    ASSERT_TRUE(t_small > 0.0);
    ASSERT_TRUE(t_big   > 0.0);
    /* Ratio is O(1): the big-batch Tr(S) is a better estimator of the same
     * underlying quantity, so the two traces must land in the same ball
     * park. 3x slack accommodates MC noise at N=4. */
    double ratio = t_big / t_small;
    printf("# MC consistency: small(512)=%.4f  big(4096)=%.4f  ratio=%.3f\n",
           t_small, t_big, ratio);
    ASSERT_TRUE(ratio > 0.33 && ratio < 3.0);

    nqs_sampler_free(s_small); nqs_ansatz_free(a_small);
    nqs_sampler_free(s_big);   nqs_ansatz_free(a_big);
}

/* ---- Bipartite phase probe on kagome ------------------------------------ */

static void test_kagome_bond_phase_basic(void) {
    /* 2×2 PBC kagome (N=12). Freshly-initialised complex RBM. The probe
     * must return finite per-class Re/Im, with non-zero counts (random
     * samples hit opposite-spin bonds of each class), and the complex
     * magnitude of each class's mean ratio must be positive and finite. */
    int Lx = 2, Ly = 2, N = 3 * Lx * Ly;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz           = NQS_ANSATZ_COMPLEX_RBM;
    cfg.rbm_hidden_units = 12;
    cfg.rbm_init_scale   = 0.05;
    cfg.hamiltonian      = NQS_HAM_KAGOME_HEISENBERG;
    cfg.j_coupling       = 1.0;
    cfg.kagome_pbc       = 1;
    cfg.num_samples      = 512;
    cfg.num_thermalize   = 256;
    cfg.num_decorrelate  = 2;
    cfg.rng_seed         = 0xC0BBu;
    nqs_ansatz_t  *a = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a);
    ASSERT_TRUE(a && s);

    double re[NQS_KAGOME_NUM_BOND_CLASSES] = {0};
    double im[NQS_KAGOME_NUM_BOND_CLASSES] = {0};
    long   cnt[NQS_KAGOME_NUM_BOND_CLASSES] = {0};
    int rc = nqs_compute_kagome_bond_phase(&cfg, Lx, Ly, a, s, re, im, cnt);
    ASSERT_EQ_INT(rc, 0);

    for (int c = 0; c < NQS_KAGOME_NUM_BOND_CLASSES; c++) {
        ASSERT_TRUE(isfinite(re[c]));
        ASSERT_TRUE(isfinite(im[c]));
        ASSERT_TRUE(cnt[c] > 0);
        printf("# kagome 2x2 PBC class %d (%s): ⟨r⟩ = %+.4f %+.4fi  |r|=%.4f  n=%ld\n",
               c, (const char*[]){"A-B","A-C","B-C"}[c],
               re[c], im[c], sqrt(re[c]*re[c] + im[c]*im[c]), cnt[c]);
    }
    nqs_sampler_free(s);
    nqs_ansatz_free(a);
}

static void test_kagome_bond_phase_rejects_non_kagome(void) {
    /* Helper must refuse non-kagome Hamiltonians — its bond enumeration
     * is kagome-specific and applying it to a TFIM/Heisenberg square
     * lattice would return meaningless numbers. */
    int Lx = 2, Ly = 2, N = Lx * Ly;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz           = NQS_ANSATZ_COMPLEX_RBM;
    cfg.rbm_hidden_units = 4;
    cfg.hamiltonian      = NQS_HAM_TFIM;   /* wrong kernel */
    cfg.num_samples      = 64;
    cfg.num_thermalize   = 64;
    cfg.num_decorrelate  = 1;
    cfg.rng_seed         = 0xC0BCu;
    nqs_ansatz_t  *a = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a);
    ASSERT_TRUE(a && s);

    double re[3] = {0}, im[3] = {0};
    int rc = nqs_compute_kagome_bond_phase(&cfg, Lx, Ly, a, s, re, im, NULL);
    ASSERT_TRUE(rc != 0);

    nqs_sampler_free(s); nqs_ansatz_free(a);
}

int main(void) {
    TEST_RUN(test_chi_F_complex_rbm_tfim);
    TEST_RUN(test_chi_F_legacy_mlp_real_path);
    TEST_RUN(test_chi_F_bad_args);
    TEST_RUN(test_chi_F_batch_monte_carlo_consistency);
    TEST_RUN(test_kagome_bond_phase_basic);
    TEST_RUN(test_kagome_bond_phase_rejects_non_kagome);
    TEST_SUMMARY();
}
