/*
 * tests/test_nqs_vit.c
 *
 * Factored-attention ViT NQS v0 — Rende et al. 2024 (arXiv:2310.05715),
 * single-head + patch_size=1 + real-amplitude slice.
 *
 * Tests:
 *   - lifecycle (create / num_params / free) on a small grid
 *   - forward returns a finite log|ψ| on a non-trivial config
 *   - analytic gradient matches a 5-point finite-difference reference
 *     for every parameter, on a randomly-initialised ansatz
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "nqs/nqs_config.h"
#include "nqs/nqs_ansatz.h"

static void test_vit_lifecycle(void) {
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_FACTORED_VIT;
    cfg.width  = 4;
    cfg.rng_seed = 0xCAFEFACEu;
    int N = 6;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    ASSERT_TRUE(a != NULL);
    /* P = 3·d + d² + N + 1  with d = 4, N = 6 → 12 + 16 + 6 + 1 = 35 */
    ASSERT_EQ_INT((int)nqs_ansatz_num_params(a), 35);
    nqs_ansatz_free(a);
}

static void test_vit_forward_finite(void) {
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_FACTORED_VIT;
    cfg.width  = 6;
    cfg.rng_seed = 0xBEEFu;
    int N = 8;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    ASSERT_TRUE(a != NULL);

    int spins[8] = { +1, -1, +1, -1, +1, +1, -1, -1 };
    double lp, arg;
    nqs_ansatz_log_amp(spins, N, a, &lp, &arg);
    ASSERT_TRUE(isfinite(lp));
    ASSERT_NEAR(arg, 0.0, 1e-12);    /* real-amplitude path */
    nqs_ansatz_free(a);
}

/* Finite-difference reference: 5-point stencil
 *   df/dθ ≈ ( -f(θ+2h) + 8 f(θ+h) - 8 f(θ-h) + f(θ-2h) ) / (12 h)
 * has truncation error O(h⁴), much better than central differences.
 */
static double fd_grad_at(nqs_ansatz_t *a, const int *spins, int num_sites,
                         long k, double h) {
    double *p = nqs_ansatz_params_raw(a);
    double saved = p[k];
    double f[4];
    double offsets[4] = { +2*h, +h, -h, -2*h };
    for (int s = 0; s < 4; s++) {
        p[k] = saved + offsets[s];
        double lp, arg;
        nqs_ansatz_log_amp(spins, num_sites, a, &lp, &arg);
        f[s] = lp;
    }
    p[k] = saved;
    return (-f[0] + 8.0*f[1] - 8.0*f[2] + f[3]) / (12.0 * h);
}

static void test_vit_gradient_matches_finite_difference(void) {
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_FACTORED_VIT;
    cfg.width  = 4;
    cfg.rng_seed = 0xC0FFEEu;
    int N = 6;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    ASSERT_TRUE(a != NULL);

    int spins[6] = { +1, -1, +1, +1, -1, -1 };
    long P = nqs_ansatz_num_params(a);
    double *grad_an = malloc((size_t)P * sizeof(double));
    ASSERT_EQ_INT(nqs_ansatz_logpsi_gradient(a, spins, N, grad_an), 0);

    double max_err = 0.0;
    long max_err_k = -1;
    for (long k = 0; k < P; k++) {
        double g_fd = fd_grad_at(a, spins, N, k, 1e-4);
        double err  = fabs(g_fd - grad_an[k]);
        if (err > max_err) { max_err = err; max_err_k = k; }
    }
    printf("# ViT v0: %ld params, max |grad_an − grad_fd| = %.3e at k=%ld\n",
           P, max_err, max_err_k);
    /* 5-point stencil at h=1e-4 with O(1) parameter scale gives
     * truncation ~h⁴ = 1e-16; round-off floor adds a few × 1e-10. */
    ASSERT_TRUE(max_err < 1e-7);

    free(grad_an);
    nqs_ansatz_free(a);
}

int main(void) {
    TEST_RUN(test_vit_lifecycle);
    TEST_RUN(test_vit_forward_finite);
    TEST_RUN(test_vit_gradient_matches_finite_difference);
    TEST_SUMMARY();
}
