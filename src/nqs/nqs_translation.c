/*
 * src/nqs/nqs_translation.c
 *
 * k = 0 translation-projection wrapper. Averages the base ansatz
 * over all |G| = L_x · L_y cyclic shifts. Log-sum-exp keeps the sum
 * numerically stable; the cos(arg_τ) factor handles phase wrappers
 * like Marshall that emit arg ∈ {0, π}.
 *
 * Cost per log_amp evaluation: |G| base-ansatz calls. For a 6x6
 * lattice that is 36× the bare cost; for 1D chains it scales with
 * chain length. Future: cache shifted-spin buffers per sampler step.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "nqs/nqs_translation.h"
#include "nqs/nqs_ansatz.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void nqs_translation_log_amp(const int *spins, int num_sites,
                              void *user,
                              double *out_log_abs,
                              double *out_arg) {
    nqs_translation_wrapper_t *w = (nqs_translation_wrapper_t *)user;
    int Lx = w->size_x;
    int Ly = w->size_y;
    int N = Lx * Ly;
    if (N != num_sites) { if (out_log_abs) *out_log_abs = 0; if (out_arg) *out_arg = 0; return; }

    int *shifted = malloc((size_t)N * sizeof(int));
    long G = (long)Lx * Ly;
    double *lp_arr  = malloc((size_t)G * sizeof(double));
    double *cos_arr = malloc((size_t)G * sizeof(double));

    double max_lp = -1e300;
    long idx = 0;
    for (int tx = 0; tx < Lx; tx++) {
        for (int ty = 0; ty < Ly; ty++) {
            /* shift spins by (tx, ty) on the L_x × L_y lattice. */
            for (int x = 0; x < Lx; x++) {
                int xs = (x + tx) % Lx;
                for (int y = 0; y < Ly; y++) {
                    int ys = (y + ty) % Ly;
                    shifted[x * Ly + y] = spins[xs * Ly + ys];
                }
            }
            double lp, arg;
            w->base_log_amp(shifted, N, w->base_user, &lp, &arg);
            lp_arr[idx]  = lp;
            cos_arr[idx] = cos(arg);   /* ±1 for real wrappers */
            if (lp > max_lp) max_lp = lp;
            idx++;
        }
    }

    double sum = 0.0;
    for (long i = 0; i < G; i++) {
        sum += cos_arr[i] * exp(lp_arr[i] - max_lp);
    }
    double abs_sum = fabs(sum);
    if (abs_sum > 0.0) {
        if (out_log_abs) *out_log_abs = log(abs_sum) + max_lp - 0.5 * log((double)G);
        if (out_arg)     *out_arg     = (sum < 0.0) ? M_PI : 0.0;
    } else {
        if (out_log_abs) *out_log_abs = -1e300;
        if (out_arg)     *out_arg     = 0.0;
    }
    free(shifted); free(lp_arr); free(cos_arr);
}

int nqs_translation_gradient(void *grad_user,
                              nqs_ansatz_t *ansatz,
                              const int *spins, int num_sites,
                              double *out_grad) {
    if (!grad_user || !ansatz || !spins || !out_grad) return -1;
    nqs_translation_wrapper_t *w = (nqs_translation_wrapper_t *)grad_user;
    int Lx = w->size_x;
    int Ly = w->size_y;
    int N = Lx * Ly;
    if (N != num_sites) return -1;
    long P = nqs_ansatz_num_params(ansatz);
    long G = (long)Lx * Ly;

    /* Step 1: evaluate base log_amp at every shift and collect
     * weighted coefficients w_τ = χ(τ) · ψ_base(T^τ s) / (sum). Use
     * log-sum-exp for numerical stability. */
    int *shifted = malloc((size_t)N * sizeof(int));
    double *lp_arr  = malloc((size_t)G * sizeof(double));
    double *cos_arr = malloc((size_t)G * sizeof(double));
    int   **shifted_store = malloc((size_t)G * sizeof(int *));
    double max_lp = -1e300;
    long idx = 0;
    for (int tx = 0; tx < Lx; tx++) {
        for (int ty = 0; ty < Ly; ty++) {
            shifted_store[idx] = malloc((size_t)N * sizeof(int));
            for (int x = 0; x < Lx; x++) {
                int xs = (x + tx) % Lx;
                for (int y = 0; y < Ly; y++) {
                    int ys = (y + ty) % Ly;
                    shifted_store[idx][x * Ly + y] = spins[xs * Ly + ys];
                }
            }
            double lp, arg;
            w->base_log_amp(shifted_store[idx], N, w->base_user, &lp, &arg);
            lp_arr[idx]  = lp;
            cos_arr[idx] = cos(arg);
            if (lp > max_lp) max_lp = lp;
            idx++;
        }
    }
    /* Weighted denominator (signed). */
    double denom = 0.0;
    double *contrib = malloc((size_t)G * sizeof(double));
    for (long i = 0; i < G; i++) {
        contrib[i] = cos_arr[i] * exp(lp_arr[i] - max_lp);
        denom += contrib[i];
    }
    if (denom == 0.0) {
        /* Degenerate; return zero gradient. */
        memset(out_grad, 0, (size_t)P * sizeof(double));
        for (long i = 0; i < G; i++) free(shifted_store[i]);
        free(shifted_store); free(shifted); free(lp_arr); free(cos_arr); free(contrib);
        return 0;
    }

    /* Step 2: weighted sum of base gradients at each shifted config.
     * Since Marshall is a sign-only transformation, ∂ log|ψ_marshall|/∂θ
     * = ∂ log|ψ_base|/∂θ; we can always call nqs_ansatz_logpsi_gradient
     * on the shifted spins. */
    double *tmp_grad = malloc((size_t)P * sizeof(double));
    memset(out_grad, 0, (size_t)P * sizeof(double));
    for (long i = 0; i < G; i++) {
        double w_i = contrib[i] / denom;
        if (w_i == 0.0) continue;
        nqs_ansatz_logpsi_gradient(ansatz, shifted_store[i], N, tmp_grad);
        for (long k = 0; k < P; k++) out_grad[k] += w_i * tmp_grad[k];
    }
    free(tmp_grad);
    for (long i = 0; i < G; i++) free(shifted_store[i]);
    free(shifted_store); free(shifted); free(lp_arr); free(cos_arr); free(contrib);
    return 0;
}
