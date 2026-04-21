/*
 * src/thermodynamic/rbm_cd.c
 *
 * Restricted Boltzmann machine trained via contrastive divergence
 * (Hinton 2002). Pure binary {0,1} visible/hidden spins. Block-Gibbs
 * sampling; mean-field hidden activations inside the CD gradient
 * (standard variance-reduction trick).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "thermodynamic/rbm_cd.h"

static double xs_uniform(unsigned long long *st) {
    unsigned long long x = *st ? *st : 0x9E3779B97F4A7C15ULL;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *st = x;
    return (double)(x >> 11) / 9007199254740992.0;
}

static double sigmoid(double x) {
    if (x >= 0) {
        double e = exp(-x);
        return 1.0 / (1.0 + e);
    } else {
        double e = exp(x);
        return e / (1.0 + e);
    }
}

static double softplus(double x) {
    if (x > 30) return x;
    if (x < -30) return exp(x);
    return log1p(exp(x));
}

rbm_cd_t *rbm_cd_create(int num_visible, int num_hidden,
                         double weight_scale,
                         unsigned long long seed) {
    if (num_visible <= 0 || num_hidden <= 0) return NULL;
    rbm_cd_t *r = calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->num_visible = num_visible;
    r->num_hidden  = num_hidden;
    size_t W_elems = (size_t)num_visible * (size_t)num_hidden;
    r->W = calloc(W_elems, sizeof(double));
    r->a = calloc((size_t)num_visible, sizeof(double));
    r->b = calloc((size_t)num_hidden,  sizeof(double));
    if (!r->W || !r->a || !r->b) { rbm_cd_free(r); return NULL; }
    r->rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < W_elems; i++) {
        /* Gaussian via Box-Muller over two uniforms. */
        double u1 = xs_uniform(&r->rng);
        double u2 = xs_uniform(&r->rng);
        if (u1 < 1e-12) u1 = 1e-12;
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
        r->W[i] = weight_scale * z;
    }
    return r;
}

void rbm_cd_free(rbm_cd_t *rbm) {
    if (!rbm) return;
    free(rbm->W); free(rbm->a); free(rbm->b);
    free(rbm);
}

void rbm_cd_mean_h_given_v(const rbm_cd_t *rbm,
                            const int *v, double *mean_h) {
    int N = rbm->num_visible, M = rbm->num_hidden;
    for (int j = 0; j < M; j++) {
        double z = rbm->b[j];
        for (int i = 0; i < N; i++) z += (double)v[i] * rbm->W[i * M + j];
        mean_h[j] = sigmoid(z);
    }
}

void rbm_cd_mean_v_given_h(const rbm_cd_t *rbm,
                            const int *h, double *mean_v) {
    int N = rbm->num_visible, M = rbm->num_hidden;
    for (int i = 0; i < N; i++) {
        double z = rbm->a[i];
        for (int j = 0; j < M; j++) z += (double)h[j] * rbm->W[i * M + j];
        mean_v[i] = sigmoid(z);
    }
}

void rbm_cd_sample_h_given_v(rbm_cd_t *rbm, const int *v, int *h) {
    int M = rbm->num_hidden;
    double *mh = malloc((size_t)M * sizeof(double));
    rbm_cd_mean_h_given_v(rbm, v, mh);
    for (int j = 0; j < M; j++) h[j] = (xs_uniform(&rbm->rng) < mh[j]) ? 1 : 0;
    free(mh);
}

void rbm_cd_sample_v_given_h(rbm_cd_t *rbm, const int *h, int *v) {
    int N = rbm->num_visible;
    double *mv = malloc((size_t)N * sizeof(double));
    rbm_cd_mean_v_given_h(rbm, h, mv);
    for (int i = 0; i < N; i++) v[i] = (xs_uniform(&rbm->rng) < mv[i]) ? 1 : 0;
    free(mv);
}

int rbm_cd_train_step(rbm_cd_t *rbm,
                       const int *v_data,
                       int k_gibbs,
                       double learning_rate) {
    if (!rbm || !v_data || k_gibbs <= 0 || learning_rate <= 0) return -1;
    int N = rbm->num_visible, M = rbm->num_hidden;

    double *mh_data  = malloc((size_t)M * sizeof(double));
    double *mh_model = malloc((size_t)M * sizeof(double));
    int    *v_model  = malloc((size_t)N * sizeof(int));
    int    *h_sample = malloc((size_t)M * sizeof(int));
    if (!mh_data || !mh_model || !v_model || !h_sample) {
        free(mh_data); free(mh_model); free(v_model); free(h_sample);
        return -1;
    }

    /* Positive phase: mean-field P(h|v_data). */
    rbm_cd_mean_h_given_v(rbm, v_data, mh_data);

    /* Negative phase: k Gibbs steps starting from h ~ P(h|v_data). */
    for (int j = 0; j < M; j++) h_sample[j] = (xs_uniform(&rbm->rng) < mh_data[j]) ? 1 : 0;
    memcpy(v_model, v_data, (size_t)N * sizeof(int));
    for (int step = 0; step < k_gibbs; step++) {
        rbm_cd_sample_v_given_h(rbm, h_sample, v_model);
        rbm_cd_sample_h_given_v(rbm, v_model, h_sample);
    }
    /* Mean-field for the negative statistic (variance reduction). */
    rbm_cd_mean_h_given_v(rbm, v_model, mh_model);

    /* Updates. */
    for (int i = 0; i < N; i++) {
        double dvi = (double)v_data[i] - (double)v_model[i];
        for (int j = 0; j < M; j++) {
            double vdata_times_hdata = (double)v_data[i]  * mh_data[j];
            double vmodel_times_hmodel = (double)v_model[i] * mh_model[j];
            rbm->W[i * M + j] += learning_rate * (vdata_times_hdata - vmodel_times_hmodel);
        }
        rbm->a[i] += learning_rate * dvi;
    }
    for (int j = 0; j < M; j++) rbm->b[j] += learning_rate * (mh_data[j] - mh_model[j]);

    free(mh_data); free(mh_model); free(v_model); free(h_sample);
    return 0;
}

int rbm_cd_train_batch(rbm_cd_t *rbm,
                        const int *v_data_batch, int num_patterns,
                        int k_gibbs, double learning_rate) {
    if (!rbm || !v_data_batch || num_patterns <= 0) return -1;
    int N = rbm->num_visible;
    for (int p = 0; p < num_patterns; p++) {
        int rc = rbm_cd_train_step(rbm, &v_data_batch[(size_t)p * N],
                                    k_gibbs, learning_rate);
        if (rc != 0) return rc;
    }
    return 0;
}

int rbm_cd_sample(rbm_cd_t *rbm,
                   int *out_samples, int num_samples,
                   int burn_in, int thin) {
    if (!rbm || !out_samples || num_samples <= 0) return -1;
    int N = rbm->num_visible, M = rbm->num_hidden;
    int *v = malloc((size_t)N * sizeof(int));
    int *h = malloc((size_t)M * sizeof(int));
    if (!v || !h) { free(v); free(h); return -1; }
    for (int i = 0; i < N; i++) v[i] = (xs_uniform(&rbm->rng) < 0.5) ? 1 : 0;
    for (int s = 0; s < burn_in; s++) {
        rbm_cd_sample_h_given_v(rbm, v, h);
        rbm_cd_sample_v_given_h(rbm, h, v);
    }
    for (int n = 0; n < num_samples; n++) {
        for (int s = 0; s < thin; s++) {
            rbm_cd_sample_h_given_v(rbm, v, h);
            rbm_cd_sample_v_given_h(rbm, h, v);
        }
        memcpy(&out_samples[(size_t)n * N], v, (size_t)N * sizeof(int));
    }
    free(v); free(h);
    return 0;
}

double rbm_cd_free_energy(const rbm_cd_t *rbm, const int *v) {
    int N = rbm->num_visible, M = rbm->num_hidden;
    double F = 0.0;
    for (int i = 0; i < N; i++) F -= rbm->a[i] * (double)v[i];
    for (int j = 0; j < M; j++) {
        double z = rbm->b[j];
        for (int i = 0; i < N; i++) z += (double)v[i] * rbm->W[i * M + j];
        F -= softplus(z);
    }
    return F;
}
