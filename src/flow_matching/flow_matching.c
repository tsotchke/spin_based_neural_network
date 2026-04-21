/*
 * src/flow_matching/flow_matching.c
 *
 * Reference two-state CTMC sampler with constant flip rate per unit
 * time. Two flavours:
 *
 *   - conditional: absorbing-state rate
 *         wrong → right at rate c; right → wrong at rate 0.
 *     Survival probability of a wrong spin at t=1 is exp(-c).
 *
 *   - unconditional: detailed-balanced reversible rates
 *         - → + at rate c · (1+b)/2
 *         + → - at rate c · (1-b)/2
 *     Stationary P(+) = (1+b)/2; relaxation time 1/c.
 *
 * The constant c is chosen per-config via `epsilon`, which for this
 * sampler is reinterpreted as the total integrated rate ∫₀¹ λ(t) dt.
 * The default c = 5 gives ≈ 0.7% residual after full relaxation. A
 * learned rate network (v0.5+) replaces the constant with a predicted
 * per-site, per-time schedule behind the same integrator.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "flow_matching/flow_matching.h"

static double xorshift_uniform(unsigned long long *state) {
    unsigned long long x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return (double)(x >> 11) / 9007199254740992.0;   /* [0, 1) */
}

static int sign_to_pm1(int b) { return b > 0 ? +1 : -1; }

static double integrated_rate(const flow_matching_config_t *cfg) {
    return (cfg->integrated_rate > 0.0) ? cfg->integrated_rate : 2.0;
}

int flow_matching_sample_conditional(const flow_matching_config_t *cfg,
                                     const int *target, int *out) {
    if (!cfg || !target || !out) return -1;
    if (cfg->num_sites <= 0 || cfg->num_steps <= 0) return -1;
    int N = cfg->num_sites;
    int S = cfg->num_steps;
    unsigned long long rng = cfg->seed ? cfg->seed : 1ULL;
    for (int i = 0; i < N; i++) {
        double u = xorshift_uniform(&rng);
        out[i] = (u < 0.5) ? +1 : -1;
    }
    double c = integrated_rate(cfg);
    double dt = 1.0 / (double)S;
    double p_flip = 1.0 - exp(-c * dt);
    for (int s = 0; s < S; s++) {
        for (int i = 0; i < N; i++) {
            int tgt = sign_to_pm1(target[i]);
            if (out[i] != tgt) {
                double u = xorshift_uniform(&rng);
                if (u < p_flip) out[i] = tgt;
            }
        }
    }
    return 0;
}

int flow_matching_fit_rates_to_magnetisation(const flow_matching_config_t *cfg,
                                              const double *bias,
                                              const double *m_target,
                                              double *out_rates) {
    if (!cfg || !bias || !m_target || !out_rates) return -1;
    for (int i = 0; i < cfg->num_sites; i++) {
        double b = bias[i];
        if (b < -1.0) b = -1.0;
        if (b >  1.0) b =  1.0;
        double m = m_target[i];
        if (fabs(b) < 1e-12) {
            out_rates[i] = 0.0;   /* symmetric stationary, no target possible */
            continue;
        }
        double ratio = m / b;
        /* Require m to have same sign as bias and |m| < |b|. */
        if (ratio <= 0.0) { out_rates[i] = 0.0; continue; }
        if (ratio >= 1.0) ratio = 0.9999;
        out_rates[i] = -log(1.0 - ratio);
    }
    return 0;
}

int flow_matching_sample_biased_rates(const flow_matching_config_t *cfg,
                                       const double *bias,
                                       const double *per_site_rates,
                                       int *out) {
    if (!cfg || !bias || !per_site_rates || !out) return -1;
    if (cfg->num_sites <= 0 || cfg->num_steps <= 0) return -1;
    int N = cfg->num_sites;
    int S = cfg->num_steps;
    unsigned long long rng = cfg->seed ? cfg->seed : 1ULL;
    for (int i = 0; i < N; i++) {
        double u = xorshift_uniform(&rng);
        out[i] = (u < 0.5) ? +1 : -1;
    }
    double dt = 1.0 / (double)S;
    for (int s = 0; s < S; s++) {
        for (int i = 0; i < N; i++) {
            double c = per_site_rates[i];
            if (c <= 0.0) continue;
            double b = bias[i];
            if (b < -1.0) b = -1.0;
            if (b >  1.0) b =  1.0;
            double p_plus  = 0.5 * (1.0 + b);
            double p_minus = 1.0 - p_plus;
            double rate = (out[i] == +1) ? (c * p_minus) : (c * p_plus);
            double p_flip = 1.0 - exp(-rate * dt);
            double u = xorshift_uniform(&rng);
            if (u < p_flip) out[i] = -out[i];
        }
    }
    return 0;
}

int flow_matching_sample_unconditional(const flow_matching_config_t *cfg,
                                       const double *bias, int *out) {
    if (!cfg || !bias || !out) return -1;
    if (cfg->num_sites <= 0 || cfg->num_steps <= 0) return -1;
    int N = cfg->num_sites;
    int S = cfg->num_steps;
    unsigned long long rng = cfg->seed ? cfg->seed : 1ULL;
    for (int i = 0; i < N; i++) {
        double u = xorshift_uniform(&rng);
        out[i] = (u < 0.5) ? +1 : -1;
    }
    double c = integrated_rate(cfg);
    double dt = 1.0 / (double)S;
    for (int s = 0; s < S; s++) {
        for (int i = 0; i < N; i++) {
            double b = bias[i];
            if (b < -1.0) b = -1.0;
            if (b >  1.0) b =  1.0;
            double p_plus  = 0.5 * (1.0 + b);
            double p_minus = 1.0 - p_plus;
            double rate = (out[i] == +1) ? (c * p_minus) : (c * p_plus);
            double p_flip = 1.0 - exp(-rate * dt);
            double u = xorshift_uniform(&rng);
            if (u < p_flip) out[i] = -out[i];
        }
    }
    return 0;
}
