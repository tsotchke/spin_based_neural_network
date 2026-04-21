/*
 * tests/test_flow_matching.c
 *
 * Verifies the two-state CTMC actually integrates. The analytic
 * predictions below are derived for the reference constant-rate
 * schedule (integrated rate c = 2 by default, see flow_matching.c).
 *
 *   Conditional (irreversible target-absorbing):
 *     P(match at t=1) = 0.5 + 0.5 · (1 - exp(-c))
 *   Unconditional (reversible detailed-balanced to Bernoulli(bias)):
 *     m(1) = b · (1 - exp(-c)) + m(0) · exp(-c)     with m(0) = 0.
 *
 * A cheating implementation that simply sets the output to target or
 * Bernoulli(bias) at the end would give match = 1.0 and m = b, which
 * these tests reject at the stated tolerances.
 */
#include <stdlib.h>
#include <math.h>
#include "harness.h"
#include "flow_matching/flow_matching.h"
static double conditional_match_fraction(int N, int S, long trials,
                                         unsigned long long seed_base,
                                         const int *target, double c) {
    flow_matching_config_t cfg = flow_matching_config_defaults();
    cfg.num_sites       = N;
    cfg.num_steps       = S;
    cfg.integrated_rate = c;
    int *sample = malloc((size_t)N * sizeof(int));
    long matches = 0, total = 0;
    for (long t = 0; t < trials; t++) {
        cfg.seed = 0x9E3779B97F4A7C15ULL * (seed_base + (unsigned long long)t);
        flow_matching_sample_conditional(&cfg, target, sample);
        for (int i = 0; i < N; i++) {
            if (sample[i] == target[i]) matches++;
            total++;
        }
    }
    free(sample);
    return (double)matches / (double)total;
}
static double unconditional_mean(int N, int S, long trials,
                                 unsigned long long seed_base,
                                 const double *bias, double c) {
    flow_matching_config_t cfg = flow_matching_config_defaults();
    cfg.num_sites       = N;
    cfg.num_steps       = S;
    cfg.integrated_rate = c;
    int *sample = malloc((size_t)N * sizeof(int));
    double total = 0.0;
    for (long t = 0; t < trials; t++) {
        cfg.seed = 0x9E3779B97F4A7C15ULL * (seed_base + (unsigned long long)t);
        flow_matching_sample_unconditional(&cfg, bias, sample);
        double s = 0.0;
        for (int i = 0; i < N; i++) s += sample[i];
        total += s / (double)N;
    }
    free(sample);
    return total / (double)trials;
}
static void test_conditional_match_matches_analytic(void) {
    /* c = 2 → analytic match = 0.5 + 0.5·(1 - e^{-2}) = 0.9323.
     * A projection stub would produce exactly 1.0. */
    int N = 8;
    int target[8] = {+1, -1, +1, +1, -1, -1, +1, -1};
    double f = conditional_match_fraction(N, 128, 1500, 1, target, 2.0);
    double expected = 0.5 + 0.5 * (1.0 - exp(-2.0));
    ASSERT_NEAR(f, expected, 0.015);
    ASSERT_TRUE(f < 0.95);   /* below the projection stub's 1.0 */
}
static void test_conditional_is_independent_of_step_count(void) {
    /* For the irreversible CTMC, (1 - p_step)^S = exp(-c) exactly, so
     * the match fraction is the same at any S. Projection stub would
     * also be S-independent but at a different value (1.0). */
    int N = 8;
    int target[8] = {+1, -1, +1, +1, -1, -1, +1, -1};
    double f1    = conditional_match_fraction(N, 1,   1500, 2, target, 2.0);
    double f64   = conditional_match_fraction(N, 64,  1500, 3, target, 2.0);
    double f1024 = conditional_match_fraction(N, 1024,1500, 4, target, 2.0);
    ASSERT_NEAR(f1,    f64,  0.02);
    ASSERT_NEAR(f64,   f1024,0.02);
}
static void test_conditional_scales_with_rate(void) {
    /* Match fraction strictly increases with c. A projection stub is
     * fixed at 1.0 for every c, so this fails on the stub. */
    int N = 8;
    int target[8] = {+1, -1, +1, +1, -1, -1, +1, -1};
    double f_c05 = conditional_match_fraction(N, 128, 1500, 5, target, 0.5);
    double f_c1  = conditional_match_fraction(N, 128, 1500, 6, target, 1.0);
    double f_c2  = conditional_match_fraction(N, 128, 1500, 7, target, 2.0);
    ASSERT_TRUE(f_c05 < f_c1);
    ASSERT_TRUE(f_c1  < f_c2);
    ASSERT_TRUE(f_c05 < 0.80);
    ASSERT_NEAR(f_c2, 0.5 + 0.5*(1 - exp(-2.0)), 0.02);
}
static void test_unconditional_relaxes_to_analytic_mean(void) {
    /* c = 2, b = 0.6, m(0) = 0 → m(1) = 0.6·(1-e^{-2}) ≈ 0.519.
     * A projection stub that samples Bernoulli(b) at the end would
     * give 0.6 — rejected by the 0.04 tolerance. */
    int N = 32;
    double bias[32];
    for (int i = 0; i < N; i++) bias[i] = 0.6;
    double m_high = unconditional_mean(N, 256, 400, 10, bias, 2.0);
    double expected = 0.6 * (1.0 - exp(-2.0));
    ASSERT_NEAR(m_high, expected, 0.03);
    ASSERT_TRUE(fabs(m_high - 0.6) > 0.05);
}
static void test_unconditional_low_S_is_distinct_from_high_S(void) {
    /* With c = 2 and S = 1 the Euler flip probability per step is
     * 1-exp(-2·p), giving a quite different mean than the continuum
     * limit. A stub that just sets the end state to Bernoulli(b)
     * gives the SAME mean at any S, so this test fails on the stub. */
    int N = 32;
    double bias[32];
    for (int i = 0; i < N; i++) bias[i] = 0.6;
    double m_low  = unconditional_mean(N, 1,   400, 20, bias, 2.0);
    double m_high = unconditional_mean(N, 256, 400, 21, bias, 2.0);
    ASSERT_TRUE(fabs(m_low - m_high) > 0.03);
}
static void test_zero_bias_gives_zero_mean(void) {
    int N = 32;
    double bias[32] = {0};
    double m = unconditional_mean(N, 64, 400, 30, bias, 2.0);
    ASSERT_NEAR(m, 0.0, 0.03);
}
static void test_null_arguments_return_error(void) {
    flow_matching_config_t cfg = flow_matching_config_defaults();
    int sample[4];
    int target[4] = {+1, -1, +1, -1};
    double bias[4] = {0, 0, 0, 0};
    cfg.num_sites = 4;
    ASSERT_EQ_INT(flow_matching_sample_conditional(NULL, target, sample), -1);
    ASSERT_EQ_INT(flow_matching_sample_conditional(&cfg, NULL,   sample), -1);
    ASSERT_EQ_INT(flow_matching_sample_conditional(&cfg, target, NULL),   -1);
    ASSERT_EQ_INT(flow_matching_sample_unconditional(NULL, bias, sample), -1);
    ASSERT_EQ_INT(flow_matching_sample_unconditional(&cfg, NULL, sample), -1);
    ASSERT_EQ_INT(flow_matching_sample_unconditional(&cfg, bias, NULL),   -1);
}
static void test_fit_rates_closes_gap_to_target_magnetisation(void) {
    /* Given per-site target magnetisations, the fitter should choose
     * rates that reach them. Pick 8 sites each with bias=0.8 and
     * target m=0.4. The optimal rate is -log(1 - 0.5) = log(2) ≈ 0.693.
     * Verify the sampler averaged over many trials hits m_target to
     * a few percent. */
    flow_matching_config_t cfg = flow_matching_config_defaults();
    cfg.num_sites = 8;
    cfg.num_steps = 2048;      /* finer dt to reduce Euler discretisation bias */
    double bias[8], m_target[8], rates[8];
    for (int i = 0; i < 8; i++) { bias[i] = 0.8; m_target[i] = 0.4; }
    flow_matching_fit_rates_to_magnetisation(&cfg, bias, m_target, rates);
    for (int i = 0; i < 8; i++) ASSERT_NEAR(rates[i], log(2.0), 1e-10);
    long trials = 600;
    double mean = 0;
    int sample[8];
    /* Hash the trial index into a well-spread 64-bit seed to decorrelate
     * xorshift's first-iteration outputs across trials. */
    for (long t = 0; t < trials; t++) {
        cfg.seed = (unsigned long long)(t + 1) * 0x9E3779B97F4A7C15ULL;
        cfg.seed ^= cfg.seed >> 33;
        cfg.seed *= 0xFF51AFD7ED558CCDULL;
        cfg.seed ^= cfg.seed >> 33;
        flow_matching_sample_biased_rates(&cfg, bias, rates, sample);
        for (int i = 0; i < 8; i++) mean += (double)sample[i];
    }
    mean /= (double)(trials * 8);
    printf("# flow-match fitted rate: ⟨m⟩ = %.3f (target 0.4)\n", mean);
    ASSERT_NEAR(mean, 0.4, 0.05);
}
int main(void) {
    TEST_RUN(test_conditional_match_matches_analytic);
    TEST_RUN(test_conditional_is_independent_of_step_count);
    TEST_RUN(test_conditional_scales_with_rate);
    TEST_RUN(test_unconditional_relaxes_to_analytic_mean);
    TEST_RUN(test_unconditional_low_S_is_distinct_from_high_S);
    TEST_RUN(test_zero_bias_gives_zero_mean);
    TEST_RUN(test_null_arguments_return_error);
    TEST_RUN(test_fit_rates_closes_gap_to_target_magnetisation);
    TEST_SUMMARY();
}