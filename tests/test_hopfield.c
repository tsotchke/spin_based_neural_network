/*
 * tests/test_hopfield.c
 *
 * Hopfield associative memory — thermodynamic-computing baseline for
 * pillar P2.9. Validates:
 *   (1) A single stored pattern is a fixed point of the zero-T dynamics.
 *   (2) A noisy-version-of-a-stored-pattern recalls to the clean
 *       pattern (capacity regime: K/N ≪ 0.138).
 *   (3) Capacity: storing K ≈ 0.1·N patterns still achieves > 90%
 *       per-spin recall overlap on average.
 *   (4) Metropolis at finite β samples near stored attractors.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "thermodynamic/hopfield.h"
static void random_pattern(int *xi, int N, unsigned *seed) {
    for (int i = 0; i < N; i++) {
        *seed = (*seed) * 1103515245u + 12345u;
        xi[i] = ((*seed) & 1) ? +1 : -1;
    }
}
static void flip_fraction(int *xi, int *out, int N, double frac, unsigned *seed) {
    memcpy(out, xi, (size_t)N * sizeof(int));
    int num_flips = (int)(frac * (double)N + 0.5);
    for (int k = 0; k < num_flips; k++) {
        *seed = (*seed) * 1103515245u + 12345u;
        int i = (int)((*seed) % (unsigned)N);
        out[i] = -out[i];
    }
}
static void test_single_pattern_is_fixed_point(void) {
    /* Storing one pattern must make it a stable fixed point. */
    int N = 32;
    int pattern[32];
    unsigned seed = 0xA5A5;
    random_pattern(pattern, N, &seed);
    hopfield_t *net = hopfield_create(N);
    hopfield_store_patterns(net, pattern, 1);
    int state[32];
    memcpy(state, pattern, sizeof(state));
    int changed = hopfield_sync_update(net, state);
    ASSERT_EQ_INT(changed, 0);
    double ov = hopfield_overlap(net, pattern, state);
    ASSERT_NEAR(ov, 1.0, 1e-12);
    hopfield_free(net);
}
static void test_noisy_input_recalls_to_stored_pattern(void) {
    /* Flip 10% of bits; a single-pattern Hopfield must recover. */
    int N = 64;
    int pattern[64];
    unsigned seed = 0xBEEF;
    random_pattern(pattern, N, &seed);
    hopfield_t *net = hopfield_create(N);
    hopfield_store_patterns(net, pattern, 1);
    int noisy[64];
    flip_fraction(pattern, noisy, N, 0.10, &seed);
    double ov_before = hopfield_overlap(net, pattern, noisy);
    hopfield_recall(net, noisy, 100);
    double ov_after = hopfield_overlap(net, pattern, noisy);
    printf("# Hopfield noisy recall: overlap %.3f → %.3f\n",
           ov_before, ov_after);
    ASSERT_TRUE(ov_before < 0.95);
    ASSERT_NEAR(ov_after, 1.0, 1e-12);
    hopfield_free(net);
}
static void test_capacity_below_amit_bound(void) {
    /* Amit-Gutfreund-Sompolinsky bound: K/N < 0.138 for reliable
     * recall. At K/N = 0.1 the mean recall overlap should exceed 0.9. */
    int N = 100;
    int K = 10;
    int *patterns = malloc((size_t)K * (size_t)N * sizeof(int));
    unsigned seed = 0xC0FFEE;
    for (int mu = 0; mu < K; mu++) {
        random_pattern(&patterns[mu * N], N, &seed);
    }
    hopfield_t *net = hopfield_create(N);
    hopfield_store_patterns(net, patterns, K);
    double sum_ov = 0.0;
    int *state = malloc((size_t)N * sizeof(int));
    for (int mu = 0; mu < K; mu++) {
        flip_fraction(&patterns[mu * N], state, N, 0.05, &seed);
        hopfield_recall(net, state, 200);
        sum_ov += fabs(hopfield_overlap(net, &patterns[mu * N], state));
    }
    double mean_ov = sum_ov / (double)K;
    printf("# Hopfield capacity N=%d K=%d (K/N=%.2f): "
           "mean |overlap| after recall = %.3f\n", N, K, (double)K/N, mean_ov);
    ASSERT_TRUE(mean_ov > 0.9);
    free(state);
    free(patterns);
    hopfield_free(net);
}
static void test_metropolis_samples_near_attractor(void) {
    /* At low temperature (β = 4), Metropolis sweeps starting from a
     * stored pattern stay close to it (overlap ≫ 0). At very high
     * temperature (β = 0.01) the overlap drops toward ~0. */
    int N = 40;
    int pattern[40];
    unsigned seed = 0xD00D;
    random_pattern(pattern, N, &seed);
    hopfield_t *net = hopfield_create(N);
    hopfield_store_patterns(net, pattern, 1);
    int state[40];
    memcpy(state, pattern, sizeof(state));
    unsigned long long rng = 0x1234ULL;
    for (int s = 0; s < 50; s++) hopfield_metropolis_sweep(net, state, 4.0, &rng);
    double ov_lowT = fabs(hopfield_overlap(net, pattern, state));
    /* High T: start at pattern, expect drift. */
    memcpy(state, pattern, sizeof(state));
    for (int s = 0; s < 200; s++) hopfield_metropolis_sweep(net, state, 0.01, &rng);
    double ov_highT = fabs(hopfield_overlap(net, pattern, state));
    printf("# Hopfield MC: ⟨|m|⟩ β=4.0 → %.3f, β=0.01 → %.3f\n",
           ov_lowT, ov_highT);
    ASSERT_TRUE(ov_lowT > 0.8);
    ASSERT_TRUE(ov_highT < 0.5);
    hopfield_free(net);
}
static void test_energy_minimum_at_stored_pattern(void) {
    /* Among a stored pattern, its bit-flip neighbours, and random
     * configurations, the stored pattern has the lowest energy. */
    int N = 32;
    int pattern[32];
    unsigned seed = 0xE42;
    random_pattern(pattern, N, &seed);
    hopfield_t *net = hopfield_create(N);
    hopfield_store_patterns(net, pattern, 1);
    double E_store = hopfield_energy(net, pattern);
    int neighbour[32];
    memcpy(neighbour, pattern, sizeof(neighbour));
    neighbour[0] = -neighbour[0];
    double E_nei = hopfield_energy(net, neighbour);
    int rnd[32];
    random_pattern(rnd, N, &seed);
    double E_rnd = hopfield_energy(net, rnd);
    printf("# Hopfield energies: stored=%.3f  neighbour=%.3f  random=%.3f\n",
           E_store, E_nei, E_rnd);
    ASSERT_TRUE(E_store <= E_nei);
    ASSERT_TRUE(E_store <= E_rnd);
    hopfield_free(net);
}
int main(void) {
    TEST_RUN(test_single_pattern_is_fixed_point);
    TEST_RUN(test_noisy_input_recalls_to_stored_pattern);
    TEST_RUN(test_capacity_below_amit_bound);
    TEST_RUN(test_metropolis_samples_near_attractor);
    TEST_RUN(test_energy_minimum_at_stored_pattern);
    TEST_SUMMARY();
}