/*
 * tests/test_rbm_cd.c
 *
 * Contrastive-divergence RBM as a generative model on ±1 spin patterns.
 * Validates:
 *   (1) Before training, samples are ~uniform: free energy is roughly
 *       the same for training patterns and random patterns.
 *   (2) After training, free energy of training patterns drops well
 *       below that of random patterns (model has learned the data).
 *   (3) Samples from the trained RBM hit the training patterns at rate
 *       well above chance (1 / 2^N).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "thermodynamic/rbm_cd.h"
static void test_free_energy_drops_on_training_patterns(void) {
    /* Three fixed 8-bit patterns form the training set; the RBM should
     * push their free energy below that of a random held-out pattern. */
    int N = 8, M = 6;
    int train[3 * 8] = {
        1, 0, 1, 0, 1, 0, 1, 0,  /* alternating */
        1, 1, 1, 1, 0, 0, 0, 0,  /* half-split  */
        0, 0, 1, 1, 0, 0, 1, 1   /* double-step */
    };
    int random[8] = { 1, 0, 0, 1, 1, 0, 0, 1 };  /* not in train set */
    rbm_cd_t *rbm = rbm_cd_create(N, M, 0.05, 0xC001ULL);
    double F_train_before = 0;
    for (int p = 0; p < 3; p++) F_train_before += rbm_cd_free_energy(rbm, &train[p*N]);
    F_train_before /= 3.0;
    double F_rand_before = rbm_cd_free_energy(rbm, random);
    /* CD-1 training for many epochs. */
    for (int epoch = 0; epoch < 3000; epoch++) {
        rbm_cd_train_batch(rbm, train, 3, 1, 0.05);
    }
    double F_train_after = 0;
    for (int p = 0; p < 3; p++) F_train_after += rbm_cd_free_energy(rbm, &train[p*N]);
    F_train_after /= 3.0;
    double F_rand_after = rbm_cd_free_energy(rbm, random);
    printf("# RBM CD-1: F(train) %.3f→%.3f  F(rand) %.3f→%.3f  "
           "gap %.3f\n",
           F_train_before, F_train_after, F_rand_before, F_rand_after,
           F_rand_after - F_train_after);
    /* After training, training-pattern free energy must be strictly
     * below a random pattern's — the model has learned the support. */
    ASSERT_TRUE(F_train_after < F_rand_after - 0.5);
    /* Before training, they should be within O(1). */
    ASSERT_TRUE(fabs(F_train_before - F_rand_before) < 2.0);
    rbm_cd_free(rbm);
}
static int pattern_matches(const int *a, const int *b, int N) {
    for (int i = 0; i < N; i++) if (a[i] != b[i]) return 0;
    return 1;
}
static void test_samples_concentrate_on_training_patterns(void) {
    /* 4-bit, 2-pattern dataset → 16 states total. Chance rate of
     * landing on either training pattern is 2/16 = 12.5%. After
     * training, the RBM should land on one of them far more often. */
    int N = 4, M = 4;
    int train[2 * 4] = {
        1, 0, 1, 0,
        0, 1, 0, 1
    };
    rbm_cd_t *rbm = rbm_cd_create(N, M, 0.05, 0xFEEDULL);
    for (int epoch = 0; epoch < 5000; epoch++) {
        rbm_cd_train_batch(rbm, train, 2, 1, 0.1);
    }
    int num_samples = 2000;
    int *samples = malloc((size_t)num_samples * N * sizeof(int));
    rbm_cd_sample(rbm, samples, num_samples, 500, 5);
    int hits = 0;
    for (int s = 0; s < num_samples; s++) {
        int *v = &samples[s * N];
        if (pattern_matches(v, &train[0], N) || pattern_matches(v, &train[N], N)) hits++;
    }
    double hit_rate = (double)hits / (double)num_samples;
    printf("# RBM trained hit rate: %.3f  (chance = %.3f)\n",
           hit_rate, 2.0 / 16.0);
    /* Well above chance. */
    ASSERT_TRUE(hit_rate > 0.4);
    free(samples);
    rbm_cd_free(rbm);
}
static void test_untrained_rbm_samples_are_diffuse(void) {
    /* With small random weights, the RBM's distribution is nearly
     * uniform — hit rate on any two specific patterns should be ~2/16. */
    int N = 4, M = 4;
    int train[2 * 4] = { 1, 0, 1, 0,  0, 1, 0, 1 };
    rbm_cd_t *rbm = rbm_cd_create(N, M, 0.01, 0xBEEFULL);
    int num_samples = 2000;
    int *samples = malloc((size_t)num_samples * N * sizeof(int));
    rbm_cd_sample(rbm, samples, num_samples, 200, 5);
    int hits = 0;
    for (int s = 0; s < num_samples; s++) {
        int *v = &samples[s * N];
        if (pattern_matches(v, &train[0], N) || pattern_matches(v, &train[N], N)) hits++;
    }
    double hit_rate = (double)hits / (double)num_samples;
    printf("# Untrained RBM hit rate on two specific patterns: %.3f "
           "(chance 0.125)\n", hit_rate);
    /* Well below the trained hit-rate; roughly near chance. */
    ASSERT_TRUE(hit_rate < 0.3);
    free(samples);
    rbm_cd_free(rbm);
}
static void test_null_args_return_error(void) {
    ASSERT_TRUE(rbm_cd_create(0, 4, 0.1, 0) == NULL);
    ASSERT_TRUE(rbm_cd_create(4, 0, 0.1, 0) == NULL);
    ASSERT_EQ_INT(rbm_cd_train_step(NULL, NULL, 1, 0.1), -1);
    ASSERT_EQ_INT(rbm_cd_train_batch(NULL, NULL, 1, 1, 0.1), -1);
    ASSERT_EQ_INT(rbm_cd_sample(NULL, NULL, 1, 1, 1), -1);
}
int main(void) {
    TEST_RUN(test_untrained_rbm_samples_are_diffuse);
    TEST_RUN(test_free_energy_drops_on_training_patterns);
    TEST_RUN(test_samples_concentrate_on_training_patterns);
    TEST_RUN(test_null_args_return_error);
    TEST_SUMMARY();
}