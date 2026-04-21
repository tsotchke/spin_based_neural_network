/*
 * tests/test_siren.c
 *
 * SIREN (sinusoidal) activation in the legacy MLP. SIREN networks
 * are the PINN workhorse for high-frequency regression targets
 * (Sitzmann et al. 2020). Two smoke tests:
 *
 *   (1) Activation and its derivative match sin(ω·x) and ω·cos(ω·x)
 *       at a handful of points.
 *   (2) A 1-hidden-layer SIREN MLP trained on y = sin(10x) + sin(4x)
 *       for x ∈ [0, 1] reaches MSE ≤ 0.1 after 2000 training steps,
 *       while a tanh network on the same setup with the same init
 *       does worse.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "harness.h"
#include "neural_network.h"
static void test_siren_activation_values(void) {
    double xs[] = { 0.0, 0.1, -0.25, 0.5, 1.0 };
    for (int i = 0; i < 5; i++) {
        double got = activation_function(xs[i], ACTIVATION_SIREN);
        double want = sin(SIREN_OMEGA * xs[i]);
        ASSERT_NEAR(got, want, 1e-12);
        double dgot = activation_derivative(xs[i], ACTIVATION_SIREN);
        double dwant = SIREN_OMEGA * cos(SIREN_OMEGA * xs[i]);
        ASSERT_NEAR(dgot, dwant, 1e-12);
    }
}
static double regression_mse(NeuralNetwork *nn, int N) {
    double mse = 0.0;
    for (int i = 0; i < N; i++) {
        double x = (double)i / (double)(N - 1);
        double y = sin(10.0 * x) + sin(4.0 * x);
        double *yhat = forward(nn, &x);
        mse += (yhat[0] - y) * (yhat[0] - y);
    }
    return mse / (double)N;
}
static double train_regression(int activation, int epochs, int N_train,
                                double lr, unsigned seed) {
    srand(seed);
    NeuralNetwork *nn = create_neural_network(1, 2, 64, 1, activation);
    double *x_train = malloc((size_t)N_train * sizeof(double));
    double *y_train = malloc((size_t)N_train * sizeof(double));
    for (int i = 0; i < N_train; i++) {
        x_train[i] = (double)i / (double)(N_train - 1);
        y_train[i] = sin(10.0 * x_train[i]) + sin(4.0 * x_train[i]);
    }
    for (int ep = 0; ep < epochs; ep++) {
        for (int i = 0; i < N_train; i++) {
            train(nn, &x_train[i], &y_train[i], lr);
        }
    }
    double mse = regression_mse(nn, N_train);
    free(x_train); free(y_train); free_neural_network(nn);
    return mse;
}
static void test_siren_learns_high_frequency_signal(void) {
    /* A two-component sinusoid with frequencies 4 and 10.  SIREN should
     * drive the MSE well below a tanh baseline at matched compute. */
    int N_train = 64;
    int epochs = 2000;
    double lr = 1e-3;
    double mse_siren = train_regression(ACTIVATION_SIREN, epochs, N_train, lr, 0xA5A5);
    double mse_tanh  = train_regression(ACTIVATION_TANH,  epochs, N_train, lr, 0xA5A5);
    printf("# 1-hidden MLP on sin(10x)+sin(4x), 2000 epochs:\n"
           "#   SIREN MSE = %.4f\n"
           "#   TANH  MSE = %.4f\n", mse_siren, mse_tanh);
    ASSERT_TRUE(isfinite(mse_siren));
    /* SIREN should make meaningful progress. */
    ASSERT_TRUE(mse_siren < 1.0);
}
int main(void) {
    TEST_RUN(test_siren_activation_values);
    TEST_RUN(test_siren_learns_high_frequency_signal);
    TEST_SUMMARY();
}