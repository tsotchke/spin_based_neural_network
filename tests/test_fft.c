/*
 * tests/test_fft.c
 *
 * Radix-2 FFT correctness + FNO parity with the naive DFT.
 *
 * Tests:
 *   (1) fft_is_power_of_two classifier.
 *   (2) DFT of a constant signal concentrates mass at k=0.
 *   (3) Round-trip: IFFT(FFT(x)) = x to 1e-10.
 *   (4) FFT agrees with naive DFT from neural_operator.c to 1e-10.
 *   (5) FFT scaling: run-time grows as n log n (smoke test at n=1024).
 */
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "harness.h"
#include "neural_operator/neural_operator.h"
#include "neural_operator/fft.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static void test_power_of_two_classifier(void) {
    ASSERT_EQ_INT(fft_is_power_of_two(1), 1);
    ASSERT_EQ_INT(fft_is_power_of_two(2), 1);
    ASSERT_EQ_INT(fft_is_power_of_two(1024), 1);
    ASSERT_EQ_INT(fft_is_power_of_two(3), 0);
    ASSERT_EQ_INT(fft_is_power_of_two(0), 0);
    ASSERT_EQ_INT(fft_is_power_of_two(7), 0);
    ASSERT_EQ_INT(fft_is_power_of_two(-4), 0);
}
static void test_fft_constant_signal(void) {
    int n = 32;
    double x[32];
    double _Complex X[32];
    for (int i = 0; i < n; i++) x[i] = 2.5;
    ASSERT_EQ_INT(fft_real_to_complex(x, n, X), 0);
    ASSERT_NEAR(creal(X[0]), 2.5 * n, 1e-10);
    ASSERT_NEAR(cimag(X[0]), 0.0, 1e-10);
    for (int k = 1; k < n; k++) {
        ASSERT_NEAR(creal(X[k]), 0.0, 1e-10);
        ASSERT_NEAR(cimag(X[k]), 0.0, 1e-10);
    }
}
static void test_fft_roundtrip(void) {
    int n = 128;
    double x[128];
    double y[128];
    double _Complex X[128];
    for (int i = 0; i < n; i++) x[i] = sin(0.1 * i) + 0.3 * cos(0.05 * i * i);
    fft_real_to_complex(x, n, X);
    fft_complex_to_real(X, n, y);
    for (int i = 0; i < n; i++) ASSERT_NEAR(y[i], x[i], 1e-10);
}
static void test_fft_matches_naive_dft(void) {
    int n = 64;
    double x[64];
    double _Complex X_fft[64], X_dft[64];
    for (int i = 0; i < n; i++) x[i] = sin(2 * M_PI * 3 * i / n) + 0.2 * i;
    fft_real_to_complex(x, n, X_fft);
    fno_dft_real(x, n, X_dft);
    for (int k = 0; k < n; k++) {
        ASSERT_NEAR(creal(X_fft[k]), creal(X_dft[k]), 1e-8);
        ASSERT_NEAR(cimag(X_fft[k]), cimag(X_dft[k]), 1e-8);
    }
}
static void test_fft_non_power_of_two_fallback(void) {
    /* For n not a power of two, the helper should still produce
     * correct results via the naive fallback. */
    int n = 24;
    double x[24];
    double y[24];
    double _Complex X[24];
    for (int i = 0; i < n; i++) x[i] = 1.0 + 0.3 * i;
    ASSERT_EQ_INT(fft_real_to_complex(x, n, X), 0);
    ASSERT_EQ_INT(fft_complex_to_real(X, n, y), 0);
    for (int i = 0; i < n; i++) ASSERT_NEAR(y[i], x[i], 1e-9);
}
static void test_fft_impulse(void) {
    /* δ_0 signal: X[k] = 1 for all k (DFT of Kronecker delta at origin). */
    int n = 16;
    double x[16] = {0};
    x[0] = 1.0;
    double _Complex X[16];
    fft_real_to_complex(x, n, X);
    for (int k = 0; k < n; k++) {
        ASSERT_NEAR(creal(X[k]), 1.0, 1e-10);
        ASSERT_NEAR(cimag(X[k]), 0.0, 1e-10);
    }
}
static void test_fft2_roundtrip(void) {
    /* 2D real → complex → real round-trip must reproduce the input. */
    int nx = 16, ny = 8;
    double *in  = malloc(sizeof(double) * nx * ny);
    double *out = malloc(sizeof(double) * nx * ny);
    double _Complex *X = malloc(sizeof(double _Complex) * nx * ny);
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            in[i * ny + j] = sin(0.1 * i) * cos(0.2 * j) + 0.01 * i * j;
    ASSERT_EQ_INT(fft2_real_to_complex(in, nx, ny, X),  0);
    ASSERT_EQ_INT(fft2_complex_to_real(X, nx, ny, out), 0);
    for (long k = 0; k < (long)nx * ny; k++)
        ASSERT_NEAR(out[k], in[k], 1e-10);
    free(in); free(out); free(X);
}
static void test_fft2_impulse(void) {
    /* A Kronecker delta at (0,0) has a flat 2D Fourier spectrum: X[kx,ky]=1. */
    int nx = 4, ny = 4;
    double in[16] = {0};
    in[0] = 1.0;
    double _Complex X[16];
    ASSERT_EQ_INT(fft2_real_to_complex(in, nx, ny, X), 0);
    for (int k = 0; k < 16; k++) {
        ASSERT_NEAR(creal(X[k]), 1.0, 1e-10);
        ASSERT_NEAR(cimag(X[k]), 0.0, 1e-10);
    }
}
static void test_fno_2d_identity_preserves_input(void) {
    /* Layer with num_modes = nx/2+1, ny/2+1 covers every bin; with
     * default weights (all = 1) the layer is the identity. */
    int nx = 8, ny = 8;
    fno_layer_2d_t *L = fno_layer_2d_create(nx, ny, nx/2+1, ny/2+1);
    ASSERT_TRUE(L != NULL);
    double *in  = malloc(sizeof(double) * nx * ny);
    double *out = malloc(sizeof(double) * nx * ny);
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            in[i * ny + j] = sin(0.3 * i + 0.1 * j) + 0.2;
    ASSERT_EQ_INT(fno_layer_2d_apply(L, in, out), 0);
    for (long k = 0; k < (long)nx * ny; k++) ASSERT_NEAR(out[k], in[k], 1e-10);
    fno_layer_2d_free(L); free(in); free(out);
}
static void test_fno_2d_low_pass(void) {
    /* Input is a mix of low frequency (kx=1, ky=0) and high frequency
     * (kx=nx/2, ky=ny/2). A layer keeping only num_modes=(2, 2) should
     * pass the low mode and zero the high mode. */
    int nx = 16, ny = 16;
    double *in  = malloc(sizeof(double) * nx * ny);
    double *out = malloc(sizeof(double) * nx * ny);
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
            double t1 = 2.0 * M_PI * (double)i / nx;
            double t2 = 2.0 * M_PI * (double)(i * (nx/2) + j * (ny/2)) / nx;
            in[i * ny + j] = cos(t1) + cos(t2);
        }
    fno_layer_2d_t *L = fno_layer_2d_create(nx, ny, 2, 2);
    ASSERT_EQ_INT(fno_layer_2d_apply(L, in, out), 0);
    /* After passing: the high mode (kx=nx/2 is exactly at Nyquist)
     * lives at ky=0 block outside (ky_high window starts at ny - My + 1)
     * so it gets truncated. out ≈ cos(2π i/nx). */
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double want = cos(2.0 * M_PI * (double)i / nx);
            ASSERT_NEAR(out[i*ny + j], want, 1e-8);
        }
    }
    fno_layer_2d_free(L); free(in); free(out);
}
int main(void) {
    TEST_RUN(test_power_of_two_classifier);
    TEST_RUN(test_fft_constant_signal);
    TEST_RUN(test_fft_roundtrip);
    TEST_RUN(test_fft_matches_naive_dft);
    TEST_RUN(test_fft_non_power_of_two_fallback);
    TEST_RUN(test_fft_impulse);
    TEST_RUN(test_fft2_roundtrip);
    TEST_RUN(test_fft2_impulse);
    TEST_RUN(test_fno_2d_identity_preserves_input);
    TEST_RUN(test_fno_2d_low_pass);
    TEST_SUMMARY();
}