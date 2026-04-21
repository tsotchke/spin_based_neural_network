/*
 * tests/test_neural_operator.c
 *
 * Covers the reference FNO layer. Checks:
 *   - DFT of a constant signal is concentrated at k=0.
 *   - DFT/IDFT round-trip is the identity (to machine precision for n=8).
 *   - Identity weights give the low-pass-filtered input back.
 *   - Zeroing a weight zeroes the corresponding Fourier component.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "neural_operator/neural_operator.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static void test_dft_of_constant(void) {
    int n = 16;
    double *x = malloc((size_t)n * sizeof(double));
    double _Complex *X = malloc((size_t)n * sizeof(double _Complex));
    for (int i = 0; i < n; i++) x[i] = 3.5;
    fno_dft_real(x, n, X);
    /* F[0] = sum(x) = n·3.5 = 56; all other bins ≈ 0. */
    ASSERT_NEAR(creal(X[0]), 56.0, 1e-10);
    ASSERT_NEAR(cimag(X[0]),  0.0, 1e-10);
    for (int k = 1; k < n; k++) {
        ASSERT_NEAR(creal(X[k]), 0.0, 1e-10);
        ASSERT_NEAR(cimag(X[k]), 0.0, 1e-10);
    }
    free(x); free(X);
}
static void test_dft_idft_roundtrip(void) {
    int n = 8;
    double *x  = malloc((size_t)n * sizeof(double));
    double *y  = malloc((size_t)n * sizeof(double));
    double _Complex *X = malloc((size_t)n * sizeof(double _Complex));
    for (int i = 0; i < n; i++) x[i] = (double)(i * i) - 5.0;
    fno_dft_real(x, n, X);
    fno_idft(X, n, y);
    for (int i = 0; i < n; i++) ASSERT_NEAR(y[i], x[i], 1e-10);
    free(x); free(y); free(X);
}
static void test_layer_identity_weights_preserve_low_modes(void) {
    /* With num_modes = n/2+1 the layer is the full identity: output equals
     * input to numerical precision. */
    int n = 8;
    fno_layer_t *L = fno_layer_create(n, n / 2 + 1);
    ASSERT_TRUE(L != NULL);
    double in[8], out[8];
    for (int i = 0; i < n; i++) in[i] = sin(2.0 * M_PI * (double)i / (double)n) + 0.3;
    ASSERT_EQ_INT(fno_layer_apply(L, in, out), 0);
    for (int i = 0; i < n; i++) ASSERT_NEAR(out[i], in[i], 1e-9);
    fno_layer_free(L);
}
static void test_layer_zero_weights_produces_zero(void) {
    int n = 16;
    fno_layer_t *L = fno_layer_create(n, 4);
    ASSERT_TRUE(L != NULL);
    for (int k = 0; k < L->num_modes; k++) L->weights[k] = 0.0 + 0.0 * _Complex_I;
    double in[16], out[16];
    for (int i = 0; i < n; i++) in[i] = (double)i - 7.5;
    ASSERT_EQ_INT(fno_layer_apply(L, in, out), 0);
    for (int i = 0; i < n; i++) ASSERT_NEAR(out[i], 0.0, 1e-9);
    fno_layer_free(L);
}
static void test_layer_low_pass_on_pure_tones(void) {
    /* Build a signal with two pure cosine tones at k=1 and k=6.
     * Keeping num_modes=3 drops the k=6 component. */
    int n = 16;
    double in[16], out[16];
    for (int i = 0; i < n; i++) {
        double t = 2.0 * M_PI * (double)i / (double)n;
        in[i] = cos(t) + cos(6.0 * t);
    }
    fno_layer_t *L = fno_layer_create(n, 3);  /* keeps k = 0, 1, 2 */
    ASSERT_TRUE(L != NULL);
    ASSERT_EQ_INT(fno_layer_apply(L, in, out), 0);
    for (int i = 0; i < n; i++) {
        double t = 2.0 * M_PI * (double)i / (double)n;
        double want = cos(t);   /* only the k=1 tone survives */
        ASSERT_NEAR(out[i], want, 1e-8);
    }
    fno_layer_free(L);
}
static void test_layer_imaginary_weight_rotates_phase(void) {
    /* Applying w_1 = i to the k=1 Fourier mode (with conj(w_1) = -i
     * on the mirror bin k = n-1 to preserve Hermitian symmetry)
     * rotates cos(2π j / n) into -sin(2π j / n). A stub that simply
     * scales each bin by |w| would NOT produce this sign flip in the
     * sine component, so this test rejects magnitude-only scaling. */
    int n = 16;
    fno_layer_t *L = fno_layer_create(n, 2);    /* keep k=0, k=1 */
    ASSERT_TRUE(L != NULL);
    L->weights[0] = 1.0 + 0.0 * _Complex_I;
    L->weights[1] = 0.0 + 1.0 * _Complex_I;     /* w_1 = i */
    double in[16], out[16];
    for (int i = 0; i < n; i++)
        in[i] = cos(2.0 * M_PI * (double)i / (double)n);
    ASSERT_EQ_INT(fno_layer_apply(L, in, out), 0);
    for (int i = 0; i < n; i++) {
        double t = 2.0 * M_PI * (double)i / (double)n;
        ASSERT_NEAR(out[i], -sin(t), 1e-8);
    }
    fno_layer_free(L);
}
static void test_layer_fit_learns_spatial_derivative(void) {
    /* Spatial derivative in Fourier space is multiplication by i·k.
     * Give the fitter a batch of random-amplitude sine/cosine inputs
     * paired with their derivatives; the closed-form optimal weights
     * should end up approximating i·2π·k/n for the low-frequency
     * modes. */
    int n = 32;
    int M = n / 2 + 1;
    fno_layer_t *L = fno_layer_create(n, M);
    int num_pairs = 20;
    double *phi_batch = malloc(sizeof(double) * num_pairs * n);
    double *psi_batch = malloc(sizeof(double) * num_pairs * n);
    unsigned long long rng = 0x2222ULL;
    for (int p = 0; p < num_pairs; p++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        int kk = 1 + (int)((rng >> 3) % 5);          /* freqs 1..5 */
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        double amp = 0.5 + (double)(rng >> 11) / 9007199254740992.0;
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        double phase = 2.0 * M_PI * (double)(rng >> 11) / 9007199254740992.0;
        for (int i = 0; i < n; i++) {
            double t = 2.0 * M_PI * (double)i / (double)n;
            double omega = (double)kk;
            phi_batch[p * n + i] = amp * cos(omega * t + phase);
            /* d/dx (amp · cos(ωt + phase)) = -amp·ω · sin(ωt + phase). */
            psi_batch[p * n + i] = -amp * omega * sin(omega * t + phase);
        }
    }
    ASSERT_EQ_INT(fno_layer_fit(L, phi_batch, psi_batch, num_pairs, 1e-8), 0);
    /* Apply the trained layer to a held-out cosine and compare
     * against its analytic derivative. */
    double test_in[32], test_out[32];
    for (int i = 0; i < n; i++) {
        double t = 2.0 * M_PI * (double)i / (double)n;
        test_in[i] = 0.7 * cos(3.0 * t + 0.5);
    }
    fno_layer_apply(L, test_in, test_out);
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double t = 2.0 * M_PI * (double)i / (double)n;
        double want = -0.7 * 3.0 * sin(3.0 * t + 0.5);
        double err = fabs(test_out[i] - want);
        if (err > max_err) max_err = err;
    }
    printf("# fitted FNO derivative: max error = %.3e\n", max_err);
    ASSERT_TRUE(max_err < 1e-6);
    fno_layer_free(L);
    free(phi_batch); free(psi_batch);
}
int main(void) {
    TEST_RUN(test_dft_of_constant);
    TEST_RUN(test_dft_idft_roundtrip);
    TEST_RUN(test_layer_identity_weights_preserve_low_modes);
    TEST_RUN(test_layer_zero_weights_produces_zero);
    TEST_RUN(test_layer_low_pass_on_pure_tones);
    TEST_RUN(test_layer_imaginary_weight_rotates_phase);
    TEST_RUN(test_layer_fit_learns_spatial_derivative);
    TEST_SUMMARY();
}