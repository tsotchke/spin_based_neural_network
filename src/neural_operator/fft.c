/*
 * src/neural_operator/fft.c
 *
 * Radix-2 Cooley-Tukey FFT with bit-reversal permutation and
 * iterative butterfly stages. Pure C, no BLAS/FFTW dependency.
 * For n = 1024 this runs at ~10⁶ flops vs 10⁹ for the naive DFT.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "neural_operator/fft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int fft_is_power_of_two(int n) {
    return (n > 0 && (n & (n - 1)) == 0);
}

/* Reverse the bits of i, keeping only `bits` bits. */
static unsigned reverse_bits(unsigned i, int bits) {
    unsigned r = 0;
    for (int b = 0; b < bits; b++) {
        r = (r << 1) | (i & 1);
        i >>= 1;
    }
    return r;
}

int fft_complex_inplace(double _Complex *x, int n, int inverse) {
    if (!x || n <= 0 || !fft_is_power_of_two(n)) return -1;

    /* Bit-reverse permutation. */
    int bits = 0;
    { int m = n; while (m > 1) { m >>= 1; bits++; } }
    for (unsigned i = 0; i < (unsigned)n; i++) {
        unsigned j = reverse_bits(i, bits);
        if (j > i) {
            double _Complex tmp = x[i];
            x[i] = x[j];
            x[j] = tmp;
        }
    }

    /* Iterative butterflies over stages m = 2, 4, 8, ..., n. */
    double sign = inverse ? +1.0 : -1.0;
    for (int m = 2; m <= n; m <<= 1) {
        int half = m >> 1;
        double theta = sign * 2.0 * M_PI / (double)m;
        double _Complex w_m = cos(theta) + sin(theta) * _Complex_I;
        for (int k = 0; k < n; k += m) {
            double _Complex w = 1.0 + 0.0 * _Complex_I;
            for (int j = 0; j < half; j++) {
                double _Complex t = w * x[k + j + half];
                double _Complex u = x[k + j];
                x[k + j]        = u + t;
                x[k + j + half] = u - t;
                w *= w_m;
            }
        }
    }
    return 0;
}

int fft_real_to_complex(const double *x_real, int n, double _Complex *X_out) {
    if (!x_real || !X_out || n <= 0) return -1;
    for (int i = 0; i < n; i++) X_out[i] = x_real[i] + 0.0 * _Complex_I;
    if (fft_is_power_of_two(n)) {
        return fft_complex_inplace(X_out, n, 0);
    }
    /* Fallback: Bluestein / naive DFT. Since the FNO layer only calls
     * power-of-two sizes in practice, fall back to naive-DFT here. */
    double _Complex *temp = malloc((size_t)n * sizeof(double _Complex));
    if (!temp) return -1;
    for (int k = 0; k < n; k++) {
        double re = 0.0, im = 0.0;
        double arg0 = -2.0 * M_PI * (double)k / (double)n;
        for (int j = 0; j < n; j++) {
            double arg = arg0 * (double)j;
            re += x_real[j] * cos(arg);
            im += x_real[j] * sin(arg);
        }
        temp[k] = re + im * _Complex_I;
    }
    memcpy(X_out, temp, (size_t)n * sizeof(double _Complex));
    free(temp);
    return 0;
}

int fft2_complex_inplace(double _Complex *x, int nx, int ny, int inverse) {
    if (!x || nx <= 0 || ny <= 0) return -1;
    if (!fft_is_power_of_two(nx) || !fft_is_power_of_two(ny)) return -1;
    /* Row transforms. */
    for (int i = 0; i < nx; i++) {
        if (fft_complex_inplace(&x[(size_t)i * ny], ny, inverse) != 0) return -1;
    }
    /* Column transforms: gather each column, transform, scatter. */
    double _Complex *col = malloc((size_t)nx * sizeof(double _Complex));
    if (!col) return -1;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) col[i] = x[(size_t)i * ny + j];
        if (fft_complex_inplace(col, nx, inverse) != 0) { free(col); return -1; }
        for (int i = 0; i < nx; i++) x[(size_t)i * ny + j] = col[i];
    }
    free(col);
    return 0;
}

int fft2_real_to_complex(const double *x_real, int nx, int ny,
                          double _Complex *X_out) {
    if (!x_real || !X_out || nx <= 0 || ny <= 0) return -1;
    for (long k = 0; k < (long)nx * ny; k++)
        X_out[k] = x_real[k] + 0.0 * _Complex_I;
    return fft2_complex_inplace(X_out, nx, ny, 0);
}

int fft2_complex_to_real(const double _Complex *X_in, int nx, int ny,
                          double *x_out) {
    if (!X_in || !x_out || nx <= 0 || ny <= 0) return -1;
    double _Complex *buf = malloc((size_t)nx * ny * sizeof(double _Complex));
    if (!buf) return -1;
    memcpy(buf, X_in, (size_t)nx * ny * sizeof(double _Complex));
    int rc = fft2_complex_inplace(buf, nx, ny, 1);
    if (rc != 0) { free(buf); return rc; }
    double inv = 1.0 / ((double)nx * ny);
    for (long k = 0; k < (long)nx * ny; k++) x_out[k] = creal(buf[k]) * inv;
    free(buf);
    return 0;
}

int fft_complex_to_real(const double _Complex *X_in, int n, double *x_out) {
    if (!X_in || !x_out || n <= 0) return -1;
    double _Complex *buf = malloc((size_t)n * sizeof(double _Complex));
    if (!buf) return -1;
    memcpy(buf, X_in, (size_t)n * sizeof(double _Complex));
    int rc;
    if (fft_is_power_of_two(n)) {
        rc = fft_complex_inplace(buf, n, 1);
    } else {
        /* Naive inverse DFT fallback. */
        double _Complex *tmp = malloc((size_t)n * sizeof(double _Complex));
        if (!tmp) { free(buf); return -1; }
        for (int j = 0; j < n; j++) {
            double re = 0.0, im = 0.0;
            double arg0 = 2.0 * M_PI * (double)j / (double)n;
            for (int k = 0; k < n; k++) {
                double arg = arg0 * (double)k;
                re += creal(X_in[k]) * cos(arg) - cimag(X_in[k]) * sin(arg);
                im += creal(X_in[k]) * sin(arg) + cimag(X_in[k]) * cos(arg);
            }
            tmp[j] = re + im * _Complex_I;
        }
        memcpy(buf, tmp, (size_t)n * sizeof(double _Complex));
        free(tmp);
        rc = 0;
    }
    if (rc != 0) { free(buf); return rc; }
    double inv_n = 1.0 / (double)n;
    for (int j = 0; j < n; j++) x_out[j] = creal(buf[j]) * inv_n;
    free(buf);
    return 0;
}
