/*
 * include/neural_operator/fft.h
 *
 * In-place radix-2 Cooley-Tukey FFT for the Fourier neural operator
 * pillar (P1.4). The naive O(n²) DFT in neural_operator.c was fine
 * for validation at n ≤ 64; this header ships the production
 * path used when n is a power of two (the common FNO case).
 *
 *   fft_complex_inplace(x, n, inverse)
 *     x ∈ C^n, n power of two.
 *     inverse = 0 computes X = DFT(x);
 *     inverse = 1 computes X = IDFT(x)   (no extra 1/n scaling).
 *
 *   fft_real_to_complex(x_real, n, X_out)
 *     Convenience: copies real input into a complex buffer and
 *     performs forward FFT.
 *
 *   fft_complex_to_real(X_in, n, x_out)
 *     Inverse FFT, extracts the real part, divides by n.
 *
 * The implementation uses precomputed twiddle factors for the
 * innermost butterfly; memory layout is interleaved (re, im).
 */
#ifndef NEURAL_OPERATOR_FFT_H
#define NEURAL_OPERATOR_FFT_H

#include <complex.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 if n > 0 is a power of two, 0 otherwise. */
int fft_is_power_of_two(int n);

/* In-place radix-2 FFT. n must be a power of two. Forward (inverse=0)
 * uses the -i·2π·k·j/n convention; inverse (inverse=1) uses the
 * conjugate kernel and does NOT apply the 1/n normalisation. Returns
 * 0 on success, -1 on argument error. */
int fft_complex_inplace(double _Complex *x, int n, int inverse);

/* Forward FFT of a real input. Returns 0 on success. */
int fft_real_to_complex(const double *x_real, int n, double _Complex *X_out);

/* Inverse FFT. Writes (1/n)·Re(IFFT(X_in)) into x_out. Returns 0 on
 * success. */
int fft_complex_to_real(const double _Complex *X_in, int n, double *x_out);

/* 2D FFT via separable 1D transforms (row-then-column). Both nx, ny
 * must be powers of two. Buffers are row-major of shape (nx, ny):
 * index(i, j) = i * ny + j. In-place. */
int fft2_complex_inplace(double _Complex *x, int nx, int ny, int inverse);

/* Convenience: forward 2D FFT of a real field; inverse 2D FFT to a
 * real field, dividing by (nx * ny). */
int fft2_real_to_complex(const double *x_real, int nx, int ny,
                          double _Complex *X_out);
int fft2_complex_to_real(const double _Complex *X_in, int nx, int ny,
                          double *x_out);

#ifdef __cplusplus
}
#endif

#endif /* NEURAL_OPERATOR_FFT_H */
