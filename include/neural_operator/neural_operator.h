/*
 * include/neural_operator/neural_operator.h
 *
 * Fourier-Neural-Operator spectral-convolution layer for v0.5 pillar
 * P1.4. v0.4 ships a compact reference implementation that works on
 * 1D real signals: forward DFT → pointwise multiply by learned complex
 * weights (truncated to the lowest `num_modes` Fourier modes) →
 * inverse DFT. Scales to 3D with Tucker decomposition in v0.5+.
 *
 *     (K φ)(x) = IDFT[ W(k) · DFT[φ](k) ]
 *
 * This is the core of Li et al. 2021 (ICLR) and the spectral backbone
 * of the NeuralMAG micromagnetic surrogate.
 */
#ifndef NEURAL_OPERATOR_H
#define NEURAL_OPERATOR_H

#include <complex.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n;                    /* signal length */
    int num_modes;            /* Fourier modes kept (≤ n/2 + 1) */
    double _Complex *weights; /* learned complex weights, length num_modes */
} fno_layer_t;

/* Allocate an FNO layer over signals of length n, keeping num_modes
 * lowest Fourier components. Weights are initialised to 1+0i (identity
 * pass-through). */
fno_layer_t *fno_layer_create(int n, int num_modes);
void         fno_layer_free  (fno_layer_t *L);

/* Naive O(n²) DFT / inverse DFT on a real input. No FFT dependency in
 * v0.4 — adequate for n ≤ 256, upgraded to FFTW / KissFFT in v0.5. */
void fno_dft_real (const double *in, int n, double _Complex *out);
void fno_idft     (const double _Complex *in, int n, double *out);

/* Apply the layer to a real 1D signal in-place: out[n] ← K · in[n]. */
int fno_layer_apply(const fno_layer_t *L, const double *in, double *out);

/* Closed-form least-squares fit: given `num_pairs` input/output
 * trajectories `{phi_i, psi_i}` (each a row-major length-n signal),
 * set each spectral weight W(k) to the MSE-optimal value
 *     W(k) = Σ_i F[phi_i](k)* · F[psi_i](k) / Σ_i |F[phi_i](k)|²
 * For a pure spectral-conv layer, this minimises the sum-of-squares
 * error across the batch; no gradient descent needed. When the
 * denominator is below `reg` (no signal at that frequency), the
 * corresponding weight stays at its previous value (ridge-like
 * regularisation). Returns 0 on success. */
int fno_layer_fit(fno_layer_t *L,
                   const double *phi_batch, const double *psi_batch,
                   int num_pairs, double reg);

/* 2D FNO spectral-convolution layer.
 *
 *   (K φ)(x, y) = IFFT₂[ W(kx, ky) · FFT₂[φ](kx, ky) ]
 *
 * Weights are stored only for the retained modes:
 *   kx ∈ [0, num_modes_x)  or  (nx - num_modes_x + 1, nx)   (mirrors)
 *   ky ∈ [0, num_modes_y)  or  (ny - num_modes_y + 1, ny)   (mirrors)
 * giving 4·num_modes_x·num_modes_y - 2(num_modes_x + num_modes_y) + 1
 * non-trivial complex weights (at most). The layer keeps the weights
 * for the four quadrants in a flat buffer of length 4·M_x·M_y and the
 * apply routine zeroes the truncated high-frequency tail. */
typedef struct {
    int nx, ny;
    int num_modes_x;
    int num_modes_y;
    double _Complex *weights;   /* length 4 · num_modes_x · num_modes_y */
} fno_layer_2d_t;

fno_layer_2d_t *fno_layer_2d_create(int nx, int ny,
                                      int num_modes_x, int num_modes_y);
void            fno_layer_2d_free (fno_layer_2d_t *L);

/* Apply 2D FNO layer: out[nx,ny] ← K · in[nx,ny]. Both buffers are
 * row-major of shape (nx, ny). Returns 0 on success. */
int fno_layer_2d_apply(const fno_layer_2d_t *L,
                         const double *in, double *out);

#ifdef __cplusplus
}
#endif

#endif /* NEURAL_OPERATOR_H */
