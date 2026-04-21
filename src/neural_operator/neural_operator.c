/*
 * src/neural_operator/neural_operator.c
 *
 * Naive reference implementation of a 1D Fourier Neural Operator layer.
 * O(n²) DFT / IDFT by definition — fine for n ≤ 256 and keeps the
 * translation unit free of FFT dependencies. v0.5 swaps the DFT pair
 * for KissFFT or FFTW behind the same public signatures.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "neural_operator/neural_operator.h"
#include "neural_operator/fft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

fno_layer_t *fno_layer_create(int n, int num_modes) {
    if (n <= 0 || num_modes <= 0) return NULL;
    int cap = n / 2 + 1;
    if (num_modes > cap) num_modes = cap;
    fno_layer_t *L = calloc(1, sizeof(*L));
    if (!L) return NULL;
    L->n = n;
    L->num_modes = num_modes;
    L->weights = calloc((size_t)num_modes, sizeof(double _Complex));
    if (!L->weights) { free(L); return NULL; }
    for (int k = 0; k < num_modes; k++) {
        L->weights[k] = 1.0 + 0.0 * _Complex_I;  /* identity pass-through */
    }
    return L;
}

void fno_layer_free(fno_layer_t *L) {
    if (!L) return;
    free(L->weights);
    free(L);
}

void fno_dft_real(const double *in, int n, double _Complex *out) {
    if (!in || !out || n <= 0) return;
    for (int k = 0; k < n; k++) {
        double re = 0.0, im = 0.0;
        double arg0 = -2.0 * M_PI * (double)k / (double)n;
        for (int j = 0; j < n; j++) {
            double arg = arg0 * (double)j;
            re += in[j] * cos(arg);
            im += in[j] * sin(arg);
        }
        out[k] = re + im * _Complex_I;
    }
}

void fno_idft(const double _Complex *in, int n, double *out) {
    if (!in || !out || n <= 0) return;
    double inv_n = 1.0 / (double)n;
    for (int j = 0; j < n; j++) {
        double re = 0.0;
        double arg0 = 2.0 * M_PI * (double)j / (double)n;
        for (int k = 0; k < n; k++) {
            double arg = arg0 * (double)k;
            double c = cos(arg), s = sin(arg);
            re += creal(in[k]) * c - cimag(in[k]) * s;
        }
        out[j] = re * inv_n;
    }
}

int fno_layer_fit(fno_layer_t *L,
                   const double *phi_batch, const double *psi_batch,
                   int num_pairs, double reg) {
    if (!L || !phi_batch || !psi_batch || num_pairs <= 0) return -1;
    int n = L->n;
    int M = L->num_modes;
    double _Complex *num   = calloc((size_t)n, sizeof(double _Complex));
    double          *denom = calloc((size_t)n, sizeof(double));
    double _Complex *Phi = malloc((size_t)n * sizeof(double _Complex));
    double _Complex *Psi = malloc((size_t)n * sizeof(double _Complex));
    if (!num || !denom || !Phi || !Psi) {
        free(num); free(denom); free(Phi); free(Psi); return -1;
    }
    /* Accumulate per-k sums across the training batch. */
    for (int i = 0; i < num_pairs; i++) {
        const double *phi = &phi_batch[(size_t)i * (size_t)n];
        const double *psi = &psi_batch[(size_t)i * (size_t)n];
        if (fft_is_power_of_two(n)) {
            fft_real_to_complex(phi, n, Phi);
            fft_real_to_complex(psi, n, Psi);
        } else {
            fno_dft_real(phi, n, Phi);
            fno_dft_real(psi, n, Psi);
        }
        for (int k = 0; k < n; k++) {
            num[k]   += conj(Phi[k]) * Psi[k];
            denom[k] += creal(Phi[k] * conj(Phi[k]));   /* |Φ|² */
        }
    }
    /* Optimal W(k) for the retained low-frequency block. */
    for (int k = 0; k < M; k++) {
        if (denom[k] < reg) continue;    /* keep existing weight */
        L->weights[k] = num[k] / denom[k];
    }
    free(num); free(denom); free(Phi); free(Psi);
    return 0;
}

/* -------------------------- 2D FNO ----------------------------------- */

fno_layer_2d_t *fno_layer_2d_create(int nx, int ny,
                                      int num_modes_x, int num_modes_y) {
    if (nx <= 0 || ny <= 0 || num_modes_x <= 0 || num_modes_y <= 0) return NULL;
    if (num_modes_x > nx / 2 + 1) num_modes_x = nx / 2 + 1;
    if (num_modes_y > ny / 2 + 1) num_modes_y = ny / 2 + 1;
    fno_layer_2d_t *L = calloc(1, sizeof(*L));
    if (!L) return NULL;
    L->nx = nx; L->ny = ny;
    L->num_modes_x = num_modes_x;
    L->num_modes_y = num_modes_y;
    /* Store weights for 4 quadrants (low-low, low-high, high-low,
     * high-high) in a flat buffer indexed by (qx·Mx + mx, qy·My + my)
     * where qx, qy ∈ {0, 1}. */
    /* size_t product is overflow-safe on every supported target (LP64 and
     * LLP64); long on LLP64 is 32-bit and could overflow for very large
     * mode counts. */
    size_t nw = (size_t)4 * (size_t)num_modes_x * (size_t)num_modes_y;
    L->weights = calloc(nw, sizeof(double _Complex));
    if (!L->weights) { free(L); return NULL; }
    for (size_t i = 0; i < nw; i++) L->weights[i] = 1.0 + 0.0 * _Complex_I;
    return L;
}

void fno_layer_2d_free(fno_layer_2d_t *L) {
    if (!L) return;
    free(L->weights);
    free(L);
}

/* Quadrant-aware weight lookup. Returns 1+0i when the (kx, ky) bin is
 * in a retained quadrant, else 0+0i (truncated). The quadrant layout
 * is a 2-block structure:
 *      qx=0: kx ∈ [0, Mx)
 *      qx=1: kx ∈ (nx - Mx, nx)   mapped to the "high" block
 *   weights[ (qx · Mx + mx) · (2 · My) + (qy · My + my) ] */
static double _Complex fno2d_weight_for(const fno_layer_2d_t *L,
                                         int kx, int ky) {
    int Mx = L->num_modes_x, My = L->num_modes_y;
    int nx = L->nx, ny = L->ny;
    int qx, mx;
    if (kx < Mx) { qx = 0; mx = kx; }
    else if (kx > nx - Mx && kx < nx) { qx = 1; mx = kx - (nx - Mx); }
    else return 0.0 + 0.0 * _Complex_I;
    int qy, my;
    if (ky < My) { qy = 0; my = ky; }
    else if (ky > ny - My && ky < ny) { qy = 1; my = ky - (ny - My); }
    else return 0.0 + 0.0 * _Complex_I;
    long idx = ((long)(qx * Mx + mx)) * (2L * My) + (long)(qy * My + my);
    return L->weights[idx];
}

int fno_layer_2d_apply(const fno_layer_2d_t *L,
                         const double *in, double *out) {
    if (!L || !in || !out) return -1;
    int nx = L->nx, ny = L->ny;
    long n = (long)nx * ny;
    double _Complex *F = calloc((size_t)n, sizeof(double _Complex));
    if (!F) return -1;
    if (fft2_real_to_complex(in, nx, ny, F) != 0) { free(F); return -1; }
    /* Apply weights per (kx, ky). */
    for (int kx = 0; kx < nx; kx++) {
        for (int ky = 0; ky < ny; ky++) {
            double _Complex w = fno2d_weight_for(L, kx, ky);
            F[(long)kx * ny + ky] *= w;
        }
    }
    if (fft2_complex_to_real(F, nx, ny, out) != 0) { free(F); return -1; }
    free(F);
    return 0;
}

int fno_layer_apply(const fno_layer_t *L, const double *in, double *out) {
    if (!L || !in || !out) return -1;
    int n = L->n;
    double _Complex *F = calloc((size_t)n, sizeof(double _Complex));
    if (!F) return -1;
    /* Use radix-2 FFT when n is a power of two (O(n log n)); the
     * naive O(n²) DFT is the fallback for arbitrary sizes. */
    if (fft_is_power_of_two(n)) {
        fft_real_to_complex(in, n, F);
    } else {
        fno_dft_real(in, n, F);
    }

    /* Multiply low-mode coefficients by learned weights; zero the rest
     * (spectral truncation — the operator keeps only the first
     * `num_modes` frequencies, matching Li et al. 2021). Hermitian
     * symmetry F[n-k] = conj(F[k]) is preserved by also scaling the
     * mirrored high-frequency bin. */
    for (int k = 0; k < n; k++) {
        if (k < L->num_modes) {
            F[k] = F[k] * L->weights[k];
        } else if (k > n - L->num_modes && k < n) {
            /* mirror: corresponds to frequency n-k ∈ [1, num_modes-1] */
            int km = n - k;
            F[k] = F[k] * conj(L->weights[km]);
        } else {
            F[k] = 0.0 + 0.0 * _Complex_I;
        }
    }

    if (fft_is_power_of_two(n)) {
        fft_complex_to_real(F, n, out);
    } else {
        fno_idft(F, n, out);
    }
    free(F);
    return 0;
}
