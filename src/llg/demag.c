/*
 * src/llg/demag.c
 *
 * Demagnetisation field via FFT convolution. The k-space zz kernel
 * is the Fourier transform of the real-space dipolar Green's
 * function G_zz(r) = (3 z² − r²) / r⁵.  For a 2D lattice of
 * thickness-zero dipoles we take z = 0 and the kernel becomes
 *
 *     G_zz(r) = 1/r³        r > 0 ; G_zz(0) = 0.
 *
 * Stored as N_zz(k) = FFT[G_zz](k). The demag field at site r is
 *     H_dz(r) = -Σ_{r'} G_zz(r - r') m_z(r')
 * (minus sign: demag opposes). In k-space: H_dz(k) = -N_zz(k) M_z(k).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "llg/demag.h"
#include "neural_operator/fft.h"

llg_demag_2d_t *llg_demag_2d_create(int Lx, int Ly) {
    if (Lx <= 0 || Ly <= 0) return NULL;
    if (!fft_is_power_of_two(Lx) || !fft_is_power_of_two(Ly)) return NULL;
    llg_demag_2d_t *d = calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->Lx = Lx; d->Ly = Ly;
    d->N_zz = calloc((size_t)Lx * Ly, sizeof(double));
    if (!d->N_zz) { free(d); return NULL; }

    /* Build the real-space Green's function with minimum-image
     * distance under periodic BCs. */
    double *G = calloc((size_t)Lx * Ly, sizeof(double));
    for (int x = 0; x < Lx; x++) {
        for (int y = 0; y < Ly; y++) {
            int dx = x;              if (dx > Lx / 2) dx -= Lx;
            int dy = y;              if (dy > Ly / 2) dy -= Ly;
            double r2 = (double)dx * dx + (double)dy * dy;
            if (r2 == 0.0) { G[x * Ly + y] = 0.0; continue; }
            double r  = sqrt(r2);
            G[x * Ly + y] = 1.0 / (r2 * r);
        }
    }
    /* FFT the Green's function; store real part (G is real+even so
     * N_zz is real). */
    double _Complex *Gk = malloc((size_t)Lx * Ly * sizeof(double _Complex));
    fft2_real_to_complex(G, Lx, Ly, Gk);
    for (long k = 0; k < (long)Lx * Ly; k++) d->N_zz[k] = creal(Gk[k]);
    free(Gk); free(G);
    return d;
}

void llg_demag_2d_free(llg_demag_2d_t *d) {
    if (!d) return;
    free(d->N_zz);
    free(d);
}

int llg_demag_2d_apply_z(const llg_demag_2d_t *d,
                          const double *m, double *out_Hdz) {
    if (!d || !m || !out_Hdz) return -1;
    int Lx = d->Lx, Ly = d->Ly;
    long N = (long)Lx * Ly;
    double *mz = malloc((size_t)N * sizeof(double));
    double _Complex *Mk = malloc((size_t)N * sizeof(double _Complex));
    for (long k = 0; k < N; k++) mz[k] = m[3 * k + 2];
    fft2_real_to_complex(mz, Lx, Ly, Mk);
    for (long k = 0; k < N; k++) Mk[k] *= -d->N_zz[k];  /* H_dz = -N · m_z */
    fft2_complex_to_real(Mk, Lx, Ly, out_Hdz);
    free(mz); free(Mk);
    return 0;
}
