/*
 * include/llg/demag.h
 *
 * Demagnetisation field for a 2D micromagnetic sample. In free space
 * the magnetostatic potential satisfies
 *     ∇² Φ_d = ∇·M
 * so the demag field H_d = -∇Φ_d is given by the convolution of the
 * dipolar Green's function with the magnetisation. On a periodic (or
 * zero-padded) grid this is O(N log N) via FFT:
 *
 *     H_d(k) = -N_tensor(k) · M(k)
 *
 * where N_tensor(k) is the 3x3 k-space demag tensor. This module
 * computes the z-component of H_d for a 2D (Lx × Ly) lattice under
 * periodic boundary conditions — the in-plane components come out
 * of the same machinery but are not needed for thin-film samples
 * with out-of-plane anisotropy.
 *
 * The model used here is a 1/r³ dipole-dipole kernel truncated at
 * a cutoff radius R_cut. For small systems this is equivalent to
 * direct summation and establishes a reference the FFT path can be
 * checked against.
 */
#ifndef LLG_DEMAG_H
#define LLG_DEMAG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Precomputed Green's function lookup for a 2D (Lx × Ly) lattice
 * with periodic boundary conditions. Stores the k-space demag
 * tensor components needed for a thin-film geometry with M
 * predominantly out-of-plane. */
typedef struct {
    int Lx, Ly;
    /* K-space demag kernel entries for the zz component, one per
     * (kx, ky) bin on the (Lx × Ly) grid, row-major. */
    double *N_zz;
} llg_demag_2d_t;

/* Build the demag kernel for a 2D thin-film sample of dimensions
 * (Lx × Ly) sites. Cell spacing is assumed uniform. Returns NULL on
 * failure. Caller frees via llg_demag_2d_free. */
llg_demag_2d_t *llg_demag_2d_create(int Lx, int Ly);
void            llg_demag_2d_free  (llg_demag_2d_t *d);

/* Compute the z-component of the demag field:
 *   H_dz[i] = -Σ_j N_zz(i, j) · m_z[j]
 * via 2D FFT convolution. `m` is the magnetisation in the (Lx, Ly, 3)
 * layout (Lx × Ly sites, 3 components per site). `out_Hdz` is length
 * Lx × Ly. Returns 0 on success. */
int llg_demag_2d_apply_z(const llg_demag_2d_t *d,
                          const double *m, double *out_Hdz);

#ifdef __cplusplus
}
#endif

#endif /* LLG_DEMAG_H */
