/*
 * include/llg/exchange_field.h
 *
 * Ready-made effective-field callbacks for the LLG integrator. v0.4
 * ships a 1D Heisenberg exchange field (nearest-neighbour, periodic
 * boundary conditions) and a uniform Zeeman field. Both satisfy the
 * llg_effective_field_fn_t signature and plug directly into
 * llg_config_t::field_fn.
 *
 *   Heisenberg:   E = -J Σ_<ij> m_i · m_j   →   B_eff,i = J (m_{i-1} + m_{i+1})
 *   Zeeman:       E = -Σ_i m_i · B_ext      →   B_eff,i = B_ext (constant)
 *
 * Additional modules (uniaxial anisotropy, DMI, demag) land with the
 * equivariant-GNN replacement in pillar P1.2.
 */
#ifndef LLG_EXCHANGE_FIELD_H
#define LLG_EXCHANGE_FIELD_H

#include "llg/llg.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double J;           /* exchange coupling, J > 0 → ferromagnet            */
    double Bx, By, Bz;  /* uniform Zeeman field, added to every site         */
    /* Uniaxial anisotropy along z:   E_aniso = -K_z Σ_i (m_z)²
     *   ∂E/∂m_z = -2 K_z m_z   →   B_eff_z contribution = +2 K_z m_z.
     * K_z > 0 → easy-axis along z (favours m ∥ ẑ or ∥ -ẑ).
     * K_z < 0 → easy-plane (favours in-plane orientation). */
    double Kz;
    /* 1D Dzyaloshinskii-Moriya interaction along z:
     *   E_DMI = D Σ_i ẑ · (m_i × m_{i+1}) = D Σ_i (m_i_x m_{i+1}_y - m_i_y m_{i+1}_x)
     * Open boundary conditions use whichever neighbour exists; periodic
     * boundary conditions (the default for llg_heisenberg_1d_field)
     * include both (i-1) and (i+1). */
    double D_dmi;
} llg_heisenberg_1d_t;

/* Heisenberg + uniform Zeeman on a 1D chain with periodic boundary
 * conditions. Signature matches llg_effective_field_fn_t. */
void llg_heisenberg_1d_field(const double *m, double *b_eff,
                              long num_sites, void *user_data);

/* Full 2D LLG effective field: exchange (4 nearest neighbours),
 * uniaxial anisotropy along z, interfacial DMI (the variant that
 * stabilises Néel-type skyrmions), and uniform Zeeman field. The
 * demagnetising field (non-local dipolar) is added separately via
 * llg_demag_2d_apply_z() and composed with this field in the
 * caller's sweep. */
typedef struct {
    int Lx, Ly;                      /* lattice shape; sites stored in
                                        row-major (Lx × Ly) order */
    double J;                         /* Heisenberg exchange, J > 0 → FM  */
    double Kz;                        /* easy-axis uniaxial anisotropy     */
    double D_dmi;                     /* interfacial DMI strength          */
    double Bx, By, Bz;                /* uniform applied field             */
} llg_2d_config_t;

/* Signature matches llg_effective_field_fn_t. Expects num_sites =
 * cfg->Lx * cfg->Ly and `m` in length-3 per-site layout. */
void llg_2d_field(const double *m, double *b_eff,
                   long num_sites, void *user_data);

#ifdef __cplusplus
}
#endif

#endif /* LLG_EXCHANGE_FIELD_H */
