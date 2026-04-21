/*
 * include/mps/mps.h
 *
 * Matrix-product-state representation and 2-site DMRG driver.
 *
 * v0.4 scaffolds the interface. The concrete tensor-network
 * implementation lands with pillar P2.2 in v0.6+; for now,
 * `mps_ground_state_dense` solves small systems via exact
 * diagonalisation-by-Lanczos on the full Hilbert space, which is
 * enough to validate the framework and to provide a reference
 * ground-state energy for NQS convergence tests.
 *
 * The full tensor-network path (bond dimensions χ, left/right
 * canonicalisation, two-site DMRG sweeps) will ship behind the same
 * `mps_t` handle so call sites don't change when it lands.
 */
#ifndef MPS_MPS_H
#define MPS_MPS_H

#include <stddef.h>
#include "mps/lanczos.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MPS_HAM_TFIM       = 0,   /* H = -J Σ σ^z σ^z - Γ Σ σ^x  */
    MPS_HAM_HEISENBERG = 1,   /* H = J Σ S · S              */
    MPS_HAM_XXZ        = 2    /* H = J (Sx·Sx + Sy·Sy) + Jz Sz·Sz */
} mps_hamiltonian_kind_t;

typedef struct {
    int num_sites;             /* 1D chain length */
    mps_hamiltonian_kind_t ham;
    double J;                  /* nearest-neighbour coupling */
    double Gamma;               /* transverse field (TFIM only) */
    double Jz;                  /* XXZ Jz (XXZ only) */
    /* Reserved for v0.6+ tensor-network DMRG: */
    int    max_bond_dim;        /* target χ */
    int    num_sweeps;
    double sweep_tol;           /* energy convergence threshold */
    /* Lanczos parameters for the eigensolve step: */
    int    lanczos_max_iters;
    double lanczos_tol;
} mps_config_t;

static inline mps_config_t mps_config_defaults(void) {
    mps_config_t c;
    c.num_sites         = 4;
    c.ham               = MPS_HAM_HEISENBERG;
    c.J                 = 1.0;
    c.Gamma             = 1.0;
    c.Jz                = 1.0;
    c.max_bond_dim      = 32;
    c.num_sweeps        = 10;
    c.sweep_tol         = 1e-8;
    c.lanczos_max_iters = 100;
    c.lanczos_tol       = 1e-9;
    return c;
}

/* Ground-state driver that uses a dense Hamiltonian-vector product
 * (valid for num_sites ≤ ~14). Writes the ground-state energy into
 * *out_energy and, when out_state is non-NULL, the eigenvector into a
 * buffer of length 2^num_sites. Returns 0 on success.
 *
 * This is the substrate the v0.6+ tensor-network DMRG will replace
 * behind the same API. */
int mps_ground_state_dense(const mps_config_t *cfg,
                           double *out_energy,
                           double *out_state,
                           lanczos_result_t *out_info);

#ifdef __cplusplus
}
#endif

#endif /* MPS_MPS_H */
