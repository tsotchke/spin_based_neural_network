/*
 * include/nqs/nqs_config.h
 *
 * Compile-time and runtime configuration for the Neural Network Quantum
 * States (NQS) pillar (v0.5 P1.1). An NQS represents a quantum
 * wavefunction ψ(s) on spin configurations s as a parametric neural
 * network evaluated on patch-tokenised lattice input.
 *
 * v0.4 ships this header with the data structures and the ansatz /
 * sampler / local-energy / optimizer interfaces. The concrete
 * transformer ansatz is backed by the legacy MLP today; the full
 * factored-ViT implementation lands with the external NN engine
 * (eshkol-transformers) in v0.5+.
 */
#ifndef NQS_CONFIG_H
#define NQS_CONFIG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Ansatz families --------------------------------------------------- */
typedef enum {
    NQS_ANSATZ_LEGACY_MLP   = 0,  /* v0.4: flat MLP; always available.     */
    NQS_ANSATZ_VIT          = 1,  /* v0.5 target: vision-transformer.      */
    NQS_ANSATZ_FACTORED_VIT = 2,  /* v0.5 target: symmetry-factored ViT.   */
    NQS_ANSATZ_AUTOREGRESSIVE = 3,/* v0.5 target: gauge-invariant AR NQS.  */
    NQS_ANSATZ_KAN          = 4,  /* v0.6 target: Kolmogorov-Arnold.       */
    NQS_ANSATZ_RBM          = 5,  /* Real-valued Restricted Boltzmann
                                     wavefunction (Carleo & Troyer 2017). */
    NQS_ANSATZ_COMPLEX_RBM  = 6   /* Complex-valued RBM: non-stoquastic
                                     / frustrated systems; parameters
                                     are (real, imag) pairs. */
} nqs_ansatz_kind_t;

/* --- Symmetry groups applied to the wavefunction amplitude ------------- */
typedef enum {
    NQS_SYM_NONE             = 0,
    NQS_SYM_TRANSLATION      = 1 << 0,
    NQS_SYM_SPIN_FLIP        = 1 << 1,  /* Z_2 up <-> down                 */
    NQS_SYM_U1               = 1 << 2,  /* magnetisation conservation      */
    NQS_SYM_POINT_GROUP      = 1 << 3,  /* C_4v etc (v0.5+)                */
    NQS_SYM_SU2              = 1 << 4   /* full rotational (v0.5+)         */
} nqs_symmetry_flags_t;

/* --- Hamiltonian the local-energy estimator evaluates against ---------- */
typedef enum {
    NQS_HAM_TFIM             = 0,       /* transverse-field Ising          */
    NQS_HAM_HEISENBERG       = 1,       /* Heisenberg                      */
    NQS_HAM_J1_J2            = 2,       /* frustrated square lattice       */
    NQS_HAM_KITAEV_HONEYCOMB = 3,       /* Kitaev honeycomb (exact at isotropic) */
    NQS_HAM_XXZ              = 4        /* XXZ: J(SxSx + SySy) + Jz SzSz   */
} nqs_hamiltonian_kind_t;

/* --- Top-level configuration struct ----------------------------------- */
typedef struct {
    /* Ansatz */
    nqs_ansatz_kind_t    ansatz;
    int                   patch_size;        /* e.g. 2 for 2x2 patches     */
    int                   depth;             /* transformer layers         */
    int                   width;             /* embedding dim              */
    int                   heads;             /* attention heads            */
    int                   rbm_hidden_units;  /* M in the RBM ansatz        */
    double                rbm_init_scale;    /* weight init std-dev        */
    nqs_symmetry_flags_t  symmetries;

    /* Hamiltonian */
    nqs_hamiltonian_kind_t hamiltonian;
    double                 j_coupling;       /* J_1 (Heisenberg: J; XXZ: Jxy) */
    double                 j2_coupling;      /* J_2 for J1-J2              */
    double                 j_z_coupling;     /* Jz for XXZ (unused elsewhere) */
    double                 transverse_field; /* Γ for TFIM                 */

    /* Sampling */
    int     num_samples;           /* Metropolis samples per gradient step */
    int     num_thermalize;        /* burn-in steps                        */
    int     num_decorrelate;       /* steps between recorded samples       */
    int     cluster_moves;         /* 1 = enable Swendsen-Wang-like moves  */
    unsigned rng_seed;

    /* Optimisation */
    int     num_iterations;
    double  learning_rate;
    double  sr_diag_shift;         /* Tikhonov shift on the QGT            */
    int     sr_cg_max_iters;       /* CG iterations inside stochastic-reconf */
    double  sr_cg_tol;
} nqs_config_t;

/* Return a configuration with reasonable defaults for a 6x6 TFIM. */
static inline nqs_config_t nqs_config_defaults(void) {
    nqs_config_t c;
    c.ansatz            = NQS_ANSATZ_LEGACY_MLP;
    c.patch_size        = 1;
    c.depth             = 3;
    c.width             = 128;
    c.heads             = 4;
    c.rbm_hidden_units  = 0;      /* 0 → default to 2 × num_sites */
    c.rbm_init_scale    = 0.01;
    c.symmetries        = (nqs_symmetry_flags_t)(NQS_SYM_SPIN_FLIP | NQS_SYM_TRANSLATION);
    c.hamiltonian       = NQS_HAM_TFIM;
    c.j_coupling        = 1.0;
    c.j2_coupling       = 0.0;
    c.j_z_coupling      = 1.0;          /* XXZ: default to isotropic */
    c.transverse_field  = 1.0;
    c.num_samples       = 1024;
    c.num_thermalize    = 256;
    c.num_decorrelate   = 4;
    c.cluster_moves     = 0;
    c.rng_seed          = 0xC0FFEEu;
    c.num_iterations    = 200;
    c.learning_rate     = 1e-2;
    c.sr_diag_shift     = 1e-3;
    c.sr_cg_max_iters   = 100;
    c.sr_cg_tol         = 1e-6;
    return c;
}

#ifdef __cplusplus
}
#endif

#endif /* NQS_CONFIG_H */
