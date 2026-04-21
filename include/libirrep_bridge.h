/*
 * include/libirrep_bridge.h
 *
 * Lazy bridge to the libirrep library (SO(3) / SU(2) / O(3) irrep
 * machinery — Clebsch-Gordan, spherical harmonics, Wigner-D, tensor
 * products). libirrep supplies the math backbone that the v0.5
 * equivariant-LLG pillar (P1.2) and Fibonacci-anyon universal gates
 * (P1.3b) both need.
 *
 * The bridge is engine-neutral in the same sense as engine_adapter
 * and eshkol_bridge: it compiles without libirrep present (every entry
 * point returns IRREP_BRIDGE_EDISABLED) and only calls into the real
 * library when -DSPIN_NN_HAS_IRREP=1 is set at build time. Enable via
 *
 *     make IRREP_ENABLE=1 IRREP_ROOT=/path/to/libirrep/install
 */
#ifndef LIBIRREP_BRIDGE_H
#define LIBIRREP_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    IRREP_BRIDGE_OK         =  0,
    IRREP_BRIDGE_EDISABLED  = -1, /* built without SPIN_NN_HAS_IRREP   */
    IRREP_BRIDGE_ENOT_READY = -2, /* libirrep_bridge_init() not called */
    IRREP_BRIDGE_ELIB       = -3, /* underlying library error          */
    IRREP_BRIDGE_EARG       = -4
} libirrep_bridge_status_t;

/* Refcounted init/shutdown (mirrors the engine_adapter pattern). */
int libirrep_bridge_init(void);
int libirrep_bridge_shutdown(void);

/* 1 iff built with SPIN_NN_HAS_IRREP and the library linked in. */
int libirrep_bridge_is_available(void);

/* Returns the libirrep version string (e.g. "0.3.0") when built in,
 * or NULL otherwise. Pointer to static storage; do not free. */
const char *libirrep_bridge_version(void);

/* Evaluate a real spherical harmonic Y_l^m(theta, phi). Signs follow
 * the Condon-Shortley phase convention. Returns IRREP_BRIDGE_OK and
 * writes the result to *out. */
int libirrep_bridge_sph_harm_real(int l, int m,
                                  double theta, double phi,
                                  double *out);

/* Evaluate a Clebsch-Gordan coefficient <j1 m1 ; j2 m2 | J M>. All
 * arguments are integers (half-integer angular momenta in v0.4 are
 * represented via the 2j convention in the caller — see libirrep's
 * clebsch_gordan.h for the doubled-integer API). */
int libirrep_bridge_clebsch_gordan(int j1, int m1,
                                   int j2, int m2,
                                   int J, int M,
                                   double *out);

/* Evaluate a (small-d) Wigner-d element d^j_{m',m}(beta) for integer j. */
int libirrep_bridge_wigner_d_small(int j, int mp, int m, double beta,
                                   double *out);

/* Return the total block dimension of an e3nn-style multiset spec
 * (e.g. "1x0e + 1x1o + 1x2e" → 1 + 3 + 5 = 9). Useful for allocating
 * per-node feature buffers without instantiating a multiset. */
int libirrep_bridge_multiset_dim(const char *spec, int *out_dim);

/* ---- NequIP-style E(3)-equivariant message-passing layer ------------
 *
 * Thin wrapper around libirrep's NequIP layer. Hides the multiset and
 * tensor-product types behind e3nn-style string specs. A single layer
 * runs one round of
 *     h_out[i] = Σ_{j ∈ nbrs(i)} TP(h_in[j], Y(r̂_ij)) · φ(r_ij)
 * where Y are cartesian real spherical harmonics up to `l_sh_max`,
 * φ is an n_radial Bessel RBF with polynomial cutoff at r_cut, and
 * TP is the weighted uvw tensor product.
 *
 * Stack several layers + a final l=1 readout to build an equivariant
 * torque/force predictor (pillar P1.2 LLG integrator).
 */
typedef struct libirrep_bridge_nequip libirrep_bridge_nequip_t;

/* Build a single NequIP layer given in/out irrep specs, spherical-
 * harmonic order, radial basis size, cutoff radius, and cutoff-
 * polynomial order (0 → cosine cutoff). */
int libirrep_bridge_nequip_build(const char *hidden_in_spec,
                                  int l_sh_max,
                                  int n_radial,
                                  double r_cut,
                                  int cutoff_poly_p,
                                  const char *hidden_out_spec,
                                  libirrep_bridge_nequip_t **out_layer);

/* Free a layer created by `_build`. Safe to call with NULL. */
int libirrep_bridge_nequip_free(libirrep_bridge_nequip_t *layer);

/* Number of learnable tp weights this layer expects. */
int libirrep_bridge_nequip_num_weights(const libirrep_bridge_nequip_t *layer,
                                        int *out_num_weights);

/* Forward pass. Caller pre-zeros h_out. */
int libirrep_bridge_nequip_apply(const libirrep_bridge_nequip_t *layer,
                                  const double *tp_weights,
                                  int n_nodes, int n_edges,
                                  const int *edge_src,
                                  const int *edge_dst,
                                  const double *edge_vec,
                                  const double *h_in,
                                  double *h_out);

/* Backward pass. Caller pre-zeros grad_h_in and grad_tp_weights. */
int libirrep_bridge_nequip_apply_backward(const libirrep_bridge_nequip_t *layer,
                                           const double *tp_weights,
                                           int n_nodes, int n_edges,
                                           const int *edge_src,
                                           const int *edge_dst,
                                           const double *edge_vec,
                                           const double *h_in,
                                           const double *grad_h_out,
                                           double *grad_h_in,
                                           double *grad_tp_weights);

#ifdef __cplusplus
}
#endif

#endif /* LIBIRREP_BRIDGE_H */
