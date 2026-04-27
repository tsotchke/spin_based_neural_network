/*
 * include/equivariant_gnn/torque_net_irrep.h
 *
 * libirrep-backed NequIP-tower torque network for the LLG integrator.
 *
 * Replaces the homegrown 9-term Cartesian basis (torque_net.h) with a
 * proper E(3)-equivariant tensor-product layer:
 *
 *   1. Edge embedding via real spherical harmonics Y_l^m(r̂_ij) up to
 *      l_sh_max (default 2 — captures up to L=2 quadrupolar features
 *      from the geometry side, vs. the homegrown basis's manual
 *      cross/dot/triple products).
 *   2. Bessel-RBF radial basis φ_n(r_ij) on the bond length, n_radial
 *      learned scalar channels.
 *   3. Polynomial or cosine cutoff smoothly to zero at r_cut.
 *   4. Path-indexed UVW tensor product of h_j with Y(r̂_ij), producing
 *      one edge message per neighbour pair.
 *   5. Aggregation Σ_j on the destination node, scaled by RBF · cutoff.
 *
 * Gated behind SPIN_NN_HAS_IRREP=1.  Build with libirrep available
 * via the vendored submodule:
 *
 *     git submodule update --init --recursive
 *     make IRREP_ENABLE=1 test_torque_net_irrep
 *
 * The graph format is the same `torque_net_graph_t` as the homegrown
 * torque_net (so the LLG integrator wiring carries over without
 * change).  The hidden-feature layout is determined by the irrep spec
 * passed at construction time:
 *   "1x1o -> 1x1o"           one l=1 vector per node, mapping m → τ
 *   "1x1o -> 1x0e + 1x1o"    output a scalar (e.g. energy density)
 *                            alongside the torque
 *
 * The library accepts e3nn-style multiset strings; see libirrep
 * `irrep_multiset_from_spec` for the grammar.
 */
#ifndef TORQUE_NET_IRREP_H
#define TORQUE_NET_IRREP_H

#include "equivariant_gnn/torque_net.h"     /* torque_net_graph_t */

#ifdef __cplusplus
extern "C" {
#endif

#ifdef SPIN_NN_HAS_IRREP

typedef struct torque_net_irrep torque_net_irrep_t;

/* Construct a NequIP-backed layer.  hidden_in_spec / hidden_out_spec
 * are e3nn-style irrep multisets ("1x1o" for one l=1 odd vector,
 * "2x0e + 1x1o" for two scalars + one vector, etc.).
 *
 * Defaults if you pass NULL spec strings: "1x1o" -> "1x1o".
 *
 * Returns NULL on construction failure (mismatched spec, bad params,
 * libirrep error). */
torque_net_irrep_t *torque_net_irrep_create(const char *hidden_in_spec,
                                             const char *hidden_out_spec,
                                             int    l_sh_max,
                                             int    n_radial,
                                             double r_cut,
                                             int    cutoff_poly_p);

void torque_net_irrep_free(torque_net_irrep_t *net);

/* Number of learnable TP weights this layer expects.  Allocate a
 * weight buffer of this length and pass it to forward / backward. */
int torque_net_irrep_num_weights(const torque_net_irrep_t *net);

/* Per-node hidden-feature dimension for the input / output multisets.
 * For "1x1o" both are 3.  For "2x0e + 1x1o" the dim is 2 + 3 = 5. */
int torque_net_irrep_in_dim(const torque_net_irrep_t *net);
int torque_net_irrep_out_dim(const torque_net_irrep_t *net);

/* Forward: h_in is [num_nodes * in_dim], h_out is [num_nodes * out_dim].
 * h_out is zeroed internally.  Edge geometry comes from the same
 * torque_net_graph_t as the homegrown net.  Returns 0 on success. */
int torque_net_irrep_forward(const torque_net_irrep_t *net,
                              const double *weights,
                              const torque_net_graph_t *graph,
                              const double *h_in,
                              double *h_out);

/* Backward through hidden features and weights.  grad_h_in and
 * grad_tp_weights are accumulated (+=); caller pre-zeros. */
int torque_net_irrep_backward(const torque_net_irrep_t *net,
                               const double *weights,
                               const torque_net_graph_t *graph,
                               const double *h_in,
                               const double *grad_h_out,
                               double *grad_h_in,
                               double *grad_weights);

#endif /* SPIN_NN_HAS_IRREP */

#ifdef __cplusplus
}
#endif

#endif /* TORQUE_NET_IRREP_H */
