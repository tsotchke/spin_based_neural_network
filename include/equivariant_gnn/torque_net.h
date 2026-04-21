/*
 * include/equivariant_gnn/torque_net.h
 *
 * Minimal E(3)-equivariant torque predictor for the micromagnetic LLG
 * integrator (pillar P1.2). Pure C, no libirrep dependency — uses
 * only geometrically-invariant primitive building blocks (cross,
 * dot, triple product) whose equivariance under SO(3) rotations of
 * the input node features is trivially verifiable.
 *
 * This is the "pragmatic baseline" promised by architecture_v0.4.md
 * §P1.2; a full NequIP / MACE tower using the libirrep NequIP layer
 * lands once libirrep ≥ 1.1 is vendored. The torque predicted here
 * is demonstrably rotation-covariant, meaning it can replace the
 * analytic B_eff in the LLG integrator without breaking the equations'
 * symmetry group.
 *
 * Architecture:
 *   For each node i with vector feature m_i ∈ R³, neighbours j with
 *   m_j, and connecting displacement r_ij ∈ R³:
 *
 *       τ_i = Σ_j [
 *           w0 · (m_j · r̂_ij) · m_i
 *         + w1 · (m_j × r̂_ij)
 *         + w2 · (m_i × m_j)
 *         + w3 · ((m_i · m_j) · m_i)
 *         + w4 · m_j
 *       ] · φ(||r_ij||)
 *
 *   Each term in brackets is a proper vector under SO(3); the radial
 *   modulator φ is a scalar function of the bond length. A linear
 *   combination of SO(3) vectors remains an SO(3) vector, hence τ_i
 *   is equivariant.
 *
 * The five weights w0..w4 are the only learnable parameters (plus an
 * exponent controlling the radial cutoff). A full 2-layer TP tower
 * would generalise this to arbitrary irrep content via libirrep.
 */
#ifndef TORQUE_NET_H
#define TORQUE_NET_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int     num_nodes;
    int     num_edges;
    const int    *edge_src;     /* [num_edges]                  */
    const int    *edge_dst;     /* [num_edges]                  */
    const double *edge_vec;     /* [3 * num_edges]  r_j - r_i   */
} torque_net_graph_t;

typedef struct {
    double w0;                  /* (m_j·r̂) m_i      — polar even      */
    double w1;                  /* m_j × r̂           — axial odd       */
    double w2;                  /* m_i × m_j         — axial odd       */
    double w3;                  /* (m_i·m_j) m_i     — polar even      */
    double w4;                  /* m_j               — polar even      */
    double r_cut;               /* radial cutoff     — scalar          */
    double radial_order;        /* polynomial order for cutoff         */
} torque_net_params_t;

/* Forward pass. Caller pre-allocates `out_torque` of length 3·num_nodes
 * and is responsible for zeroing it if needed (the function always
 * overwrites — no accumulation). */
int torque_net_forward(const torque_net_graph_t *g,
                        const double *m_in, /* [3 * num_nodes] */
                        const torque_net_params_t *p,
                        double *out_torque   /* [3 * num_nodes] */);

/* Sanity check: apply an SO(3) rotation R to the full graph (vectors
 * m_in and edge_vec) and verify the torque rotates by the same R.
 * Caller supplies R as a 3×3 row-major matrix. Returns the L∞ residual
 *     max_i ||τ(R·m, R·r) - R·τ(m, r)||_∞ / ||τ||_∞.
 * Should be ≤ 1e-10 on any valid rotation. */
double torque_net_equivariance_residual(const torque_net_graph_t *g,
                                         const double *m_in,
                                         const torque_net_params_t *p,
                                         const double *R);

/* Convenience: build a 2D nearest-neighbour grid graph with periodic
 * bonds. Positions live at integer coordinates (so r_ij is a unit
 * vector along x or y). Edges are directed: each bond i↔j produces
 * two directed edges i→j and j→i. Caller owns/frees all output
 * arrays. Returns 0 on success. */
int torque_net_build_grid(int Lx, int Ly, int periodic,
                           int **out_edge_src,
                           int **out_edge_dst,
                           double **out_edge_vec,
                           int *out_num_edges);

/* Closed-form least-squares fit of the five linear weights {w0..w4}
 * to a labelled dataset {(m^(s), τ_target^(s))}. With r_cut and
 * radial_order held fixed, τ(m) is linear in the five weights, so the
 * optimal weights are obtained by solving a 5×5 normal-equations
 * system in one pass — no gradient descent needed.
 *
 * Input layout:
 *   num_samples samples, each of 3·num_nodes doubles, concatenated in
 *   `m_batch` (row-major over samples) and `tau_batch`.
 *   r_cut, radial_order from `p_template`.
 *
 * Writes the five fitted weights into `p_out` (unchanged radial
 * parameters); also reports the training residual
 *     rms = √(mean |τ_fit - τ_target|²)
 * into *out_residual if non-NULL. Returns 0 on success. */
int torque_net_fit_weights(const torque_net_graph_t *g,
                            const double *m_batch,
                            const double *tau_batch,
                            int num_samples,
                            const torque_net_params_t *p_template,
                            torque_net_params_t *p_out,
                            double *out_residual);

#ifdef __cplusplus
}
#endif

#endif /* TORQUE_NET_H */
