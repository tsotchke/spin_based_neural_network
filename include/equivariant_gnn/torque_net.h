/*
 * include/equivariant_gnn/torque_net.h
 *
 * SO(3)-covariant torque predictor for the micromagnetic LLG integrator
 * (pillar P1.2 — "pragmatic baseline" tier).  Pure C, no libirrep
 * dependency.  Uses geometrically-covariant primitives (cross, dot,
 * triple product) so the predicted torque rotates correctly under any
 * rigid-body rotation of the input field.
 *
 * Scope and limits — what this is, and what it is NOT:
 *
 *   * Covariant under proper rotations R ∈ SO(3) — verified to ~1e-10
 *     by torque_net_equivariance_residual on random rotations.  Each
 *     of the five terms is a polar or axial vector; their sum stays a
 *     vector under SO(3).
 *
 *   * NOT a full E(3)-equivariant network: there is no machinery for
 *     translations beyond using relative displacements, no enforcement
 *     of time-reversal parity (m is t-odd, torque is t-odd, but the
 *     individual w0..w4 mix even and odd terms with no parity gating),
 *     and no irrep tensor-product tower (no l > 1 features, no MACE-
 *     style many-body messages).  Calling this "E(3)-equivariant" in
 *     the NequIP / MACE sense would be wrong.
 *
 *   * Five learnable scalar weights only.  Function class is the linear
 *     span of the five labelled-vector terms below; expressivity is
 *     limited.  For real micromagnetic ground states this fits a few
 *     leading terms (exchange-driven, DMI-like, Zeeman); finer features
 *     require the libirrep tower (planned for v0.5 P1.2).
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
 *   Each term is a proper vector under SO(3); the radial modulator φ
 *   is a scalar function of the bond length.  A linear combination of
 *   covariant vectors remains covariant, hence τ_i is SO(3)-covariant.
 *
 * Upgrade path: replace this header's contract with a libirrep-backed
 * variant (`torque_net_irrep.h`) once libirrep ≥ 1.1 ships full NequIP
 * tensor-product layers; that variant will support arbitrary irrep
 * content (l ≤ L_max), parity gating, and many-body MACE messages.
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

/*
 * Nine-term linear basis.  Terms 0–4 are the original L=1 vector basis;
 * terms 5–8 add L=2 quadrupolar contractions to L=1 (the symmetric
 * traceless part of m ⊗ m projected onto r̂ or m).  All nine terms are
 * proper SO(3) vectors under rotation of (m_i, m_j, r̂_ij).
 *
 * Time-reversal classification.  The torque_net output is fed to the
 * LLG integrator as an effective field B_eff.  B_eff is t-odd (it is
 * itself a magnetic-field-like quantity), so a strict micromagnetic
 * model uses only t-odd terms — i.e. odd power of m.  Terms with even
 * power of m (w0, w2, w5, w7) are t-even and break time-reversal
 * symmetry.  They are kept in the basis as a useful function-class
 * extension (e.g. for itinerant-electron effective fields where the
 * physics genuinely is t-symmetry-breaking), but a clean conservative
 * LLG run should zero them — see torque_net_zero_t_even_weights().
 *
 *   w0: (m_j · r̂_ij) · m_i               — L=1, t-even (m²)
 *   w1:  m_j × r̂_ij                       — L=1, t-odd  (m¹)
 *   w2:  m_i × m_j                         — L=1, t-even (m²)
 *   w3: (m_i · m_j) · m_i                  — L=1, t-odd  (m³)
 *   w4:  m_j                               — L=1, t-odd  (m¹)
 *   w5: (m_i · r̂_ij) · m_j               — L=2, t-even (m²)
 *   w6: (m_i · m_j) · m_j                  — L=2, t-odd  (m³)
 *   w7: (m_i · m_j) · r̂_ij                — L=2, t-even (m²)
 *   w8: (m_j · r̂_ij) · r̂_ij              — L=2, t-odd  (m¹)
 */
typedef struct {
    double w0;                  /* (m_j·r̂) m_i        — polar           */
    double w1;                  /* m_j × r̂             — axial           */
    double w2;                  /* m_i × m_j           — axial           */
    double w3;                  /* (m_i·m_j) m_i       — polar           */
    double w4;                  /* m_j                 — polar           */
    double w5;                  /* (m_i·r̂) m_j        — L=2, polar      */
    double w6;                  /* (m_i·m_j) m_j       — L=2, polar      */
    double w7;                  /* (m_i·m_j) r̂        — L=2, polar      */
    double w8;                  /* (m_j·r̂) r̂          — L=2, polar      */
    double r_cut;               /* radial cutoff       — scalar          */
    double radial_order;        /* polynomial order for cutoff           */
} torque_net_params_t;

/* Number of linear basis terms.  Used by the fitter to size the
 * normal-equations system. */
#define TORQUE_NET_NUM_BASIS 9

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

/* Time-reversal residual on the t-odd projection.  m is t-odd, r̂ is
 * t-even, so a strictly t-odd output (B_eff for conservative LLG) must
 * satisfy τ(−m) = −τ(m).  Returns
 *     max_i ||τ(−m) + τ(m)||_∞ / ||τ(m)||_∞.
 * Should be ≤ 1e-12 if all t-even weights (w0, w2, w5, w7) are zero;
 * non-zero quantifies the t-symmetry breaking from those terms.  See
 * the t-parity classification at the top of this header. */
double torque_net_time_reversal_residual(const torque_net_graph_t *g,
                                          const double *m_in,
                                          const torque_net_params_t *p);

/* In-place: zero the t-even weights {w0, w2, w5, w7}, leaving only the
 * t-odd basis {w1, w3, w4, w6, w8} active.  Use to enforce strict
 * time-reversal symmetry on a trained or hand-set parameter vector
 * before passing to the LLG integrator. */
void torque_net_zero_t_even_weights(torque_net_params_t *p);

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

/* Closed-form least-squares fit of all TORQUE_NET_NUM_BASIS linear
 * weights {w0..w8} to a labelled dataset {(m^(s), τ_target^(s))}.
 * With r_cut and radial_order held fixed, τ(m) is linear in the nine
 * weights, so the optimal weights are obtained by solving a 9×9
 * normal-equations system in one pass — no gradient descent needed.
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
