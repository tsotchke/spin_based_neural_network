/*
 * src/equivariant_gnn/torque_net_irrep.c
 *
 * libirrep-backed NequIP-tower torque network.  Wraps
 * libirrep_bridge_nequip_{build,apply,apply_backward,free} with our
 * graph format.
 *
 * Gated behind SPIN_NN_HAS_IRREP — translation unit is empty when
 * libirrep is not linked.
 */
#include "equivariant_gnn/torque_net_irrep.h"

#ifdef SPIN_NN_HAS_IRREP

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libirrep_bridge.h"

struct torque_net_irrep {
    libirrep_bridge_nequip_t *layer;
    int in_dim;
    int out_dim;
    int num_weights;
};

/* Probe the bridge for a multiset's total dimension.  If it fails we
 * fall back to 3 (the "1x1o" default) — caller will get an error
 * later if the spec was actually invalid. */
static int spec_total_dim(const char *spec, int fallback) {
    int d = fallback;
    if (libirrep_bridge_multiset_dim(spec, &d) != 0) return fallback;
    return d;
}

torque_net_irrep_t *torque_net_irrep_create(const char *hidden_in_spec,
                                             const char *hidden_out_spec,
                                             int    l_sh_max,
                                             int    n_radial,
                                             double r_cut,
                                             int    cutoff_poly_p) {
    /* Defaults: "1x1o" → "1x1o", l_sh_max = 2, n_radial = 4, r_cut = 1.5,
     * polynomial cutoff with p = 6. */
    if (!hidden_in_spec)  hidden_in_spec  = "1x1o";
    if (!hidden_out_spec) hidden_out_spec = "1x1o";
    if (l_sh_max      <= 0)   l_sh_max      = 2;
    if (n_radial      <= 0)   n_radial      = 4;
    if (r_cut         <= 0.0) r_cut         = 1.5;
    if (cutoff_poly_p <  0)   cutoff_poly_p = 6;

    if (libirrep_bridge_init() != 0) return NULL;

    libirrep_bridge_nequip_t *layer = NULL;
    int rc = libirrep_bridge_nequip_build(hidden_in_spec, l_sh_max,
                                           n_radial, r_cut, cutoff_poly_p,
                                           hidden_out_spec, &layer);
    if (rc != 0 || !layer) return NULL;

    torque_net_irrep_t *net = (torque_net_irrep_t *)calloc(1, sizeof(*net));
    if (!net) {
        libirrep_bridge_nequip_free(layer);
        return NULL;
    }
    net->layer   = layer;
    net->in_dim  = spec_total_dim(hidden_in_spec,  3);
    net->out_dim = spec_total_dim(hidden_out_spec, 3);
    int nw = 0;
    libirrep_bridge_nequip_num_weights(layer, &nw);
    net->num_weights = nw;
    return net;
}

void torque_net_irrep_free(torque_net_irrep_t *net) {
    if (!net) return;
    if (net->layer) libirrep_bridge_nequip_free(net->layer);
    free(net);
}

int torque_net_irrep_num_weights(const torque_net_irrep_t *net) {
    return net ? net->num_weights : 0;
}
int torque_net_irrep_in_dim(const torque_net_irrep_t *net) {
    return net ? net->in_dim : 0;
}
int torque_net_irrep_out_dim(const torque_net_irrep_t *net) {
    return net ? net->out_dim : 0;
}

int torque_net_irrep_forward(const torque_net_irrep_t *net,
                              const double *weights,
                              const torque_net_graph_t *g,
                              const double *h_in,
                              double *h_out) {
    if (!net || !net->layer || !weights || !g || !h_in || !h_out) return -1;
    /* Bridge zeros h_out internally; we still wipe to be defensive. */
    memset(h_out, 0, (size_t)g->num_nodes * (size_t)net->out_dim * sizeof(double));
    /* Edge-direction convention swap: our torque_net_graph_t labels
     * each edge as (src = this site, dst = neighbour); libirrep labels
     * (src = neighbour where the message comes from, dst = receiver).
     * The libirrep API expects messages to land at `dst`, so we feed
     * our edge_dst as libirrep's src and our edge_src as libirrep's dst.
     * edge_vec = r_neighbour − r_this matches libirrep's r_j − r_i
     * direction, so it is reused unchanged. */
    return libirrep_bridge_nequip_apply(net->layer, weights,
                                         g->num_nodes, g->num_edges,
                                         g->edge_dst,    /* libirrep src */
                                         g->edge_src,    /* libirrep dst */
                                         g->edge_vec,
                                         h_in, h_out);
}

int torque_net_irrep_backward(const torque_net_irrep_t *net,
                               const double *weights,
                               const torque_net_graph_t *g,
                               const double *h_in,
                               const double *grad_h_out,
                               double *grad_h_in,
                               double *grad_weights) {
    if (!net || !net->layer || !weights || !g || !h_in || !grad_h_out
        || !grad_h_in || !grad_weights) return -1;
    /* Same src/dst swap as forward — see torque_net_irrep_forward. */
    return libirrep_bridge_nequip_apply_backward(net->layer, weights,
                                                  g->num_nodes, g->num_edges,
                                                  g->edge_dst,    /* libirrep src */
                                                  g->edge_src,    /* libirrep dst */
                                                  g->edge_vec, h_in,
                                                  grad_h_out,
                                                  grad_h_in, grad_weights);
}

#endif /* SPIN_NN_HAS_IRREP */
