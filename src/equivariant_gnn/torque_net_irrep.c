/*
 * src/equivariant_gnn/torque_net_irrep.c
 *
 * libirrep-backed NequIP-tower torque network.  Wraps
 * libirrep_bridge_nequip_{build,apply,apply_backward,free} and stacks
 * up to N layers in a chain.
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

#define MAX_LAYERS 8

struct torque_net_irrep {
    int num_layers;
    libirrep_bridge_nequip_t *layers[MAX_LAYERS];
    int layer_dim_in [MAX_LAYERS + 1];   /* feature dim at each interface */
    int layer_weights[MAX_LAYERS];       /* weights for each layer        */
    int weight_offsets[MAX_LAYERS + 1];  /* prefix sums                   */
    int total_weights;
};

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
    /* Single-layer convenience wrapper. */
    return torque_net_irrep_create_multilayer(hidden_in_spec, NULL,
                                               hidden_out_spec, 1,
                                               l_sh_max, n_radial,
                                               r_cut, cutoff_poly_p);
}

torque_net_irrep_t *torque_net_irrep_create_multilayer(
    const char *in_spec,
    const char *const *hidden_specs,
    const char *out_spec,
    int    num_layers,
    int    l_sh_max,
    int    n_radial,
    double r_cut,
    int    cutoff_poly_p)
{
    if (!in_spec)  in_spec  = "1x1o";
    if (!out_spec) out_spec = "1x1o";
    if (l_sh_max      <= 0)   l_sh_max      = 2;
    if (n_radial      <= 0)   n_radial      = 4;
    if (r_cut         <= 0.0) r_cut         = 1.5;
    if (cutoff_poly_p <  0)   cutoff_poly_p = 6;
    if (num_layers <= 0 || num_layers > MAX_LAYERS) return NULL;
    if (num_layers > 1 && !hidden_specs) return NULL;

    if (libirrep_bridge_init() != 0) return NULL;

    torque_net_irrep_t *net = (torque_net_irrep_t *)calloc(1, sizeof(*net));
    if (!net) return NULL;
    net->num_layers = num_layers;

    /* Build the chain of NequIP layers.  Layer k maps spec_k → spec_{k+1}.
     * spec_0 = in_spec; spec_{num_layers} = out_spec; intermediates from
     * hidden_specs[0..num_layers-2]. */
    const char *prev_spec = in_spec;
    net->layer_dim_in[0] = spec_total_dim(in_spec, 3);
    for (int k = 0; k < num_layers; k++) {
        const char *next_spec = (k == num_layers - 1)
                                  ? out_spec
                                  : hidden_specs[k];
        libirrep_bridge_nequip_t *layer = NULL;
        int rc = libirrep_bridge_nequip_build(prev_spec, l_sh_max,
                                                n_radial, r_cut, cutoff_poly_p,
                                                next_spec, &layer);
        if (rc != 0 || !layer) {
            for (int kk = 0; kk < k; kk++)
                libirrep_bridge_nequip_free(net->layers[kk]);
            free(net);
            return NULL;
        }
        net->layers[k] = layer;
        net->layer_dim_in[k + 1] = spec_total_dim(next_spec,
                                                  net->layer_dim_in[k]);
        int nw = 0;
        libirrep_bridge_nequip_num_weights(layer, &nw);
        net->layer_weights[k] = nw;
        prev_spec = next_spec;
    }
    /* Prefix sums for the flat weight buffer. */
    net->weight_offsets[0] = 0;
    for (int k = 0; k < num_layers; k++)
        net->weight_offsets[k + 1] = net->weight_offsets[k] + net->layer_weights[k];
    net->total_weights = net->weight_offsets[num_layers];
    return net;
}

void torque_net_irrep_free(torque_net_irrep_t *net) {
    if (!net) return;
    for (int k = 0; k < net->num_layers; k++) {
        if (net->layers[k]) libirrep_bridge_nequip_free(net->layers[k]);
    }
    free(net);
}

int torque_net_irrep_num_weights(const torque_net_irrep_t *net) {
    return net ? net->total_weights : 0;
}
int torque_net_irrep_in_dim(const torque_net_irrep_t *net) {
    return net ? net->layer_dim_in[0] : 0;
}
int torque_net_irrep_out_dim(const torque_net_irrep_t *net) {
    return net ? net->layer_dim_in[net->num_layers] : 0;
}
int torque_net_irrep_num_layers(const torque_net_irrep_t *net) {
    return net ? net->num_layers : 0;
}
int torque_net_irrep_layer_offset(const torque_net_irrep_t *net, int k) {
    if (!net || k < 0 || k > net->num_layers) return -1;
    return net->weight_offsets[k];
}

/* Forward through the layer chain, allocating per-layer scratch
 * buffers for intermediate features.  For 1-layer nets this matches
 * the single-layer apply call exactly. */
int torque_net_irrep_forward(const torque_net_irrep_t *net,
                              const double *weights,
                              const torque_net_graph_t *g,
                              const double *h_in,
                              double *h_out) {
    if (!net || !weights || !g || !h_in || !h_out) return -1;

    int N = g->num_nodes;
    /* Find the maximum interface dim so we can size two ping-pong scratch
     * buffers. */
    int max_dim = 0;
    for (int k = 0; k <= net->num_layers; k++)
        if (net->layer_dim_in[k] > max_dim) max_dim = net->layer_dim_in[k];

    double *buf_a = (double *)calloc((size_t)N * max_dim, sizeof(double));
    double *buf_b = (double *)calloc((size_t)N * max_dim, sizeof(double));
    if (!buf_a || !buf_b) { free(buf_a); free(buf_b); return -1; }

    /* Copy h_in into buf_a (length N · dim_in[0]). */
    memcpy(buf_a, h_in, (size_t)N * net->layer_dim_in[0] * sizeof(double));

    const double *src = buf_a;
    double       *dst = buf_b;

    for (int k = 0; k < net->num_layers; k++) {
        int dim_out = net->layer_dim_in[k + 1];
        memset(dst, 0, (size_t)N * dim_out * sizeof(double));
        const double *layer_weights = &weights[net->weight_offsets[k]];
        int rc = libirrep_bridge_nequip_apply(net->layers[k],
                                                layer_weights,
                                                N, g->num_edges,
                                                g->edge_dst, /* libirrep src = our edge_dst */
                                                g->edge_src, /* libirrep dst = our edge_src */
                                                g->edge_vec,
                                                src, dst);
        if (rc != 0) { free(buf_a); free(buf_b); return rc; }

        /* Swap ping-pong buffers for the next layer. */
        const double *tmp = src;
        src = dst;
        dst = (double *)tmp;
    }

    /* Final result is in `src` after the last swap.  Copy out. */
    memcpy(h_out, src, (size_t)N * net->layer_dim_in[net->num_layers] * sizeof(double));
    free(buf_a); free(buf_b);
    return 0;
}

int torque_net_irrep_backward(const torque_net_irrep_t *net,
                               const double *weights,
                               const torque_net_graph_t *g,
                               const double *h_in,
                               const double *grad_h_out,
                               double *grad_h_in,
                               double *grad_weights) {
    if (!net || !weights || !g || !h_in || !grad_h_out || !grad_h_in
        || !grad_weights) return -1;

    /* For multi-layer backward we need the FORWARD activations at each
     * interface.  Re-run the forward, capturing every intermediate. */
    int N = g->num_nodes;
    int Nl = net->num_layers;
    double **fwd = (double **)calloc((size_t)Nl + 1, sizeof(double *));
    if (!fwd) return -1;
    for (int k = 0; k <= Nl; k++) {
        fwd[k] = (double *)calloc((size_t)N * net->layer_dim_in[k], sizeof(double));
        if (!fwd[k]) {
            for (int kk = 0; kk < k; kk++) free(fwd[kk]);
            free(fwd);
            return -1;
        }
    }
    memcpy(fwd[0], h_in, (size_t)N * net->layer_dim_in[0] * sizeof(double));
    for (int k = 0; k < Nl; k++) {
        const double *layer_w = &weights[net->weight_offsets[k]];
        int rc = libirrep_bridge_nequip_apply(net->layers[k], layer_w,
                                                N, g->num_edges,
                                                g->edge_dst, g->edge_src,
                                                g->edge_vec,
                                                fwd[k], fwd[k + 1]);
        if (rc != 0) {
            for (int kk = 0; kk <= Nl; kk++) free(fwd[kk]);
            free(fwd);
            return rc;
        }
    }

    /* Backward: walk layers in reverse, accumulating into grad_weights
     * and propagating grad_h backwards through the chain. */
    double *grad_curr = (double *)calloc((size_t)N * net->layer_dim_in[Nl],
                                          sizeof(double));
    if (!grad_curr) {
        for (int kk = 0; kk <= Nl; kk++) free(fwd[kk]);
        free(fwd);
        return -1;
    }
    memcpy(grad_curr, grad_h_out,
           (size_t)N * net->layer_dim_in[Nl] * sizeof(double));

    for (int k = Nl - 1; k >= 0; k--) {
        int dim_in_k = net->layer_dim_in[k];
        double *grad_prev = (double *)calloc((size_t)N * dim_in_k, sizeof(double));
        if (!grad_prev) {
            free(grad_curr);
            for (int kk = 0; kk <= Nl; kk++) free(fwd[kk]);
            free(fwd);
            return -1;
        }
        const double *layer_w = &weights[net->weight_offsets[k]];
        double       *layer_gw = &grad_weights[net->weight_offsets[k]];
        int rc = libirrep_bridge_nequip_apply_backward(
            net->layers[k], layer_w,
            N, g->num_edges,
            g->edge_dst, g->edge_src, g->edge_vec,
            fwd[k], grad_curr, grad_prev, layer_gw);
        if (rc != 0) {
            free(grad_prev); free(grad_curr);
            for (int kk = 0; kk <= Nl; kk++) free(fwd[kk]);
            free(fwd);
            return rc;
        }
        free(grad_curr);
        grad_curr = grad_prev;
    }
    /* `grad_curr` now holds grad w.r.t. h_in; accumulate. */
    for (long c = 0; c < (long)N * net->layer_dim_in[0]; c++)
        grad_h_in[c] += grad_curr[c];
    free(grad_curr);
    for (int kk = 0; kk <= Nl; kk++) free(fwd[kk]);
    free(fwd);
    return 0;
}

#endif /* SPIN_NN_HAS_IRREP */
