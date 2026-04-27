/*
 * src/libirrep_bridge.c
 *
 * Lazy libirrep wrapper. Compiles without libirrep present — all
 * entry points return IRREP_BRIDGE_EDISABLED. When built with
 * -DSPIN_NN_HAS_IRREP=1, forwards to the real library.
 */
#include <stdio.h>
#include <stdlib.h>
#include "libirrep_bridge.h"

#ifdef SPIN_NN_HAS_IRREP
#include <irrep/irrep.h>
#include <irrep/rdm.h>
/* nequip.h is only in libirrep >= 1.1 (shipped with pillar P1.2 work).
 * The umbrella irrep.h in 1.0 does not pull it in, so include it
 * explicitly when the caller opts into NequIP via SPIN_NN_HAS_IRREP_NEQUIP. */
#ifdef SPIN_NN_HAS_IRREP_NEQUIP
#include <irrep/nequip.h>
#endif
static int g_refcount = 0;
#endif

int libirrep_bridge_is_available(void) {
#ifdef SPIN_NN_HAS_IRREP
    return 1;
#else
    return 0;
#endif
}

const char *libirrep_bridge_version(void) {
#ifdef SPIN_NN_HAS_IRREP
    static char buf[32];
    snprintf(buf, sizeof(buf), "%d.%d.%d",
             IRREP_VERSION_MAJOR, IRREP_VERSION_MINOR, IRREP_VERSION_PATCH);
    return buf;
#else
    return NULL;
#endif
}

int libirrep_bridge_init(void) {
#ifdef SPIN_NN_HAS_IRREP
    g_refcount++;
    return IRREP_BRIDGE_OK;
#else
    return IRREP_BRIDGE_EDISABLED;
#endif
}

int libirrep_bridge_shutdown(void) {
#ifdef SPIN_NN_HAS_IRREP
    if (g_refcount <= 0) return IRREP_BRIDGE_ENOT_READY;
    g_refcount--;
    return IRREP_BRIDGE_OK;
#else
    return IRREP_BRIDGE_EDISABLED;
#endif
}

int libirrep_bridge_sph_harm_real(int l, int m,
                                  double theta, double phi,
                                  double *out) {
    if (!out) return IRREP_BRIDGE_EARG;
#ifdef SPIN_NN_HAS_IRREP
    *out = irrep_sph_harm_real(l, m, theta, phi);
    return IRREP_BRIDGE_OK;
#else
    (void)l; (void)m; (void)theta; (void)phi;
    return IRREP_BRIDGE_EDISABLED;
#endif
}

int libirrep_bridge_clebsch_gordan(int j1, int m1,
                                   int j2, int m2,
                                   int J, int M,
                                   double *out) {
    if (!out) return IRREP_BRIDGE_EARG;
#ifdef SPIN_NN_HAS_IRREP
    *out = irrep_cg(j1, m1, j2, m2, J, M);
    return IRREP_BRIDGE_OK;
#else
    (void)j1; (void)m1; (void)j2; (void)m2; (void)J; (void)M;
    return IRREP_BRIDGE_EDISABLED;
#endif
}

int libirrep_bridge_wigner_d_small(int j, int mp, int m, double beta,
                                   double *out) {
    if (!out) return IRREP_BRIDGE_EARG;
#ifdef SPIN_NN_HAS_IRREP
    *out = irrep_wigner_d_small(j, mp, m, beta);
    return IRREP_BRIDGE_OK;
#else
    (void)j; (void)mp; (void)m; (void)beta;
    return IRREP_BRIDGE_EDISABLED;
#endif
}

int libirrep_bridge_multiset_dim(const char *spec, int *out_dim) {
    if (!spec || !out_dim) return IRREP_BRIDGE_EARG;
#ifdef SPIN_NN_HAS_IRREP
    irrep_multiset_t *m = irrep_multiset_parse(spec);
    if (!m) return IRREP_BRIDGE_ELIB;
    *out_dim = irrep_multiset_dim(m);
    irrep_multiset_free(m);
    return IRREP_BRIDGE_OK;
#else
    (void)spec;
    return IRREP_BRIDGE_EDISABLED;
#endif
}

#if defined(SPIN_NN_HAS_IRREP) && defined(SPIN_NN_HAS_IRREP_NEQUIP)
struct libirrep_bridge_nequip {
    irrep_nequip_layer_t *layer;
    irrep_multiset_t     *hidden_in;
    irrep_multiset_t     *hidden_out;
    int                   in_dim;
    int                   out_dim;
};
#endif

int libirrep_bridge_nequip_build(const char *hidden_in_spec,
                                  int l_sh_max,
                                  int n_radial,
                                  double r_cut,
                                  int cutoff_poly_p,
                                  const char *hidden_out_spec,
                                  libirrep_bridge_nequip_t **out_layer) {
    if (!hidden_in_spec || !hidden_out_spec || !out_layer) return IRREP_BRIDGE_EARG;
    if (l_sh_max < 0 || n_radial <= 0 || r_cut <= 0.0) return IRREP_BRIDGE_EARG;
#if defined(SPIN_NN_HAS_IRREP) && defined(SPIN_NN_HAS_IRREP_NEQUIP)
    libirrep_bridge_nequip_t *b = calloc(1, sizeof(*b));
    if (!b) return IRREP_BRIDGE_ELIB;
    b->hidden_in  = irrep_multiset_parse(hidden_in_spec);
    b->hidden_out = irrep_multiset_parse(hidden_out_spec);
    if (!b->hidden_in || !b->hidden_out) {
        irrep_multiset_free(b->hidden_in);
        irrep_multiset_free(b->hidden_out);
        free(b);
        return IRREP_BRIDGE_ELIB;
    }
    irrep_nequip_cutoff_t ck = (cutoff_poly_p == 0)
                                ? IRREP_NEQUIP_CUTOFF_COSINE
                                : IRREP_NEQUIP_CUTOFF_POLYNOMIAL;
    b->layer = irrep_nequip_layer_build(b->hidden_in, l_sh_max, n_radial,
                                         r_cut, ck, cutoff_poly_p, b->hidden_out);
    if (!b->layer) {
        irrep_multiset_free(b->hidden_in);
        irrep_multiset_free(b->hidden_out);
        free(b);
        return IRREP_BRIDGE_ELIB;
    }
    b->in_dim  = irrep_multiset_dim(b->hidden_in);
    b->out_dim = irrep_multiset_dim(b->hidden_out);
    *out_layer = b;
    return IRREP_BRIDGE_OK;
#else
    (void)hidden_in_spec; (void)l_sh_max; (void)n_radial; (void)r_cut;
    (void)cutoff_poly_p; (void)hidden_out_spec;
    *out_layer = NULL;
    return IRREP_BRIDGE_EDISABLED;
#endif
}

int libirrep_bridge_nequip_free(libirrep_bridge_nequip_t *layer) {
    if (!layer) return IRREP_BRIDGE_OK;
#if defined(SPIN_NN_HAS_IRREP) && defined(SPIN_NN_HAS_IRREP_NEQUIP)
    irrep_nequip_layer_free(layer->layer);
    irrep_multiset_free(layer->hidden_in);
    irrep_multiset_free(layer->hidden_out);
    free(layer);
    return IRREP_BRIDGE_OK;
#else
    (void)layer;
    return IRREP_BRIDGE_EDISABLED;
#endif
}

int libirrep_bridge_nequip_num_weights(const libirrep_bridge_nequip_t *layer,
                                        int *out_num_weights) {
    if (!layer || !out_num_weights) return IRREP_BRIDGE_EARG;
#if defined(SPIN_NN_HAS_IRREP) && defined(SPIN_NN_HAS_IRREP_NEQUIP)
    *out_num_weights = irrep_nequip_layer_num_weights(layer->layer);
    return IRREP_BRIDGE_OK;
#else
    (void)layer; (void)out_num_weights;
    return IRREP_BRIDGE_EDISABLED;
#endif
}

int libirrep_bridge_nequip_apply(const libirrep_bridge_nequip_t *layer,
                                  const double *tp_weights,
                                  int n_nodes, int n_edges,
                                  const int *edge_src,
                                  const int *edge_dst,
                                  const double *edge_vec,
                                  const double *h_in,
                                  double *h_out) {
    if (!layer || !tp_weights || !edge_src || !edge_dst || !edge_vec ||
        !h_in || !h_out) return IRREP_BRIDGE_EARG;
    if (n_nodes <= 0 || n_edges < 0) return IRREP_BRIDGE_EARG;
#if defined(SPIN_NN_HAS_IRREP) && defined(SPIN_NN_HAS_IRREP_NEQUIP)
    irrep_nequip_layer_apply(layer->layer, tp_weights, n_nodes, n_edges,
                              edge_src, edge_dst, edge_vec, h_in, h_out);
    return IRREP_BRIDGE_OK;
#else
    (void)layer; (void)tp_weights; (void)n_nodes; (void)n_edges;
    (void)edge_src; (void)edge_dst; (void)edge_vec; (void)h_in; (void)h_out;
    return IRREP_BRIDGE_EDISABLED;
#endif
}

int libirrep_bridge_nequip_apply_backward(const libirrep_bridge_nequip_t *layer,
                                           const double *tp_weights,
                                           int n_nodes, int n_edges,
                                           const int *edge_src,
                                           const int *edge_dst,
                                           const double *edge_vec,
                                           const double *h_in,
                                           const double *grad_h_out,
                                           double *grad_h_in,
                                           double *grad_tp_weights) {
    if (!layer || !tp_weights || !edge_src || !edge_dst || !edge_vec ||
        !h_in || !grad_h_out || !grad_h_in || !grad_tp_weights)
        return IRREP_BRIDGE_EARG;
    if (n_nodes <= 0 || n_edges < 0) return IRREP_BRIDGE_EARG;
#if defined(SPIN_NN_HAS_IRREP) && defined(SPIN_NN_HAS_IRREP_NEQUIP)
    irrep_nequip_layer_apply_backward(layer->layer, tp_weights,
                                       n_nodes, n_edges,
                                       edge_src, edge_dst, edge_vec,
                                       h_in, grad_h_out,
                                       grad_h_in, grad_tp_weights);
    return IRREP_BRIDGE_OK;
#else
    (void)layer; (void)tp_weights; (void)n_nodes; (void)n_edges;
    (void)edge_src; (void)edge_dst; (void)edge_vec; (void)h_in;
    (void)grad_h_out; (void)grad_h_in; (void)grad_tp_weights;
    return IRREP_BRIDGE_EDISABLED;
#endif
}

/* ---- RDM / entropy ---------------------------------------------------- */

int libirrep_bridge_partial_trace_spin_half(int num_sites,
                                             const double _Complex *psi,
                                             const int *sites_A, int nA,
                                             double _Complex *rho_A) {
    if (!psi || !rho_A || num_sites <= 0 || nA < 0 || nA > num_sites)
        return IRREP_BRIDGE_EARG;
    if (nA > 0 && !sites_A) return IRREP_BRIDGE_EARG;
#ifdef SPIN_NN_HAS_IRREP
    irrep_status_t rc = irrep_partial_trace(num_sites, /*local_dim*/ 2,
                                             psi, sites_A, nA, rho_A);
    return (rc == IRREP_OK) ? IRREP_BRIDGE_OK : IRREP_BRIDGE_ELIB;
#else
    (void)num_sites; (void)psi; (void)sites_A; (void)nA; (void)rho_A;
    return IRREP_BRIDGE_EDISABLED;
#endif
}

int libirrep_bridge_entropy_vonneumann(const double _Complex *rho, int n,
                                        double *out_S) {
    if (!rho || !out_S || n <= 0) return IRREP_BRIDGE_EARG;
#ifdef SPIN_NN_HAS_IRREP
    *out_S = irrep_entropy_vonneumann(rho, n);
    return IRREP_BRIDGE_OK;
#else
    (void)rho; (void)n; (void)out_S;
    return IRREP_BRIDGE_EDISABLED;
#endif
}

int libirrep_bridge_entropy_renyi(const double _Complex *rho, int n,
                                   double alpha, double *out_S) {
    if (!rho || !out_S || n <= 0 || alpha <= 0.0)
        return IRREP_BRIDGE_EARG;
#ifdef SPIN_NN_HAS_IRREP
    *out_S = irrep_entropy_renyi(rho, n, alpha);
    return IRREP_BRIDGE_OK;
#else
    (void)rho; (void)n; (void)alpha; (void)out_S;
    return IRREP_BRIDGE_EDISABLED;
#endif
}
