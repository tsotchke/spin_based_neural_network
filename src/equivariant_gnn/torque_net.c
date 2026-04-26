/*
 * src/equivariant_gnn/torque_net.c
 *
 * Minimal SO(3)-equivariant torque predictor.  Each term in the
 * summation is a proper rank-1 tensor under rotations, so the output
 * τ_i transforms as a vector by construction.  We verify that
 * property numerically in torque_net_equivariance_residual.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "equivariant_gnn/torque_net.h"

static double poly_cutoff(double r, double r_cut, double p) {
    if (r >= r_cut) return 0.0;
    double x = r / r_cut;
    /* Smooth polynomial cutoff (Schütt 2017 / NequIP paper):
     *   φ(r) = 1 - (p+1)(p+2)/2 · x^p
     *            + p(p+2) · x^{p+1}
     *            - p(p+1)/2 · x^{p+2}
     * Goes smoothly from 1 at r=0 to 0 at r=r_cut with zero slope. */
    double xp  = pow(x, p);
    double xp1 = xp * x;
    double xp2 = xp1 * x;
    return 1.0 - 0.5 * (p + 1.0) * (p + 2.0) * xp
                + p * (p + 2.0) * xp1
                - 0.5 * p * (p + 1.0) * xp2;
}

static void cross3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

static double dot3(const double a[3], const double b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

int torque_net_forward(const torque_net_graph_t *g,
                        const double *m_in,
                        const torque_net_params_t *p,
                        double *out_torque) {
    if (!g || !m_in || !p || !out_torque) return -1;
    if (g->num_nodes <= 0 || g->num_edges < 0) return -1;
    memset(out_torque, 0, (size_t)3 * (size_t)g->num_nodes * sizeof(double));
    double rc = p->r_cut > 0.0 ? p->r_cut : 1.5;
    double po = p->radial_order > 0.0 ? p->radial_order : 6.0;

    for (int e = 0; e < g->num_edges; e++) {
        int i = g->edge_src[e];
        int j = g->edge_dst[e];
        if (i < 0 || i >= g->num_nodes || j < 0 || j >= g->num_nodes) return -1;
        double r[3] = { g->edge_vec[3*e], g->edge_vec[3*e+1], g->edge_vec[3*e+2] };
        double r_len = sqrt(dot3(r, r));
        if (r_len < 1e-12) continue;
        double r_hat[3] = { r[0]/r_len, r[1]/r_len, r[2]/r_len };
        double phi = poly_cutoff(r_len, rc, po);
        if (phi == 0.0) continue;

        const double *mi = &m_in[3*i];
        const double *mj = &m_in[3*j];

        double mj_dot_rhat = dot3(mj, r_hat);
        double mi_dot_rhat = dot3(mi, r_hat);
        double mi_dot_mj   = dot3(mi, mj);
        double mj_cross_rhat[3], mi_cross_mj[3];
        cross3(mj, r_hat, mj_cross_rhat);
        cross3(mi, mj,    mi_cross_mj);

        double *ti = &out_torque[3*i];
        for (int k = 0; k < 3; k++) {
            ti[k] += phi * (
                p->w0 * mj_dot_rhat * mi[k]
              + p->w1 * mj_cross_rhat[k]
              + p->w2 * mi_cross_mj[k]
              + p->w3 * mi_dot_mj  * mi[k]
              + p->w4 * mj[k]
              /* L=2 quadrupolar contractions to L=1 */
              + p->w5 * mi_dot_rhat * mj[k]
              + p->w6 * mi_dot_mj   * mj[k]
              + p->w7 * mi_dot_mj   * r_hat[k]
              + p->w8 * mj_dot_rhat * r_hat[k]);
        }
    }
    return 0;
}

static void rot3(const double R[9], const double v[3], double out[3]) {
    out[0] = R[0]*v[0] + R[1]*v[1] + R[2]*v[2];
    out[1] = R[3]*v[0] + R[4]*v[1] + R[5]*v[2];
    out[2] = R[6]*v[0] + R[7]*v[1] + R[8]*v[2];
}

double torque_net_equivariance_residual(const torque_net_graph_t *g,
                                         const double *m_in,
                                         const torque_net_params_t *p,
                                         const double *R) {
    int N = g->num_nodes, E = g->num_edges;
    double *m_rot     = malloc((size_t)3 * N * sizeof(double));
    double *edge_rot  = malloc((size_t)3 * E * sizeof(double));
    double *tau       = malloc((size_t)3 * N * sizeof(double));
    double *tau_rot   = malloc((size_t)3 * N * sizeof(double));
    double *tau_rotated_forward = malloc((size_t)3 * N * sizeof(double));

    /* Apply R to input: m_rot[i] = R · m_in[i]; edge_rot[e] = R · edge_vec[e]. */
    for (int i = 0; i < N; i++) rot3(R, &m_in[3*i],    &m_rot[3*i]);
    for (int e = 0; e < E; e++) rot3(R, &g->edge_vec[3*e], &edge_rot[3*e]);
    torque_net_graph_t g_rot = *g;
    g_rot.edge_vec = edge_rot;

    torque_net_forward(g, m_in, p, tau);
    torque_net_forward(&g_rot, m_rot, p, tau_rot);

    for (int i = 0; i < N; i++) rot3(R, &tau[3*i], &tau_rotated_forward[3*i]);

    double num = 0.0, den = 0.0;
    for (int i = 0; i < 3 * N; i++) {
        double d = fabs(tau_rotated_forward[i] - tau_rot[i]);
        if (d > num) num = d;
        double t = fabs(tau[i]);
        if (t > den) den = t;
    }
    free(m_rot); free(edge_rot); free(tau); free(tau_rot); free(tau_rotated_forward);
    return (den > 0.0) ? num / den : num;
}

double torque_net_time_reversal_residual(const torque_net_graph_t *g,
                                          const double *m_in,
                                          const torque_net_params_t *p) {
    int N = g->num_nodes;
    double *m_neg = malloc((size_t)3 * N * sizeof(double));
    double *tau_pos = malloc((size_t)3 * N * sizeof(double));
    double *tau_neg = malloc((size_t)3 * N * sizeof(double));
    if (!m_neg || !tau_pos || !tau_neg) {
        free(m_neg); free(tau_pos); free(tau_neg);
        return 1.0;
    }
    for (int k = 0; k < 3 * N; k++) m_neg[k] = -m_in[k];
    torque_net_forward(g, m_in,  p, tau_pos);
    torque_net_forward(g, m_neg, p, tau_neg);
    /* For t-odd output: τ(−m) = −τ(m), so τ(m) + τ(−m) ≡ 0. */
    double num = 0.0, den = 0.0;
    for (int k = 0; k < 3 * N; k++) {
        double d = fabs(tau_pos[k] + tau_neg[k]);
        if (d > num) num = d;
        if (fabs(tau_pos[k]) > den) den = fabs(tau_pos[k]);
    }
    free(m_neg); free(tau_pos); free(tau_neg);
    return (den > 0.0) ? num / den : num;
}

void torque_net_zero_t_even_weights(torque_net_params_t *p) {
    if (!p) return;
    p->w0 = 0.0;     /* (m_j·r̂) m_i        — t-even (m²) */
    p->w2 = 0.0;     /* m_i × m_j           — t-even (m²) */
    p->w5 = 0.0;     /* (m_i·r̂) m_j        — t-even (m²) */
    p->w7 = 0.0;     /* (m_i·m_j) r̂         — t-even (m²) */
}

/* Per-site contribution of a single basis term k at site i, given
 * the current magnetization batch. Returns τ_i for that basis only,
 * accumulating over neighbours. 3 components per site.  */
static void torque_net_basis_component(const torque_net_graph_t *g,
                                        const double *m,
                                        const torque_net_params_t *p_template,
                                        int which_k,
                                        double *out_tau) {
    memset(out_tau, 0, (size_t)3 * (size_t)g->num_nodes * sizeof(double));
    double rc = p_template->r_cut > 0.0 ? p_template->r_cut : 1.5;
    double po = p_template->radial_order > 0.0 ? p_template->radial_order : 6.0;
    for (int e = 0; e < g->num_edges; e++) {
        int i = g->edge_src[e];
        int j = g->edge_dst[e];
        double r[3] = { g->edge_vec[3*e], g->edge_vec[3*e+1], g->edge_vec[3*e+2] };
        double r_len = sqrt(dot3(r, r));
        if (r_len < 1e-12) continue;
        double r_hat[3] = { r[0]/r_len, r[1]/r_len, r[2]/r_len };
        double phi = poly_cutoff(r_len, rc, po);
        if (phi == 0.0) continue;
        const double *mi = &m[3*i];
        const double *mj = &m[3*j];
        double *ti = &out_tau[3*i];
        switch (which_k) {
            case 0: {
                double d = dot3(mj, r_hat);
                ti[0] += phi * d * mi[0];
                ti[1] += phi * d * mi[1];
                ti[2] += phi * d * mi[2];
            } break;
            case 1: {
                double c[3]; cross3(mj, r_hat, c);
                ti[0] += phi * c[0]; ti[1] += phi * c[1]; ti[2] += phi * c[2];
            } break;
            case 2: {
                double c[3]; cross3(mi, mj, c);
                ti[0] += phi * c[0]; ti[1] += phi * c[1]; ti[2] += phi * c[2];
            } break;
            case 3: {
                double d = dot3(mi, mj);
                ti[0] += phi * d * mi[0];
                ti[1] += phi * d * mi[1];
                ti[2] += phi * d * mi[2];
            } break;
            case 4: {
                ti[0] += phi * mj[0]; ti[1] += phi * mj[1]; ti[2] += phi * mj[2];
            } break;
            case 5: {
                /* (m_i · r̂) m_j */
                double d = dot3(mi, r_hat);
                ti[0] += phi * d * mj[0];
                ti[1] += phi * d * mj[1];
                ti[2] += phi * d * mj[2];
            } break;
            case 6: {
                /* (m_i · m_j) m_j */
                double d = dot3(mi, mj);
                ti[0] += phi * d * mj[0];
                ti[1] += phi * d * mj[1];
                ti[2] += phi * d * mj[2];
            } break;
            case 7: {
                /* (m_i · m_j) r̂ */
                double d = dot3(mi, mj);
                ti[0] += phi * d * r_hat[0];
                ti[1] += phi * d * r_hat[1];
                ti[2] += phi * d * r_hat[2];
            } break;
            case 8: {
                /* (m_j · r̂) r̂ */
                double d = dot3(mj, r_hat);
                ti[0] += phi * d * r_hat[0];
                ti[1] += phi * d * r_hat[1];
                ti[2] += phi * d * r_hat[2];
            } break;
            default: break;
        }
    }
}

/* Gauss-Jordan elimination on an n×n symmetric positive-definite system
 * with partial pivoting on the diagonal.  In-place; on success x = A⁻¹ b
 * (b unchanged at the call site since we work on copies via the caller). */
static int solve_dense(double *A, double *b, double *x, int n) {
    for (int i = 0; i < n; i++) {
        /* Partial pivoting: find largest |A[k][i]| in column i below row i. */
        int piv_row = i;
        double piv_val = fabs(A[i * n + i]);
        for (int k = i + 1; k < n; k++) {
            double v = fabs(A[k * n + i]);
            if (v > piv_val) { piv_val = v; piv_row = k; }
        }
        if (piv_val < 1e-18) return -1;
        if (piv_row != i) {
            for (int j = 0; j < n; j++) {
                double tmp = A[i * n + j]; A[i * n + j] = A[piv_row * n + j]; A[piv_row * n + j] = tmp;
            }
            double tmp = b[i]; b[i] = b[piv_row]; b[piv_row] = tmp;
        }
        double piv = A[i * n + i];
        for (int j = i; j < n; j++) A[i * n + j] /= piv;
        b[i] /= piv;
        for (int k = 0; k < n; k++) {
            if (k == i) continue;
            double fac = A[k * n + i];
            for (int j = i; j < n; j++) A[k * n + j] -= fac * A[i * n + j];
            b[k] -= fac * b[i];
        }
    }
    for (int i = 0; i < n; i++) x[i] = b[i];
    return 0;
}

int torque_net_fit_weights(const torque_net_graph_t *g,
                            const double *m_batch,
                            const double *tau_batch,
                            int num_samples,
                            const torque_net_params_t *p_template,
                            torque_net_params_t *p_out,
                            double *out_residual) {
    if (!g || !m_batch || !tau_batch || num_samples <= 0 ||
        !p_template || !p_out) return -1;
    int N = g->num_nodes;
    int K = TORQUE_NET_NUM_BASIS;
    long comp_per_sample = (long)3 * N;

    /* For each sample, evaluate all K basis vectors once. */
    double *bases = malloc((size_t)K * comp_per_sample * num_samples * sizeof(double));
    if (!bases) return -1;
    for (int s = 0; s < num_samples; s++) {
        const double *m_s = &m_batch[(size_t)s * comp_per_sample];
        for (int k = 0; k < K; k++) {
            double *slot = &bases[(((size_t)s * K) + k) * comp_per_sample];
            torque_net_basis_component(g, m_s, p_template, k, slot);
        }
    }

    /* Normal equations A^T A · w = A^T b. */
    double *AtA = calloc((size_t)K * K, sizeof(double));
    double *Atb = calloc((size_t)K,     sizeof(double));
    double *w   = calloc((size_t)K,     sizeof(double));
    if (!AtA || !Atb || !w) {
        free(bases); free(AtA); free(Atb); free(w);
        return -1;
    }
    for (int s = 0; s < num_samples; s++) {
        for (int k = 0; k < K; k++) {
            const double *bk = &bases[(((size_t)s * K) + k) * comp_per_sample];
            for (int l = k; l < K; l++) {
                const double *bl = &bases[(((size_t)s * K) + l) * comp_per_sample];
                double acc = 0.0;
                for (long c = 0; c < comp_per_sample; c++) acc += bk[c] * bl[c];
                AtA[k * K + l] += acc;
                if (l != k) AtA[l * K + k] += acc;
            }
            const double *tau_s = &tau_batch[(size_t)s * comp_per_sample];
            double acc = 0.0;
            for (long c = 0; c < comp_per_sample; c++) acc += bk[c] * tau_s[c];
            Atb[k] += acc;
        }
    }

    int rc = solve_dense(AtA, Atb, w, K);
    if (rc != 0) {
        free(bases); free(AtA); free(Atb); free(w);
        return rc;
    }

    *p_out = *p_template;
    p_out->w0 = w[0]; p_out->w1 = w[1]; p_out->w2 = w[2];
    p_out->w3 = w[3]; p_out->w4 = w[4];
    p_out->w5 = w[5]; p_out->w6 = w[6]; p_out->w7 = w[7]; p_out->w8 = w[8];

    if (out_residual) {
        double sq = 0.0; long ncomp = 0;
        for (int s = 0; s < num_samples; s++) {
            double *tau_fit = malloc((size_t)comp_per_sample * sizeof(double));
            torque_net_forward(g, &m_batch[(size_t)s * comp_per_sample],
                                p_out, tau_fit);
            const double *tau_s = &tau_batch[(size_t)s * comp_per_sample];
            for (long c = 0; c < comp_per_sample; c++) {
                double d = tau_fit[c] - tau_s[c];
                sq += d * d;
                ncomp++;
            }
            free(tau_fit);
        }
        *out_residual = sqrt(sq / (double)ncomp);
    }

    free(bases); free(AtA); free(Atb); free(w);
    return 0;
}

int torque_net_build_grid(int Lx, int Ly, int periodic,
                           int **out_edge_src,
                           int **out_edge_dst,
                           double **out_edge_vec,
                           int *out_num_edges) {
    if (Lx <= 0 || Ly <= 0 || !out_edge_src || !out_edge_dst ||
        !out_edge_vec || !out_num_edges) return -1;
    /* Each node has up to 4 outgoing bonds (E, W, N, S), all directed. */
    int N = Lx * Ly;
    int cap = 4 * N;
    int *src = malloc((size_t)cap * sizeof(int));
    int *dst = malloc((size_t)cap * sizeof(int));
    double *vec = malloc((size_t)3 * cap * sizeof(double));
    if (!src || !dst || !vec) { free(src); free(dst); free(vec); return -1; }
    int n_edges = 0;
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++) {
        int i = x * Ly + y;
        int dxs[4] = { +1, -1,  0,  0 };
        int dys[4] = {  0,  0, +1, -1 };
        for (int d = 0; d < 4; d++) {
            int nx = x + dxs[d];
            int ny = y + dys[d];
            if (periodic) {
                nx = (nx + Lx) % Lx;
                ny = (ny + Ly) % Ly;
            } else if (nx < 0 || nx >= Lx || ny < 0 || ny >= Ly) continue;
            int j = nx * Ly + ny;
            src[n_edges] = i;
            dst[n_edges] = j;
            /* Minimum-image convention for periodic bonds. */
            double dx = (double)dxs[d];
            double dy = (double)dys[d];
            vec[3*n_edges]   = dx;
            vec[3*n_edges+1] = dy;
            vec[3*n_edges+2] = 0.0;
            n_edges++;
        }
    }
    *out_edge_src = src;
    *out_edge_dst = dst;
    *out_edge_vec = vec;
    *out_num_edges = n_edges;
    return 0;
}
