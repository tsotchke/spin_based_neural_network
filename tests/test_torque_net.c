/*
 * tests/test_torque_net.c
 *
 * Minimal SO(3)-equivariant torque predictor for LLG — pillar P1.2
 * pragmatic baseline (no libirrep dependency). The whole point of the
 * module is rotation equivariance:
 *
 *     τ(R·m, R·r) = R · τ(m, r)     for every R ∈ SO(3).
 *
 * We check this to machine precision on random rotations. A naïve
 * per-component MLP on (m_x, m_y, m_z) would break it.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "equivariant_gnn/torque_net.h"
static void random_unit_vectors(int N, double *v, unsigned *seed) {
    for (int i = 0; i < N; i++) {
        *seed = (*seed) * 1103515245u + 12345u;
        double u1 = ((*seed) % 10000) / 10000.0;
        *seed = (*seed) * 1103515245u + 12345u;
        double u2 = ((*seed) % 10000) / 10000.0;
        double theta = 2.0 * M_PI * u2;
        double z = 2.0 * u1 - 1.0;
        double s = sqrt(1.0 - z*z);
        v[3*i]   = s * cos(theta);
        v[3*i+1] = s * sin(theta);
        v[3*i+2] = z;
    }
}
static void rotation_matrix_from_axis_angle(const double axis[3], double angle,
                                             double R[9]) {
    double n[3] = { axis[0], axis[1], axis[2] };
    double a = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    for (int i = 0; i < 3; i++) n[i] /= a;
    double c = cos(angle), s = sin(angle), t = 1.0 - c;
    R[0] = c + n[0]*n[0]*t;
    R[1] = n[0]*n[1]*t - n[2]*s;
    R[2] = n[0]*n[2]*t + n[1]*s;
    R[3] = n[1]*n[0]*t + n[2]*s;
    R[4] = c + n[1]*n[1]*t;
    R[5] = n[1]*n[2]*t - n[0]*s;
    R[6] = n[2]*n[0]*t - n[1]*s;
    R[7] = n[2]*n[1]*t + n[0]*s;
    R[8] = c + n[2]*n[2]*t;
}
static void test_grid_build(void) {
    int *src, *dst; double *vec; int E;
    int rc = torque_net_build_grid(3, 3, 1, &src, &dst, &vec, &E);
    ASSERT_EQ_INT(rc, 0);
    /* 3×3 periodic grid: each node has 4 outgoing directed edges = 36. */
    ASSERT_EQ_INT(E, 36);
    free(src); free(dst); free(vec);
    rc = torque_net_build_grid(2, 2, 0, &src, &dst, &vec, &E);
    ASSERT_EQ_INT(rc, 0);
    /* 2×2 open BC: each node has 2 neighbours → 8 directed edges. */
    ASSERT_EQ_INT(E, 8);
    free(src); free(dst); free(vec);
}
static void test_equivariance_random_rotations(void) {
    /* 4×4 periodic grid. 10 random rotations. */
    int Lx = 4, Ly = 4, N = Lx * Ly;
    int *src, *dst; double *vec; int E;
    torque_net_build_grid(Lx, Ly, 1, &src, &dst, &vec, &E);
    torque_net_graph_t g = {.num_nodes = N,.num_edges = E,
.edge_src = src,.edge_dst = dst,
.edge_vec = vec };
    torque_net_params_t p = {
.w0 = 0.7,.w1 = -1.3,.w2 = 0.4,.w3 = 0.9,.w4 = -0.5,
.r_cut = 1.5,.radial_order = 6.0
    };
    double *m = malloc((size_t)3 * N * sizeof(double));
    unsigned seed = 0xA1B1;
    random_unit_vectors(N, m, &seed);
    double max_residual = 0.0;
    for (int t = 0; t < 10; t++) {
        double axis[3];
        random_unit_vectors(1, axis, &seed);
        seed = seed * 1103515245u + 12345u;
        double angle = 2.0 * M_PI * ((seed % 10000) / 10000.0);
        double R[9];
        rotation_matrix_from_axis_angle(axis, angle, R);
        double r = torque_net_equivariance_residual(&g, m, &p, R);
        if (r > max_residual) max_residual = r;
    }
    printf("# torque net equivariance: max residual over 10 rotations = %.3e\n",
           max_residual);
    ASSERT_TRUE(max_residual < 1e-10);
    free(m); free(src); free(dst); free(vec);
}
static void test_trivial_identity_rotation(void) {
    /* Rotation = identity → zero residual. */
    int Lx = 2, Ly = 2, N = 4;
    int *src, *dst; double *vec; int E;
    torque_net_build_grid(Lx, Ly, 0, &src, &dst, &vec, &E);
    torque_net_graph_t g = {.num_nodes = N,.num_edges = E,
.edge_src = src,.edge_dst = dst,
.edge_vec = vec };
    torque_net_params_t p = { .w0 = 0.5, .w1 = -0.3, .w2 = 1.1, .w3 = -0.7, .w4 = 0.2, .r_cut = 1.5, .radial_order = 6.0 };
    double m[12];
    unsigned seed = 0x1234;
    random_unit_vectors(N, m, &seed);
    double R[9] = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };
    double res = torque_net_equivariance_residual(&g, m, &p, R);
    ASSERT_TRUE(res < 1e-12);
    free(src); free(dst); free(vec);
}
static void test_null_args_return_error(void) {
    torque_net_params_t p = { 0 };
    int rc = torque_net_forward(NULL, NULL, &p, NULL);
    ASSERT_EQ_INT(rc, -1);
    int *ss; int *dd; double *vv; int EE;
    rc = torque_net_build_grid(-1, 2, 0, &ss, &dd, &vv, &EE);
    ASSERT_EQ_INT(rc, -1);
}
static void test_trivial_state_gives_consistent_torque(void) {
    /* All-parallel m (aligned with +z): symmetry implies τ_i aligned
     * with z-axis (or zero) for every site. */
    int Lx = 4, Ly = 4, N = Lx * Ly;
    int *src, *dst; double *vec; int E;
    torque_net_build_grid(Lx, Ly, 1, &src, &dst, &vec, &E);
    torque_net_graph_t g = {.num_nodes = N,.num_edges = E,
.edge_src = src,.edge_dst = dst,
.edge_vec = vec };
    torque_net_params_t p = { .w0 = 1.0, .w1 = 0.0, .w2 = 1.0, .w3 = 1.0, .w4 = 1.0, .r_cut = 1.5, .radial_order = 6.0 };
    double *m = malloc((size_t)3 * N * sizeof(double));
    for (int i = 0; i < N; i++) { m[3*i] = 0; m[3*i+1] = 0; m[3*i+2] = 1; }
    double *tau = malloc((size_t)3 * N * sizeof(double));
    torque_net_forward(&g, m, &p, tau);
    double max_xy = 0;
    for (int i = 0; i < N; i++) {
        double xy = sqrt(tau[3*i]*tau[3*i] + tau[3*i+1]*tau[3*i+1]);
        if (xy > max_xy) max_xy = xy;
    }
    printf("# uniform-z config: max |τ_xy| across sites = %.3e\n", max_xy);
    ASSERT_TRUE(max_xy < 1e-12);
    free(m); free(tau); free(src); free(dst); free(vec);
}
static void test_fit_recovers_synthetic_weights(void) {
    /* Generate training data from a known parameter set and verify
     * the fitter recovers those weights to machine precision. Since
     * the forward pass is linear in {w0..w4}, the 5×5 normal
     * equations should solve exactly in one pass. */
    int Lx = 3, Ly = 3, N = Lx * Ly;
    int *src, *dst; double *vec; int E;
    torque_net_build_grid(Lx, Ly, 1, &src, &dst, &vec, &E);
    torque_net_graph_t g = {.num_nodes = N,.num_edges = E,
.edge_src = src,.edge_dst = dst,
.edge_vec = vec };
    torque_net_params_t p_true = {
.w0 = 0.37,.w1 = -1.15,.w2 = 0.42,.w3 = 0.89,.w4 = -0.5,
.r_cut = 1.5,.radial_order = 6.0
    };
    int num_samples = 40;
    double *m_batch   = malloc((size_t)3 * N * num_samples * sizeof(double));
    double *tau_batch = malloc((size_t)3 * N * num_samples * sizeof(double));
    unsigned seed = 0xCAFE;
    random_unit_vectors(N * num_samples, m_batch, &seed);
    for (int s = 0; s < num_samples; s++) {
        double *ms   = &m_batch[(size_t)s * 3 * N];
        double *ts   = &tau_batch[(size_t)s * 3 * N];
        torque_net_forward(&g, ms, &p_true, ts);
    }
    torque_net_params_t p_template = { .r_cut = 1.5, .radial_order = 6.0 };
    torque_net_params_t p_fit;
    double residual = -1.0;
    int rc = torque_net_fit_weights(&g, m_batch, tau_batch, num_samples,
                                     &p_template, &p_fit, &residual);
    ASSERT_EQ_INT(rc, 0);
    printf("# torque net fit:\n"
           "#   true  w = [%6.3f  %6.3f  %6.3f  %6.3f  %6.3f]\n"
           "#   fit   w = [%6.3f  %6.3f  %6.3f  %6.3f  %6.3f]  residual=%.3e\n",
           p_true.w0, p_true.w1, p_true.w2, p_true.w3, p_true.w4,
           p_fit.w0,  p_fit.w1,  p_fit.w2,  p_fit.w3,  p_fit.w4,
           residual);
    ASSERT_NEAR(p_fit.w0, p_true.w0, 1e-10);
    ASSERT_NEAR(p_fit.w1, p_true.w1, 1e-10);
    ASSERT_NEAR(p_fit.w2, p_true.w2, 1e-10);
    ASSERT_NEAR(p_fit.w3, p_true.w3, 1e-10);
    ASSERT_NEAR(p_fit.w4, p_true.w4, 1e-10);
    ASSERT_TRUE(residual < 1e-10);
    free(m_batch); free(tau_batch);
    free(src); free(dst); free(vec);
}
/* The torque_net output is fed to LLG as B_eff, which is t-odd.
 * The basis splits: t-odd terms are {w1, w3, w4, w6, w8}; t-even
 * terms are {w0, w2, w5, w7}.  A strict-t-odd parameter vector must
 * pass τ(−m) + τ(m) = 0; mixed t-parity does not.  zero_t_even fixes
 * a mixed parameter vector by zeroing the four t-even weights. */
static void test_t_odd_subset_passes_t_reversal(void) {
    int Lx = 3, Ly = 3, N = Lx * Ly;
    int *src, *dst; double *vec; int E;
    torque_net_build_grid(Lx, Ly, 1, &src, &dst, &vec, &E);
    torque_net_graph_t g = {.num_nodes = N,.num_edges = E,
.edge_src = src,.edge_dst = dst,
.edge_vec = vec };
    double *m = malloc((size_t)3 * N * sizeof(double));
    unsigned seed = 0xCAFEBEEF;
    random_unit_vectors(N, m, &seed);

    /* Strict-t-odd parameters: w0=w2=w5=w7=0. */
    torque_net_params_t p_odd = {
        .w1 = -0.45, .w3 = 0.18, .w4 = -0.27, .w6 = -0.19, .w8 = -0.36,
        .r_cut = 1.5, .radial_order = 6.0
    };
    double r_odd = torque_net_time_reversal_residual(&g, m, &p_odd);
    printf("# t-odd subset residual: %.3e\n", r_odd);
    ASSERT_TRUE(r_odd < 1e-12);

    /* Strict-t-even parameters give the OPPOSITE residual identity:
     * τ(−m) = +τ(m), so τ(m) + τ(−m) = 2τ(m) → residual = 2.0 once
     * normalised by ||τ(m)||. */
    torque_net_params_t p_even = {
        .w0 = 0.31, .w2 = 0.62, .w5 = 0.42, .w7 = 0.53,
        .r_cut = 1.5, .radial_order = 6.0
    };
    double r_even = torque_net_time_reversal_residual(&g, m, &p_even);
    printf("# t-even subset residual: %.3e (expect ~2)\n", r_even);
    ASSERT_TRUE(fabs(r_even - 2.0) < 1e-10);

    /* zero_t_even on a mixed param vector must produce a t-odd result. */
    torque_net_params_t p_mixed = {
        .w0 = 0.31, .w1 = -0.45, .w2 = 0.62, .w3 = 0.18, .w4 = -0.27,
        .w5 = 0.42, .w6 = -0.19, .w7 = 0.53, .w8 = -0.36,
        .r_cut = 1.5, .radial_order = 6.0
    };
    torque_net_zero_t_even_weights(&p_mixed);
    double r_zeroed = torque_net_time_reversal_residual(&g, m, &p_mixed);
    printf("# zero_t_even residual: %.3e\n", r_zeroed);
    ASSERT_TRUE(r_zeroed < 1e-12);
    ASSERT_NEAR(p_mixed.w0, 0.0, 0.0);
    ASSERT_NEAR(p_mixed.w2, 0.0, 0.0);
    ASSERT_NEAR(p_mixed.w5, 0.0, 0.0);
    ASSERT_NEAR(p_mixed.w7, 0.0, 0.0);

    free(m); free(src); free(dst); free(vec);
}

/* L=2 quadrupolar features (w5..w8) must each remain SO(3)-covariant
 * on their own.  We isolate them by zeroing w0..w4 and testing
 * equivariance term-by-term. */
static void test_l2_terms_each_equivariant(void) {
    int Lx = 3, Ly = 3, N = Lx * Ly;
    int *src, *dst; double *vec; int E;
    torque_net_build_grid(Lx, Ly, 1, &src, &dst, &vec, &E);
    torque_net_graph_t g = {.num_nodes = N,.num_edges = E,
.edge_src = src,.edge_dst = dst,
.edge_vec = vec };
    double *m = malloc((size_t)3 * N * sizeof(double));
    unsigned seed = 0x1234;
    random_unit_vectors(N, m, &seed);

    /* Build a 90° rotation about z. */
    double R[9] = { 0, -1, 0,
                    1,  0, 0,
                    0,  0, 1 };

    /* Test each L=2 weight in isolation. */
    double weights[4] = { 0.7, -0.4, 1.1, 0.3 };
    for (int idx = 0; idx < 4; idx++) {
        torque_net_params_t p = { .r_cut = 1.5, .radial_order = 6.0 };
        switch (idx) {
            case 0: p.w5 = weights[idx]; break;
            case 1: p.w6 = weights[idx]; break;
            case 2: p.w7 = weights[idx]; break;
            case 3: p.w8 = weights[idx]; break;
        }
        double resid = torque_net_equivariance_residual(&g, m, &p, R);
        ASSERT_TRUE(resid < 1e-10);
    }

    free(m); free(src); free(dst); free(vec);
}

/* The 9-term fitter must recover all 9 weights from data generated by
 * the same 9-term forward pass. */
static void test_fit_recovers_l2_weights(void) {
    int Lx = 3, Ly = 3, N = Lx * Ly;
    int *src, *dst; double *vec; int E;
    torque_net_build_grid(Lx, Ly, 1, &src, &dst, &vec, &E);
    torque_net_graph_t g = {.num_nodes = N,.num_edges = E,
.edge_src = src,.edge_dst = dst,
.edge_vec = vec };
    torque_net_params_t p_true = {
.w0 = 0.21,.w1 = -0.55,.w2 = 0.33,.w3 = 0.66,.w4 = -0.18,
.w5 = 0.42,.w6 = -0.27,.w7 = 0.13,.w8 = -0.31,
.r_cut = 1.5,.radial_order = 6.0
    };
    int num_samples = 80;          /* > 9 weights, well-conditioned */
    double *m_batch   = malloc((size_t)3 * N * num_samples * sizeof(double));
    double *tau_batch = malloc((size_t)3 * N * num_samples * sizeof(double));
    unsigned seed = 0xBADCAFE;
    random_unit_vectors(N * num_samples, m_batch, &seed);
    for (int s = 0; s < num_samples; s++) {
        double *ms = &m_batch[(size_t)s * 3 * N];
        double *ts = &tau_batch[(size_t)s * 3 * N];
        torque_net_forward(&g, ms, &p_true, ts);
    }
    torque_net_params_t p_template = { .r_cut = 1.5, .radial_order = 6.0 };
    torque_net_params_t p_fit;
    double residual = -1.0;
    int rc = torque_net_fit_weights(&g, m_batch, tau_batch, num_samples,
                                     &p_template, &p_fit, &residual);
    ASSERT_EQ_INT(rc, 0);
    printf("# 9-term fit residual = %.3e\n", residual);
    ASSERT_NEAR(p_fit.w0, p_true.w0, 1e-9);
    ASSERT_NEAR(p_fit.w1, p_true.w1, 1e-9);
    ASSERT_NEAR(p_fit.w2, p_true.w2, 1e-9);
    ASSERT_NEAR(p_fit.w3, p_true.w3, 1e-9);
    ASSERT_NEAR(p_fit.w4, p_true.w4, 1e-9);
    ASSERT_NEAR(p_fit.w5, p_true.w5, 1e-9);
    ASSERT_NEAR(p_fit.w6, p_true.w6, 1e-9);
    ASSERT_NEAR(p_fit.w7, p_true.w7, 1e-9);
    ASSERT_NEAR(p_fit.w8, p_true.w8, 1e-9);
    ASSERT_TRUE(residual < 1e-10);
    free(m_batch); free(tau_batch);
    free(src); free(dst); free(vec);
}

int main(void) {
    TEST_RUN(test_grid_build);
    TEST_RUN(test_equivariance_random_rotations);
    TEST_RUN(test_trivial_identity_rotation);
    TEST_RUN(test_null_args_return_error);
    TEST_RUN(test_trivial_state_gives_consistent_torque);
    TEST_RUN(test_fit_recovers_synthetic_weights);
    TEST_RUN(test_l2_terms_each_equivariant);
    TEST_RUN(test_fit_recovers_l2_weights);
    TEST_RUN(test_t_odd_subset_passes_t_reversal);
    TEST_SUMMARY();
}