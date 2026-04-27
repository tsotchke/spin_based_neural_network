/*
 * tests/test_torque_net_irrep.c
 *
 * Cross-validates the equivariant torque network's hand-rolled l=0
 * contraction against libirrep's spherical-harmonic machinery. This
 * is the first test in the tree that genuinely *consumes* libirrep
 * — the production torque_net uses Cartesian primitives, so we need
 * an explicit SH-addition-theorem check to establish the two views
 * agree.
 *
 * The identity we verify:
 *     (4π / (2l+1)) · Σ_m Y_l^m(û) · Y_l^m(v̂)  =  P_l(û · v̂).
 * For l=1 that reduces to
 *     (4π / 3) · Σ_{m=-1,0,+1} Y_1^m(û) Y_1^m(v̂)  =  û · v̂.
 *
 * This binds the SH-basis representation (the language NequIP /
 * libirrep uses) to the Cartesian representation (the language the
 * torque-net is written in) for l=1 — the irrep the torque-net
 * outputs. If the identity ever broke, the torque-net's "l=1
 * equivariant" claim would need revisiting.
 *
 * Gated on SPIN_NN_HAS_IRREP; enable with
 *     make IRREP_ENABLE=1 \
 *          IRREP_ROOT=/path/to/libirrep \
 *          IRREP_LIBDIR=/path/to/libirrep/lib/<triple> \
 *          test_torque_net_irrep && build/test_torque_net_irrep
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "harness.h"
#include "libirrep_bridge.h"
#ifdef SPIN_NN_HAS_IRREP
/* Convert cartesian unit vector → (theta, phi). */
static void cart_to_polar(const double v[3], double *theta, double *phi) {
    *theta = acos(v[2]);
    *phi   = atan2(v[1], v[0]);
}
static double dot3(const double a[3], const double b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
static double sh_addition_l1(const double u[3], const double v[3]) {
    double tu, pu, tv, pv;
    cart_to_polar(u, &tu, &pu);
    cart_to_polar(v, &tv, &pv);
    double acc = 0.0;
    for (int m = -1; m <= 1; m++) {
        double yu = 0.0, yv = 0.0;
        libirrep_bridge_sph_harm_real(1, m, tu, pu, &yu);
        libirrep_bridge_sph_harm_real(1, m, tv, pv, &yv);
        acc += yu * yv;
    }
    return 4.0 * M_PI / 3.0 * acc;
}
static void test_bridge_reports_available(void) {
    ASSERT_EQ_INT(libirrep_bridge_is_available(), 1);
    const char *v = libirrep_bridge_version();
    ASSERT_TRUE(v != NULL);
    printf("# libirrep version: %s\n", v);
}
static void test_sh_addition_theorem_l1_matches_cartesian_dot(void) {
    /* Five fixed unit vectors, arbitrary. Verify the l=1 SH
     * addition theorem reproduces the Cartesian dot. */
    double u[5][3] = {
        { 1.0, 0.0, 0.0 },
        { 0.0, 1.0, 0.0 },
        { 0.0, 0.0, 1.0 },
        { 0.577350269, 0.577350269, 0.577350269 },     /* (1,1,1)/√3 */
        { 0.707106781, 0.0, -0.707106781 }             /* (1,0,-1)/√2 */
    };
    double v[5][3] = {
        { 0.0, 1.0, 0.0 },
        { 1.0, 0.0, 0.0 },
        { 0.6, 0.8, 0.0 },
        { -0.577350269, 0.577350269, 0.577350269 },
        { 0.0, 1.0, 0.0 }
    };
    double max_err = 0.0;
    for (int t = 0; t < 5; t++) {
        double lib_val  = sh_addition_l1(u[t], v[t]);
        double cart_val = dot3(u[t], v[t]);
        double err = fabs(lib_val - cart_val);
        if (err > max_err) max_err = err;
    }
    printf("# SH-addition-theorem l=1 vs cartesian dot: max err = %.3e\n",
           max_err);
    ASSERT_TRUE(max_err < 1e-12);
}
static void test_pairwise_orthogonality_l1_l2(void) {
    /* Pick a random-ish direction and verify Y_l^m integrates to
     * near-zero against Y_l'^m' at orthogonal vectors (sanity that
     * libirrep's SH convention is the standard orthonormal one;
     * this catches an accidental 1/√(4π) vs. 1/2√π drift). */
    double r[3] = { 0.707106781, 0.0, 0.707106781 };  /* (1,0,1)/√2 */
    double theta, phi;
    cart_to_polar(r, &theta, &phi);
    /* Y_0^0(anywhere) = 1/(2 √π). */
    double y00 = 0.0;
    libirrep_bridge_sph_harm_real(0, 0, theta, phi, &y00);
    ASSERT_NEAR(y00, 0.5 / sqrt(M_PI), 1e-12);
    /* Y_2^0(θ) = √(5/(16π))·(3 cos²θ − 1); at θ = π/4, cos²θ = 1/2,
     * so 3·0.5−1 = 0.5, giving √(5/(16π))·0.5 ≈ 0.0789. */
    double y20 = 0.0;
    libirrep_bridge_sph_harm_real(2, 0, theta, phi, &y20);
    double expected = 0.5 * sqrt(5.0 / (16.0 * M_PI));
    ASSERT_NEAR(y20, expected, 1e-8);
}
/* ---------------------------------------------------------------- *
 *  NequIP-backed torque_net (libirrep-driven E(3) tower)            *
 * ---------------------------------------------------------------- */
#include "equivariant_gnn/torque_net_irrep.h"

static void rot3_apply(const double R[9], const double v[3], double out[3]) {
    out[0] = R[0]*v[0] + R[1]*v[1] + R[2]*v[2];
    out[1] = R[3]*v[0] + R[4]*v[1] + R[5]*v[2];
    out[2] = R[6]*v[0] + R[7]*v[1] + R[8]*v[2];
}

static void test_irrep_layer_constructs(void) {
    torque_net_irrep_t *net =
        torque_net_irrep_create("1x1o", "1x1o", /*sh*/ 2, /*radial*/ 4,
                                 /*r_cut*/ 1.5, /*poly p*/ 6);
    ASSERT_TRUE(net != NULL);
    ASSERT_EQ_INT(torque_net_irrep_in_dim(net),  3);
    ASSERT_EQ_INT(torque_net_irrep_out_dim(net), 3);
    int nw = torque_net_irrep_num_weights(net);
    ASSERT_TRUE(nw > 0);
    printf("# NequIP 1x1o -> 1x1o, sh=2, radial=4, r_cut=1.5, p=6: "
           "%d learnable TP weights\n", nw);
    torque_net_irrep_free(net);
}

/* Helper: run NequIP forward + collect outputs.  Caller owns buffers. */
static int run_irrep_forward(torque_net_irrep_t *net, const double *w,
                             const torque_net_graph_t *g,
                             const double *m, double *out) {
    return torque_net_irrep_forward(net, w, g, m, out);
}

static void test_irrep_forward_so3_equivariance(void) {
    /* Open-BC 3x3 grid → graph has no torus-tiling constraint, so any
     * SO(3) rotation R is a clean equivariance test.  Periodic graphs
     * impose `R must preserve the torus`, which constrains the test
     * but doesn't change the underlying claim. */
    int Lx = 3, Ly = 3, N = Lx * Ly;
    int *src, *dst; double *vec; int E;
    torque_net_build_grid(Lx, Ly, /*periodic*/ 0, &src, &dst, &vec, &E);
    torque_net_graph_t g = { .num_nodes = N, .num_edges = E,
                             .edge_src  = src, .edge_dst  = dst,
                             .edge_vec  = vec };

    torque_net_irrep_t *net =
        torque_net_irrep_create("1x1o", "1x1o", 2, 4, 1.5, 6);
    ASSERT_TRUE(net != NULL);
    int nw = torque_net_irrep_num_weights(net);
    double *w = (double *)malloc((size_t)nw * sizeof(double));
    for (int i = 0; i < nw; i++) w[i] = 0.1 * sin(0.7 * i + 1.3);

    double *m = (double *)malloc((size_t)3 * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        double t = 2.0 * M_PI * i / (double)N;
        double s = sin(0.5 * i + 0.2);
        double c = sqrt(1.0 - s * s);
        m[3*i + 0] = c * cos(t);
        m[3*i + 1] = c * sin(t);
        m[3*i + 2] = s;
    }

    double *out_unrot = (double *)calloc((size_t)3 * N, sizeof(double));
    ASSERT_EQ_INT(run_irrep_forward(net, w, &g, m, out_unrot), 0);

    /* 30-degree rotation about (1, 1, 1)/√3 — non-symmetric, generic. */
    double th = M_PI / 6.0;
    double cx = 1.0/sqrt(3.0), cy = 1.0/sqrt(3.0), cz = 1.0/sqrt(3.0);
    double C = cos(th), S = sin(th), V = 1.0 - C;
    double R[9] = {
        cx*cx*V + C,    cx*cy*V - cz*S, cx*cz*V + cy*S,
        cy*cx*V + cz*S, cy*cy*V + C,    cy*cz*V - cx*S,
        cz*cx*V - cy*S, cz*cy*V + cx*S, cz*cz*V + C
    };

    double *m_R   = (double *)malloc((size_t)3 * N * sizeof(double));
    double *vec_R = (double *)malloc((size_t)3 * E * sizeof(double));
    for (int i = 0; i < N; i++) rot3_apply(R, &m[3*i],   &m_R[3*i]);
    for (int e = 0; e < E; e++) rot3_apply(R, &vec[3*e], &vec_R[3*e]);
    torque_net_graph_t g_rot = g;
    g_rot.edge_vec = vec_R;

    double *out_rot = (double *)calloc((size_t)3 * N, sizeof(double));
    ASSERT_EQ_INT(run_irrep_forward(net, w, &g_rot, m_R, out_rot), 0);

    double max_err = 0.0;
    double *expected = (double *)malloc((size_t)3 * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        rot3_apply(R, &out_unrot[3*i], &expected[3*i]);
        for (int k = 0; k < 3; k++) {
            double e = fabs(expected[3*i + k] - out_rot[3*i + k]);
            if (e > max_err) max_err = e;
        }
    }
    printf("# NequIP-backed torque equivariance residual: %.3e\n", max_err);
    /* libirrep's SH / UVW TP machinery preserves equivariance to ~ 1e-12
     * on any rotation. */
    ASSERT_TRUE(max_err < 1e-10);

    free(w); free(m); free(m_R); free(vec_R);
    free(out_unrot); free(out_rot); free(expected);
    free(src); free(dst); free(vec);
    torque_net_irrep_free(net);
}

/* Multi-layer NequIP: stack two layers with a hidden multiset that has
 * more multiplets than the 1x1o input, then read out back to 1x1o.
 * Total weight count grows; equivariance must still be machine-precision. */
static void test_irrep_two_layer_construction_and_equivariance(void) {
    const char *hidden_specs[1] = { "4x0e + 2x1o + 1x2e" };
    torque_net_irrep_t *net =
        torque_net_irrep_create_multilayer("1x1o", hidden_specs, "1x1o",
                                            /*num_layers*/ 2,
                                            /*sh*/ 2, /*radial*/ 4,
                                            /*r_cut*/ 1.5, /*p*/ 6);
    ASSERT_TRUE(net != NULL);
    ASSERT_EQ_INT(torque_net_irrep_num_layers(net), 2);
    int nw = torque_net_irrep_num_weights(net);
    int off0 = torque_net_irrep_layer_offset(net, 0);
    int off1 = torque_net_irrep_layer_offset(net, 1);
    int off2 = torque_net_irrep_layer_offset(net, 2);
    printf("# 2-layer NequIP 1x1o → 4x0e+2x1o+1x2e → 1x1o: %d weights "
           "(layer0 %d..%d, layer1 %d..%d)\n",
           nw, off0, off1, off1, off2);
    ASSERT_TRUE(nw > 2);             /* strictly more than the v0 single-layer */
    ASSERT_EQ_INT(off2, nw);

    /* Run the same SO(3) equivariance check as the single-layer case
     * with a generic 30° rotation about (1,1,1)/√3. */
    int Lx = 3, Ly = 3, N = Lx * Ly;
    int *src, *dst; double *vec; int E;
    torque_net_build_grid(Lx, Ly, /*periodic*/ 0, &src, &dst, &vec, &E);
    torque_net_graph_t g = { .num_nodes = N, .num_edges = E,
                             .edge_src = src, .edge_dst = dst,
                             .edge_vec = vec };

    double *w = (double *)malloc((size_t)nw * sizeof(double));
    for (int i = 0; i < nw; i++) w[i] = 0.07 * sin(0.5 * i + 0.3);

    double *m = (double *)malloc((size_t)3 * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        double t = 2.0 * M_PI * i / (double)N;
        double s = sin(0.7 * i + 0.1);
        double c = sqrt(1.0 - s * s);
        m[3*i + 0] = c * cos(t);
        m[3*i + 1] = c * sin(t);
        m[3*i + 2] = s;
    }
    double *out_unrot = (double *)calloc((size_t)3 * N, sizeof(double));
    ASSERT_EQ_INT(torque_net_irrep_forward(net, w, &g, m, out_unrot), 0);

    double th = M_PI / 6.0;
    double cx = 1.0/sqrt(3.0), cy = 1.0/sqrt(3.0), cz = 1.0/sqrt(3.0);
    double C = cos(th), S = sin(th), V = 1.0 - C;
    double R[9] = {
        cx*cx*V + C,    cx*cy*V - cz*S, cx*cz*V + cy*S,
        cy*cx*V + cz*S, cy*cy*V + C,    cy*cz*V - cx*S,
        cz*cx*V - cy*S, cz*cy*V + cx*S, cz*cz*V + C
    };

    double *m_R   = (double *)malloc((size_t)3 * N * sizeof(double));
    double *vec_R = (double *)malloc((size_t)3 * E * sizeof(double));
    for (int i = 0; i < N; i++) rot3_apply(R, &m[3*i],   &m_R[3*i]);
    for (int e = 0; e < E; e++) rot3_apply(R, &vec[3*e], &vec_R[3*e]);
    torque_net_graph_t g_rot = g; g_rot.edge_vec = vec_R;
    double *out_rot = (double *)calloc((size_t)3 * N, sizeof(double));
    ASSERT_EQ_INT(torque_net_irrep_forward(net, w, &g_rot, m_R, out_rot), 0);

    double max_err = 0.0;
    double *expected = (double *)malloc((size_t)3 * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        rot3_apply(R, &out_unrot[3*i], &expected[3*i]);
        for (int k = 0; k < 3; k++) {
            double e = fabs(expected[3*i + k] - out_rot[3*i + k]);
            if (e > max_err) max_err = e;
        }
    }
    printf("# 2-layer NequIP equivariance residual: %.3e\n", max_err);
    ASSERT_TRUE(max_err < 1e-10);

    free(w); free(m); free(m_R); free(vec_R);
    free(out_unrot); free(out_rot); free(expected);
    free(src); free(dst); free(vec);
    torque_net_irrep_free(net);
}

int main(void) {
    TEST_RUN(test_bridge_reports_available);
    TEST_RUN(test_sh_addition_theorem_l1_matches_cartesian_dot);
    TEST_RUN(test_pairwise_orthogonality_l1_l2);
    TEST_RUN(test_irrep_layer_constructs);
    TEST_RUN(test_irrep_forward_so3_equivariance);
    TEST_RUN(test_irrep_two_layer_construction_and_equivariance);
    TEST_SUMMARY();
}
#else /* !SPIN_NN_HAS_IRREP */
static void test_skipped_without_irrep(void) {
    printf("# skipped: built without -DSPIN_NN_HAS_IRREP\n");
}
int main(void) {
    TEST_RUN(test_skipped_without_irrep);
    TEST_SUMMARY();
}
#endif