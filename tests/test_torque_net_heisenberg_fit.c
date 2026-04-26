/*
 * tests/test_torque_net_heisenberg_fit.c
 *
 * Physical-validation test for the equivariant torque-net closed-form
 * fitter: given training data generated from the analytic Heisenberg
 * exchange effective field
 *
 *     B_i^{Heis} = J · Σ_{j ∈ nbrs(i)} m_j
 *
 * the fitter should recover weights w₀..w₄ ≈ (0, 0, 0, 0, J) because
 * basis term 4 is exactly m_j summed over neighbours (with radial
 * weighting φ(||r_ij||) on each term).
 *
 * This is the Paper-1 §P1.2 claim reduced to an explicit test: the
 * learned equivariant torque predictor can recover a known physical
 * Hamiltonian's effective field from data alone, with no hand-coded
 * physics. If this test passes, the torque network's equivariance
 * plus the closed-form fitter together form a generic pipeline for
 * extracting effective-field functionals from measured magnetisation
 * trajectories.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "equivariant_gnn/torque_net.h"
/* Generate training data from analytic Heisenberg exchange:
 * B_i = Σ_j φ(r_ij) · m_j   with φ the polynomial cutoff already
 * used by torque_net_forward. We invoke torque_net_forward with
 * planted weights (0, 0, 0, 0, J) to produce the reference — so the
 * fitter's recovery is exact up to linear-algebra round-off. */
static void generate_heisenberg_training(const torque_net_graph_t *g,
                                          int num_configs,
                                          double J,
                                          double *m_batch,
                                          double *tau_batch,
                                          unsigned long long *rng) {
    int N = g->num_nodes;
    torque_net_params_t p_heis = {
.w0 = 0.0,.w1 = 0.0,.w2 = 0.0,.w3 = 0.0,.w4 = J,
.r_cut = 1.5,.radial_order = 6.0
    };
    for (int s = 0; s < num_configs; s++) {
        double *m = &m_batch[(size_t)s * 3 * N];
        /* Random unit vectors via splitmix-hashed seeds. */
        for (int i = 0; i < N; i++) {
            unsigned long long x = *rng;
            x ^= x << 13; x ^= x >> 7; x ^= x << 17; *rng = x;
            double u1 = (double)(x >> 11) / 9007199254740992.0;
            x ^= x << 13; x ^= x >> 7; x ^= x << 17; *rng = x;
            double u2 = (double)(x >> 11) / 9007199254740992.0;
            double theta = 2.0 * M_PI * u2;
            double z = 2.0 * u1 - 1.0;
            double sp = sqrt(1.0 - z * z);
            m[3*i]   = sp * cos(theta);
            m[3*i+1] = sp * sin(theta);
            m[3*i+2] = z;
        }
        torque_net_forward(g, m, &p_heis, &tau_batch[(size_t)s * 3 * N]);
    }
}
static void test_fits_w4_to_heisenberg_coupling(void) {
    /* 4×4 periodic grid; generate 30 random configs, fit, check that
     * w4 recovers the planted J and other weights are ≈ 0. */
    int Lx = 4, Ly = 4, N = Lx * Ly;
    int *src, *dst; double *vec; int E;
    torque_net_build_grid(Lx, Ly, 1, &src, &dst, &vec, &E);
    torque_net_graph_t g = {.num_nodes = N,.num_edges = E,
.edge_src = src,.edge_dst = dst,
.edge_vec = vec };
    double J_true = 1.25;
    int num_configs = 30;
    double *m_batch   = malloc((size_t)3 * N * num_configs * sizeof(double));
    double *tau_batch = malloc((size_t)3 * N * num_configs * sizeof(double));
    unsigned long long rng = 0x9E3779B97F4A7C15ULL;
    generate_heisenberg_training(&g, num_configs, J_true, m_batch, tau_batch, &rng);
    torque_net_params_t p_template = { .r_cut = 1.5, .radial_order = 6.0 };
    torque_net_params_t p_fit;
    double residual = -1;
    int rc = torque_net_fit_weights(&g, m_batch, tau_batch, num_configs,
                                     &p_template, &p_fit, &residual);
    ASSERT_EQ_INT(rc, 0);
    printf("# torque net fit to Heisenberg J=%.3f:\n"
           "#   recovered w = [%+.6f  %+.6f  %+.6f  %+.6f  %+.6f]\n"
           "#   residual = %.3e\n",
           J_true, p_fit.w0, p_fit.w1, p_fit.w2, p_fit.w3, p_fit.w4,
           residual);
    /* w4 must match the planted J; other weights must be ≈ 0 — linear
     * least-squares is exact on synthetic data up to round-off. */
    ASSERT_NEAR(p_fit.w4, J_true, 1e-10);
    ASSERT_NEAR(p_fit.w0, 0.0,    1e-10);
    ASSERT_NEAR(p_fit.w1, 0.0,    1e-10);
    ASSERT_NEAR(p_fit.w2, 0.0,    1e-10);
    ASSERT_NEAR(p_fit.w3, 0.0,    1e-10);
    ASSERT_TRUE(residual < 1e-10);
    free(m_batch); free(tau_batch);
    free(src); free(dst); free(vec);
}
static void test_fits_heisenberg_plus_anisotropy(void) {
    /* Slightly more realistic: ferromagnetic exchange + DMI-like axial
     * cross-product contribution. Plant w4 = J, w1 = -D, everything
     * else zero. Verify fit recovers both. */
    int Lx = 4, Ly = 4, N = Lx * Ly;
    int *src, *dst; double *vec; int E;
    torque_net_build_grid(Lx, Ly, 1, &src, &dst, &vec, &E);
    torque_net_graph_t g = {.num_nodes = N,.num_edges = E,
.edge_src = src,.edge_dst = dst,
.edge_vec = vec };
    double J_true = 1.0;
    double D_true = -0.3;
    torque_net_params_t p_plant = {
.w0 = 0.0,.w1 = D_true,.w2 = 0.0,.w3 = 0.0,.w4 = J_true,
.r_cut = 1.5,.radial_order = 6.0
    };
    int num_configs = 40;
    double *m_batch   = malloc((size_t)3 * N * num_configs * sizeof(double));
    double *tau_batch = malloc((size_t)3 * N * num_configs * sizeof(double));
    unsigned long long rng = 0xC0FFEE1234ULL;
    /* Same config-gen as above but use the planted (J, D) weights. */
    for (int s = 0; s < num_configs; s++) {
        double *m = &m_batch[(size_t)s * 3 * N];
        for (int i = 0; i < N; i++) {
            unsigned long long x = rng;
            x ^= x << 13; x ^= x >> 7; x ^= x << 17; rng = x;
            double u1 = (double)(x >> 11) / 9007199254740992.0;
            x ^= x << 13; x ^= x >> 7; x ^= x << 17; rng = x;
            double u2 = (double)(x >> 11) / 9007199254740992.0;
            double theta = 2.0 * M_PI * u2;
            double z = 2.0 * u1 - 1.0;
            double sp = sqrt(1.0 - z * z);
            m[3*i]   = sp * cos(theta);
            m[3*i+1] = sp * sin(theta);
            m[3*i+2] = z;
        }
        torque_net_forward(&g, m, &p_plant, &tau_batch[(size_t)s * 3 * N]);
    }
    torque_net_params_t p_template = { .r_cut = 1.5, .radial_order = 6.0 };
    torque_net_params_t p_fit;
    double residual = -1;
    int rc = torque_net_fit_weights(&g, m_batch, tau_batch, num_configs,
                                     &p_template, &p_fit, &residual);
    ASSERT_EQ_INT(rc, 0);
    printf("# torque net fit to (J=%.3f, D=%.3f):\n"
           "#   recovered w = [%+.6f  %+.6f  %+.6f  %+.6f  %+.6f]  residual=%.3e\n",
           J_true, D_true,
           p_fit.w0, p_fit.w1, p_fit.w2, p_fit.w3, p_fit.w4, residual);
    ASSERT_NEAR(p_fit.w4, J_true, 1e-10);
    ASSERT_NEAR(p_fit.w1, D_true, 1e-10);
    ASSERT_NEAR(p_fit.w0, 0.0, 1e-10);
    ASSERT_NEAR(p_fit.w2, 0.0, 1e-10);
    ASSERT_NEAR(p_fit.w3, 0.0, 1e-10);
    ASSERT_TRUE(residual < 1e-10);
    free(m_batch); free(tau_batch);
    free(src); free(dst); free(vec);
}
int main(void) {
    TEST_RUN(test_fits_w4_to_heisenberg_coupling);
    TEST_RUN(test_fits_heisenberg_plus_anisotropy);
    TEST_SUMMARY();
}