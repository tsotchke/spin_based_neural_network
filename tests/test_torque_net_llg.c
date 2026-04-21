/*
 * tests/test_torque_net_llg.c
 *
 * End-to-end: LLG integrator driven by the equivariant torque
 * predictor. Exercises the adapter that plugs torque-net output into
 * the llg_effective_field_fn callback slot and verifies basic
 * physical consistency:
 *
 *   (1) At zero Gilbert damping, ⟨m_z⟩ averaged over a rotationally
 *       symmetric initial condition is conserved over short time.
 *   (2) Renormalisation projects |m| = 1 after each step.
 *   (3) Zero torque → zero motion.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "llg/llg.h"
#include "equivariant_gnn/torque_net.h"
#include "equivariant_gnn/llg_adapter.h"
static double mean_mz(const double *m, long N) {
    double s = 0.0;
    for (long i = 0; i < N; i++) s += m[3*i + 2];
    return s / (double)N;
}
static double max_unit_drift(const double *m, long N) {
    double d = 0.0;
    for (long i = 0; i < N; i++) {
        double x = m[3*i], y = m[3*i+1], z = m[3*i+2];
        double n = sqrt(x*x + y*y + z*z);
        double e = fabs(n - 1.0);
        if (e > d) d = e;
    }
    return d;
}
static void test_unit_norm_preserved_under_learned_torque(void) {
    /* Random torque weights, RK4 steps; |m|=1 must stay to 1e-10. */
    int Lx = 4, Ly = 4, N = Lx * Ly;
    int *src, *dst; double *vec; int E;
    torque_net_build_grid(Lx, Ly, 1, &src, &dst, &vec, &E);
    torque_net_graph_t g = {.num_nodes = N,.num_edges = E,
.edge_src = src,.edge_dst = dst,
.edge_vec = vec };
    torque_net_params_t p = { 0.4, -0.2, 0.5, 0.1, -0.3, 1.5, 6.0 };
    llg_torque_user_t user = {.graph = &g,.params = &p };
    double *m = malloc((size_t)3 * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        /* Random unit vector via spherical angles, deterministic. */
        double u = (double)i / (double)N;
        double theta = 2.0 * M_PI * u;
        double phi   = M_PI * (0.25 + 0.5 * u);
        m[3*i]   = sin(phi) * cos(theta);
        m[3*i+1] = sin(phi) * sin(theta);
        m[3*i+2] = cos(phi);
    }
    llg_config_t cfg = llg_config_defaults();
    cfg.gamma = 1.0;    /* normalise to O(1) units for numerical test  */
    cfg.alpha = 0.0;
    cfg.dt    = 1e-3;
    cfg.field_fn = llg_torque_field_fn;
    cfg.field_user_data = &user;
    for (int s = 0; s < 200; s++) llg_rk4_step(&cfg, m, N);
    double drift = max_unit_drift(m, N);
    printf("# LLG-τnet 200 RK4 steps: max ||m|-1| = %.3e\n", drift);
    ASSERT_TRUE(drift < 1e-10);
    free(m); free(src); free(dst); free(vec);
}
static void test_zero_torque_means_zero_motion(void) {
    int Lx = 3, Ly = 3, N = Lx * Ly;
    int *src, *dst; double *vec; int E;
    torque_net_build_grid(Lx, Ly, 1, &src, &dst, &vec, &E);
    torque_net_graph_t g = {.num_nodes = N,.num_edges = E,
.edge_src = src,.edge_dst = dst,
.edge_vec = vec };
    torque_net_params_t p = { 0, 0, 0, 0, 0, 1.5, 6.0 };     /* zero weights */
    llg_torque_user_t user = {.graph = &g,.params = &p };
    double m0[27], m[27];
    for (int i = 0; i < N; i++) {
        m0[3*i]   =  sin((double)i);
        m0[3*i+1] =  cos((double)i);
        m0[3*i+2] =  0.3;
    }
    /* Normalise once so m0 lies on the sphere. */
    for (int i = 0; i < N; i++) {
        double nn = sqrt(m0[3*i]*m0[3*i] + m0[3*i+1]*m0[3*i+1] + m0[3*i+2]*m0[3*i+2]);
        m0[3*i] /= nn; m0[3*i+1] /= nn; m0[3*i+2] /= nn;
    }
    memcpy(m, m0, sizeof(m));
    llg_config_t cfg = llg_config_defaults();
    cfg.gamma = 1.0;
    cfg.alpha = 0.01;
    cfg.dt    = 1e-3;
    cfg.field_fn = llg_torque_field_fn;
    cfg.field_user_data = &user;
    for (int s = 0; s < 100; s++) llg_rk4_step(&cfg, m, N);
    double max_diff = 0;
    for (int i = 0; i < 3*N; i++) {
        double d = fabs(m[i] - m0[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("# LLG-τnet zero weights: max ||m - m₀||_∞ = %.3e\n", max_diff);
    ASSERT_TRUE(max_diff < 1e-10);
    free(src); free(dst); free(vec);
}
static void test_uniform_z_stays_along_z(void) {
    /* A configuration with the rotational symmetry of the torque-net
     * output (see test_trivial_state_gives_consistent_torque in the
     * torque-net test) is a fixed point of the LLG equation: τ has
     * no component in the tangent space orthogonal to m. */
    int Lx = 4, Ly = 4, N = Lx * Ly;
    int *src, *dst; double *vec; int E;
    torque_net_build_grid(Lx, Ly, 1, &src, &dst, &vec, &E);
    torque_net_graph_t g = {.num_nodes = N,.num_edges = E,
.edge_src = src,.edge_dst = dst,
.edge_vec = vec };
    torque_net_params_t p = { 1.0, 0.0, 1.0, 1.0, 1.0, 1.5, 6.0 };
    llg_torque_user_t user = {.graph = &g,.params = &p };
    double m[48];
    for (int i = 0; i < N; i++) { m[3*i] = 0; m[3*i+1] = 0; m[3*i+2] = 1; }
    llg_config_t cfg = llg_config_defaults();
    cfg.gamma = 1.0; cfg.alpha = 0.0;
    cfg.dt = 1e-3;
    cfg.field_fn = llg_torque_field_fn;
    cfg.field_user_data = &user;
    for (int s = 0; s < 100; s++) llg_rk4_step(&cfg, m, N);
    double mz = mean_mz(m, N);
    double max_xy = 0;
    for (int i = 0; i < N; i++) {
        double xy = sqrt(m[3*i]*m[3*i] + m[3*i+1]*m[3*i+1]);
        if (xy > max_xy) max_xy = xy;
    }
    printf("# LLG-τnet uniform-z: ⟨m_z⟩=%.6f, max |m_xy|=%.3e\n", mz, max_xy);
    ASSERT_NEAR(mz, 1.0, 1e-10);
    ASSERT_TRUE(max_xy < 1e-10);
    free(src); free(dst); free(vec);
}
int main(void) {
    TEST_RUN(test_zero_torque_means_zero_motion);
    TEST_RUN(test_uniform_z_stays_along_z);
    TEST_RUN(test_unit_norm_preserved_under_learned_torque);
    TEST_SUMMARY();
}