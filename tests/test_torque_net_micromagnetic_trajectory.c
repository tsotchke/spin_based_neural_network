/*
 * tests/test_torque_net_micromagnetic_trajectory.c
 *
 * End-to-end LLG-pillar trajectory benchmark — the µMAG-lite slice.
 *
 * For a small periodic grid, generate a reference LLG trajectory from
 * a known analytic effective field (Heisenberg exchange + Zeeman),
 * accumulate (m_t, B_t) pairs along the trajectory, fit torque_net to
 * those pairs, then run torque_net-driven LLG from the same initial
 * condition and compare trajectories.
 *
 * The Heisenberg exchange B_i^{Heis} = J · Σ_{j∈NN} m_j is exactly
 * the torque_net w4 basis term; the Zeeman B_i = h ẑ is captured by
 * the constant offset.  The fit must therefore recover the trajectory
 * to integration precision, not just static-config precision.
 *
 * This is the equivariant-LLG pillar's first end-to-end physical
 * validation:  if torque_net fits a *trajectory* (not just snapshots)
 * and the LLG integrator reproduces the reference dynamics, the
 * pipeline is sound.
 *
 * v0.5 µMAG #1, #3, #4 will lift this to standard reference problems
 * (currently µMAG-lite uses a synthetic but well-defined target).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "equivariant_gnn/torque_net.h"
#include "llg/llg.h"

typedef struct {
    const torque_net_graph_t *graph;
    torque_net_params_t       params;
    double                    h_ext_z;   /* Zeeman field along z */
} ref_field_user_t;

/* Reference effective field: torque_net forward with planted Heisenberg
 * weights, plus a per-site Zeeman ẑ contribution. */
static void ref_field_fn(const double *m, double *b_eff,
                         long num_sites, void *user) {
    ref_field_user_t *u = (ref_field_user_t *)user;
    torque_net_forward(u->graph, m, &u->params, b_eff);
    for (long i = 0; i < num_sites; i++) {
        b_eff[3 * i + 2] += u->h_ext_z;
    }
}

static void random_unit_vectors(int N, double *out, unsigned long long *rng) {
    for (int i = 0; i < N; i++) {
        unsigned long long x = *rng;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17; *rng = x;
        double u1 = (double)(x >> 11) / 9007199254740992.0;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17; *rng = x;
        double u2 = (double)(x >> 11) / 9007199254740992.0;
        double theta = 2.0 * M_PI * u2;
        double z     = 2.0 * u1 - 1.0;
        double sp    = sqrt(1.0 - z * z);
        out[3 * i]     = sp * cos(theta);
        out[3 * i + 1] = sp * sin(theta);
        out[3 * i + 2] = z;
    }
}

/* Run LLG forward num_steps RK4 steps, saving snapshots at every step. */
static void roll_trajectory(const llg_config_t *cfg,
                            const double *m_init, int N, int num_steps,
                            double *snapshots /* [num_steps+1, 3*N] */) {
    long stride = 3 * N;
    memcpy(snapshots, m_init, (size_t)stride * sizeof(double));
    double *m = (double *)malloc((size_t)stride * sizeof(double));
    memcpy(m, m_init, (size_t)stride * sizeof(double));
    for (int s = 0; s < num_steps; s++) {
        llg_rk4_step(cfg, m, N);
        memcpy(&snapshots[(size_t)(s + 1) * stride], m,
               (size_t)stride * sizeof(double));
    }
    free(m);
}

/* End-to-end: reference run → fit → fitted-net run → compare. */
static void test_heisenberg_zeeman_trajectory_recovers_reference(void) {
    int Lx = 3, Ly = 3, N = Lx * Ly;
    int *src, *dst; double *vec; int E;
    torque_net_build_grid(Lx, Ly, 1, &src, &dst, &vec, &E);
    torque_net_graph_t g = { .num_nodes = N, .num_edges = E,
                             .edge_src = src, .edge_dst = dst,
                             .edge_vec = vec };

    /* Planted physics: Heisenberg w4 = +J = 0.8 plus Zeeman h_z = 0.3. */
    const double J     = 0.8;
    const double h_ext = 0.3;
    ref_field_user_t ref_u = {
        .graph   = &g,
        .params  = { .w4 = J, .r_cut = 1.5, .radial_order = 6.0 },
        .h_ext_z = h_ext
    };

    /* Reference LLG run.  Modest dt and damping so the trajectory is
     * smooth and finite-precision is dominated by integration error,
     * not stiffness. */
    int    num_steps = 40;
    double dt        = 1e-3;
    llg_config_t cfg_ref = llg_config_defaults();
    cfg_ref.gamma           = 1.0;
    cfg_ref.alpha           = 0.05;
    cfg_ref.dt              = dt;
    cfg_ref.field_fn        = ref_field_fn;
    cfg_ref.field_user_data = &ref_u;

    double *m0 = (double *)malloc((size_t)3 * N * sizeof(double));
    unsigned long long rng = 0xC0FFEEFACEULL;
    random_unit_vectors(N, m0, &rng);

    long stride = 3 * N;
    double *snap_ref = (double *)malloc((size_t)(num_steps + 1) * stride * sizeof(double));
    roll_trajectory(&cfg_ref, m0, N, num_steps, snap_ref);

    /* Build the training set from the reference trajectory: each
     * snapshot is one (m, B_eff) pair.  The Zeeman component is
     * subtracted out so the fitter only sees the Heisenberg part —
     * the Zeeman bias is added back at inference time. */
    double *m_batch  = (double *)malloc((size_t)(num_steps + 1) * stride * sizeof(double));
    double *b_batch  = (double *)malloc((size_t)(num_steps + 1) * stride * sizeof(double));
    for (int s = 0; s <= num_steps; s++) {
        const double *ms = &snap_ref[(size_t)s * stride];
        double *bs       = &b_batch [(size_t)s * stride];
        memcpy(&m_batch[(size_t)s * stride], ms, (size_t)stride * sizeof(double));
        ref_field_fn(ms, bs, N, &ref_u);
        /* Strip Zeeman to leave the Heisenberg-only signal. */
        for (int i = 0; i < N; i++) bs[3 * i + 2] -= h_ext;
    }

    /* Fit torque_net to the trajectory data. */
    torque_net_params_t p_template = { .r_cut = 1.5, .radial_order = 6.0 };
    torque_net_params_t p_fit;
    double residual = -1.0;
    int rc = torque_net_fit_weights(&g, m_batch, b_batch,
                                     num_steps + 1,
                                     &p_template, &p_fit, &residual);
    ASSERT_EQ_INT(rc, 0);
    printf("# trajectory-fit residual = %.3e   w4 fit = %.6f (truth = %.6f)\n",
           residual, p_fit.w4, J);
    ASSERT_TRUE(residual < 1e-9);
    ASSERT_NEAR(p_fit.w4, J, 1e-6);

    /* Re-run LLG from the same initial condition with the trained
     * net's effective field, plus the same Zeeman bias. */
    ref_field_user_t fit_u = {
        .graph   = &g,
        .params  = p_fit,
        .h_ext_z = h_ext
    };
    llg_config_t cfg_fit = cfg_ref;
    cfg_fit.field_user_data = &fit_u;

    double *snap_fit = (double *)malloc((size_t)(num_steps + 1) * stride * sizeof(double));
    roll_trajectory(&cfg_fit, m0, N, num_steps, snap_fit);

    /* Trajectory comparison: per-snapshot L∞ on the difference. */
    double max_diff = 0.0;
    for (int s = 0; s <= num_steps; s++) {
        const double *mr = &snap_ref[(size_t)s * stride];
        const double *mf = &snap_fit[(size_t)s * stride];
        for (long k = 0; k < stride; k++) {
            double d = fabs(mr[k] - mf[k]);
            if (d > max_diff) max_diff = d;
        }
    }
    printf("# µMAG-lite trajectory L∞ error after %d RK4 steps: %.3e\n",
           num_steps, max_diff);
    /* Expect machine-precision agreement: same physics, same integrator,
     * same starting state.  Allow a generous floor for FP-rounding
     * accumulation across 40 RK4 steps. */
    ASSERT_TRUE(max_diff < 1e-9);

    free(m0); free(snap_ref); free(snap_fit);
    free(m_batch); free(b_batch);
    free(src); free(dst); free(vec);
}

int main(void) {
    TEST_RUN(test_heisenberg_zeeman_trajectory_recovers_reference);
    TEST_SUMMARY();
}
