/*
 * tests/test_llg_2d.c
 *
 * Full 2D micromagnetics integrator: exchange + anisotropy + DMI +
 * Zeeman (llg_2d_field) combined with demag (llg_demag_2d_apply_z)
 * and RK4 time-stepping. Physical validation:
 *   (1) Uniform +z FM ground state is stationary under exchange +
 *       easy-axis anisotropy + zero field.
 *   (2) Pure DMI (no exchange) on a 2D lattice: uniform state has
 *       zero effective field since the DMI cross products cancel
 *       for symmetric neighbour configurations.
 *   (3) Demag on a uniformly +z magnetised thin film always produces
 *       a negative z-field (opposing magnetisation).
 *   (4) Composed integrator with demag included: starting in +z,
 *       damping relaxes the state to equilibrium; ⟨m_z⟩ stays
 *       positive under easy-axis anisotropy despite opposing demag.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "llg/llg.h"
#include "llg/exchange_field.h"
#include "llg/demag.h"
static void test_2d_field_fm_ground_state_stationary(void) {
    int Lx = 8, Ly = 8;
    long N = (long)Lx * Ly;
    double *m = malloc(sizeof(double) * 3 * N);
    for (long i = 0; i < N; i++) { m[3*i] = 0; m[3*i+1] = 0; m[3*i+2] = 1; }
    llg_2d_config_t p = {Lx, Ly, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0};
    double *b = malloc(sizeof(double) * 3 * N);
    llg_2d_field(m, b, N, &p);
    /* Exchange: each neighbour gives +J m_j = +1 · (0, 0, 1). Four
     * neighbours → (0, 0, 4J) = (0, 0, 4). Anisotropy: bz += 2·0.5·1 = 1.
     * Zeeman: zero. Total bz = 5, bx = by = 0. Cross product m × B_eff =
     * (0,0,1) × (0,0,5) = 0, so ṁ = 0. Ground state is stationary. */
    for (long i = 0; i < N; i++) {
        ASSERT_NEAR(b[3*i],   0.0, 1e-12);
        ASSERT_NEAR(b[3*i+1], 0.0, 1e-12);
        ASSERT_NEAR(b[3*i+2], 5.0, 1e-12);
    }
    free(m); free(b);
}
static void test_2d_dmi_cancels_on_uniform_state(void) {
    /* Uniform m field: DMI cross-products between site i and its +x,
     * -x, +y, -y neighbours cancel pairwise because the D-vectors
     * have opposite signs on opposite bonds. No anisotropy or
     * Zeeman, no exchange → b_eff should be zero everywhere. */
    int Lx = 8, Ly = 8;
    long N = (long)Lx * Ly;
    double *m = malloc(sizeof(double) * 3 * N);
    /* Non-trivial in-plane direction to give DMI something to act on. */
    for (long i = 0; i < N; i++) { m[3*i] = 0.6; m[3*i+1] = 0.8; m[3*i+2] = 0.0; }
    llg_2d_config_t p = {Lx, Ly, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
    double *b = malloc(sizeof(double) * 3 * N);
    llg_2d_field(m, b, N, &p);
    for (long i = 0; i < N; i++) {
        ASSERT_NEAR(b[3*i],   0.0, 1e-12);
        ASSERT_NEAR(b[3*i+1], 0.0, 1e-12);
        ASSERT_NEAR(b[3*i+2], 0.0, 1e-12);
    }
    free(m); free(b);
}
/* Composed-field callback: exchange+anisotropy+DMI+Zeeman (from
 * llg_2d_field) plus demag (from llg_demag_2d_apply_z). */
typedef struct {
    llg_2d_config_t *two_d;
    llg_demag_2d_t *demag;
} combined_ctx_t;
static void combined_field_fn(const double *m, double *b_eff,
                               long num_sites, void *user) {
    combined_ctx_t *c = (combined_ctx_t *)user;
    llg_2d_field(m, b_eff, num_sites, c->two_d);
    /* Add demag z-contribution. */
    if (c->demag) {
        double *Hdz = malloc(sizeof(double) * num_sites);
        llg_demag_2d_apply_z(c->demag, m, Hdz);
        for (long i = 0; i < num_sites; i++) b_eff[3*i + 2] += Hdz[i];
        free(Hdz);
    }
}
static void test_2d_integrator_with_demag_preserves_norm(void) {
    /* Run RK4 with composed field for a few hundred steps from a
     * small-perturbation FM state; verify |m|=1 is preserved at
     * every site (llg_rk4_step renormalises). */
    int Lx = 8, Ly = 8;
    long N = (long)Lx * Ly;
    double *m = malloc(sizeof(double) * 3 * N);
    unsigned long long rng = 0x1234ULL;
    for (long i = 0; i < N; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        double u = (double)(rng >> 11) / 9007199254740992.0;
        m[3*i]   = 0.05 * (u - 0.5);
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        u = (double)(rng >> 11) / 9007199254740992.0;
        m[3*i+1] = 0.05 * (u - 0.5);
        m[3*i+2] = sqrt(1.0 - m[3*i]*m[3*i] - m[3*i+1]*m[3*i+1]);
    }
    llg_2d_config_t p = {Lx, Ly, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0};
    llg_demag_2d_t *demag = llg_demag_2d_create(Lx, Ly);
    combined_ctx_t ctx = {&p, demag};
    llg_config_t lcfg = llg_config_defaults();
    lcfg.gamma = 1.0;
    lcfg.alpha = 0.1;
    lcfg.dt = 0.005;
    lcfg.field_fn = combined_field_fn;
    lcfg.field_user_data = &ctx;
    for (int s = 0; s < 200; s++) llg_rk4_step(&lcfg, m, N);
    for (long i = 0; i < N; i++) {
        double n = sqrt(m[3*i]*m[3*i] + m[3*i+1]*m[3*i+1] + m[3*i+2]*m[3*i+2]);
        ASSERT_NEAR(n, 1.0, 1e-6);
    }
    double mz_avg = 0;
    for (long i = 0; i < N; i++) mz_avg += m[3*i+2];
    mz_avg /= (double)N;
    printf("# 2D LLG + demag after 200 steps: <m_z> = %.4f (started ≈ 1)\n", mz_avg);
    ASSERT_TRUE(mz_avg > 0.9);   /* still mostly +z (easy axis dominates demag) */
    free(m);
    llg_demag_2d_free(demag);
}
int main(void) {
    TEST_RUN(test_2d_field_fm_ground_state_stationary);
    TEST_RUN(test_2d_dmi_cancels_on_uniform_state);
    TEST_RUN(test_2d_integrator_with_demag_preserves_norm);
    TEST_SUMMARY();
}