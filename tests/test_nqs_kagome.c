/*
 * tests/test_nqs_kagome.c
 *
 * NQS local-energy kernel for the Heisenberg model on the kagome
 * lattice (NQS_HAM_KAGOME_HEISENBERG). Geometry: Lx_cells × Ly_cells
 * unit cells, 3 sublattices (A, B, C) per cell, coordination four
 * under PBC. Flat site index i = 3·(cx·Ly_cells + cy) + s, s ∈ {0,1,2}.
 *
 * Convention implemented in src/nqs/nqs_gradient.c
 * local_energy_kagome_heisenberg:
 *
 *     H = J · Σ_⟨ij⟩ S_i · S_j  =  (J/4) Σ s_i s_j + (J/2) Σ flip-pair
 *
 * Bond list per unit cell (cx, cy):
 *   Up-triangle : A–B, A–C, B–C (all within cell)
 *   Down-triangle (anchored at A of the cell):
 *     A(cx, cy) – B(cx−1, cy),
 *     A(cx, cy) – C(cx, cy−1),
 *     B(cx−1, cy) – C(cx, cy−1).
 * Under PBC the cell indices wrap. Under OBC a down-triangle is
 * skipped entirely if either required neighbour cell is out of range.
 *
 * For the 2×2 PBC cluster: 4 up-triangles + 4 down-triangles = 24
 * bonds, each site has coordination 4.
 *
 * Checkpoint identity used below:
 *   For a uniform ansatz ψ ≡ 1 (so ψ(s')/ψ(s) = 1 everywhere), the
 *   Heisenberg local energy reduces to
 *
 *       E_loc(s) = (J/4)·(N_par − N_anti) + (J/2)·N_anti
 *                = (J/4)·(N_par + N_anti) = (J/4)·N_bonds,
 *
 *   i.e. independent of the spin configuration s. Exercising this
 *   invariance on several configurations verifies the bond count
 *   and the diagonal/off-diagonal balance.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "harness.h"
#include "nqs/nqs_config.h"
#include "nqs/nqs_gradient.h"

/* Uniform log-amp: ψ ≡ 1 for every configuration. */
static void uniform_log_amp(const int *spins, int num_sites, void *user,
                             double *out_log_abs, double *out_arg) {
    (void)spins; (void)num_sites; (void)user;
    if (out_log_abs) *out_log_abs = 0.0;
    if (out_arg)     *out_arg     = 0.0;
}

static void test_kagome_pbc_all_up_2x2(void) {
    /* 2x2 PBC: 24 bonds; every bond sasb = +1 → diag = J/4 · 24 = 6J.
     * Off-diagonal vanishes (no antiparallel pairs). With J = 1 → E = 6. */
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_KAGOME_HEISENBERG;
    cfg.j_coupling  = 1.0;
    cfg.kagome_pbc  = 1;
    int spins[12];
    for (int i = 0; i < 12; i++) spins[i] = +1;
    double E = nqs_local_energy(&cfg, 2, 2, spins, uniform_log_amp, NULL);
    ASSERT_NEAR(E, 6.0, 1e-12);
}

static void test_kagome_pbc_uniform_invariance(void) {
    /* Uniform-ψ identity: for kagome Heisenberg with ψ ≡ 1, the local
     * energy is (J/4)·|bonds| independent of spin configuration. Flip
     * a specific subset to verify (single-site flip, two-site flip,
     * checkerboard-ish on the 3-sublattice set). */
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_KAGOME_HEISENBERG;
    cfg.j_coupling  = 1.0;
    cfg.kagome_pbc  = 1;
    const double target = 6.0;

    int spins1[12]; for (int i = 0; i < 12; i++) spins1[i] = +1;
    spins1[0] = -1;                               /* flip A of cell (0,0) */
    ASSERT_NEAR(nqs_local_energy(&cfg, 2, 2, spins1, uniform_log_amp, NULL),
                target, 1e-12);

    int spins2[12]; for (int i = 0; i < 12; i++) spins2[i] = +1;
    spins2[0] = -1; spins2[7] = -1;               /* flip A(0,0) and B(1,0) */
    ASSERT_NEAR(nqs_local_energy(&cfg, 2, 2, spins2, uniform_log_amp, NULL),
                target, 1e-12);

    int spins3[12]; /* Polarise one sublattice up, others down. */
    for (int c = 0; c < 4; c++) {
        spins3[3*c + 0] = +1;  /* A */
        spins3[3*c + 1] = -1;  /* B */
        spins3[3*c + 2] = -1;  /* C */
    }
    ASSERT_NEAR(nqs_local_energy(&cfg, 2, 2, spins3, uniform_log_amp, NULL),
                target, 1e-12);
}

static void test_kagome_scales_linearly_with_J(void) {
    /* Same all-up configuration, J = 2 → E = 12. */
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_KAGOME_HEISENBERG;
    cfg.j_coupling  = 2.0;
    cfg.kagome_pbc  = 1;
    int spins[12]; for (int i = 0; i < 12; i++) spins[i] = +1;
    double E = nqs_local_energy(&cfg, 2, 2, spins, uniform_log_amp, NULL);
    ASSERT_NEAR(E, 12.0, 1e-12);
}

static void test_kagome_obc_all_up_2x2(void) {
    /* 2x2 OBC: 4 up-triangles unchanged; down-triangles require cxm ≥ 0
     * AND cym ≥ 0, so only the one anchored at (1, 1) survives.
     * Total = 4·3 + 1·3 = 15 bonds. All-up → E = J/4 · 15 = 3.75. */
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_KAGOME_HEISENBERG;
    cfg.j_coupling  = 1.0;
    cfg.kagome_pbc  = 0;
    int spins[12]; for (int i = 0; i < 12; i++) spins[i] = +1;
    double E = nqs_local_energy(&cfg, 2, 2, spins, uniform_log_amp, NULL);
    ASSERT_NEAR(E, 3.75, 1e-12);
}

static void test_kagome_1x1_pbc_has_degenerate_self_bonds(void) {
    /* Regression guard for the 1×1 PBC edge case.  With Lx=Ly=1 the
     * down-triangle anchored at A(0,0) wraps back to the same cell,
     * so its three bonds coincide with the up-triangle. The current
     * kernel counts both, producing an effective multiplicity of 2
     * on every bond.
     *
     * This is a geometric quirk of the 1×1 PBC kagome; normal
     * research-size clusters (Lx, Ly ≥ 2) don't hit it. The test
     * exists to pin the behaviour so it doesn't silently change:
     * E(all-up, J=1, Lx=Ly=1, PBC) = (J/4) · 6 = 1.5. If a future
     * refactor deduplicates bonds it will return 0.75 and this test
     * will deliberately fail, signalling that callers of 1×1 PBC
     * need a new expected value. */
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_KAGOME_HEISENBERG;
    cfg.j_coupling  = 1.0;
    cfg.kagome_pbc  = 1;
    int spins[3] = {+1, +1, +1};
    double E = nqs_local_energy(&cfg, 1, 1, spins, uniform_log_amp, NULL);
    ASSERT_NEAR(E, 1.5, 1e-12);
}

/* ------------------------------------------------------------------
 * Non-uniform ansatz test. The uniform-ψ tests above verify the bond
 * enumeration and diagonal/off-diagonal balance but share a symmetry
 * that can mask bugs in the amplitude-ratio path: with ψ ≡ 1 every
 * flip-pair ratio ψ(s')/ψ(s) = 1, so a kernel that multiplied the
 * off-diagonal contribution by the wrong constant but happened to
 * absorb the error into a matching diagonal bug would pass. This
 * test exercises the ratio path with a non-trivial per-site ansatz.
 *
 * Ansatz: ψ(s) = exp(Σ_i α_i s_i) — log ψ is linear in s, so the
 * flip-pair ratio on bond (u, v) reduces to
 *   ψ(s ⊕ {u,v}) / ψ(s)  =  exp(-2·(α_u s_u + α_v s_v))
 * which depends on the specific pair flipped. Distinct α_i per site
 * breaks the total-magnetisation invariance that the simpler
 * exp(α·Σ s_i) choice would preserve.
 * ------------------------------------------------------------------ */
typedef struct { const double *alpha; } linear_ansatz_ctx_t;

static void linear_log_amp(const int *spins, int num_sites, void *user,
                            double *out_log_abs, double *out_arg) {
    linear_ansatz_ctx_t *ctx = (linear_ansatz_ctx_t *)user;
    double lp = 0.0;
    for (int i = 0; i < num_sites; i++) lp += ctx->alpha[i] * (double)spins[i];
    if (out_log_abs) *out_log_abs = lp;
    if (out_arg)     *out_arg     = 0.0;
}

static void test_kagome_nonuniform_ansatz_1x1_pbc(void) {
    /* Uses the 1×1 PBC degenerate geometry where the kernel counts
     * each of the 3 bonds twice (up-triangle and down-triangle coincide
     * under wrap). With per-site α = (0.1, 0.2, 0.3) and spins
     * s = (+1, −1, +1), J = 1:
     *
     *   Bond (0, 1): s_0 s_1 = −1 (antipar). Each of 2 occurrences:
     *     diag = (J/4)·(−1) = −0.25, off ratio
     *     = exp(−2(0.1·(+1) + 0.2·(−1))) = exp(+0.2).
     *     per-bond off = (J/2)·exp(+0.2). Two bonds contribute
     *     2·0.5·exp(+0.2) = exp(+0.2).
     *   Bond (0, 2): s_0 s_2 = +1 (par). diag per bond = +0.25, off = 0.
     *     Two bonds: diag total +0.5.
     *   Bond (1, 2): s_1 s_2 = −1 (antipar). Each of 2 occurrences:
     *     diag = −0.25, off ratio
     *     = exp(−2(0.2·(−1) + 0.3·(+1))) = exp(−0.2).
     *     Two bonds contribute exp(−0.2).
     *
     *   Total: diag = −0.5 + 0.5 + (−0.5) = −0.5.
     *          off  = exp(+0.2) + exp(−0.2) = 2·cosh(0.2).
     *          E_loc = −0.5 + 2·cosh(0.2). */
    double alpha[3] = {0.1, 0.2, 0.3};
    linear_ansatz_ctx_t ctx = { .alpha = alpha };
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_KAGOME_HEISENBERG;
    cfg.j_coupling  = 1.0;
    cfg.kagome_pbc  = 1;
    int spins[3] = {+1, -1, +1};
    double E = nqs_local_energy(&cfg, 1, 1, spins, linear_log_amp, &ctx);
    double expected = -0.5 + 2.0 * cosh(0.2);
    ASSERT_NEAR(E, expected, 1e-12);
}

static void test_kagome_nonuniform_ansatz_2x2_pbc(void) {
    /* Non-uniform linear ansatz on the research-standard 2×2 PBC
     * geometry (N=12, 24 unique bonds, coord 4). Expected local
     * energy is computed inline by independently iterating the same
     * bond list the kernel uses — not a clean cross-check against
     * an external source, but catches off-by-one / mispaired-bond
     * bugs by splitting the bond enumeration into two independently
     * written pieces. Both must agree for the test to pass.
     *
     * Spin config: flip two sites (0 and 7) to -1, rest +1.
     * Linear ansatz α_i = 0.05·(i+1). */
    double alpha[12];
    for (int i = 0; i < 12; i++) alpha[i] = 0.05 * (double)(i + 1);
    linear_ansatz_ctx_t ctx = { .alpha = alpha };
    int spins[12];
    for (int i = 0; i < 12; i++) spins[i] = +1;
    spins[0] = -1;
    spins[7] = -1;

    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_KAGOME_HEISENBERG;
    cfg.j_coupling  = 1.0;
    cfg.kagome_pbc  = 1;

    /* Independent bond list enumeration. The 24 bonds on a 2×2 PBC
     * kagome cluster under our convention (up-triangle intra-cell +
     * down-triangle anchored at A(cx,cy) with PBC wrap). */
    int bonds[24][2] = {
        /* up-triangles per cell (0..3) */
        { 0, 1}, { 0, 2}, { 1, 2},     /* cell (0,0) */
        { 3, 4}, { 3, 5}, { 4, 5},     /* cell (0,1) */
        { 6, 7}, { 6, 8}, { 7, 8},     /* cell (1,0) */
        { 9,10}, { 9,11}, {10,11},     /* cell (1,1) */
        /* down-triangles anchored at A(cx,cy), cxm=(cx-1)%Lx, cym=(cy-1)%Ly */
        { 0, 7}, { 0, 5}, { 7, 5},     /* at (0,0): {A=0, B(1,0)=7, C(0,1)=5} */
        { 3,10}, { 3, 2}, {10, 2},     /* at (0,1): {A=3, B(1,1)=10, C(0,0)=2} */
        { 6, 1}, { 6,11}, { 1,11},     /* at (1,0): {A=6, B(0,0)=1, C(1,1)=11} */
        { 9, 4}, { 9, 8}, { 4, 8},     /* at (1,1): {A=9, B(0,1)=4, C(1,0)=8} */
    };

    double J = 1.0;
    double expected_diag = 0.0, expected_off = 0.0;
    for (int b = 0; b < 24; b++) {
        int u = bonds[b][0], v = bonds[b][1];
        int su = spins[u], sv = spins[v];
        expected_diag += 0.25 * J * (double)(su * sv);
        if (su != sv) {
            /* ratio ψ(s')/ψ(s) = exp(-2 α_u s_u - 2 α_v s_v) */
            double r = exp(-2.0 * (alpha[u] * (double)su + alpha[v] * (double)sv));
            expected_off += 0.5 * J * r;
        }
    }
    double expected = expected_diag + expected_off;

    double E = nqs_local_energy(&cfg, 2, 2, spins, linear_log_amp, &ctx);
    ASSERT_NEAR(E, expected, 1e-12);
}

int main(void) {
    TEST_RUN(test_kagome_pbc_all_up_2x2);
    TEST_RUN(test_kagome_pbc_uniform_invariance);
    TEST_RUN(test_kagome_scales_linearly_with_J);
    TEST_RUN(test_kagome_obc_all_up_2x2);
    TEST_RUN(test_kagome_1x1_pbc_has_degenerate_self_bonds);
    TEST_RUN(test_kagome_nonuniform_ansatz_1x1_pbc);
    TEST_RUN(test_kagome_nonuniform_ansatz_2x2_pbc);
    TEST_SUMMARY();
}
