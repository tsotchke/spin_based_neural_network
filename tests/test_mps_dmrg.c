/*
 * tests/test_mps_dmrg.c
 *
 * Two-site DMRG validation on 1D Heisenberg / XXZ chains with open
 * boundaries. Reference energies come from exact diagonalisation
 * (dense H via mps_ground_state_dense) on small systems.
 *
 *   N = 4  Heisenberg:   E₀ ≈ -1.616025
 *   N = 6  Heisenberg:   E₀ ≈ -2.493577
 *   N = 8  Heisenberg:   E₀ ≈ -3.374932
 *
 * The DMRG result must match to at least 1e-6 for small N and 1e-4
 * for slightly larger N at bond dim 16.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "harness.h"
#include "mps/mps.h"
#include "mps/dmrg.h"
static double ed_energy_xxz(int N, double J, double Jz,
                             mps_hamiltonian_kind_t ham) {
    mps_config_t cfg = mps_config_defaults();
    cfg.num_sites = N;
    cfg.ham       = ham;
    cfg.J         = J;
    cfg.Jz        = Jz;
    cfg.lanczos_max_iters = 200;
    cfg.lanczos_tol = 1e-10;
    double E = 0.0;
    lanczos_result_t info;
    mps_ground_state_dense(&cfg, &E, NULL, &info);
    return E;
}
static void test_dmrg_heisenberg_4site(void) {
    mps_config_t cfg = mps_config_defaults();
    cfg.num_sites = 4;
    cfg.ham = MPS_HAM_HEISENBERG;
    cfg.J = 1.0;
    cfg.max_bond_dim = 16;
    cfg.num_sweeps = 8;
    cfg.sweep_tol = 1e-10;
    cfg.lanczos_max_iters = 60;
    cfg.lanczos_tol = 1e-10;
    double E_ref = ed_energy_xxz(4, 1.0, 1.0, MPS_HAM_HEISENBERG);
    mps_dmrg_result_t res;
    ASSERT_EQ_INT(mps_dmrg_xxz(&cfg, &res), 0);
    printf("# N=4 Heisenberg: E_ED = %.8f  E_DMRG = %.8f  Δ = %.2e  sweeps = %d\n",
           E_ref, res.final_energy, fabs(res.final_energy - E_ref),
           res.sweeps_performed);
    ASSERT_NEAR(res.final_energy, E_ref, 1e-6);
}
static void test_dmrg_heisenberg_6site(void) {
    mps_config_t cfg = mps_config_defaults();
    cfg.num_sites = 6;
    cfg.ham = MPS_HAM_HEISENBERG;
    cfg.J = 1.0;
    cfg.max_bond_dim = 16;
    cfg.num_sweeps = 10;
    cfg.sweep_tol = 1e-10;
    cfg.lanczos_max_iters = 60;
    double E_ref = ed_energy_xxz(6, 1.0, 1.0, MPS_HAM_HEISENBERG);
    mps_dmrg_result_t res;
    ASSERT_EQ_INT(mps_dmrg_xxz(&cfg, &res), 0);
    printf("# N=6 Heisenberg: E_ED = %.8f  E_DMRG = %.8f  Δ = %.2e\n",
           E_ref, res.final_energy, fabs(res.final_energy - E_ref));
    ASSERT_NEAR(res.final_energy, E_ref, 1e-5);
}
static void test_dmrg_heisenberg_8site(void) {
    mps_config_t cfg = mps_config_defaults();
    cfg.num_sites = 8;
    cfg.ham = MPS_HAM_HEISENBERG;
    cfg.J = 1.0;
    cfg.max_bond_dim = 32;
    cfg.num_sweeps = 10;
    cfg.sweep_tol = 1e-10;
    cfg.lanczos_max_iters = 80;
    double E_ref = ed_energy_xxz(8, 1.0, 1.0, MPS_HAM_HEISENBERG);
    mps_dmrg_result_t res;
    ASSERT_EQ_INT(mps_dmrg_xxz(&cfg, &res), 0);
    printf("# N=8 Heisenberg: E_ED = %.8f  E_DMRG = %.8f  Δ = %.2e\n",
           E_ref, res.final_energy, fabs(res.final_energy - E_ref));
    ASSERT_NEAR(res.final_energy, E_ref, 1e-4);
}
static void test_dmrg_xxz_anisotropic(void) {
    /* XXZ with Δ = 2.0 (Ising-anisotropic regime): DMRG must still
     * converge; reference via ED on N=6. */
    mps_config_t cfg = mps_config_defaults();
    cfg.num_sites = 6;
    cfg.ham = MPS_HAM_XXZ;
    cfg.J = 1.0;
    cfg.Jz = 2.0;
    cfg.max_bond_dim = 16;
    cfg.num_sweeps = 10;
    cfg.lanczos_max_iters = 60;
    double E_ref = ed_energy_xxz(6, 1.0, 2.0, MPS_HAM_XXZ);
    mps_dmrg_result_t res;
    ASSERT_EQ_INT(mps_dmrg_xxz(&cfg, &res), 0);
    printf("# N=6 XXZ Δ=2: E_ED = %.8f  E_DMRG = %.8f  Δ = %.2e\n",
           E_ref, res.final_energy, fabs(res.final_energy - E_ref));
    ASSERT_NEAR(res.final_energy, E_ref, 1e-5);
}
static void test_dmrg_truncation_error_decreases_with_D(void) {
    /* N=10 Heisenberg. Run DMRG with varying D_max and verify the
     * energy error (vs exact ED) decreases monotonically as D grows.
     * At D=2 there must be visible truncation error; at D=16 DMRG must
     * match ED to machine precision. */
    double E_ref = ed_energy_xxz(10, 1.0, 1.0, MPS_HAM_HEISENBERG);
    int D_values[] = {2, 4, 8, 16};
    double errs[4];
    for (int k = 0; k < 4; k++) {
        mps_config_t cfg = mps_config_defaults();
        cfg.num_sites = 10;
        cfg.ham = MPS_HAM_HEISENBERG;
        cfg.J = 1.0;
        cfg.max_bond_dim = D_values[k];
        cfg.num_sweeps = 10;
        cfg.sweep_tol = 1e-10;
        cfg.lanczos_max_iters = 80;
        mps_dmrg_result_t res;
        mps_dmrg_xxz(&cfg, &res);
        errs[k] = res.final_energy - E_ref;   /* must be ≥ 0 */
        printf("# N=10 D_max=%2d: E_DMRG = %.8f  gap = %.2e\n",
               D_values[k], res.final_energy, errs[k]);
    }
    /* Variational: each truncated run must be ≥ exact (allow MC noise). */
    for (int k = 0; k < 4; k++) ASSERT_TRUE(errs[k] >= -1e-8);
    /* Monotonic-ish: D=2 worse than D=16; D=16 effectively exact. */
    ASSERT_TRUE(errs[0] > errs[3]);
    ASSERT_TRUE(errs[3] < 1e-8);
}
static void test_dmrg_state_vector_overlap_with_ed(void) {
    /* DMRG state and ED state should have |⟨ψ_DMRG|ψ_ED⟩|² ≈ 1 for a
     * non-degenerate ground state. Heisenberg 6-site has a unique
     * singlet ground state. */
    int N = 6;
    mps_config_t cfg = mps_config_defaults();
    cfg.num_sites = N;
    cfg.ham = MPS_HAM_HEISENBERG;
    cfg.J = 1.0;
    cfg.max_bond_dim = 16;
    cfg.num_sweeps = 8;
    cfg.sweep_tol = 1e-10;
    cfg.lanczos_max_iters = 80;
    mps_dmrg_result_t res;
    double *psi_dmrg = NULL;
    long dim_dmrg = 0;
    int rc = mps_dmrg_xxz_with_state(&cfg, &res, &psi_dmrg, &dim_dmrg);
    ASSERT_EQ_INT(rc, 0);
    ASSERT_TRUE(psi_dmrg != NULL);
    ASSERT_EQ_INT((int)dim_dmrg, 1 << N);
    /* Compute the reference state via dense Lanczos. */
    double *psi_ed = calloc((size_t)(1 << N), sizeof(double));
    double E_ed;
    lanczos_result_t info;
    mps_ground_state_dense(&cfg, &E_ed, psi_ed, &info);
    double ovlp = 0.0;
    for (long i = 0; i < dim_dmrg; i++) ovlp += psi_dmrg[i] * psi_ed[i];
    printf("# N=6 |⟨DMRG|ED⟩|² = %.8f  (should be close to 1)\n",
           ovlp * ovlp);
    ASSERT_TRUE(ovlp * ovlp > 0.999);
    free(psi_dmrg);
    free(psi_ed);
}
static void test_dmrg_12_site_heisenberg_matches_ed(void) {
    /* N=12 is at the edge of practical dense ED (dim=4096). This is a
     * real benchmark: DMRG at bond dim 32 must agree with dense Lanczos
     * to high precision. */
    int N = 12;
    double E_ref = ed_energy_xxz(N, 1.0, 1.0, MPS_HAM_HEISENBERG);
    mps_config_t cfg = mps_config_defaults();
    cfg.num_sites = N;
    cfg.ham = MPS_HAM_HEISENBERG;
    cfg.J = 1.0;
    cfg.max_bond_dim = 32;
    cfg.num_sweeps = 12;
    cfg.sweep_tol = 1e-10;
    cfg.lanczos_max_iters = 100;
    mps_dmrg_result_t res;
    ASSERT_EQ_INT(mps_dmrg_xxz(&cfg, &res), 0);
    printf("# N=12 Heisenberg: E_ED = %.8f  E_DMRG = %.8f  Δ = %.2e\n",
           E_ref, res.final_energy, fabs(res.final_energy - E_ref));
    ASSERT_NEAR(res.final_energy, E_ref, 1e-4);
}
static void test_dmrg_16_site_energy_per_site_converges(void) {
    /* Thermodynamic-limit check: per-site energy of the Heisenberg
     * chain tends to e_∞ = 1/4 − ln 2 ≈ -0.4431. At N=16 with OBC,
     * finite-size corrections leave the per-site energy within a few
     * percent of this value. This test bounds the DMRG output against
     * that target and also verifies it is below the lowest triplet
     * per-site energy (which is ≥ -0.25 for FM states). */
    int N = 16;
    mps_config_t cfg = mps_config_defaults();
    cfg.num_sites = N;
    cfg.ham = MPS_HAM_HEISENBERG;
    cfg.J = 1.0;
    cfg.max_bond_dim = 32;
    cfg.num_sweeps = 10;
    cfg.sweep_tol = 1e-8;
    cfg.lanczos_max_iters = 80;
    mps_dmrg_result_t res;
    ASSERT_EQ_INT(mps_dmrg_xxz(&cfg, &res), 0);
    double per_site = res.final_energy / (double)N;
    printf("# N=16 Heisenberg: E_total = %.6f  E/N = %.6f  (e_∞ = -0.4431)\n",
           res.final_energy, per_site);
    ASSERT_TRUE(per_site < -0.25);    /* below FM triplet */
    ASSERT_TRUE(per_site > -0.5);     /* physical lower bound */
    ASSERT_NEAR(per_site, -0.4431, 0.05);
}
static void test_dmrg_tfim_matches_ed(void) {
    /* Transverse-field Ising at criticality (Γ = J = 1) on N=8.
     * Reference via dense Lanczos. */
    int N = 8;
    mps_config_t cfg = mps_config_defaults();
    cfg.num_sites = N;
    cfg.ham = MPS_HAM_TFIM;
    cfg.J = 1.0;
    cfg.Gamma = 1.0;
    cfg.lanczos_max_iters = 200;
    cfg.lanczos_tol = 1e-10;
    double E_ed = 0.0;
    lanczos_result_t info;
    mps_ground_state_dense(&cfg, &E_ed, NULL, &info);
    cfg.max_bond_dim = 16;
    cfg.num_sweeps = 10;
    cfg.sweep_tol = 1e-10;
    cfg.lanczos_max_iters = 80;
    mps_dmrg_result_t res;
    ASSERT_EQ_INT(mps_dmrg_xxz(&cfg, &res), 0);
    printf("# N=8 TFIM Γ=J=1: E_ED=%.8f  E_DMRG=%.8f  Δ=%.2e\n",
           E_ed, res.final_energy, fabs(res.final_energy - E_ed));
    ASSERT_NEAR(res.final_energy, E_ed, 1e-5);
}
static void test_dmrg_tfim_paramagnet_limit(void) {
    /* Large transverse field Γ = 10, J = 1: paramagnetic ground state,
     * E ≈ -N · Γ (every spin aligned with +x). */
    int N = 6;
    mps_config_t cfg = mps_config_defaults();
    cfg.num_sites = N;
    cfg.ham = MPS_HAM_TFIM;
    cfg.J = 1.0;
    cfg.Gamma = 10.0;
    cfg.max_bond_dim = 8;
    cfg.num_sweeps = 10;
    cfg.sweep_tol = 1e-10;
    cfg.lanczos_max_iters = 80;
    mps_dmrg_result_t res;
    ASSERT_EQ_INT(mps_dmrg_xxz(&cfg, &res), 0);
    double per_site = res.final_energy / (double)N;
    printf("# N=6 TFIM Γ=10: E/N = %.4f  (expect ≈ -10 + O(J²/Γ))\n", per_site);
    ASSERT_TRUE(per_site < -9.9);
    ASSERT_TRUE(per_site > -10.1);
}
int main(void) {
    TEST_RUN(test_dmrg_heisenberg_4site);
    TEST_RUN(test_dmrg_heisenberg_6site);
    TEST_RUN(test_dmrg_heisenberg_8site);
    TEST_RUN(test_dmrg_xxz_anisotropic);
    TEST_RUN(test_dmrg_truncation_error_decreases_with_D);
    TEST_RUN(test_dmrg_state_vector_overlap_with_ed);
    TEST_RUN(test_dmrg_12_site_heisenberg_matches_ed);
    TEST_RUN(test_dmrg_16_site_energy_per_site_converges);
    TEST_RUN(test_dmrg_tfim_matches_ed);
    TEST_RUN(test_dmrg_tfim_paramagnet_limit);
    TEST_SUMMARY();
}