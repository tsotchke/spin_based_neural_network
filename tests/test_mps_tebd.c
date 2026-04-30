/*
 * tests/test_mps_tebd.c
 *
 * Imaginary-time TEBD on 1D XXZ / Heisenberg / TFIM chains. The
 * energy after many imaginary-time steps must converge to the
 * ground-state energy computed by dense ED (same known-answer
 * target as the DMRG tests).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "harness.h"
#include "mps/mps.h"
#include "mps/tebd.h"
static double ed_energy(int N, double J, double Jz, double Gamma,
                         mps_hamiltonian_kind_t ham) {
    mps_config_t cfg = mps_config_defaults();
    cfg.num_sites = N;
    cfg.ham = ham;
    cfg.J = J;
    cfg.Jz = Jz;
    cfg.Gamma = Gamma;
    cfg.lanczos_max_iters = 200;
    cfg.lanczos_tol = 1e-10;
    double E;
    lanczos_result_t info;
    mps_ground_state_dense(&cfg, &E, NULL, &info);
    return E;
}
static void test_tebd_heisenberg_4site_converges_to_ground_state(void) {
    /* N=4 Heisenberg chain: E₀ = -1.616025. Imaginary-time TEBD
     * should drive the energy down to this value within a few
     * hundred sweeps. */
    int N = 4;
    double E_ref = ed_energy(N, 1.0, 1.0, 0.0, MPS_HAM_HEISENBERG);
    mps_config_t cfg = mps_config_defaults();
    cfg.num_sites = N;
    cfg.ham = MPS_HAM_HEISENBERG;
    cfg.J = 1.0;
    cfg.max_bond_dim = 16;
    double E_final;
    int num_sweeps = 400;
    double *trace = malloc(sizeof(double) * num_sweeps);
    int rc = mps_tebd_imaginary_run(&cfg, 0.05, num_sweeps, trace, &E_final);
    ASSERT_EQ_INT(rc, 0);
    printf("# TEBD Heisenberg N=4: E_init=%.4f  E_final=%.4f  (E_ED=%.4f)\n",
           trace[0], E_final, E_ref);
    ASSERT_NEAR(E_final, E_ref, 0.05);
    ASSERT_TRUE(trace[num_sweeps - 1] <= trace[0]);    /* monotone-ish descent */
    free(trace);
}
static void test_tebd_tfim_8site_converges_to_ground_state(void) {
    /* N=8 TFIM at Γ=J=1 (critical). E₀ ≈ -9.83795. */
    int N = 8;
    double E_ref = ed_energy(N, 1.0, 1.0, 1.0, MPS_HAM_TFIM);
    mps_config_t cfg = mps_config_defaults();
    cfg.num_sites = N;
    cfg.ham = MPS_HAM_TFIM;
    cfg.J = 1.0;
    cfg.Gamma = 1.0;
    cfg.max_bond_dim = 16;
    double E_final;
    int num_sweeps = 200;
    double *trace = malloc(sizeof(double) * num_sweeps);
    mps_tebd_imaginary_run(&cfg, 0.05, num_sweeps, trace, &E_final);
    printf("# TEBD TFIM N=8 Γ=J=1: E_init=%.4f  E_final=%.4f  (E_ED=%.4f)\n",
           trace[0], E_final, E_ref);
    ASSERT_NEAR(E_final, E_ref, 0.2);
    free(trace);
}
static void test_tebd_energy_monotone_descent(void) {
    /* Imaginary-time evolution under e^{-Hτ} (with renormalisation)
     * must monotonically reduce the energy up to Trotter-step error.
     * Tolerate a small slack for the second-order Trotter O(τ³)
     * artefact. */
    int N = 6;
    mps_config_t cfg = mps_config_defaults();
    cfg.num_sites = N;
    cfg.ham = MPS_HAM_HEISENBERG;
    cfg.J = 1.0;
    cfg.max_bond_dim = 16;
    int num_sweeps = 100;
    double *trace = malloc(sizeof(double) * num_sweeps);
    double E_final;
    mps_tebd_imaginary_run(&cfg, 0.05, num_sweeps, trace, &E_final);
    /* Average the first 10 and last 10 values: the tail must be lower. */
    double head = 0, tail = 0;
    for (int i = 0; i < 10; i++) head += trace[i];
    for (int i = 0; i < 10; i++) tail += trace[num_sweeps - 10 + i];
    head /= 10; tail /= 10;
    printf("# TEBD monotone: head=%.4f  tail=%.4f\n", head, tail);
    ASSERT_TRUE(tail < head);
    free(trace);
}
static void test_real_time_conserves_norm(void) {
    /* Under Schrödinger evolution ||ψ||² is conserved. RK4 on dim=16
     * should keep it to 10^-6 over 100 steps at dt = 0.02. */
    int N = 4;
    mps_config_t cfg = mps_config_defaults();
    cfg.num_sites = N;
    cfg.ham = MPS_HAM_HEISENBERG;
    cfg.J = 1.0;
    double init[12];
    for (int i = 0; i < N; i++) {   /* start in +z product state */
        init[3*i] = 0.0; init[3*i+1] = 0.0; init[3*i+2] = 1.0;
    }
    int steps = 100;
    double *los = malloc(sizeof(double) * steps);
    mps_tebd_real_time_run(&cfg, init, 0.02, steps, NULL, los);
    /* For a +z product state under pure Heisenberg H, |ψ(t)⟩ = |ψ(0)⟩
     * since |↑↑...↑⟩ is an eigenstate → Loschmidt echo stays at 1. */
    for (int s = 0; s < steps; s++) ASSERT_NEAR(los[s], 1.0, 1e-8);
    free(los);
}
static void test_real_time_tfim_quench_loschmidt(void) {
    /* Start from |↑↑↑↑↑↑↑↑⟩ (σ^z = +1 everywhere), quench to TFIM
     * with Γ = 1.5, J = 1. Initial state is NOT an eigenstate, so
     * Loschmidt echo |⟨ψ(0)|ψ(t)⟩|² should drop below 1 over time
     * and oscillate / decay characteristically. */
    int N = 6;
    mps_config_t cfg = mps_config_defaults();
    cfg.num_sites = N;
    cfg.ham = MPS_HAM_TFIM;
    cfg.J = 1.0;
    cfg.Gamma = 1.5;
    double init[18];
    for (int i = 0; i < N; i++) {
        init[3*i] = 0.0; init[3*i+1] = 0.0; init[3*i+2] = 1.0;
    }
    int steps = 200;
    double *los = malloc(sizeof(double) * steps);
    mps_tebd_real_time_run(&cfg, init, 0.02, steps, NULL, los);
    /* Start at 1.0, then DECAY. */
    ASSERT_NEAR(los[0], 1.0, 1e-10);
    double min_echo = 1.0;
    for (int s = 0; s < steps; s++) if (los[s] < min_echo) min_echo = los[s];
    printf("# TFIM quench N=6 Γ=1.5: Loschmidt min = %.4f\n", min_echo);
    ASSERT_TRUE(min_echo < 0.2);    /* clear decay out of initial state */
    free(los);
}
static void test_real_time_magnetisation_tracks(void) {
    /* Single-site check: start with spin pointing along +x, no
     * coupling (J=0) and Γ=1 (single-site σ^x eigenstate).
     *   |+x⟩ = (|↑⟩ + |↓⟩)/√2
     * Under -Γ σ^x, |+x⟩ is an eigenstate with energy -Γ. No time
     * evolution in the m_z direction — ⟨σ^z⟩ stays 0. */
    int N = 1;
    mps_config_t cfg = mps_config_defaults();
    cfg.num_sites = N;
    cfg.ham = MPS_HAM_TFIM;
    cfg.J = 0.0;
    cfg.Gamma = 1.0;
    double init[3] = {1.0, 0.0, 0.0};   /* +x */
    int steps = 50;
    double *mz = malloc(sizeof(double) * steps);
    mps_tebd_real_time_run(&cfg, init, 0.02, steps, mz, NULL);
    for (int s = 0; s < steps; s++) ASSERT_NEAR(mz[s], 0.0, 1e-8);
    free(mz);
}
int main(void) {
    TEST_RUN(test_tebd_heisenberg_4site_converges_to_ground_state);
    TEST_RUN(test_tebd_tfim_8site_converges_to_ground_state);
    TEST_RUN(test_tebd_energy_monotone_descent);
    TEST_RUN(test_real_time_conserves_norm);
    TEST_RUN(test_real_time_tfim_quench_loschmidt);
    TEST_RUN(test_real_time_magnetisation_tracks);
    TEST_SUMMARY();
}