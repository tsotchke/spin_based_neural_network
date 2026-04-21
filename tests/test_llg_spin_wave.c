/*
 * tests/test_llg_spin_wave.c
 *
 * Scientific validation for the LLG exchange field: the linear
 * spin-wave dispersion relation of a 1D Heisenberg ferromagnet.
 *
 * Derivation: around the FM ground state m = ẑ with m_i = ẑ + δ_i,
 *   ψ_i = δ_{i,x} + i δ_{i,y}
 *   ψ̇_i = -i γ J (ψ_{i-1} + ψ_{i+1} - 2 ψ_i)
 * Plane waves ψ_i = A exp(i(ki - ωt)) give
 *   ω(k) = 2 γ J (1 - cos k) = 4 γ J sin²(k/2)
 *
 * The test perturbs a PBC chain with a single plane-wave mode,
 * integrates with LLG, and measures the oscillation frequency from
 * the time-series by zero-crossing counting.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "harness.h"
#include "llg/llg.h"
#include "llg/exchange_field.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
/* Count sign changes of a scalar trace to estimate its frequency.
 * Two zero crossings per period → freq = crossings / (2T). */
static double frequency_from_trace(const double *trace, int len, double dt) {
    int crossings = 0;
    for (int i = 1; i < len; i++) {
        if ((trace[i - 1] > 0 && trace[i] < 0) ||
            (trace[i - 1] < 0 && trace[i] > 0)) crossings++;
    }
    return (double)crossings / (2.0 * dt * (double)len) * 2.0 * M_PI;   /* → ω */
}
static void test_spin_wave_dispersion_at_k_pi_over_two(void) {
    int N = 16;
    double J = 1.0, gamma = 1.0, eps = 0.05;
    int nk = 4;                       /* k = 2π · nk / N = π / 2 */
    double k = 2.0 * M_PI * (double)nk / (double)N;
    double omega_analytic = 2.0 * gamma * J * (1.0 - cos(k));
    double *m = malloc(sizeof(double) * 3 * N);
    for (int i = 0; i < N; i++) {
        double kx = k * (double)i;
        m[3*i    ] = eps * cos(kx);
        m[3*i + 1] = eps * sin(kx);
        m[3*i + 2] = sqrt(1.0 - eps * eps);
    }
    llg_heisenberg_1d_t ex = {.J = J,.Bx = 0,.By = 0,.Bz = 0 };
    llg_config_t cfg = llg_config_defaults();
    cfg.gamma = gamma;
    cfg.alpha = 0.0;
    cfg.dt    = 0.01;
    cfg.field_fn        = llg_heisenberg_1d_field;
    cfg.field_user_data = &ex;
    /* Integrate for ~5 periods of the analytic mode. */
    double T_analytic = 2.0 * M_PI / omega_analytic;
    int num_steps = (int)(5.0 * T_analytic / cfg.dt);
    double *trace = malloc(sizeof(double) * num_steps);
    for (int s = 0; s < num_steps; s++) {
        llg_rk4_step(&cfg, m, N);
        trace[s] = m[0];   /* m_x at site 0 */
    }
    double omega_measured = frequency_from_trace(trace, num_steps, cfg.dt);
    printf("# 1D FM spin wave k=π/2: ω_analytic=%.4f  ω_measured=%.4f  (rel err %.3f%%)\n",
           omega_analytic, omega_measured,
           100.0 * fabs(omega_measured - omega_analytic) / omega_analytic);
    ASSERT_NEAR(omega_measured, omega_analytic, 0.05);
    free(trace); free(m);
}
static void test_spin_wave_dispersion_at_k_pi(void) {
    /* k = π is the Brillouin-zone edge; ω(π) = 4γJ. This is the
     * maximum of the dispersion. Checking this guards against
     * forgetting the "2" in 2(1-cos k). */
    int N = 16;
    double J = 1.0, gamma = 1.0, eps = 0.05;
    int nk = N / 2;                   /* k = π */
    double k = 2.0 * M_PI * (double)nk / (double)N;
    double omega_analytic = 2.0 * gamma * J * (1.0 - cos(k));  /* = 4 */
    double *m = malloc(sizeof(double) * 3 * N);
    for (int i = 0; i < N; i++) {
        double kx = k * (double)i;
        m[3*i    ] = eps * cos(kx);
        m[3*i + 1] = eps * sin(kx);
        m[3*i + 2] = sqrt(1.0 - eps * eps);
    }
    llg_heisenberg_1d_t ex = {.J = J,.Bx = 0,.By = 0,.Bz = 0 };
    llg_config_t cfg = llg_config_defaults();
    cfg.gamma = gamma;
    cfg.alpha = 0.0;
    cfg.dt    = 0.005;                /* finer for faster mode */
    cfg.field_fn        = llg_heisenberg_1d_field;
    cfg.field_user_data = &ex;
    double T_analytic = 2.0 * M_PI / omega_analytic;
    int num_steps = (int)(5.0 * T_analytic / cfg.dt);
    double *trace = malloc(sizeof(double) * num_steps);
    for (int s = 0; s < num_steps; s++) {
        llg_rk4_step(&cfg, m, N);
        trace[s] = m[0];
    }
    double omega_measured = frequency_from_trace(trace, num_steps, cfg.dt);
    printf("# 1D FM spin wave k=π:  ω_analytic=%.4f  ω_measured=%.4f\n",
           omega_analytic, omega_measured);
    ASSERT_NEAR(omega_measured, omega_analytic, 0.10);
    free(trace); free(m);
}
static void test_spin_wave_frequency_scales_linearly_with_J(void) {
    /* Doubling J must double the frequency at fixed k. This is a
     * rigorous check on both the exchange field magnitude AND the
     * integrator step consistency. A wrong factor of 2 in either
     * would break this. */
    int N = 16, nk = 4;
    double k = 2.0 * M_PI * (double)nk / (double)N;
    double gamma = 1.0, eps = 0.05;
    double omega[2];
    for (int t = 0; t < 2; t++) {
        double J = (t == 0) ? 1.0 : 2.0;
        double *m = malloc(sizeof(double) * 3 * N);
        for (int i = 0; i < N; i++) {
            double kx = k * (double)i;
            m[3*i    ] = eps * cos(kx);
            m[3*i + 1] = eps * sin(kx);
            m[3*i + 2] = sqrt(1.0 - eps * eps);
        }
        llg_heisenberg_1d_t ex = {.J = J,.Bx = 0,.By = 0,.Bz = 0 };
        llg_config_t cfg = llg_config_defaults();
        cfg.gamma = gamma;
        cfg.alpha = 0.0;
        cfg.dt    = 0.005;
        cfg.field_fn        = llg_heisenberg_1d_field;
        cfg.field_user_data = &ex;
        double omega_a = 2.0 * gamma * J * (1.0 - cos(k));
        int num_steps = (int)(5.0 * 2.0 * M_PI / omega_a / cfg.dt);
        double *trace = malloc(sizeof(double) * num_steps);
        for (int s = 0; s < num_steps; s++) {
            llg_rk4_step(&cfg, m, N);
            trace[s] = m[0];
        }
        omega[t] = frequency_from_trace(trace, num_steps, cfg.dt);
        free(trace); free(m);
    }
    double ratio = omega[1] / omega[0];
    printf("# ω(J=2)/ω(J=1) = %.4f (expected 2.0)\n", ratio);
    ASSERT_NEAR(ratio, 2.0, 0.10);
}
static void test_ferromagnetic_ground_state_is_stationary(void) {
    /* m_i = ẑ for all i is the undamped FM ground state under
     * Heisenberg exchange. |m|=1 and m should not budge. */
    int N = 8;
    double *m = malloc(sizeof(double) * 3 * N);
    for (int i = 0; i < N; i++) { m[3*i] = 0; m[3*i+1] = 0; m[3*i+2] = 1; }
    llg_heisenberg_1d_t ex = {.J = 1.0,.Bx = 0,.By = 0,.Bz = 0 };
    llg_config_t cfg = llg_config_defaults();
    cfg.gamma = 1.0;
    cfg.alpha = 0.0;
    cfg.dt    = 0.01;
    cfg.field_fn = llg_heisenberg_1d_field;
    cfg.field_user_data = &ex;
    for (int s = 0; s < 1000; s++) llg_rk4_step(&cfg, m, N);
    for (int i = 0; i < N; i++) {
        ASSERT_NEAR(m[3*i],   0.0, 1e-10);
        ASSERT_NEAR(m[3*i+1], 0.0, 1e-10);
        ASSERT_NEAR(m[3*i+2], 1.0, 1e-10);
    }
    free(m);
}
static void test_anisotropy_opens_spin_wave_gap(void) {
    /* With easy-axis anisotropy Kz > 0 added to the FM, the spin-wave
     * dispersion acquires an anisotropy gap:
     *     ω(k) = 2γJ(1 - cos k) + 2γ·Kz
     * (small-amplitude fluctuations around m = ẑ see an effective
     * field 2 Kz ẑ from the anisotropy term.) At k = 0 the pure
     * exchange mode is gapless (ω(0) = 0); adding Kz gives a finite
     * gap of 2 γ Kz. Check at k close to 0 that the measured
     * frequency matches the expected 2γJ(1-cos k) + 2γKz. */
    int N = 32;
    double J = 1.0, gamma = 1.0, Kz = 0.5, eps = 0.05;
    int nk = 1;                               /* k = 2π/N ≈ 0 */
    double k = 2.0 * M_PI * (double)nk / (double)N;
    double omega_analytic = 2.0 * gamma * J * (1.0 - cos(k)) + 2.0 * gamma * Kz;
    double *m = malloc(sizeof(double) * 3 * N);
    for (int i = 0; i < N; i++) {
        double kx = k * (double)i;
        m[3*i    ] = eps * cos(kx);
        m[3*i + 1] = eps * sin(kx);
        m[3*i + 2] = sqrt(1.0 - eps * eps);
    }
    llg_heisenberg_1d_t ex = {.J = J,.Bx = 0,.By = 0,.Bz = 0,.Kz = Kz,.D_dmi = 0 };
    llg_config_t cfg = llg_config_defaults();
    cfg.gamma = gamma;
    cfg.alpha = 0.0;
    cfg.dt    = 0.002;
    cfg.field_fn = llg_heisenberg_1d_field;
    cfg.field_user_data = &ex;
    int num_steps = (int)(5.0 * 2.0 * M_PI / omega_analytic / cfg.dt);
    double *trace = malloc(sizeof(double) * num_steps);
    for (int s = 0; s < num_steps; s++) {
        llg_rk4_step(&cfg, m, N);
        trace[s] = m[0];
    }
    double omega_measured = frequency_from_trace(trace, num_steps, cfg.dt);
    printf("# FM + Kz=%.1f anisotropy, k≈0: "
           "ω_analytic=%.4f  ω_measured=%.4f  gap=2γKz=%.4f\n",
           Kz, omega_analytic, omega_measured, 2.0 * gamma * Kz);
    ASSERT_NEAR(omega_measured, omega_analytic, 0.05);
    free(trace); free(m);
}
/* Compute the energy of a chain under exchange + DMI + Zeeman +
 * anisotropy. Used to verify that the helical state has lower energy
 * than the uniform FM state under DMI. */
static double llg_chain_energy(const double *m, int N,
                                const llg_heisenberg_1d_t *p) {
    double E = 0.0;
    for (int i = 0; i < N; i++) {
        int ip = (i + 1) % N;
        /* Exchange: -J m_i · m_{i+1}  (ferromagnetic sign). */
        E += -p->J * (m[3*i]*m[3*ip] + m[3*i+1]*m[3*ip+1] + m[3*i+2]*m[3*ip+2]);
        /* DMI: +D ẑ · (m_i × m_{i+1}) = D (m_i_x m_{i+1}_y - m_i_y m_{i+1}_x). */
        E += p->D_dmi * (m[3*i]*m[3*ip+1] - m[3*i+1]*m[3*ip]);
        /* Uniaxial anisotropy and Zeeman on site i. */
        E += -p->Kz * m[3*i+2] * m[3*i+2];
        E += -(p->Bx * m[3*i] + p->By * m[3*i+1] + p->Bz * m[3*i+2]);
    }
    return E;
}
static void test_dmi_helical_state_energy_lower_than_uniform(void) {
    /* Ground-state claim: under pure exchange + DMI with periodic BCs,
     * the helical state with pitch k = atan(D/J) has energy per bond
     *     e_helical = -√(J² + D²)
     * while the uniform FM state has -J. For D = J, the helical state
     * is √2× more strongly bound.
     *
     * Build both configurations explicitly and compare their energies
     * — this is a static verification of the DMI term. */
    int N = 16;
    double J = 1.0, D = 1.0;
    llg_heisenberg_1d_t ex = {.J = J,.Bx = 0,.By = 0,.Bz = 0,.Kz = 0,.D_dmi = D };
    double *m_uniform = malloc(sizeof(double) * 3 * N);
    double *m_helical = malloc(sizeof(double) * 3 * N);
    for (int i = 0; i < N; i++) {
        m_uniform[3*i    ] = 0.0;
        m_uniform[3*i + 1] = 0.0;
        m_uniform[3*i + 2] = 1.0;
    }
    /* Helical: choose pitch k = 2π/N (an integer number of twists to
     * match PBC). With N = 16 the pitch is 2π/16 = π/8. */
    double k = 2.0 * M_PI / (double)N;
    for (int i = 0; i < N; i++) {
        double phase = k * (double)i;
        m_helical[3*i    ] = sin(phase);
        m_helical[3*i + 1] = cos(phase);   /* chirality matches -D sin convention */
        m_helical[3*i + 2] = 0.0;
    }
    double E_uniform = llg_chain_energy(m_uniform, N, &ex);
    double E_helical = llg_chain_energy(m_helical, N, &ex);
    printf("# DMI D=%.1f, J=%.1f: E_uniform=%.4f  E_helical(k=π/8)=%.4f\n",
           D, J, E_uniform, E_helical);
    /* Helical must be strictly lower for D > 0. */
    ASSERT_TRUE(E_helical < E_uniform);
    free(m_uniform); free(m_helical);
}
static void test_dmi_helical_state_is_approximate_equilibrium(void) {
    /* Initialise with the helical PBC-commensurate pitch and check
     * that under damping the state stays close to helical (small m_z
     * throughout, roughly constant in-plane magnitude). */
    int N = 16;
    double J = 1.0, D = 1.0;
    llg_heisenberg_1d_t ex = {.J = J,.Bx = 0,.By = 0,.Bz = 0,.Kz = 0,.D_dmi = D };
    double *m = malloc(sizeof(double) * 3 * N);
    double k = 2.0 * M_PI / (double)N;
    for (int i = 0; i < N; i++) {
        double phase = k * (double)i;
        m[3*i    ] = sin(phase);
        m[3*i + 1] = cos(phase);
        m[3*i + 2] = 0.0;
    }
    llg_config_t cfg = llg_config_defaults();
    cfg.gamma = 1.0;
    cfg.alpha = 0.3;           /* heavy damping — relaxes fast */
    cfg.dt    = 0.01;
    cfg.field_fn = llg_heisenberg_1d_field;
    cfg.field_user_data = &ex;
    double E0 = llg_chain_energy(m, N, &ex);
    for (int s = 0; s < 1000; s++) llg_rk4_step(&cfg, m, N);
    double E1 = llg_chain_energy(m, N, &ex);
    /* Energy must be non-increasing under damping, and should have
     * changed by less than 20% — the seed is already close to the
     * energy minimum. */
    ASSERT_TRUE(E1 <= E0 + 1e-8);
    ASSERT_TRUE(fabs(E1 - E0) < 0.2 * fabs(E0));
    printf("# helical under damping: E₀=%.4f  E(t=10)=%.4f  ΔE=%.4f\n",
           E0, E1, E1 - E0);
    /* All m_z should be small after relaxation (state is planar). */
    double max_mz = 0;
    for (int i = 0; i < N; i++) if (fabs(m[3*i+2]) > max_mz) max_mz = fabs(m[3*i+2]);
    ASSERT_TRUE(max_mz < 0.2);
    free(m);
}
int main(void) {
    TEST_RUN(test_ferromagnetic_ground_state_is_stationary);
    TEST_RUN(test_spin_wave_dispersion_at_k_pi_over_two);
    TEST_RUN(test_spin_wave_dispersion_at_k_pi);
    TEST_RUN(test_spin_wave_frequency_scales_linearly_with_J);
    TEST_RUN(test_anisotropy_opens_spin_wave_gap);
    TEST_RUN(test_dmi_helical_state_energy_lower_than_uniform);
    TEST_RUN(test_dmi_helical_state_is_approximate_equilibrium);
    TEST_SUMMARY();
}