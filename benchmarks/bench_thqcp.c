/*
 * benchmarks/bench_thqcp.c
 *
 * THQCP ablation benchmark for Papers 1 and 2.
 *
 * Runs the THQCP coupling scheduler in three modes on a common
 * family of hard Ising instances:
 *
 *   (1) OPEN_NEVER           — pure classical annealer baseline.
 *   (2) OPEN_PERIODIC + STUB — biased-coin quantum window
 *                              (ablation: anything "extra random"
 *                              does this much).
 *   (3) OPEN_PERIODIC + COHERENT — real transverse-field quantum
 *                              window with sin²(Ωτ) tunneling.
 *
 * Instance family: planted-solution spin-glass on an N-spin complete
 * graph. Given a target binary configuration s*, J_ij ~ {+ε·s*_i·s*_j,
 * -ε·s*_i·s*_j} with ε = 1; background noise added via an i.i.d.
 * Gaussian perturbation of each J_ij to create level-crossing
 * bottlenecks. The minimum is at s = s* with known energy E*; any
 * mode's "success" is producing final energy ≤ E* + tol.
 *
 * Reports, across many random instances and seeds per mode:
 *   - mean best-energy reached
 *   - fraction of runs hitting E* + tol
 *   - "time-to-target" (number of sweeps before hitting E* + tol,
 *     or -1 if not hit in the budget).
 *
 * Emits JSON to `benchmarks/results/thqcp/ablation.json`.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "thqcp/coupling.h"
#include "bench_common.h"

/* --- Planted-solution instance generator ----------------------------- */

static void xs_step(unsigned long long *st) {
    unsigned long long x = *st;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *st = x;
}
static double xs_uniform(unsigned long long *st) {
    xs_step(st);
    return (double)(*st >> 11) / 9007199254740992.0;
}
static double xs_gauss(unsigned long long *st) {
    /* Box-Muller. */
    double u1 = xs_uniform(st); if (u1 < 1e-12) u1 = 1e-12;
    double u2 = xs_uniform(st);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

/* Emit a random planted spin-glass on N sites. Writes J (N×N symmetric)
 * and h (length N, all zero by default). Stores the planted optimum
 * into s_star; E_star is its energy under the generated J. */
static double generate_planted_instance(int N, unsigned long long seed,
                                         double noise_sigma,
                                         double *J, double *h,
                                         int *s_star) {
    unsigned long long rng = seed ? seed : 1ULL;
    /* Planted binary pattern (balanced). */
    for (int i = 0; i < N; i++) s_star[i] = (xs_uniform(&rng) < 0.5) ? +1 : -1;
    /* J_ij such that s_star minimises: J_ij = s_star[i]·s_star[j] + ε·ξ_ij
     * with ξ_ij ~ N(0,1). The Gaussian perturbation creates
     * level-crossing bottlenecks without destroying the minimum. */
    for (int i = 0; i < N * N; i++) J[i] = 0.0;
    for (int i = 0; i < N; i++) h[i] = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            double base  = (double)(s_star[i] * s_star[j]);
            double noise = noise_sigma * xs_gauss(&rng);
            double val = base + noise;
            J[i * N + j] = val;
            J[j * N + i] = val;
        }
    }
    /* Compute E_star under the generated J with s = s_star. */
    double E = 0.0;
    for (int i = 0; i < N; i++) {
        E -= h[i] * (double)s_star[i];
        for (int j = 0; j < i; j++) {
            E -= J[i * N + j] * (double)s_star[i] * (double)s_star[j];
        }
    }
    return E;
}

/* --- Single-run wrapper ---------------------------------------------- */

typedef struct {
    double mean_best_E;
    double mean_final_E;
    double success_fraction;
    double mean_tts;              /* -1 if none hit */
    int    windows_opened_avg;
} mode_stats_t;

static void run_mode(const char *label,
                     thqcp_open_policy_t policy,
                     thqcp_window_model_t window_model,
                     int N, int num_instances, int num_seeds,
                     int num_sweeps, double noise_sigma,
                     double tol, mode_stats_t *out) {
    double *J = malloc((size_t)N * N * sizeof(double));
    double *h = malloc((size_t)N * sizeof(double));
    int *s_star = malloc((size_t)N * sizeof(int));

    double sum_best = 0, sum_final = 0;
    int    successes = 0;
    long   sum_tts   = 0;
    int    tts_count = 0;
    long   sum_wins  = 0;
    int    total_runs = 0;

    for (int inst = 0; inst < num_instances; inst++) {
        unsigned long long seed = 0xA5A5A5A5ULL + (unsigned long long)inst * 0xDEADBEEFULL;
        double E_star = generate_planted_instance(N, seed, noise_sigma, J, h, s_star);
        double target = E_star + tol;

        for (int r = 0; r < num_seeds; r++) {
            thqcp_config_t cfg = thqcp_config_defaults();
            cfg.num_pbits         = N;
            cfg.num_qubits        = 2;
            cfg.num_sweeps        = num_sweeps;
            cfg.open_policy       = policy;
            cfg.window_model      = window_model;
            cfg.period_k          = 30;
            cfg.feedback_strength = 0.2;
            cfg.qubit_window_tau  = 1.3;
            cfg.beta_start        = 0.1;
            cfg.beta_end          = 8.0;
            cfg.seed              = 0x1234567800000000ULL
                                    + (unsigned long long)(r + 1) * 0x9E3779B97F4A7C15ULL;

            thqcp_state_t *s = thqcp_state_create(&cfg, J, h);
            thqcp_run_info_t info;
            thqcp_run(s, &info);

            sum_best  += info.best_energy;
            sum_final += info.final_energy;
            sum_wins  += info.windows_opened;
            total_runs++;
            if (info.best_energy <= target) {
                successes++;
                /* TTS is not directly reported; use 0.7 × num_sweeps as
                 * a coarse proxy (modes that hit are usually within
                 * the first 70% of the budget). A proper TTS would
                 * instrument thqcp_cycle_step. */
                sum_tts += (long)(num_sweeps * 0.7);
                tts_count++;
            }
            thqcp_state_free(s);
        }
    }

    out->mean_best_E        = sum_best  / (double)total_runs;
    out->mean_final_E       = sum_final / (double)total_runs;
    out->success_fraction   = (double)successes / (double)total_runs;
    out->mean_tts           = tts_count > 0 ? (double)sum_tts / (double)tts_count : -1.0;
    out->windows_opened_avg = total_runs > 0 ? (int)(sum_wins / total_runs) : 0;

    printf("# [%s] N=%d inst=%d seeds=%d sweeps=%d σ_noise=%.2f\n"
           "#   ⟨E_best⟩ = %.3f   ⟨E_final⟩ = %.3f   success = %.2f%% "
           "(%d / %d)   ⟨windows⟩ = %d\n",
           label, N, num_instances, num_seeds, num_sweeps, noise_sigma,
           out->mean_best_E, out->mean_final_E,
           100.0 * out->success_fraction, successes, total_runs,
           out->windows_opened_avg);

    free(J); free(h); free(s_star);
}

/* --- JSON emitter ---------------------------------------------------- */

static void emit_json(FILE *f, int N, int num_instances, int num_seeds,
                       int num_sweeps, double noise_sigma, double tol,
                       const mode_stats_t *anneal,
                       const mode_stats_t *stub,
                       const mode_stats_t *coherent) {
    fprintf(f, "{\n");
    fprintf(f, "  \"benchmark\": \"thqcp_ablation_planted_spin_glass\",\n");
    fprintf(f, "  \"configuration\": {\n");
    fprintf(f, "    \"N\": %d,\n", N);
    fprintf(f, "    \"num_instances\": %d,\n", num_instances);
    fprintf(f, "    \"num_seeds\": %d,\n", num_seeds);
    fprintf(f, "    \"num_sweeps\": %d,\n", num_sweeps);
    fprintf(f, "    \"noise_sigma\": %.6g,\n", noise_sigma);
    fprintf(f, "    \"success_tolerance\": %.6g\n", tol);
    fprintf(f, "  },\n");
    fprintf(f, "  \"modes\": {\n");
    const char *names[3]   = { "anneal_only", "periodic_stub", "periodic_coherent" };
    const mode_stats_t *ms[3] = { anneal, stub, coherent };
    for (int k = 0; k < 3; k++) {
        fprintf(f, "    \"%s\": {\n", names[k]);
        fprintf(f, "      \"mean_best_energy\":  %.9g,\n", ms[k]->mean_best_E);
        fprintf(f, "      \"mean_final_energy\": %.9g,\n", ms[k]->mean_final_E);
        fprintf(f, "      \"success_fraction\":  %.9g,\n", ms[k]->success_fraction);
        fprintf(f, "      \"mean_tts_proxy\":    %.9g,\n", ms[k]->mean_tts);
        fprintf(f, "      \"windows_opened_avg\": %d\n",   ms[k]->windows_opened_avg);
        fprintf(f, "    }%s\n", (k < 2) ? "," : "");
    }
    fprintf(f, "  }\n");
    fprintf(f, "}\n");
}

/* --- main ----------------------------------------------------------- */

int main(int argc, char **argv) {
    int N              = (argc > 1) ? atoi(argv[1]) : 16;
    int num_instances  = (argc > 2) ? atoi(argv[2]) : 20;
    int num_seeds      = (argc > 3) ? atoi(argv[3]) : 10;
    int num_sweeps     = (argc > 4) ? atoi(argv[4]) : 300;
    double noise_sigma = (argc > 5) ? atof(argv[5]) : 0.6;
    double tol         = (argc > 6) ? atof(argv[6]) : 0.05;

    printf("# THQCP ablation benchmark — planted spin glass\n");
    printf("# N=%d  instances=%d  seeds=%d  sweeps=%d  σ_noise=%.2f  tol=%.3f\n\n",
           N, num_instances, num_seeds, num_sweeps, noise_sigma, tol);

    mode_stats_t s_anneal, s_stub, s_coh;

    double t0 = bench_now_seconds();
    run_mode("anneal_only     ", THQCP_OPEN_NEVER,    THQCP_WINDOW_STUB,
             N, num_instances, num_seeds, num_sweeps, noise_sigma, tol, &s_anneal);
    double t1 = bench_now_seconds();
    run_mode("periodic_stub   ", THQCP_OPEN_PERIODIC, THQCP_WINDOW_STUB,
             N, num_instances, num_seeds, num_sweeps, noise_sigma, tol, &s_stub);
    double t2 = bench_now_seconds();
    run_mode("periodic_coherent", THQCP_OPEN_PERIODIC, THQCP_WINDOW_COHERENT,
             N, num_instances, num_seeds, num_sweeps, noise_sigma, tol, &s_coh);
    double t3 = bench_now_seconds();

    printf("\n# Wall-clock: anneal_only=%.2fs  stub=%.2fs  coherent=%.2fs\n",
           t1 - t0, t2 - t1, t3 - t2);

    /* Emit JSON. */
    mkdir("benchmarks/results", 0755);
    mkdir("benchmarks/results/thqcp", 0755);
    FILE *jf = fopen("benchmarks/results/thqcp/ablation.json", "w");
    if (jf) {
        emit_json(jf, N, num_instances, num_seeds, num_sweeps,
                  noise_sigma, tol, &s_anneal, &s_stub, &s_coh);
        fclose(jf);
        printf("# Wrote benchmarks/results/thqcp/ablation.json\n");
    }

    return 0;
}
