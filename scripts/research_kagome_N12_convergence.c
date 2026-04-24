/*
 * scripts/research_kagome_N12_convergence.c
 *
 * Research-scale convergence driver for the open kagome Heisenberg S=1/2
 * ground-state question. Runs complex-RBM + holomorphic SR on a 2x2 PBC
 * unit-cell cluster (N = 12 sites, 24 bonds, coord 4) and compares the
 * converged variational energy against the published ED benchmark.
 *
 * Published reference (Leung & Elser 1993; Lecheminant et al. 1997):
 *     E_0 / N ≈ -0.4365 J  (infinite-size extrapolation ≈ -0.4386 J)
 *     E_0 (N=12 torus) ≈ -5.238 J
 *
 * Caller budget: several minutes on an M-series Mac.
 *
 * Output: iteration-by-iteration mean energy and running stddev,
 *         plus a final summary line with the tolerance vs ED.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nqs/nqs_config.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_sampler.h"
#include "nqs/nqs_optimizer.h"

/* Published ED / DMRG energy on the N=12 torus (per-site and total).
 * For the 2×2 PBC cluster in our brick-wall kagome indexing, the exact
 * GS total energy is within a few percent of -5.238·J for J=1. */
#define E0_PER_SITE_REF  (-0.4365)
#define E0_TOTAL_REF(N)  ((E0_PER_SITE_REF) * (double)(N))

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    /* Line-buffer stdout — research drivers take minutes; silent
     * stdout turns the wait into a "is it alive?" problem. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    int num_iterations = 1000;
    int hidden_units   = 24;        /* 2·N by default */
    int num_samples    = 1024;
    unsigned seed      = 0xC0AEEDu;
    if (argc > 1) num_iterations = atoi(argv[1]);
    if (argc > 2) hidden_units   = atoi(argv[2]);
    if (argc > 3) num_samples    = atoi(argv[3]);

    int Lx = 2, Ly = 2;
    int N  = 3 * Lx * Ly;   /* 12 sites */

    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz           = NQS_ANSATZ_COMPLEX_RBM;
    cfg.rbm_hidden_units = hidden_units;
    cfg.rbm_init_scale   = 0.02;       /* tighter init than smoke test */
    cfg.hamiltonian      = NQS_HAM_KAGOME_HEISENBERG;
    cfg.j_coupling       = 1.0;
    cfg.kagome_pbc       = 1;
    cfg.num_samples      = num_samples;
    cfg.num_thermalize   = 512;
    cfg.num_decorrelate  = 2;
    cfg.num_iterations   = num_iterations;
    cfg.learning_rate    = 0.02;       /* smaller than smoke test (0.03) */
    cfg.sr_diag_shift    = 1e-3;       /* tighter than smoke test (1e-2) */
    cfg.sr_cg_max_iters  = 200;
    cfg.sr_cg_tol        = 1e-8;
    cfg.rng_seed         = seed;

    printf("# kagome N=12 PBC Heisenberg S=1/2 convergence run\n");
    printf("# config: iters=%d, hidden=%d, samples=%d, lr=%.3g, shift=%.1e, seed=0x%x\n",
           num_iterations, hidden_units, num_samples,
           cfg.learning_rate, cfg.sr_diag_shift, seed);
    printf("# reference: E_0/N ≈ %.4f J, E_0 total ≈ %.4f J (for N=12)\n",
           E0_PER_SITE_REF, E0_TOTAL_REF(N));
    printf("# ------------------------------------------------------------\n");
    printf("# iter   E           E/N        \n");

    nqs_ansatz_t  *a = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a);
    if (!a || !s) {
        fprintf(stderr, "error: ansatz or sampler creation failed\n");
        return 2;
    }

    double *trace = malloc((size_t)num_iterations * sizeof(double));
    if (!trace) { fprintf(stderr, "error: trace alloc failed\n"); return 2; }

    double t0 = now_seconds();
    int rc = nqs_sr_run_holomorphic(&cfg, Lx, Ly, a, s,
                                     nqs_ansatz_log_amp, a, trace);
    double dt = now_seconds() - t0;
    if (rc != 0) {
        fprintf(stderr, "error: nqs_sr_run_holomorphic returned %d\n", rc);
        return 2;
    }

    /* Print every 25th iteration plus the final few. */
    for (int i = 0; i < num_iterations; i++) {
        if (i % 25 == 0 || i >= num_iterations - 5) {
            printf("%6d  %+.6f  %+.6f\n",
                   i, trace[i], trace[i] / (double)N);
        }
    }

    /* Tail statistics: last 10% of iterations. */
    int tail_start = (int)(num_iterations * 0.9);
    int tail_len   = num_iterations - tail_start;
    double mean = 0.0, sq = 0.0;
    for (int i = tail_start; i < num_iterations; i++) {
        mean += trace[i];
        sq   += trace[i] * trace[i];
    }
    mean /= (double)tail_len;
    double var = sq / (double)tail_len - mean * mean;
    double stddev = (var > 0) ? sqrt(var) : 0;

    double E_ref_total = E0_TOTAL_REF(N);
    double rel_gap = (mean - E_ref_total) / fabs(E_ref_total);

    printf("# ------------------------------------------------------------\n");
    printf("# tail mean (last %d iters): E = %+.6f  ±  %.6f (stddev)\n",
           tail_len, mean, stddev);
    printf("# tail per-site:           E/N = %+.6f\n", mean / (double)N);
    printf("# reference E (N=12):      E_0 = %+.6f\n", E_ref_total);
    printf("# relative gap:            %+.3f %% (positive = above GS)\n",
           rel_gap * 100.0);
    printf("# wall-clock: %.2f s (%.3f s/iter avg)\n",
           dt, dt / (double)num_iterations);

    free(trace);
    nqs_sampler_free(s);
    nqs_ansatz_free(a);
    /* Exit 0 if the tail mean is within 15% of ED (generous research
     * bound); exit 1 if further than that. Exit 2 on build / setup error
     * only. */
    return (fabs(rel_gap) < 0.15) ? 0 : 1;
}
