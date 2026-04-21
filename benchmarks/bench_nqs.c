/*
 * benchmarks/bench_nqs.c
 *
 * Throughput benchmarks for the v0.5 NQS scaffold: Metropolis samples
 * per second (across lattice sizes) and stochastic-reconfiguration
 * steps per second. Uses the default mean-field ansatz so the numbers
 * measure pipeline overhead, not ansatz cost.
 */
#include <stdio.h>
#include <stdlib.h>
#include "bench_common.h"
#include "nqs/nqs_config.h"
#include "nqs/nqs_sampler.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_optimizer.h"

static double sampler_throughput(int L, int samples) {
    int N = L * L;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.num_samples = samples;
    cfg.num_thermalize = 128;
    cfg.num_decorrelate = 1;

    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a);
    nqs_sampler_thermalize(s);

    int *buf = malloc((size_t)samples * (size_t)N * sizeof(int));
    double t0 = bench_now_seconds();
    nqs_sampler_batch(s, samples, buf);
    double dt = bench_now_seconds() - t0;
    free(buf);

    nqs_sampler_free(s);
    nqs_ansatz_free(a);
    return (double)samples / dt;
}

static double sr_step_throughput(int L, int samples_per_step, int steps) {
    int N = L * L;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.hamiltonian = NQS_HAM_TFIM;
    cfg.transverse_field = 1.0;
    cfg.num_samples = samples_per_step;
    cfg.num_thermalize = 128;
    cfg.num_decorrelate = 1;
    cfg.learning_rate = 1e-3;
    cfg.sr_diag_shift = 1e-2;
    cfg.sr_cg_max_iters = 20;

    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a);
    nqs_sampler_thermalize(s);

    double t0 = bench_now_seconds();
    for (int i = 0; i < steps; i++) {
        nqs_sr_step_info_t info;
        nqs_sr_step(&cfg, L, L, a, s, &info);
    }
    double dt = bench_now_seconds() - t0;

    nqs_sampler_free(s);
    nqs_ansatz_free(a);
    return (double)steps / dt;
}

int main(void) {
    int L_sizes[] = {4, 6, 8};
    int samples[] = {4096, 2048, 1024};
    for (int i = 0; i < 3; i++) {
        double sps = sampler_throughput(L_sizes[i], samples[i]);
        char name[32];
        snprintf(name, sizeof(name), "sampler_L%d", L_sizes[i]);
        bench_emitter_t em;
        bench_emit_begin(&em, "nqs", name);
        bench_emit_int(&em, "L", L_sizes[i]);
        bench_emit_int(&em, "samples", samples[i]);
        bench_emit_metric(&em, "samples_per_second", sps);
        bench_emit_end(&em);
        printf("nqs sampler L=%d: %.1f samples/sec\n", L_sizes[i], sps);
    }

    int sr_Ls[] = {4, 6};
    int sr_samples[] = {256, 128};
    int sr_steps[] = {20, 10};
    for (int i = 0; i < 2; i++) {
        double sps = sr_step_throughput(sr_Ls[i], sr_samples[i], sr_steps[i]);
        char name[32];
        snprintf(name, sizeof(name), "sr_step_L%d", sr_Ls[i]);
        bench_emitter_t em;
        bench_emit_begin(&em, "nqs", name);
        bench_emit_int(&em, "L", sr_Ls[i]);
        bench_emit_int(&em, "samples_per_step", sr_samples[i]);
        bench_emit_int(&em, "steps_run", sr_steps[i]);
        bench_emit_metric(&em, "sr_steps_per_second", sps);
        bench_emit_end(&em);
        printf("nqs SR step L=%d: %.2f steps/sec\n", sr_Ls[i], sps);
    }
    return 0;
}
