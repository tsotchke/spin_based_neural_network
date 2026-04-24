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
#include "nqs/nqs_gradient.h"

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

/* Sampler throughput on a Hamiltonian-specific site layout (kagome has
 * 3 sublattices per unit cell; every other kernel is single-site). */
static double sampler_throughput_ham(int size_x, int size_y,
                                      nqs_hamiltonian_kind_t ham, int samples) {
    int N = (ham == NQS_HAM_KAGOME_HEISENBERG)
            ? 3 * size_x * size_y
            : size_x * size_y;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz           = NQS_ANSATZ_COMPLEX_RBM;
    cfg.hamiltonian      = ham;
    cfg.kh_K             = 1.0;
    cfg.kh_J             = 1.0;
    cfg.j_coupling       = 1.0;
    cfg.kagome_pbc       = 1;
    cfg.num_samples      = samples;
    cfg.num_thermalize   = 128;
    cfg.num_decorrelate  = 1;
    cfg.rbm_hidden_units = 8;

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

/* Full holomorphic-SR step throughput on a Hamiltonian-specific
 * cluster. Exercises sampler → local-energy → gradients → QGT CG →
 * update. The complex-RBM is narrower than in a research run, but the
 * relative wall-cost across Hamiltonians is what the bench reports. */
static double sr_holomorphic_step_throughput_ham(int size_x, int size_y,
                                                  nqs_hamiltonian_kind_t ham,
                                                  int samples_per_step,
                                                  int steps) {
    int N = (ham == NQS_HAM_KAGOME_HEISENBERG)
            ? 3 * size_x * size_y
            : size_x * size_y;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz           = NQS_ANSATZ_COMPLEX_RBM;
    cfg.hamiltonian      = ham;
    cfg.kh_K             = 1.0;
    cfg.kh_J             = 1.0;
    cfg.j_coupling       = 1.0;
    cfg.kagome_pbc       = 1;
    cfg.num_samples      = samples_per_step;
    cfg.num_thermalize   = 128;
    cfg.num_decorrelate  = 1;
    cfg.rbm_hidden_units = 8;
    cfg.learning_rate    = 1e-3;
    cfg.sr_diag_shift    = 1e-2;
    cfg.sr_cg_max_iters  = 20;
    cfg.sr_cg_tol        = 1e-7;

    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a);
    nqs_sampler_thermalize(s);

    double t0 = bench_now_seconds();
    for (int i = 0; i < steps; i++) {
        nqs_sr_step_info_t info;
        nqs_sr_step_holomorphic(&cfg, size_x, size_y, a, s,
                                  nqs_ansatz_log_amp, a, &info);
    }
    double dt = bench_now_seconds() - t0;

    nqs_sampler_free(s);
    nqs_ansatz_free(a);
    return (double)steps / dt;
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

/* Local-energy evaluation throughput for a specified Hamiltonian on
 * a cluster of the requested (size_x, size_y) dims. For kagome the
 * number of sites is 3·size_x·size_y; for every other kernel it's
 * size_x·size_y. */
static double local_energy_throughput(int size_x, int size_y,
                                       nqs_hamiltonian_kind_t ham,
                                       int num_samples) {
    int N = (ham == NQS_HAM_KAGOME_HEISENBERG)
            ? 3 * size_x * size_y
            : size_x * size_y;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz           = NQS_ANSATZ_COMPLEX_RBM;
    cfg.hamiltonian      = ham;
    cfg.kh_K             = 1.0;
    cfg.kh_J             = 1.0;
    cfg.j_coupling       = 1.0;
    cfg.kagome_pbc       = 1;
    cfg.num_samples      = num_samples;
    cfg.num_thermalize   = 128;
    cfg.num_decorrelate  = 1;
    cfg.rbm_hidden_units = 8;

    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a);
    nqs_sampler_thermalize(s);

    int *batch = malloc((size_t)num_samples * (size_t)N * sizeof(int));
    nqs_sampler_batch(s, num_samples, batch);

    double *re = malloc((size_t)num_samples * sizeof(double));
    double *im = malloc((size_t)num_samples * sizeof(double));
    double t0 = bench_now_seconds();
    nqs_local_energy_batch_complex(&cfg, size_x, size_y, batch, num_samples,
                                    nqs_ansatz_log_amp, a, re, im);
    double dt = bench_now_seconds() - t0;
    free(re); free(im); free(batch);
    nqs_sampler_free(s);
    nqs_ansatz_free(a);
    return (double)num_samples / dt;
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

    /* v0.4.1: local-energy throughput for the two new Hamiltonian
     * kernels — a silent-drift canary. Kagome 2×2 PBC has N=12 sites
     * (3 per cell); KH 2×2 has N=4 sites (brick-wall honeycomb). */
    struct { const char *name; int Lx; int Ly; nqs_hamiltonian_kind_t ham; int samples; } ker[] = {
        { "kh_local_energy_2x2",     2, 2, NQS_HAM_KITAEV_HEISENBERG, 4096 },
        { "kagome_local_energy_2x2", 2, 2, NQS_HAM_KAGOME_HEISENBERG, 2048 },
    };
    for (size_t i = 0; i < sizeof(ker) / sizeof(ker[0]); i++) {
        double sps = local_energy_throughput(ker[i].Lx, ker[i].Ly, ker[i].ham, ker[i].samples);
        bench_emitter_t em;
        bench_emit_begin(&em, "nqs", ker[i].name);
        bench_emit_int(&em, "Lx", ker[i].Lx);
        bench_emit_int(&em, "Ly", ker[i].Ly);
        bench_emit_int(&em, "num_samples", ker[i].samples);
        bench_emit_metric(&em, "local_energy_per_second", sps);
        bench_emit_end(&em);
        printf("nqs %s: %.1f eval/sec\n", ker[i].name, sps);
    }

    /* Sampler + full-SR-step throughput for the KH / kagome kernels.
     * Together with the local-energy rows above, gives one record per
     * major pipeline stage so per-Hamiltonian drift across releases is
     * visible in `benchmarks/results/`. */
    struct { const char *name; int Lx; int Ly; nqs_hamiltonian_kind_t ham; int samples; } samp_ker[] = {
        { "kh_sampler_2x2",     2, 2, NQS_HAM_KITAEV_HEISENBERG, 4096 },
        { "kagome_sampler_2x2", 2, 2, NQS_HAM_KAGOME_HEISENBERG, 2048 },
    };
    for (size_t i = 0; i < sizeof(samp_ker) / sizeof(samp_ker[0]); i++) {
        double sps = sampler_throughput_ham(samp_ker[i].Lx, samp_ker[i].Ly,
                                              samp_ker[i].ham, samp_ker[i].samples);
        bench_emitter_t em;
        bench_emit_begin(&em, "nqs", samp_ker[i].name);
        bench_emit_int(&em, "Lx", samp_ker[i].Lx);
        bench_emit_int(&em, "Ly", samp_ker[i].Ly);
        bench_emit_int(&em, "samples", samp_ker[i].samples);
        bench_emit_metric(&em, "samples_per_second", sps);
        bench_emit_end(&em);
        printf("nqs %s: %.1f samples/sec\n", samp_ker[i].name, sps);
    }

    struct { const char *name; int Lx; int Ly; nqs_hamiltonian_kind_t ham;
             int samples; int steps; } sr_ker[] = {
        { "kh_sr_holomorphic_2x2",     2, 2, NQS_HAM_KITAEV_HEISENBERG, 256, 10 },
        { "kagome_sr_holomorphic_2x2", 2, 2, NQS_HAM_KAGOME_HEISENBERG, 256, 10 },
    };
    for (size_t i = 0; i < sizeof(sr_ker) / sizeof(sr_ker[0]); i++) {
        double sps = sr_holomorphic_step_throughput_ham(
            sr_ker[i].Lx, sr_ker[i].Ly, sr_ker[i].ham,
            sr_ker[i].samples, sr_ker[i].steps);
        bench_emitter_t em;
        bench_emit_begin(&em, "nqs", sr_ker[i].name);
        bench_emit_int(&em, "Lx", sr_ker[i].Lx);
        bench_emit_int(&em, "Ly", sr_ker[i].Ly);
        bench_emit_int(&em, "samples_per_step", sr_ker[i].samples);
        bench_emit_int(&em, "steps_run", sr_ker[i].steps);
        bench_emit_metric(&em, "sr_steps_per_second", sps);
        bench_emit_end(&em);
        printf("nqs %s: %.2f steps/sec\n", sr_ker[i].name, sps);
    }
    return 0;
}
