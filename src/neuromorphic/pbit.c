/*
 * src/neuromorphic/pbit.c
 *
 * Probabilistic-bit Ising-machine primitives. Pure C, deterministic
 * xorshift64 RNG so tests are reproducible. Asynchronous sweeps in
 * a Fisher-Yates-shuffled order per pass.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "neuromorphic/pbit.h"

static double xorshift_unit(unsigned long long *state) {
    unsigned long long x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x;
    return (double)(x >> 11) / 9007199254740992.0;
}

static int xorshift_range(unsigned long long *state, int n) {
    double u = xorshift_unit(state);
    int k = (int)(u * (double)n);
    if (k >= n) k = n - 1;
    return k;
}

void pbit_net_init(pbit_net_t *net, int n,
                    const double *J, const double *b,
                    double beta, unsigned long long seed) {
    net->n    = n;
    net->J    = J;
    net->b    = b;
    net->beta = beta;
    net->rng  = seed ? seed : 0xDEADBEEFULL;
}

/* Local field at site i: h_i = Σ_j J_ij s_j + b_i. */
static double local_field(const pbit_net_t *net, const int *spins, int i) {
    double h = 0.0;
    int n = net->n;
    const double *row = &net->J[(size_t)i * n];
    for (int j = 0; j < n; j++) h += row[j] * (double)spins[j];
    if (net->b) h += net->b[i];
    return h;
}

int pbit_sweep(pbit_net_t *net, int *spins, int num_sweeps) {
    if (!net || !spins || num_sweeps < 0) return -1;
    int n = net->n;
    int *order = malloc((size_t)n * sizeof(int));
    if (!order) return -1;
    for (int s = 0; s < num_sweeps; s++) {
        /* Fisher-Yates shuffle to pick visit order. */
        for (int i = 0; i < n; i++) order[i] = i;
        for (int i = n - 1; i > 0; i--) {
            int j = xorshift_range(&net->rng, i + 1);
            int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }
        for (int k = 0; k < n; k++) {
            int i = order[k];
            double h = local_field(net, spins, i);
            /* P(+1) = σ(β h). */
            double p = 1.0 / (1.0 + exp(-net->beta * h));
            double u = xorshift_unit(&net->rng);
            spins[i] = (u < p) ? +1 : -1;
        }
    }
    free(order);
    return 0;
}

double pbit_ising_energy(const pbit_net_t *net, const int *spins) {
    double E = 0.0;
    int n = net->n;
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        const double *row = &net->J[(size_t)i * n];
        for (int j = 0; j < n; j++) {
            row_sum += row[j] * (double)spins[j];
        }
        E += spins[i] * row_sum;
        if (net->b) E += 2.0 * net->b[i] * (double)spins[i];
    }
    return -0.5 * E;
}

int pbit_maxcut_couplings(int n, int num_edges,
                           const int *edges_u, const int *edges_v,
                           const double *weights,
                           double *out_J) {
    if (!edges_u || !edges_v || !out_J || n <= 0 || num_edges < 0) return -1;
    memset(out_J, 0, sizeof(double) * (size_t)n * (size_t)n);
    for (int e = 0; e < num_edges; e++) {
        int u = edges_u[e], v = edges_v[e];
        double w = weights ? weights[e] : 1.0;
        /* MaxCut → Ising: cut weight when s_u ≠ s_v: contributes
         *    (1 - s_u s_v) / 2.
         * Sum over edges ⇒ maximise. Equivalent to MINIMISING
         *    Σ_e w_e s_u s_v / 2 - const
         * i.e. Ising energy E = -½ Σ J_ij s_i s_j with J_ij = -w_e
         * per edge (both directions, J symmetric). */
        out_J[(size_t)u * n + v] -= w;
        out_J[(size_t)v * n + u] -= w;
    }
    return 0;
}

int pbit_anneal(pbit_net_t *net, int *spins,
                 double beta_start, double beta_end, int num_sweeps,
                 double *out_final_energy) {
    if (!net || !spins || num_sweeps <= 0) return -1;
    double log_ratio = (num_sweeps > 1) ?
        log(beta_end / beta_start) / (double)(num_sweeps - 1) : 0.0;
    for (int s = 0; s < num_sweeps; s++) {
        net->beta = beta_start * exp(log_ratio * (double)s);
        pbit_sweep(net, spins, 1);
    }
    if (out_final_energy) *out_final_energy = pbit_ising_energy(net, spins);
    return 0;
}
