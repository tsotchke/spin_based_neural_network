/*
 * src/thermodynamic/hopfield.c
 *
 * Hopfield associative memory — P2.9 thermodynamic-computing baseline.
 * See include/thermodynamic/hopfield.h for the algorithm.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "thermodynamic/hopfield.h"

static double xs_uniform(unsigned long long *st) {
    unsigned long long x = *st ? *st : 1ULL;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *st = x;
    return (double)(x >> 11) / 9007199254740992.0;
}

hopfield_t *hopfield_create(int num_spins) {
    if (num_spins <= 0) return NULL;
    hopfield_t *n = calloc(1, sizeof(*n));
    if (!n) return NULL;
    n->num_spins = num_spins;
    n->J = calloc((size_t)num_spins * (size_t)num_spins, sizeof(double));
    n->h = calloc((size_t)num_spins, sizeof(double));
    if (!n->J || !n->h) { hopfield_free(n); return NULL; }
    return n;
}

void hopfield_free(hopfield_t *net) {
    if (!net) return;
    free(net->J);
    free(net->h);
    free(net);
}

int hopfield_store_patterns(hopfield_t *net,
                             const int *patterns, int num_patterns) {
    if (!net || !patterns || num_patterns <= 0) return -1;
    int N = net->num_spins;
    memset(net->J, 0, (size_t)N * (size_t)N * sizeof(double));
    double inv_N = 1.0 / (double)N;
    for (int mu = 0; mu < num_patterns; mu++) {
        const int *xi = &patterns[(size_t)mu * (size_t)N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                net->J[i * N + j] += inv_N * (double)xi[i] * (double)xi[j];
            }
        }
    }
    for (int i = 0; i < N; i++) net->J[i * N + i] = 0.0;
    return 0;
}

static double local_field(const hopfield_t *net, const int *spins, int i) {
    int N = net->num_spins;
    const double *row = &net->J[(size_t)i * N];
    double f = net->h[i];
    for (int j = 0; j < N; j++) f += row[j] * (double)spins[j];
    return f;
}

int hopfield_sync_update(hopfield_t *net, int *spins) {
    int N = net->num_spins;
    int changed = 0;
    int *new_spins = malloc((size_t)N * sizeof(int));
    if (!new_spins) return 0;
    for (int i = 0; i < N; i++) {
        double f = local_field(net, spins, i);
        int v = (f >= 0.0) ? +1 : -1;
        new_spins[i] = v;
        if (v != spins[i]) changed++;
    }
    memcpy(spins, new_spins, (size_t)N * sizeof(int));
    free(new_spins);
    return changed;
}

void hopfield_metropolis_sweep(const hopfield_t *net,
                                int *spins, double beta,
                                unsigned long long *rng_state) {
    int N = net->num_spins;
    /* Fisher–Yates shuffle of visit order. */
    int *order = malloc((size_t)N * sizeof(int));
    if (!order) return;
    for (int i = 0; i < N; i++) order[i] = i;
    for (int i = N - 1; i > 0; i--) {
        int j = (int)((double)(i + 1) * xs_uniform(rng_state));
        if (j > i) j = i;
        int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }
    for (int k = 0; k < N; k++) {
        int i = order[k];
        double f = local_field(net, spins, i);
        /* Flip lowers energy by 2 s_i f; accept via Boltzmann factor. */
        double dE = 2.0 * (double)spins[i] * f;
        if (dE <= 0.0 || xs_uniform(rng_state) < exp(-beta * dE)) {
            spins[i] = -spins[i];
        }
    }
    free(order);
}

double hopfield_overlap(const hopfield_t *net,
                         const int *pattern, const int *spins) {
    int N = net->num_spins;
    long s = 0;
    for (int i = 0; i < N; i++) s += (long)pattern[i] * (long)spins[i];
    return (double)s / (double)N;
}

int hopfield_recall(hopfield_t *net, int *spins, int num_sweeps) {
    for (int k = 0; k < num_sweeps; k++) {
        int changed = hopfield_sync_update(net, spins);
        if (changed == 0) return 1;
    }
    return 0;
}

double hopfield_energy(const hopfield_t *net, const int *spins) {
    int N = net->num_spins;
    double E = 0.0;
    for (int i = 0; i < N; i++) {
        E -= net->h[i] * (double)spins[i];
        const double *row = &net->J[(size_t)i * N];
        for (int j = 0; j < N; j++) {
            E -= 0.5 * row[j] * (double)spins[i] * (double)spins[j];
        }
    }
    return E;
}
