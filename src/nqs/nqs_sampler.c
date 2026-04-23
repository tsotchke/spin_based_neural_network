/*
 * src/nqs/nqs_sampler.c
 *
 * Metropolis-Hastings sampler over spin configurations drawn from
 * |ψ(s)|^2. Uses a xorshift64 PRNG seeded from the config so the
 * state is reproducible independently of the global rand() stream.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nqs/nqs_sampler.h"

struct nqs_sampler {
    int                    num_sites;
    nqs_config_t           cfg;
    nqs_log_amp_fn_t       log_amp;
    void                  *log_amp_user;

    int                   *current;      /* length num_sites, ±1 */
    int                   *proposal;     /* length num_sites, ±1 */
    double                 current_log_abs;
    double                 current_arg;

    /* xorshift64 RNG state — avoids global rand() contention. */
    unsigned long long     rng_state;

    long                   n_accepted;
    long                   n_proposed;
};

static inline unsigned long long xorshift64_next(unsigned long long *s) {
    unsigned long long x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

static inline double xorshift64_uniform(unsigned long long *s) {
    /* Uniform in [0, 1). */
    unsigned long long x = xorshift64_next(s);
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);
}

static inline int xorshift64_range(unsigned long long *s, int n) {
    return (int)(xorshift64_next(s) % (unsigned long long)n);
}

nqs_sampler_t *nqs_sampler_create(int num_sites,
                                  const nqs_config_t *cfg,
                                  nqs_log_amp_fn_t log_amp,
                                  void *log_amp_user) {
    if (num_sites <= 0 || !cfg || !log_amp) return NULL;
    nqs_sampler_t *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->num_sites = num_sites;
    s->cfg = *cfg;
    s->log_amp = log_amp;
    s->log_amp_user = log_amp_user;
    /* xorshift requires nonzero seed. */
    s->rng_state = cfg->rng_seed ? (unsigned long long)cfg->rng_seed
                                 : 0x9E3779B97F4A7C15ULL;

    s->current  = malloc((size_t)num_sites * sizeof(int));
    s->proposal = malloc((size_t)num_sites * sizeof(int));
    if (!s->current || !s->proposal) { nqs_sampler_free(s); return NULL; }

    /* Uniform random initial configuration in {+1, -1}^N. */
    for (int i = 0; i < num_sites; i++) {
        s->current[i] = (xorshift64_next(&s->rng_state) & 1) ? 1 : -1;
    }
    log_amp(s->current, num_sites, log_amp_user,
            &s->current_log_abs, &s->current_arg);
    return s;
}

void nqs_sampler_free(nqs_sampler_t *s) {
    if (!s) return;
    free(s->current);
    free(s->proposal);
    free(s);
}

/* One single-flip Metropolis proposal. Returns 1 if accepted. */
static int nqs_sampler_step(nqs_sampler_t *s) {
    int site = xorshift64_range(&s->rng_state, s->num_sites);
    memcpy(s->proposal, s->current, (size_t)s->num_sites * sizeof(int));
    s->proposal[site] = -s->proposal[site];

    double prop_log_abs, prop_arg;
    s->log_amp(s->proposal, s->num_sites, s->log_amp_user,
               &prop_log_abs, &prop_arg);

    /* Metropolis ratio: |ψ(s')|^2 / |ψ(s)|^2 = exp(2 (log|ψ'| - log|ψ|)). */
    double log_ratio = 2.0 * (prop_log_abs - s->current_log_abs);
    int accept = 0;
    if (log_ratio >= 0.0) {
        accept = 1;
    } else {
        double u = xorshift64_uniform(&s->rng_state);
        accept = (log(u) < log_ratio);
    }

    s->n_proposed++;
    if (accept) {
        s->n_accepted++;
        /* Swap buffers — proposal becomes current, current becomes scratch. */
        int *tmp = s->current;
        s->current = s->proposal;
        s->proposal = tmp;
        s->current_log_abs = prop_log_abs;
        s->current_arg = prop_arg;
        return 1;
    }
    return 0;
}

void nqs_sampler_thermalize(nqs_sampler_t *s) {
    if (!s) return;
    for (int i = 0; i < s->cfg.num_thermalize; i++) {
        (void)nqs_sampler_step(s);
    }
}

const int *nqs_sampler_next(nqs_sampler_t *s) {
    if (!s) return NULL;
    int steps = s->cfg.num_decorrelate > 0 ? s->cfg.num_decorrelate : 1;
    for (int i = 0; i < steps; i++) {
        (void)nqs_sampler_step(s);
    }
    return s->current;
}

int nqs_sampler_batch(nqs_sampler_t *s, int n, int *out) {
    if (!s || !out || n <= 0) return -1;
    for (int b = 0; b < n; b++) {
        const int *sample = nqs_sampler_next(s);
        memcpy(&out[(size_t)b * (size_t)s->num_sites],
               sample,
               (size_t)s->num_sites * sizeof(int));
    }
    return 0;
}

double nqs_sampler_acceptance_ratio(const nqs_sampler_t *s) {
    if (!s || s->n_proposed == 0) return 0.0;
    return (double)s->n_accepted / (double)s->n_proposed;
}

const int *nqs_sampler_current(const nqs_sampler_t *s) {
    return s ? s->current : NULL;
}

int nqs_sampler_num_sites(const nqs_sampler_t *s) {
    return s ? s->num_sites : 0;
}
