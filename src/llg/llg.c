/*
 * src/llg/llg.c
 *
 * Minimal RK4 / Heun integrators for the Landau-Lifshitz-Gilbert
 * equation. The effective-field function is user-supplied so this
 * module is completely decoupled from the equivariant-GNN torque
 * network that lands with v0.5 pillar P1.2.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "llg/llg.h"

void llg_cross3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

void llg_renormalize(double *m, long num_sites) {
    if (!m) return;
    for (long i = 0; i < num_sites; i++) {
        double x = m[3*i], y = m[3*i+1], z = m[3*i+2];
        double n = sqrt(x*x + y*y + z*z);
        if (n > 0.0) {
            m[3*i]   = x / n;
            m[3*i+1] = y / n;
            m[3*i+2] = z / n;
        }
    }
}

/* Compute the time derivative dm/dt = -γ (m × B) - α γ (m × (m × B))
 * for each site, writing it into `dm` (length 3·num_sites). */
static void llg_rhs(const llg_config_t *cfg,
                    const double *m, double *dm, double *b_scratch,
                    long num_sites) {
    cfg->field_fn(m, b_scratch, num_sites, cfg->field_user_data);
    double gamma = cfg->gamma;
    double alpha = cfg->alpha;
    for (long i = 0; i < num_sites; i++) {
        const double *mi = &m[3*i];
        const double *bi = &b_scratch[3*i];
        double cross_mb[3];
        llg_cross3(mi, bi, cross_mb);
        double cross_m_mb[3];
        llg_cross3(mi, cross_mb, cross_m_mb);
        dm[3*i  ] = -gamma * cross_mb[0] - alpha * gamma * cross_m_mb[0];
        dm[3*i+1] = -gamma * cross_mb[1] - alpha * gamma * cross_m_mb[1];
        dm[3*i+2] = -gamma * cross_mb[2] - alpha * gamma * cross_m_mb[2];
    }
}

int llg_rk4_step(const llg_config_t *cfg, double *m, long num_sites) {
    if (!cfg || !cfg->field_fn || !m || num_sites <= 0) return -1;
    long N = 3 * num_sites;
    double *k1 = calloc((size_t)N, sizeof(double));
    double *k2 = calloc((size_t)N, sizeof(double));
    double *k3 = calloc((size_t)N, sizeof(double));
    double *k4 = calloc((size_t)N, sizeof(double));
    double *tmp = calloc((size_t)N, sizeof(double));
    double *b_scratch = calloc((size_t)N, sizeof(double));
    if (!k1 || !k2 || !k3 || !k4 || !tmp || !b_scratch) {
        free(k1); free(k2); free(k3); free(k4); free(tmp); free(b_scratch);
        return -1;
    }
    double dt = cfg->dt;

    /* k1 = f(m) */
    llg_rhs(cfg, m, k1, b_scratch, num_sites);
    /* k2 = f(m + dt/2 · k1) */
    for (long i = 0; i < N; i++) tmp[i] = m[i] + 0.5 * dt * k1[i];
    llg_rhs(cfg, tmp, k2, b_scratch, num_sites);
    /* k3 = f(m + dt/2 · k2) */
    for (long i = 0; i < N; i++) tmp[i] = m[i] + 0.5 * dt * k2[i];
    llg_rhs(cfg, tmp, k3, b_scratch, num_sites);
    /* k4 = f(m + dt · k3) */
    for (long i = 0; i < N; i++) tmp[i] = m[i] + dt * k3[i];
    llg_rhs(cfg, tmp, k4, b_scratch, num_sites);
    /* m ← m + dt/6 · (k1 + 2 k2 + 2 k3 + k4) */
    for (long i = 0; i < N; i++) {
        m[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    llg_renormalize(m, num_sites);
    free(k1); free(k2); free(k3); free(k4); free(tmp); free(b_scratch);
    return 0;
}

int llg_heun_step(const llg_config_t *cfg, double *m, long num_sites) {
    if (!cfg || !cfg->field_fn || !m || num_sites <= 0) return -1;
    long N = 3 * num_sites;
    double *k1 = calloc((size_t)N, sizeof(double));
    double *k2 = calloc((size_t)N, sizeof(double));
    double *tmp = calloc((size_t)N, sizeof(double));
    double *b = calloc((size_t)N, sizeof(double));
    if (!k1 || !k2 || !tmp || !b) {
        free(k1); free(k2); free(tmp); free(b); return -1;
    }
    double dt = cfg->dt;
    llg_rhs(cfg, m, k1, b, num_sites);
    for (long i = 0; i < N; i++) tmp[i] = m[i] + dt * k1[i];
    llg_renormalize(tmp, num_sites);
    llg_rhs(cfg, tmp, k2, b, num_sites);
    for (long i = 0; i < N; i++) m[i] += 0.5 * dt * (k1[i] + k2[i]);
    llg_renormalize(m, num_sites);
    free(k1); free(k2); free(tmp); free(b);
    return 0;
}
