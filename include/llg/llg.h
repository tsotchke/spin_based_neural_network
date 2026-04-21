/*
 * include/llg/llg.h
 *
 * Landau-Lifshitz-Gilbert dynamics for classical magnetization fields.
 * v0.4 ships an RK4 integrator driven by a user-supplied effective-field
 * callback; v0.5 pillar P1.2 replaces that callback with the output of
 * an E(3)-equivariant GNN (via libirrep).
 *
 *     ṁ = -γ (m × B_eff) - α γ (m × (m × B_eff))
 *
 * Magnetisation is stored as a contiguous array of 3-vectors (mx, my,
 * mz) with |m| = 1 enforced by projection after each step.
 */
#ifndef LLG_LLG_H
#define LLG_LLG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*llg_effective_field_fn_t)(const double *m, double *b_eff,
                                          long num_sites, void *user_data);

typedef struct {
    double gamma;                 /* gyromagnetic ratio (rad / (T·s))   */
    double alpha;                 /* Gilbert damping                    */
    double dt;                    /* integrator timestep                */
    llg_effective_field_fn_t field_fn;
    void *field_user_data;
} llg_config_t;

static inline llg_config_t llg_config_defaults(void) {
    llg_config_t c;
    c.gamma           = 1.760859644e11;     /* rad / (T·s), electron */
    c.alpha           = 0.01;
    c.dt              = 1e-14;              /* 10 fs */
    c.field_fn        = NULL;
    c.field_user_data = NULL;
    return c;
}

/* Project m onto the unit sphere in-place (per-site). */
void llg_renormalize(double *m, long num_sites);

/* 4th-order Runge-Kutta step for the LLG equation. Advances m by dt
 * in-place and re-normalises each spin to |m|=1. Returns 0 on success. */
int llg_rk4_step(const llg_config_t *cfg, double *m, long num_sites);

/* Heun (improved-Euler) step — second-order, cheaper, stable at small dt. */
int llg_heun_step(const llg_config_t *cfg, double *m, long num_sites);

/* Cross product c = a × b (3-vectors). */
void llg_cross3(const double a[3], const double b[3], double out[3]);

#ifdef __cplusplus
}
#endif

#endif /* LLG_LLG_H */
