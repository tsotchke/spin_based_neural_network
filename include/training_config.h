/*
 * include/training_config.h
 *
 * Runtime configuration for how topological observables and decoder
 * feedback participate in the training loop. Populated from CLI flags in
 * main.c and passed to inner routines so the loop stays thin.
 *
 * v0.4 additions:
 *   - cadence_entropy         run topological-entropy sampling every N iters
 *                             (0 = never, 1 = every iter, ...)
 *   - cadence_invariants      run Chern/winding/TKNN invariants every N iters
 *   - cadence_decoder         if non-zero, each iteration seeds a toric-code
 *                             from the Kitaev lattice, samples errors at the
 *                             configured rate, decodes greedily, and folds
 *                             (logical error yes/no) into the physics loss
 *   - decoder_error_rate      per-iteration physical error rate for P0.1
 *                             decoder-feedback sampling
 *   - lambda_topological      weight on |γ_topo - target_gamma|^2 added to loss
 *   - lambda_logical          weight on logical-error indicator from decoder
 *   - lambda_chern            weight on (C - target_chern)^2 added to loss when
 *                             invariants are evaluated (cadence_invariants > 0)
 *   - target_chern            target Chern number for the topological loss term
 *   - target_gamma            target TEE γ (defaults to log 2 for Z₂ order)
 */
#ifndef TRAINING_CONFIG_H
#define TRAINING_CONFIG_H

#include <math.h>

typedef struct {
    int    cadence_entropy;
    int    cadence_invariants;
    int    cadence_decoder;
    double decoder_error_rate;
    double lambda_topological;
    double lambda_logical;
    double lambda_chern;
    double target_chern;
    double target_gamma;
    int    verbose;
} training_config_t;

static inline training_config_t training_config_defaults(void) {
    training_config_t c = {0};
    c.cadence_entropy    = 0;      /* disabled unless CLI sets it */
    c.cadence_invariants = 0;
    c.cadence_decoder    = 0;
    c.decoder_error_rate = 0.03;
    c.lambda_topological = 0.1;
    c.lambda_logical     = 1.0;
    c.lambda_chern       = 0.0;    /* opt-in: disable Chern fold by default */
    c.target_chern       = 1.0;    /* QAH with C = +1 */
    c.target_gamma       = 0.69314718055994530942; /* log(2) — Z₂ topological order */
    c.verbose            = 0;
    return c;
}

#endif /* TRAINING_CONFIG_H */
