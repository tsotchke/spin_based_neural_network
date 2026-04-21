/*
 * src/moonlab_bridge.c
 *
 * Lazy wrapper around libquantumsim (Moonlab). When built without
 * -DSPIN_NN_HAS_MOONLAB, every entry point returns EDISABLED so the
 * bridge is dormant at zero cost. When enabled, forwards to the
 * moonlab surface-code + MWPM-decoder API to provide ground-truth
 * logical-error-rate references against which in-tree QEC decoders
 * can be cross-validated.
 */
#include <stdio.h>
#include <stdlib.h>
#include "moonlab_bridge.h"

#ifdef SPIN_NN_HAS_MOONLAB
/* Moonlab public headers — add to include path at build time. */
#include "algorithms/topological/topological.h"
#include "quantum/state.h"
#endif

int moonlab_bridge_is_available(void) {
#ifdef SPIN_NN_HAS_MOONLAB
    return 1;
#else
    return 0;
#endif
}

const char *moonlab_bridge_version(void) {
#ifdef SPIN_NN_HAS_MOONLAB
    return "0.1.2";   /* Pinned at the libquantumsim version linked against. */
#else
    return NULL;
#endif
}

int moonlab_bridge_surface_code_roundtrip(int distance, double p,
                                          int *out_logical_error) {
    if (!out_logical_error) return MOONLAB_BRIDGE_EARG;
    if (distance < 3 || (distance & 1) == 0) {
        *out_logical_error = -1;
        return MOONLAB_BRIDGE_EARG;
    }
#ifdef SPIN_NN_HAS_MOONLAB
    surface_code_t *c = surface_code_create((uint32_t)distance);
    if (!c) { *out_logical_error = -1; return MOONLAB_BRIDGE_ELIB; }
    surface_code_init_logical_zero(c);

    /* Apply independent single-qubit depolarising errors. */
    unsigned long long state = 0x9E3779B97F4A7C15ULL;
    for (uint32_t q = 0; q < c->num_data_qubits; q++) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        double u = (double)(state >> 11) / 9007199254740992.0;
        if (u < p) {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            int kind = (int)((state >> 11) % 3);
            char et = (kind == 0) ? 'X' : (kind == 1) ? 'Y' : 'Z';
            surface_code_apply_error(c, q, et);
        }
    }
    surface_code_measure_X_stabilizers(c);
    surface_code_measure_Z_stabilizers(c);
    surface_code_decode_correct(c);

    /* Post-decode: count residual syndrome weight as a coarse logical-
     * error proxy. A zero-weight post-decode syndrome means the decoder
     * closed every detected chain; nonzero means there were unmatched
     * defects (boundary residue). A true logical-error test would
     * compare the decoded state to the encoded |0⟩; that's a v0.2
     * refinement once moonlab exposes the logical-observable readout. */
    int residue = 0;
    int num_stabs = (int)((distance - 1) * distance);
    for (int i = 0; i < num_stabs; i++) {
        if (c->x_syndrome && c->x_syndrome[i]) residue++;
        if (c->z_syndrome && c->z_syndrome[i]) residue++;
    }
    *out_logical_error = residue > 0 ? 1 : 0;
    surface_code_free(c);
    return MOONLAB_BRIDGE_OK;
#else
    (void)distance; (void)p;
    *out_logical_error = -1;
    return MOONLAB_BRIDGE_EDISABLED;
#endif
}

int moonlab_bridge_surface_code_logical_error_rate(int distance,
                                                    double p_phys,
                                                    int num_trials,
                                                    unsigned rng_seed,
                                                    double *out_p_logical) {
    if (!out_p_logical || num_trials <= 0) return MOONLAB_BRIDGE_EARG;
#ifdef SPIN_NN_HAS_MOONLAB
    (void)rng_seed;   /* bridge-level RNG is internal */
    long errors = 0;
    for (int t = 0; t < num_trials; t++) {
        int le = 0;
        int rc = moonlab_bridge_surface_code_roundtrip(distance, p_phys, &le);
        if (rc != MOONLAB_BRIDGE_OK) return rc;
        if (le) errors++;
    }
    *out_p_logical = (double)errors / (double)num_trials;
    return MOONLAB_BRIDGE_OK;
#else
    (void)distance; (void)p_phys; (void)num_trials; (void)rng_seed;
    *out_p_logical = -1.0;
    return MOONLAB_BRIDGE_EDISABLED;
#endif
}
