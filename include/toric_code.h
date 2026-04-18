#ifndef TORIC_CODE_H
#define TORIC_CODE_H

#include "kitaev_model.h"

/*
 * Toric code on an L_x × L_y torus.
 *
 * Data model (physical):
 *   - 2 * L_x * L_y data qubits, one per link.
 *     link(x, y, 0) = horizontal link from (x,y) to (x+1, y)
 *     link(x, y, 1) = vertical link from (x,y) to (x, y+1)
 *     link_index(x, y, dir) = 2 * (x * L_y + y) + dir
 *   - Each data qubit accumulates X-errors and Z-errors independently
 *     in GF(2): x_errors[k], z_errors[k] ∈ {0,1}.
 *   - L_x * L_y vertex stabilizers A_v = ⊗σ_x over 4 adjacent links.
 *     Detects Z errors (anticommutes with σ_z). Syndrome +1 = clean, -1 = flagged.
 *   - L_x * L_y plaquette stabilizers B_p = ⊗σ_z over 4 adjacent links.
 *     Detects X errors (anticommutes with σ_x). Syndrome +1 = clean, -1 = flagged.
 *
 * Legacy fields (star_operators/plaquette_operators/logical_operators_x/z) are
 * retained for back-compat with pre-v0.4 callers. New code should use the
 * x_errors/z_errors arrays plus toric_code_measure_* to access syndromes.
 */
typedef struct {
    int size_x;                 /* L_x */
    int size_y;                 /* L_y */
    int num_links;              /* 2 * L_x * L_y */

    /* Physical data-qubit error accumulators (GF(2)) */
    int *x_errors;              /* size num_links, 0 or 1 */
    int *z_errors;              /* size num_links, 0 or 1 */

    /* Syndromes derived from current error state */
    int *vertex_syndrome;       /* size L_x*L_y, 0 (ok) or 1 (flagged) */
    int *plaquette_syndrome;    /* size L_x*L_y, 0 (ok) or 1 (flagged) */

    /* Legacy fields — preserved so pre-v0.4 code keeps compiling */
    int **star_operators;
    int **plaquette_operators;
    int *logical_operators_x;
    int *logical_operators_z;
} ToricCode;

typedef struct {
    int error_type;             /* 0 = X-error syndrome (plaquettes), 1 = Z-error syndrome (vertices) */
    int *error_positions;       /* flagged stabilizer indices (0..Lx*Ly) */
    int num_errors;             /* count */
} ErrorSyndrome;

/* --- Lifecycle --- */
ToricCode* initialize_toric_code(int size_x, int size_y);
void       free_toric_code(ToricCode *code);

/* --- Link indexing helpers (inline-friendly) --- */
int  toric_code_link_index(const ToricCode *code, int x, int y, int dir);
void toric_code_vertex_links(const ToricCode *code, int vx, int vy, int out_links[4]);
void toric_code_plaquette_links(const ToricCode *code, int px, int py, int out_links[4]);

/* --- Error channel and syndrome --- */
/* Apply independent X and Z errors at rate `error_rate` to each data qubit. */
void apply_random_errors(ToricCode *code, double error_rate);

/* Flip a data qubit's X or Z error bit. Also refreshes affected syndromes. */
void toric_code_apply_x_error(ToricCode *code, int link_index);
void toric_code_apply_z_error(ToricCode *code, int link_index);

/* Named correction API (same physics as _apply_x_error / _apply_z_error;
 * distinct names so decoder code reads cleanly). */
void toric_code_apply_x_correction(ToricCode *code, int link_index);
void toric_code_apply_z_correction(ToricCode *code, int link_index);

/* Recompute all syndromes from x_errors / z_errors. */
void toric_code_refresh_syndromes(ToricCode *code);

/* Measure the X- and Z-type syndromes and return legacy ErrorSyndrome structs.
 * Caller must free via free_error_syndrome. */
ErrorSyndrome* measure_error_syndrome(ToricCode *code);
ErrorSyndrome* toric_code_measure_x_syndrome(ToricCode *code);
ErrorSyndrome* toric_code_measure_z_syndrome(ToricCode *code);
void           free_error_syndrome(ErrorSyndrome *syndrome);

/* --- Decoding --- */
/* Baseline greedy-matching decoder: pairs flagged syndromes by toroidal
 * taxicab distance and applies corrections along the shortest link path.
 * Correct for low error rates (p << threshold); will be superseded by a
 * learned decoder in v0.5 (pillar P1.3). Returns 0 on success. */
int  perform_error_correction(ToricCode *code, ErrorSyndrome *syndrome);
int  toric_code_decode_greedy(ToricCode *code);

/* --- Queries --- */
/* Returns 1 iff all syndromes are +1 AND logical parities are unchanged. */
int  is_ground_state(ToricCode *code);

/* Returns 1 iff the accumulated errors form a non-contractible loop
 * (i.e. a logical error has occurred), else 0. */
int  toric_code_has_logical_error(const ToricCode *code);

/* Ground state degeneracy on a torus = 4. */
int  calculate_ground_state_degeneracy(ToricCode *code);

/* --- Kitaev-lattice coupling (legacy) --- */
void calculate_stabilizers(ToricCode *code, KitaevLattice *lattice);
void map_toric_code_to_lattice(ToricCode *code, KitaevLattice *lattice);

#endif /* TORIC_CODE_H */
