#ifndef TORIC_CODE_H
#define TORIC_CODE_H

#include "kitaev_model.h"

typedef struct {
    int size_x;                 // X dimension of toric code
    int size_y;                 // Y dimension of toric code
    int **star_operators;       // Star (vertex) operators A_v
    int **plaquette_operators;  // Plaquette operators B_p
    int *logical_operators_x;   // Logical X operators
    int *logical_operators_z;   // Logical Z operators
} ToricCode;

typedef struct {
    int error_type;             // 0 for bit-flip, 1 for phase-flip
    int *error_positions;       // Positions of errors
    int num_errors;             // Number of errors
} ErrorSyndrome;

// Initialize a toric code on a lattice
ToricCode* initialize_toric_code(int size_x, int size_y);

// Free memory allocated for toric code
void free_toric_code(ToricCode *code);

// Calculate the stabilizers (star and plaquette operators)
void calculate_stabilizers(ToricCode *code, KitaevLattice *lattice);

// Apply a random error to the toric code
void apply_random_errors(ToricCode *code, double error_rate);

// Measure the error syndrome
ErrorSyndrome* measure_error_syndrome(ToricCode *code);

// Free memory allocated for error syndrome
void free_error_syndrome(ErrorSyndrome *syndrome);

// Perform error correction
void perform_error_correction(ToricCode *code, ErrorSyndrome *syndrome);

// Calculate the ground state degeneracy
int calculate_ground_state_degeneracy(ToricCode *code);

// Map toric code onto a Kitaev lattice
void map_toric_code_to_lattice(ToricCode *code, KitaevLattice *lattice);

// Check if the toric code is in a ground state
int is_ground_state(ToricCode *code);

#endif // TORIC_CODE_H
