#ifndef MAJORANA_MODES_H
#define MAJORANA_MODES_H

#include <complex.h>
#include "kitaev_model.h"

typedef struct {
    double _Complex *operators;  // Array of Majorana operators
    int num_operators;          // Number of operators (2N for N sites)
    int num_sites;              // Number of physical sites
    double mu;                  // Chemical potential
    double t;                   // Hopping amplitude
    double delta;               // Pairing strength
    int in_topological_phase;   // Flag for topological phase
} MajoranaChain;

typedef struct {
    double coupling_strength;   // Inter-site coupling
    double chemical_potential;  // On-site potential
    double superconducting_gap; // Superconducting gap parameter
} KitaevWireParameters;

/*
 * Fermionic Fock-space state vector for N Majorana operators (N even).
 * N/2 fermion modes -> 2^(N/2)-dim complex Hilbert space, indexed by
 * occupation bitstrings (n_0, n_1, ..., n_{N/2-1}).
 * Used for real, unitary Majorana braiding physics.
 */
typedef struct {
    int num_fermion_modes;       // = N/2 where N = num Majorana operators
    int hilbert_dim;             // = 2^num_fermion_modes
    double _Complex *amplitudes; // size hilbert_dim, state vector
} MajoranaHilbertState;

// Initialize a Majorana fermion chain
MajoranaChain* initialize_majorana_chain(int num_sites, KitaevWireParameters *params);

// Free memory allocated for a Majorana chain
void free_majorana_chain(MajoranaChain *chain);

// Apply Majorana operators to lattice sites
void apply_majorana_operator(MajoranaChain *chain, int operator_index, KitaevLattice *lattice);

// Calculate the parity of a Majorana chain
int calculate_majorana_parity(MajoranaChain *chain);

// Detect zero modes at the ends of a chain
double detect_majorana_zero_modes(MajoranaChain *chain, KitaevWireParameters *params);

/*
 * Legacy operator-permutation braiding (pre-v0.4).
 * Kept for back-compat with older scripts; new code should use
 * apply_braid_unitary() on a MajoranaHilbertState.
 */
void braid_majorana_operators_legacy(MajoranaChain *chain, int mode1, int mode2);

/*
 * Back-compat alias: delegates to braid_majorana_operators_legacy().
 * Existing CLI-driven demos keep working unchanged.
 */
void braid_majorana_modes(MajoranaChain *chain, int mode1, int mode2);

// Compute the energy of a Kitaev wire
double compute_kitaev_wire_energy(MajoranaChain *chain, KitaevWireParameters *params);

// Create Majorana operators from fermionic operators
void create_majorana_operators(MajoranaChain *chain);

// Map Majorana chain onto Kitaev lattice
void map_chain_to_lattice(MajoranaChain *chain, KitaevLattice *lattice, int start_x, int start_y, int start_z, int direction);

/*
 * MajoranaHilbertState lifecycle and physics.
 * num_majoranas must be even; num_fermion_modes = num_majoranas / 2.
 * Hilbert dim = 2^num_fermion_modes — keep num_fermion_modes <= 20 on 64-bit systems.
 */
MajoranaHilbertState* initialize_majorana_hilbert_state(int num_majoranas);
void free_majorana_hilbert_state(MajoranaHilbertState *state);

/*
 * Set the state to the fermion-vacuum |0,0,...,0>.
 */
void majorana_hilbert_state_set_vacuum(MajoranaHilbertState *state);

/*
 * In-place apply γ_{op_index} to the state vector via Jordan-Wigner:
 *   γ_{2k}   = Z_0 Z_1 ... Z_{k-1} X_k
 *   γ_{2k+1} = Z_0 Z_1 ... Z_{k-1} Y_k
 * Conserves ||ψ||^2 (up to floating error) since γ^2 = 1.
 */
void apply_majorana_op_to_state(int op_index, MajoranaHilbertState *state);

/*
 * In-place apply the non-Abelian braiding unitary
 *   B_{ij} = exp(π γ_i γ_j / 4) = (1 + γ_i γ_j) / √2
 * to the state. For Ising-anyon statistics: B^2 ∝ γ_i γ_j, B^4 = -I, B^8 = I.
 * Requires i != j and both in [0, 2*num_fermion_modes).
 */
void apply_braid_unitary(MajoranaHilbertState *state, int i, int j);

/*
 * Utility inner products / norms used by tests.
 */
double majorana_state_norm_squared(const MajoranaHilbertState *state);
double _Complex majorana_states_inner_product(const MajoranaHilbertState *a,
                                              const MajoranaHilbertState *b);
void majorana_state_copy(const MajoranaHilbertState *src, MajoranaHilbertState *dst);

#endif // MAJORANA_MODES_H
