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

// Perform braiding operation on Majorana modes
void braid_majorana_modes(MajoranaChain *chain, int mode1, int mode2);

// Compute the energy of a Kitaev wire
double compute_kitaev_wire_energy(MajoranaChain *chain, KitaevWireParameters *params);

// Create Majorana operators from fermionic operators
void create_majorana_operators(MajoranaChain *chain);

// Map Majorana chain onto Kitaev lattice
void map_chain_to_lattice(MajoranaChain *chain, KitaevLattice *lattice, int start_x, int start_y, int start_z, int direction);

#endif // MAJORANA_MODES_H
