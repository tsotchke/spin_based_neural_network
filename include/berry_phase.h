#ifndef BERRY_PHASE_H
#define BERRY_PHASE_H

#include <complex.h>
#include "kitaev_model.h"
#include "majorana_modes.h"

typedef struct {
    int k_space_grid[3];           // K-space grid dimensions
    double _Complex ***connection; // Berry connection [3][kx][ky][kz]
    double ***curvature;          // Berry curvature [3][kx][ky][kz]
    double chern_number;           // Calculated Chern number
} BerryPhaseData;

typedef struct {
    double *invariants;         // Array of topological invariants
    int num_invariants;         // Number of invariants calculated
    char **invariant_names;     // Names of the invariants
} TopologicalInvariants;

// Initialize berry phase data structure
BerryPhaseData* initialize_berry_phase_data(int kx, int ky, int kz);

// Free memory allocated for berry phase data
void free_berry_phase_data(BerryPhaseData *data);

// Calculate Berry connection for a given k-point
void calculate_berry_connection(KitaevLattice *lattice, double k[3], 
                               double _Complex *connection);

// Calculate Berry curvature across the Brillouin zone
void calculate_berry_curvature(KitaevLattice *lattice, BerryPhaseData *berry_data);

// Calculate Chern number from Berry curvature
double calculate_chern_number(BerryPhaseData *berry_data);

// Calculate TKNN invariant for quantum Hall conductivity
double calculate_tknn_invariant(KitaevLattice *lattice);

// Calculate winding number for 1D systems
double calculate_winding_number(MajoranaChain *chain);

// Initialize topological invariants structure
TopologicalInvariants* initialize_topological_invariants(int num_invariants);

// Free memory allocated for topological invariants
void free_topological_invariants(TopologicalInvariants *invariants);

// Calculate all topological invariants for a system
TopologicalInvariants* calculate_all_invariants(KitaevLattice *lattice, 
                                               MajoranaChain *chain);

// Get the eigenstate at a given k-point
void get_eigenstate(KitaevLattice *lattice, double k[3], double _Complex *eigenstate, int band_index);

#endif // BERRY_PHASE_H
