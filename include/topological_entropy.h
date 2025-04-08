#ifndef TOPOLOGICAL_ENTROPY_H
#define TOPOLOGICAL_ENTROPY_H

#include <complex.h>
#include "kitaev_model.h"

typedef struct {
    int subsystem_a_coords[3];  // Starting coordinates for subsystem A
    int subsystem_a_size[3];    // Size of subsystem A
    int subsystem_b_coords[3];  // Starting coordinates for subsystem B  
    int subsystem_b_size[3];    // Size of subsystem B
    double alpha;              // Non-universal constant
    double gamma;              // Topological entanglement entropy
    double boundary_length;    // Length of the boundary between subsystems
} EntanglementData;

typedef struct {
    double quantum_dimension;  // Total quantum dimension
    int num_anyons;           // Number of anyon types
    double *anyon_dimensions;  // Individual anyon dimensions
} TopologicalOrder;

// Calculate von Neumann entropy between two subsystems
double calculate_von_neumann_entropy(KitaevLattice *lattice, 
                                    int subsystem_coords[3], 
                                    int subsystem_size[3]);

// Calculate the reduced density matrix for a subsystem
void calculate_reduced_density_matrix(KitaevLattice *lattice, 
                                     int subsystem_coords[3], 
                                     int subsystem_size[3], 
                                     double _Complex *reduced_density_matrix,
                                     int matrix_size);

// Calculate topological entanglement entropy using Kitaev-Preskill formula
double calculate_topological_entropy(KitaevLattice *lattice, 
                                    EntanglementData *entanglement_data);

// Estimate quantum dimensions from entanglement entropy
TopologicalOrder* estimate_quantum_dimensions(double topological_entropy);

// Partition a lattice into regions for Kitaev-Preskill calculation
void partition_regions(KitaevLattice *lattice, EntanglementData *entanglement_data);

// Calculate density matrix from lattice state
void calculate_density_matrix(KitaevLattice *lattice, double _Complex *density_matrix, int matrix_size);

// Perform partial trace operation
void partial_trace(double _Complex *full_density_matrix, 
                  double _Complex *reduced_density_matrix,
                  int *subsystem_sites, 
                  int subsystem_size, 
                  int full_system_size);

// Calculate von Neumann entropy from density matrix
double von_neumann_entropy(double _Complex *density_matrix, int size);

// Calculate eigenvalues of a Hermitian matrix using power iteration method
void calculate_eigenvalues(double _Complex *matrix, double *eigenvalues, int size);

// Define regions in the specific Kitaev-Preskill clover-leaf arrangement
void define_kitaev_preskill_regions(KitaevLattice *lattice, EntanglementData *entanglement_data);

// Calculate a matrix element of the Kitaev model Hamiltonian
double _Complex calculate_kitaev_matrix_element(KitaevLattice *lattice, 
                                               unsigned long long state_i, 
                                               unsigned long long state_j);

// Free memory allocated for topological order
void free_topological_order(TopologicalOrder *order);

// NEON optimization functions
int check_neon_available(void);
double von_neumann_entropy_neon(double _Complex *density_matrix, int size);

#endif // TOPOLOGICAL_ENTROPY_H
