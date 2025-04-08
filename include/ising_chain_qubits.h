#ifndef ISING_CHAIN_QUBITS_H
#define ISING_CHAIN_QUBITS_H

#include "kitaev_model.h"
#include "majorana_modes.h"

typedef struct {
    KitaevLattice *lattice;     // Underlying lattice
    MajoranaChain **chains;     // Array of Majorana chains
    int num_chains;             // Number of chains
    int (*chain_positions)[3];  // Starting positions of each chain [chain_idx][x,y,z]
    int *chain_orientations;    // Orientations (0=x, 1=y, 2=z direction)
} IsingChainQubits;

typedef struct {
    int chain1;                 // First chain index
    int chain2;                 // Second chain index
    double interaction_strength; // Interaction strength
} ChainInteraction;

// Initialize a system of Ising chain qubits
IsingChainQubits* initialize_ising_chain_qubits(KitaevLattice *lattice, 
                                               int num_chains, int chain_length,
                                               KitaevWireParameters *params);

// Free memory allocated for Ising chain qubits
void free_ising_chain_qubits(IsingChainQubits *qubits);

// Create a topological qubit from a pair of Majorana zero modes
void create_topological_qubit(IsingChainQubits *qubits, int chain_index);

// Apply X gate to a topological qubit
void apply_topological_x_gate(IsingChainQubits *qubits, int qubit_index);

// Apply Z gate to a topological qubit
void apply_topological_z_gate(IsingChainQubits *qubits, int qubit_index);

// Apply Y gate to a topological qubit
void apply_topological_y_gate(IsingChainQubits *qubits, int qubit_index);

// Perform a CNOT gate between two topological qubits
void apply_topological_cnot(IsingChainQubits *qubits, int control_qubit, int target_qubit);

// Initialize chains for qubit encoding
void initialize_chains(IsingChainQubits *qubits, int chain_length, KitaevWireParameters *params);

// Encode a qubit state in a pair of Majorana zero modes
void encode_qubit_state(IsingChainQubits *qubits, int qubit_index, int state);

// Measure a topological qubit
int measure_topological_qubit(IsingChainQubits *qubits, int qubit_index);

// Add interaction between chains
void add_chain_interaction(IsingChainQubits *qubits, int chain1, int chain2, double strength);

#endif // ISING_CHAIN_QUBITS_H
