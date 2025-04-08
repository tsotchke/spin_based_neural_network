#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include "ising_chain_qubits.h"

#define PI 3.14159265358979323846

// Initialize a system of Ising chain qubits
IsingChainQubits* initialize_ising_chain_qubits(KitaevLattice *lattice, 
                                               int num_chains, int chain_length,
                                               KitaevWireParameters *params) {
    if (!lattice || num_chains <= 0 || chain_length <= 0 || !params) {
        fprintf(stderr, "Error: Invalid parameters for initialize_ising_chain_qubits\n");
        return NULL;
    }
    
    IsingChainQubits *qubits = (IsingChainQubits *)malloc(sizeof(IsingChainQubits));
    if (!qubits) {
        fprintf(stderr, "Error: Memory allocation failed for IsingChainQubits\n");
        return NULL;
    }
    
    qubits->lattice = lattice;
    qubits->num_chains = num_chains;
    
    // Allocate memory for chains
    qubits->chains = (MajoranaChain **)malloc(num_chains * sizeof(MajoranaChain *));
    if (!qubits->chains) {
        fprintf(stderr, "Error: Memory allocation failed for chains\n");
        free(qubits);
        return NULL;
    }
    
    // Allocate memory for chain positions
    qubits->chain_positions = (int (*)[3])malloc(num_chains * sizeof(int[3]));
    if (!qubits->chain_positions) {
        fprintf(stderr, "Error: Memory allocation failed for chain_positions\n");
        free(qubits->chains);
        free(qubits);
        return NULL;
    }
    
    // Allocate memory for chain orientations
    qubits->chain_orientations = (int *)malloc(num_chains * sizeof(int));
    if (!qubits->chain_orientations) {
        fprintf(stderr, "Error: Memory allocation failed for chain_orientations\n");
        free(qubits->chain_positions);
        free(qubits->chains);
        free(qubits);
        return NULL;
    }
    
    // Initialize chains
    initialize_chains(qubits, chain_length, params);
    
    return qubits;
}

// Free memory allocated for Ising chain qubits
void free_ising_chain_qubits(IsingChainQubits *qubits) {
    if (qubits) {
        if (qubits->chains) {
            for (int i = 0; i < qubits->num_chains; i++) {
                if (qubits->chains[i]) {
                    free_majorana_chain(qubits->chains[i]);
                }
            }
            free(qubits->chains);
        }
        
        if (qubits->chain_positions) {
            free(qubits->chain_positions);
        }
        
        if (qubits->chain_orientations) {
            free(qubits->chain_orientations);
        }
        
        free(qubits);
    }
}

// Initialize chains for qubit encoding
void initialize_chains(IsingChainQubits *qubits, int chain_length, KitaevWireParameters *params) {
    if (!qubits || !params) return;
    
    KitaevLattice *lattice = qubits->lattice;
    int max_x = lattice->size_x - chain_length;
    int max_y = lattice->size_y - chain_length;
    int max_z = lattice->size_z - chain_length;
    
    if (max_x < 0 || max_y < 0 || max_z < 0) {
        fprintf(stderr, "Error: Chain length exceeds lattice dimensions\n");
        return;
    }
    
    for (int i = 0; i < qubits->num_chains; i++) {
        // Initialize Majorana chain
        qubits->chains[i] = initialize_majorana_chain(chain_length, params);
        
        // Choose random starting position
        qubits->chain_positions[i][0] = rand() % max_x;
        qubits->chain_positions[i][1] = rand() % max_y;
        qubits->chain_positions[i][2] = rand() % max_z;
        
        // Choose random orientation (0=x, 1=y, 2=z)
        qubits->chain_orientations[i] = rand() % 3;
        
        // Map chain to lattice
        map_chain_to_lattice(qubits->chains[i], lattice,
                            qubits->chain_positions[i][0],
                            qubits->chain_positions[i][1],
                            qubits->chain_positions[i][2],
                            qubits->chain_orientations[i]);
    }
}

// Create a topological qubit from a pair of Majorana zero modes
void create_topological_qubit(IsingChainQubits *qubits, int chain_index) {
    if (!qubits || chain_index < 0 || chain_index >= qubits->num_chains) {
        fprintf(stderr, "Error: Invalid parameters for create_topological_qubit\n");
        return;
    }
    
    // In a proper implementation, this would adjust the parameters
    // of the Kitaev wire to ensure it's in the topological phase
    // with Majorana zero modes at the ends
    
    KitaevWireParameters params;
    params.coupling_strength = 1.0;
    params.chemical_potential = 0.5;  // |μ| < 2|t| for topological phase
    params.superconducting_gap = 1.0;
    
    double zero_mode_strength = detect_majorana_zero_modes(qubits->chains[chain_index], &params);
    
    if (zero_mode_strength < 0.5) {
        fprintf(stderr, "Warning: Chain %d may not be in the topological phase (strength = %f)\n",
                chain_index, zero_mode_strength);
    } else {
        printf("Created topological qubit from chain %d (zero mode strength = %f)\n",
               chain_index, zero_mode_strength);
    }
}

// Encode a qubit state in a pair of Majorana zero modes
void encode_qubit_state(IsingChainQubits *qubits, int qubit_index, int state) {
    if (!qubits || qubit_index < 0 || qubit_index >= qubits->num_chains) {
        fprintf(stderr, "Error: Invalid parameters for encode_qubit_state\n");
        return;
    }
    
    // In a proper implementation, this would set the parity of the Majorana pair
    // |0⟩ corresponds to parity +1, |1⟩ corresponds to parity -1
    
    MajoranaChain *chain = qubits->chains[qubit_index];
    
    if (state == 0) {
        // Set parity to +1
        // For simplicity, we'll manipulate the first and second Majorana operators
        // to achieve the desired parity
        double _Complex temp = chain->operators[0];
        chain->operators[0] = chain->operators[1];
        chain->operators[1] = temp;
    } else if (state == 1) {
        // Set parity to -1
        // For simplicity, we'll manipulate the first and second Majorana operators
        // to achieve the desired parity
        double _Complex temp = chain->operators[0];
        chain->operators[0] = -chain->operators[1];
        chain->operators[1] = -temp;
    } else {
        fprintf(stderr, "Error: Invalid state %d (must be 0 or 1)\n", state);
    }
}

// Measure a topological qubit
int measure_topological_qubit(IsingChainQubits *qubits, int qubit_index) {
    if (!qubits || qubit_index < 0 || qubit_index >= qubits->num_chains) {
        fprintf(stderr, "Error: Invalid parameters for measure_topological_qubit\n");
        return -1;
    }
    
    // In a proper implementation, this would measure the parity of the Majorana pair
    // Parity +1 corresponds to |0⟩, parity -1 corresponds to |1⟩
    
    int parity = calculate_majorana_parity(qubits->chains[qubit_index]);
    
    // Convert parity to qubit state (0 or 1)
    return (parity < 0) ? 1 : 0;
}

// Apply X gate to a topological qubit
void apply_topological_x_gate(IsingChainQubits *qubits, int qubit_index) {
    if (!qubits || qubit_index < 0 || qubit_index >= qubits->num_chains) {
        fprintf(stderr, "Error: Invalid parameters for apply_topological_x_gate\n");
        return;
    }
    
    if (getenv("DEBUG_QUANTUM")) {
        printf("DEBUG: Applying X gate to topological qubit %d\n", qubit_index);
    }
    
    // X gate is implemented by applying the Majorana operator γ_1
    // This flips the parity eigenvalue
    
    MajoranaChain *chain = qubits->chains[qubit_index];
    if (!chain) {
        fprintf(stderr, "Error: Chain is NULL for qubit %d\n", qubit_index);
        return;
    }
    
    // Measure initial state for reporting
    int initial_state = measure_topological_qubit(qubits, qubit_index);
    if (getenv("DEBUG_QUANTUM")) {
        printf("DEBUG: Qubit %d initial state: %d\n", qubit_index, initial_state);
    }
    
    // Implement the topological X-gate by manipulating Majorana zero modes
    
    // In a real physical system, this would involve braiding operations
    // For our numerical implementation, we modify the chain operators directly:
    
    // Calculate the chain length for convenience
    int chain_length = chain->num_sites;
    
    // 1. Exchange left and right Majorana operators to flip parity
    double _Complex temp = chain->operators[0];
    chain->operators[0] = chain->operators[chain->num_operators - 1];
    chain->operators[chain->num_operators - 1] = temp;
    
    // 2. Apply phase change to implement correct gate action
    for (int i = 0; i < chain->num_operators; i++) {
        chain->operators[i] *= cexp(I * PI / 4.0); // π/4 phase for X gate
    }
    
    // 3. Update the chain's underlying representation in the Kitaev lattice
    int x = qubits->chain_positions[qubit_index][0];
    int y = qubits->chain_positions[qubit_index][1];
    int z = qubits->chain_positions[qubit_index][2];
    int orientation = qubits->chain_orientations[qubit_index];
    
    if (getenv("DEBUG_QUANTUM")) {
        printf("DEBUG: Chain %d position: (%d,%d,%d), orientation: %d\n", 
               qubit_index, x, y, z, orientation);
    }
    
    // Update spins in the lattice to reflect the new state
    for (int i = 0; i < chain_length; i++) {
        int pos_x = x, pos_y = y, pos_z = z;
        
        // Move along the chain according to orientation
        switch (orientation) {
            case 0: // x-direction
                pos_x += i;
                break;
            case 1: // y-direction
                pos_y += i;
                break;
            case 2: // z-direction
                pos_z += i;
                break;
        }
        
        // Ensure we stay within lattice bounds
        if (pos_x >= 0 && pos_x < qubits->lattice->size_x &&
            pos_y >= 0 && pos_y < qubits->lattice->size_y &&
            pos_z >= 0 && pos_z < qubits->lattice->size_z) {
            
            // Flip the relevant spin to implement the X gate
            qubits->lattice->spins[pos_x][pos_y][pos_z] *= -1;
        }
    }
    
    // Measure final state for reporting
    int final_state = measure_topological_qubit(qubits, qubit_index);
    if (getenv("DEBUG_QUANTUM")) {
        printf("DEBUG: Qubit %d final state after X gate: %d\n", qubit_index, final_state);
    }
    
    // Log the operation for debugging
    if (getenv("DEBUG")) {
        printf("Applied X gate to qubit %d: state before=%d, state after=%d\n",
               qubit_index, initial_state, final_state);
    }
}

// Apply Z gate to a topological qubit
void apply_topological_z_gate(IsingChainQubits *qubits, int qubit_index) {
    if (!qubits || qubit_index < 0 || qubit_index >= qubits->num_chains) {
        fprintf(stderr, "Error: Invalid parameters for apply_topological_z_gate\n");
        return;
    }
    
    // Z gate is implemented by braiding Majorana modes
    // For simplicity, we'll just apply a phase change conditionally on the qubit state
    
    int current_state = measure_topological_qubit(qubits, qubit_index);
    
    if (current_state == 1) {
        // Apply a phase change to the |1⟩ state
        // In a proper implementation, this would be done by braiding
        MajoranaChain *chain = qubits->chains[qubit_index];
        chain->operators[0] *= -1.0;
    }
}

// Apply Y gate to a topological qubit
void apply_topological_y_gate(IsingChainQubits *qubits, int qubit_index) {
    if (!qubits || qubit_index < 0 || qubit_index >= qubits->num_chains) {
        fprintf(stderr, "Error: Invalid parameters for apply_topological_y_gate\n");
        return;
    }
    
    // Y gate is equivalent to Z followed by X (up to a global phase)
    apply_topological_z_gate(qubits, qubit_index);
    apply_topological_x_gate(qubits, qubit_index);
}

// Perform a CNOT gate between two topological qubits
void apply_topological_cnot(IsingChainQubits *qubits, int control_qubit, int target_qubit) {
    if (!qubits || control_qubit < 0 || control_qubit >= qubits->num_chains ||
        target_qubit < 0 || target_qubit >= qubits->num_chains ||
        control_qubit == target_qubit) {
        fprintf(stderr, "Error: Invalid parameters for apply_topological_cnot\n");
        return;
    }
    
    // CNOT gate: apply X to target if control is in state |1⟩
    int control_state = measure_topological_qubit(qubits, control_qubit);
    
    if (control_state == 1) {
        apply_topological_x_gate(qubits, target_qubit);
    }
}

// Add interaction between chains
void add_chain_interaction(IsingChainQubits *qubits, int chain1, int chain2, double strength) {
    if (!qubits || chain1 < 0 || chain1 >= qubits->num_chains ||
        chain2 < 0 || chain2 >= qubits->num_chains || chain1 == chain2) {
        fprintf(stderr, "Error: Invalid parameters for add_chain_interaction\n");
        return;
    }
    
    // In a proper implementation, this would create an interaction term
    // between the Majorana modes of the two chains
    
    // For demonstration purposes, we'll just print the interaction
    printf("Added interaction between chains %d and %d with strength %f\n",
           chain1, chain2, strength);
}
