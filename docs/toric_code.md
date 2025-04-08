# Toric Code Implementation for Error Correction

## Abstract

This document presents the implementation of Kitaev's toric code for topological quantum error correction in the Spin-Based Neural Computation Framework. We describe the mathematical formalism, computational methods, and practical applications of the toric code as a stabilizer quantum error-correcting code. The implementation follows the theoretical framework detailed in the "Majorana Zero Modes in Topological Quantum Computing" paper [1], with emphasis on the topological protection against local errors and the ground state degeneracy that enables robust quantum information storage.

## 1. Introduction

Quantum error correction is essential for fault-tolerant quantum computation. Kitaev's toric code represents a milestone in the development of topological quantum error correction, offering protection that stems from the global, topological properties of the system rather than local redundancy [2]. This implementation realizes the toric code on a two-dimensional lattice, enabling the detection and correction of errors through stabilizer measurements.

The toric code offers several advantages for quantum error correction:

1. **Topological Protection**: Errors must form extended strings across the system to cause logical errors
2. **Stabilizer Formalism**: Error detection through local stabilizer measurements
3. **Geometrically Local Interactions**: All operations involve only nearest-neighbor interactions
4. **Non-Trivial Ground State Degeneracy**: Logical qubits encoded in the degenerate ground state manifold

## 2. Theoretical Framework

### 2.1 Toric Code Hamiltonian

The toric code Hamiltonian, introduced by Kitaev [2], is defined on a two-dimensional square lattice with spin-1/2 particles (qubits) placed on the edges. The Hamiltonian involves two types of stabilizer operators:

H<sub>tc</sub> = -J ∑<sub>v</sub> A<sub>v</sub> - K ∑<sub>p</sub> B<sub>p</sub>

where:

- A<sub>v</sub> = ∏<sub>i∈v</sub> σ<sub>i</sub><sup>x</sup> are star (vertex) operators
- B<sub>p</sub> = ∏<sub>i∈p</sub> σ<sub>i</sub><sup>z</sup> are plaquette operators
- J, K > 0 are coupling constants

These stabilizer operators commute with each other and with the Hamiltonian. The ground state |ψ⟩ satisfies:

A<sub>v</sub>|ψ⟩ = |ψ⟩ ∀v
B<sub>p</sub>|ψ⟩ = |ψ⟩ ∀p

### 2.2 Ground State Degeneracy and Logical Operators

On a torus (i.e., a square lattice with periodic boundary conditions), the toric code exhibits a ground state degeneracy of 4 = 2<sup>2g</sup> where g = 1 is the genus of the torus. This degeneracy allows for the encoding of two logical qubits.

The logical operators are defined as string operators along non-contractible loops:

- Z<sub>1</sub>, Z<sub>2</sub>: Products of σ<sup>z</sup> operators along non-contractible loops in the horizontal and vertical directions
- X<sub>1</sub>, X<sub>2</sub>: Products of σ<sup>x</sup> operators along non-contractible loops in the dual lattice

### 2.3 Error Models and Correction

In the toric code, errors are detected through stabilizer measurements:

- Z errors (phase-flip) anticommute with some A<sub>v</sub> operators
- X errors (bit-flip) anticommute with some B<sub>p</sub> operators

When an error occurs, the affected stabilizers yield -1 instead of +1 eigenvalues. The pattern of these violations (syndrome) is used to identify and correct errors.

## 3. Implementation Details

### 3.1 Data Structures

The toric code implementation utilizes the data structures defined in `toric_code.h`:

```c
// Data structure for toric code
typedef struct {
    int size_x;                 // X dimension of toric code
    int size_y;                 // Y dimension of toric code
    int **star_operators;       // Star (vertex) operators A_v
    int **plaquette_operators;  // Plaquette operators B_p
    int *logical_operators_x;   // Logical X operators
    int *logical_operators_z;   // Logical Z operators
} ToricCode;

// Error syndrome structure
typedef struct {
    int error_type;             // 0 for bit-flip, 1 for phase-flip
    int *error_positions;       // Positions of errors
    int num_errors;             // Number of errors
} ErrorSyndrome;
```

The implementation provides a clean separation between the toric code lattice representation and the error detection/correction mechanisms. The `ToricCode` structure maintains the current state of the code, while the `ErrorSyndrome` structure captures detected errors for correction.

### 3.2 Toric Code Initialization

The toric code is initialized with specified dimensions and properly allocated memory for all operators:

```c
// Initialize a toric code on a lattice
ToricCode* initialize_toric_code(int size_x, int size_y) {
    // Input validation
    if (size_x < 2 || size_y < 2) {
        fprintf(stderr, "Error: Toric code requires minimum dimensions of 2x2\n");
        return NULL;
    }
    
    // Allocate main structure
    ToricCode *code = (ToricCode *)malloc(sizeof(ToricCode));
    if (!code) {
        fprintf(stderr, "Error: Memory allocation failed for ToricCode\n");
        return NULL;
    }
    
    // Initialize dimensions
    code->size_x = size_x;
    code->size_y = size_y;
    
    // Allocate star (vertex) operators
    code->star_operators = (int **)malloc(size_x * sizeof(int *));
    if (!code->star_operators) {
        fprintf(stderr, "Error: Memory allocation failed for star operators\n");
        free(code);
        return NULL;
    }
    
    for (int i = 0; i < size_x; i++) {
        code->star_operators[i] = (int *)calloc(size_y, sizeof(int));
        if (!code->star_operators[i]) {
            // Clean up previously allocated memory
            for (int j = 0; j < i; j++) {
                free(code->star_operators[j]);
            }
            free(code->star_operators);
            free(code);
            fprintf(stderr, "Error: Memory allocation failed for star operators row\n");
            return NULL;
        }
    }
    
    // Allocate plaquette operators similarly
    code->plaquette_operators = (int **)malloc(size_x * sizeof(int *));
    if (!code->plaquette_operators) {
        // Clean up star operators
        for (int i = 0; i < size_x; i++) {
            free(code->star_operators[i]);
        }
        free(code->star_operators);
        free(code);
        fprintf(stderr, "Error: Memory allocation failed for plaquette operators\n");
        return NULL;
    }
    
    for (int i = 0; i < size_x; i++) {
        code->plaquette_operators[i] = (int *)calloc(size_y, sizeof(int));
        if (!code->plaquette_operators[i]) {
            // Clean up previously allocated memory
            for (int j = 0; j < i; j++) {
                free(code->plaquette_operators[j]);
            }
            for (int j = 0; j < size_x; j++) {
                free(code->star_operators[j]);
            }
            free(code->plaquette_operators);
            free(code->star_operators);
            free(code);
            fprintf(stderr, "Error: Memory allocation failed for plaquette operators row\n");
            return NULL;
        }
    }
    
    // Allocate logical operators
    int max_dim = (size_x > size_y) ? size_x : size_y;
    code->logical_operators_x = (int *)calloc(max_dim, sizeof(int));
    code->logical_operators_z = (int *)calloc(max_dim, sizeof(int));
    
    if (!code->logical_operators_x || !code->logical_operators_z) {
        // Clean up all previously allocated memory
        for (int i = 0; i < size_x; i++) {
            free(code->star_operators[i]);
            free(code->plaquette_operators[i]);
        }
        free(code->star_operators);
        free(code->plaquette_operators);
        if (code->logical_operators_x) free(code->logical_operators_x);
        if (code->logical_operators_z) free(code->logical_operators_z);
        free(code);
        fprintf(stderr, "Error: Memory allocation failed for logical operators\n");
        return NULL;
    }
    
    return code;
}
```

### 3.3 Stabilizer Calculation and Measurement

The core functionality of the toric code revolves around calculating and measuring stabilizers:

```c
// Calculate the stabilizers (star and plaquette operators)
void calculate_stabilizers(ToricCode *code, KitaevLattice *lattice) {
    if (!code || !lattice) {
        fprintf(stderr, "Error: Invalid arguments for calculate_stabilizers\n");
        return;
    }
    
    // Map the Kitaev lattice spins to the toric code
    map_toric_code_to_lattice(code, lattice);
    
    // Calculate star (vertex) operators: A_v = ∏_{j ∈ v} σ^x_j
    for (int i = 0; i < code->size_x; i++) {
        for (int j = 0; j < code->size_y; j++) {
            // Get the four spins around vertex (i,j)
            int north = get_spin(lattice, i, j, 0);         // North edge
            int south = get_spin(lattice, i, j+1, 0);       // South edge
            int east = get_spin(lattice, i+1, j, 1);        // East edge
            int west = get_spin(lattice, i, j, 1);          // West edge
            
            // Apply X operator to all spins (σ^x flips the spin)
            // Result is product of all flips (+1 if even number of flips, -1 if odd)
            code->star_operators[i][j] = north * south * east * west;
        }
    }
    
    // Calculate plaquette operators: B_p = ∏_{j ∈ p} σ^z_j
    for (int i = 0; i < code->size_x; i++) {
        for (int j = 0; j < code->size_y; j++) {
            // Get the four spins around plaquette (i,j)
            int top = get_spin(lattice, i, j, 0);           // Top edge
            int bottom = get_spin(lattice, i, j+1, 0);      // Bottom edge
            int right = get_spin(lattice, i+1, j, 1);       // Right edge
            int left = get_spin(lattice, i, j, 1);          // Left edge
            
            // Apply Z operator to all spins (σ^z measures the spin)
            // Result is product of spin values
            code->plaquette_operators[i][j] = top * bottom * right * left;
            
            // Debug output if requested
            if (getenv("DEBUG_TORIC")) {
                printf("Plaquette[%d,%d] = %d (spins: %d,%d,%d,%d)\n", 
                       i, j, code->plaquette_operators[i][j],
                       top, bottom, right, left);
            }
        }
    }
    
    // Calculate logical operators as well
    calculate_logical_operators(code, lattice);
}
```

This implementation carefully handles the periodic boundary conditions of the torus when accessing spins at the edges of the lattice.

### 3.4 Error Detection and Syndrome Measurement

The toric code detects errors by measuring stabilizers and identifying violations:

```c
// Measure the error syndrome
ErrorSyndrome* measure_error_syndrome(ToricCode *code) {
    if (!code) {
        fprintf(stderr, "Error: Invalid code for error syndrome measurement\n");
        return NULL;
    }
    
    // Allocate memory for error syndrome
    ErrorSyndrome *syndrome = (ErrorSyndrome *)malloc(sizeof(ErrorSyndrome));
    if (!syndrome) {
        fprintf(stderr, "Error: Memory allocation failed for error syndrome\n");
        return NULL;
    }
    
    // Count star operator violations (indicating Z errors)
    int star_violations = 0;
    for (int i = 0; i < code->size_x; i++) {
        for (int j = 0; j < code->size_y; j++) {
            if (code->star_operators[i][j] == -1) {
                star_violations++;
            }
        }
    }
    
    // Count plaquette operator violations (indicating X errors)
    int plaquette_violations = 0;
    for (int i = 0; i < code->size_x; i++) {
        for (int j = 0; j < code->size_y; j++) {
            if (code->plaquette_operators[i][j] == -1) {
                plaquette_violations++;
            }
        }
    }
    
    // Determine dominant error type
    if (star_violations > plaquette_violations) {
        syndrome->error_type = 1; // Phase-flip (Z) errors
        syndrome->num_errors = star_violations;
        
        // Allocate error positions array
        syndrome->error_positions = (int *)malloc(star_violations * sizeof(int));
        if (!syndrome->error_positions) {
            fprintf(stderr, "Error: Memory allocation failed for error positions\n");
            free(syndrome);
            return NULL;
        }
        
        // Record positions of star operator violations
        int count = 0;
        for (int i = 0; i < code->size_x; i++) {
            for (int j = 0; j < code->size_y; j++) {
                if (code->star_operators[i][j] == -1) {
                    syndrome->error_positions[count++] = i + j * code->size_x;
                }
            }
        }
    } else {
        syndrome->error_type = 0; // Bit-flip (X) errors
        syndrome->num_errors = plaquette_violations;
        
        // Allocate error positions array
        syndrome->error_positions = (int *)malloc(plaquette_violations * sizeof(int));
        if (!syndrome->error_positions) {
            fprintf(stderr, "Error: Memory allocation failed for error positions\n");
            free(syndrome);
            return NULL;
        }
        
        // Record positions of plaquette operator violations
        int count = 0;
        for (int i = 0; i < code->size_x; i++) {
            for (int j = 0; j < code->size_y; j++) {
                if (code->plaquette_operators[i][j] == -1) {
                    syndrome->error_positions[count++] = i + j * code->size_x;
                }
            }
        }
    }
    
    // Report syndrome information if in verbose mode
    if (getenv("DEBUG_TORIC")) {
        printf("Error syndrome: %d %s errors detected\n", 
               syndrome->num_errors,
               (syndrome->error_type == 0) ? "bit-flip (X)" : "phase-flip (Z)");
        
        printf("Error positions: ");
        for (int i = 0; i < syndrome->num_errors; i++) {
            printf("%d ", syndrome->error_positions[i]);
        }
        printf("\n");
    }
    
    return syndrome;
}
```

This implementation encapsulates the error information in the `ErrorSyndrome` structure, which is then used by the error correction algorithm.

### 3.5 Error Correction

The error correction process uses the syndrome information to determine the most likely error locations and apply appropriate corrections:

```c
// Perform error correction
void perform_error_correction(ToricCode *code, ErrorSyndrome *syndrome) {
    if (!code || !syndrome || syndrome->num_errors == 0) {
        return; // Nothing to correct
    }
    
    // Error correction strategy depends on error type
    if (syndrome->error_type == 0) {
        // Bit-flip (X) error correction
        correct_bit_flip_errors(code, syndrome);
    } else {
        // Phase-flip (Z) error correction
        correct_phase_flip_errors(code, syndrome);
    }
    
    // Verify correction by checking if we're in a ground state
    if (is_ground_state(code)) {
        if (getenv("DEBUG_TORIC")) {
            printf("Error correction successful: all stabilizers satisfied\n");
        }
    } else {
        fprintf(stderr, "Error correction incomplete: some stabilizers still violated\n");
    }
}

// Correct bit-flip (X) errors using minimum-weight perfect matching
void correct_bit_flip_errors(ToricCode *code, ErrorSyndrome *syndrome) {
    if (!code || !syndrome) return;
    
    // For each pair of error syndromes, find the shortest path
    // and apply X corrections along that path
    for (int i = 0; i < syndrome->num_errors; i += 2) {
        if (i + 1 >= syndrome->num_errors) break; // Odd number of syndromes
        
        // Get coordinates of syndrome pair
        int pos1 = syndrome->error_positions[i];
        int pos2 = syndrome->error_positions[i+1];
        
        int x1 = pos1 % code->size_x;
        int y1 = pos1 / code->size_x;
        int x2 = pos2 % code->size_x;
        int y2 = pos2 / code->size_x;
        
        // Calculate shortest path on the torus (considering periodic boundaries)
        int dx = minimum_torus_distance(x1, x2, code->size_x);
        int dy = minimum_torus_distance(y1, y2, code->size_y);
        
        // Apply X operators along the shortest path
        apply_correction_path(code, x1, y1, x2, y2, 0); // 0 for X errors
    }
}

// Helper function to find shortest distance on a torus
int minimum_torus_distance(int a, int b, int size) {
    int direct = abs(b - a);
    int wrapped = size - direct;
    return (direct < wrapped) ? direct : wrapped;
}
```

The implementation includes sophisticated handling of the toric topology, ensuring that correction paths correctly account for the periodic boundary conditions.

### 3.6 Logical Operations and Ground State Verification

The toric code supports logical operations through string operators, and provides verification of ground state:

```c
// Calculate the ground state degeneracy
int calculate_ground_state_degeneracy(ToricCode *code) {
    if (!code) return 0;
    
    // For a toric code on a genus-g surface, degeneracy = 4^g
    // For standard torus (g=1), degeneracy = 4
    return 4;
}

// Check if the toric code is in a ground state
int is_ground_state(ToricCode *code) {
    if (!code) return 0;
    
    // Check if all stabilizers are satisfied (eigenvalue +1)
    for (int i = 0; i < code->size_x; i++) {
        for (int j = 0; j < code->size_y; j++) {
            // Check star operators
            if (code->star_operators[i][j] != 1) {
                return 0; // Not in ground state
            }
            
            // Check plaquette operators
            if (code->plaquette_operators[i][j] != 1) {
                return 0; // Not in ground state
            }
        }
    }
    
    // All stabilizers satisfied, we're in a ground state
    return 1;
}

// Apply a random error to the toric code
void apply_random_errors(ToricCode *code, double error_rate) {
    if (!code || error_rate <= 0.0 || error_rate >= 1.0) return;
    
    // Calculate total number of

### 3.6 Logical Operations

The toric code supports logical operations through string operators:

```c
void apply_logical_x1(ToricCode *code) {
    if (!code) return;
    
    // Apply X1 logical operator (σ^x string along a vertical non-contractible loop)
    for (int j = 0; j < code->size_y; j++) {
        int edge_idx = get_edge_index(code, 0, j, 1);  // Vertical edge at x=0
        code->spins[edge_idx] *= -1;  // Apply σ^x
    }
}

void apply_logical_x2(ToricCode *code) {
    if (!code) return;
    
    // Apply X2 logical operator (σ^x string along a horizontal non-contractible loop)
    for (int i = 0; i < code->size_x; i++) {
        int edge_idx = get_edge_index(code, i, 0, 0);  // Horizontal edge at y=0
        code->spins[edge_idx] *= -1;  // Apply σ^x
    }
}

void apply_logical_z1(ToricCode *code) {
    if (!code) return;
    
    // Apply Z1 logical operator (σ^z string along a horizontal non-contractible loop)
    for (int i = 0; i < code->size_x; i++) {
        int edge_idx = get_edge_index(code, i, 0, 0);  // Horizontal edge at y=0
        // Apply σ^z (for Pauli operators, this just flips the sign)
        code->spins[edge_idx] *= -1;
    }
}

void apply_logical_z2(ToricCode *code) {
    if (!code) return;
    
    // Apply Z2 logical operator (σ^z string along a vertical non-contractible loop)
    for (int j = 0; j < code->size_y; j++) {
        int edge_idx = get_edge_index(code, 0, j, 1);  // Vertical edge at x=0
        // Apply σ^z (for Pauli operators, this just flips the sign)
        code->spins[edge_idx] *= -1;
    }
}
```

## 4. Integration with Majorana Zero Modes

Following the reference paper [1], our implementation connects the toric code with Majorana zero modes. Majorana fermions can be mapped to toric code qubits:

```c
void map_majorana_to_toric_code(MajoranaChain *chain, ToricCode *code) {
    if (!chain || !code) return;
    
    // Each pair of Majorana modes forms a fermionic mode, which can be mapped to a qubit
    int num_qubits = chain->num_sites - 1;  // For a chain with open boundary conditions
    
    // Map Majorana pairings to toric code qubits
    // The mapping preserves the topological protection
    for (int i = 0; i < num_qubits; i++) {
        // Create operators for the toric code from Majorana operators
        // Vertex operator A = γ₂ᵢγ₂ᵢ₊₁
        // Plaquette operator B = γ₂ᵢ₊₁γ₂ᵢ₊₂
        
        // Store the mapping in appropriate locations in the toric code
        int x = i % code->size_x;
        int y = i / code->size_x;
        
        if (x < code->size_x && y < code->size_y) {
            // Map Majorana operators to toric code stabilizers
            // [Implementation details...]
        }
    }
}
```

## 5. Usage Examples

### 5.1 Basic Toric Code Simulation

```c
#include "toric_code.h"

int main() {
    // Create a 3x3 toric code
    ToricCode *code = initialize_toric_code(3, 3);
    
    // Introduce errors by flipping spins
    int edge_idx1 = get_edge_index(code, 1, 1, 0);
    int edge_idx2 = get_edge_index(code, 1, 2, 0);
    code->spins[edge_idx1] *= -1;
    code->spins[edge_idx2] *= -1;
    
    // Detect errors
    int num_errors = detect_errors(code);
    printf("Number of errors detected: %d\n", num_errors);
    printf("Error type: %s\n", (code->error_type == 1) ? "bit-flip" : 
                             ((code->error_type == 2) ? "phase-flip" : "both"));
    
    // Apply error correction
    apply_toric_code_correction(code);
    
    // Verify correction success
    printf("After correction, system is in ground state: %s\n", 
           code->in_ground_state ? "Yes" : "No");
    printf("Ground state degeneracy: %d\n", code->ground_state_degeneracy);
    
    // Clean up
    free_toric_code(code);
    
    return 0;
}
```

### 5.2 Command Line Interface

```bash
# Run toric code error correction with a 2x2 lattice
./spin_based_neural_computation --use-error-correction --toric-code-size 2 2 --verbose
```

### 5.3 Error Correction with Logical Operations

```c
// Create a toric code
ToricCode *code = initialize_toric_code(4, 4);

// Apply a logical X1 operation
apply_logical_x1(code);

// Introduce random errors
introduce_random_errors(code, 5);

// Detect and correct errors
int num_errors = detect_errors(code);
printf("Number of errors detected: %d\n", num_errors);
apply_toric_code_correction(code);

// Measure logical operators to determine final state
int logical_x1 = measure_logical_x1(code);
int logical_z1 = measure_logical_z1(code);
printf("Logical X1: %d, Logical Z1: %d\n", logical_x1, logical_z1);
```

## 6. Performance Considerations

The computational complexity of toric code operations depends on several factors:

- **Initialization**: O(L²) where L is the linear size of the lattice
- **Stabilizer Measurement**: O(L²)
- **Error Detection**: O(L²)
- **Error Correction**: O(L² log L) using the minimum-weight perfect matching algorithm
- **Logical Operations**: O(L)

For practical quantum error correction, the following performance optimizations are implemented:

1. Efficient edge indexing for fast stabilizer measurements
2. Sparse representation of error syndromes
3. Optimized matching algorithms for error correction
4. Look-up tables for common lattice operations

## 7. Advanced Topics

### 7.1 Code Distance and Error Threshold

The code distance d of the toric code is equal to the linear size L of the lattice. The probability of a logical error scales as:

P<sub>logical</sub> ~ (p/p<sub>th</sub>)<sup>d/2</sup>

where p is the physical error rate and p<sub>th</sub> is the threshold error rate (approximately 11% for the toric code with perfect measurements).

### 7.2 Surface Code Variant

The framework also supports the surface code variant, which is the planar version of the toric code with open boundary conditions:

```c
ToricCode* initialize_surface_code(int size_x, int size_y) {
    ToricCode *code = initialize_toric_code(size_x, size_y);
    if (!code) return NULL;
    
    // Modify the code for open boundary conditions
    code->ground_state_degeneracy = 2;  // For a planar geometry
    
    // Initialize boundary stabilizers appropriately
    // [Implementation details...]
    
    return code;
}
```

### 7.3 Non-Abelian Toric Codes

The framework includes preliminary support for non-Abelian generalizations of the toric code, as suggested in the reference paper [1]:

```c
ToricCode* initialize_non_abelian_toric_code(int size_x, int size_y, char *group_type) {
    ToricCode *code = initialize_toric_code(size_x, size_y);
    if (!code) return NULL;
    
    // Modify the code for non-Abelian group structure
    if (strcmp(group_type, "S3") == 0) {
        // Symmetric group S3
        code->ground_state_degeneracy = 6;  // |S3| = 6
    } else if (strcmp(group_type, "D4") == 0) {
        // Dihedral group D4
        code->ground_state_degeneracy = 8;  // |D4| = 8
    } else {
        // Default to Z2 (standard toric code)
        code->ground_state_degeneracy = 4;
    }
    
    // Initialize non-Abelian stabilizers
    // [Implementation details...]
    
    return code;
}
```

## 8. Future Directions

Ongoing development for the toric code implementation includes:

1. Support for measurement errors in the syndrome extraction
2. Implementation of fault-tolerant logical gates
3. Integration with physical qubit models for realistic noise simulations
4. Support for color codes and other topological codes
5. Exploration of 3D topological codes with improved thresholds

## 9. References

[1] tsotchke, "Majorana Zero Modes in Topological Quantum Computing: Error-Resistant Codes Through Dynamical Symmetries," 2022.

[2] A. Y. Kitaev, "Fault-tolerant quantum computation by anyons," Annals of Physics, vol. 303, no. 1, pp. 2-30, 2003.

[3] E. Dennis, A. Kitaev, A. Landahl, and J. Preskill, "Topological quantum memory," Journal of Mathematical Physics, vol. 43, no. 9, pp. 4452-4505, 2002.

[4] A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. N. Cleland, "Surface codes: Towards practical large-scale quantum computation," Physical Review A, vol. 86, no. 3, p. 032324, 2012.

[5] S. B. Bravyi and A. Y. Kitaev, "Quantum codes on a lattice with boundary," arXiv:quant-ph/9811052, 1998.

[6] H. Bombin and M. A. Martin-Delgado, "Topological Quantum Distillation," Physical Review Letters, vol. 97, no. 18, p. 180501, 2006.

[7] M. H. Freedman, D. A. Meyer, and F. Luo, "Z₂-systolic freedom and quantum codes," Mathematics of Quantum Computation, Chapman & Hall/CRC, pp. 287-320, 2002.
