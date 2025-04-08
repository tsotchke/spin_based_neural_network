#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "toric_code.h"

// Initialize a toric code on a lattice
ToricCode* initialize_toric_code(int size_x, int size_y) {
    if (size_x <= 0 || size_y <= 0) {
        fprintf(stderr, "Error: Toric code dimensions must be positive\n");
        return NULL;
    }
    
    ToricCode *code = (ToricCode *)malloc(sizeof(ToricCode));
    if (!code) {
        fprintf(stderr, "Error: Memory allocation failed for ToricCode\n");
        return NULL;
    }
    
    code->size_x = size_x;
    code->size_y = size_y;
    
    // Allocate memory for star (vertex) operators
    // Each vertex has 4 adjacent links in a square lattice
    code->star_operators = (int **)malloc(size_x * size_y * sizeof(int *));
    if (!code->star_operators) {
        fprintf(stderr, "Error: Memory allocation failed for star operators\n");
        free(code);
        return NULL;
    }
    
    for (int i = 0; i < size_x * size_y; i++) {
        code->star_operators[i] = (int *)malloc(4 * sizeof(int));
        if (!code->star_operators[i]) {
            fprintf(stderr, "Error: Memory allocation failed for star operator %d\n", i);
            for (int j = 0; j < i; j++) {
                free(code->star_operators[j]);
            }
            free(code->star_operators);
            free(code);
            return NULL;
        }
        
        // Initialize star operators to 1 (eigenvalue +1)
        for (int j = 0; j < 4; j++) {
            code->star_operators[i][j] = 1;
        }
    }
    
    // Allocate memory for plaquette operators
    // Each plaquette has 4 adjacent links in a square lattice
    code->plaquette_operators = (int **)malloc(size_x * size_y * sizeof(int *));
    if (!code->plaquette_operators) {
        fprintf(stderr, "Error: Memory allocation failed for plaquette operators\n");
        for (int i = 0; i < size_x * size_y; i++) {
            free(code->star_operators[i]);
        }
        free(code->star_operators);
        free(code);
        return NULL;
    }
    
    for (int i = 0; i < size_x * size_y; i++) {
        code->plaquette_operators[i] = (int *)malloc(4 * sizeof(int));
        if (!code->plaquette_operators[i]) {
            fprintf(stderr, "Error: Memory allocation failed for plaquette operator %d\n", i);
            for (int j = 0; j < i; j++) {
                free(code->plaquette_operators[j]);
            }
            for (int j = 0; j < size_x * size_y; j++) {
                free(code->star_operators[j]);
            }
            free(code->plaquette_operators);
            free(code->star_operators);
            free(code);
            return NULL;
        }
        
        // Initialize plaquette operators to 1 (eigenvalue +1)
        for (int j = 0; j < 4; j++) {
            code->plaquette_operators[i][j] = 1;
        }
    }
    
    // Allocate memory for logical operators
    // For a torus, there are 2 logical X operators and 2 logical Z operators
    code->logical_operators_x = (int *)malloc(2 * sizeof(int));
    code->logical_operators_z = (int *)malloc(2 * sizeof(int));
    
    if (!code->logical_operators_x || !code->logical_operators_z) {
        fprintf(stderr, "Error: Memory allocation failed for logical operators\n");
        for (int i = 0; i < size_x * size_y; i++) {
            free(code->plaquette_operators[i]);
            free(code->star_operators[i]);
        }
        free(code->plaquette_operators);
        free(code->star_operators);
        if (code->logical_operators_x) free(code->logical_operators_x);
        if (code->logical_operators_z) free(code->logical_operators_z);
        free(code);
        return NULL;
    }
    
    // Initialize logical operators to +1 eigenvalue
    code->logical_operators_x[0] = 1;
    code->logical_operators_x[1] = 1;
    code->logical_operators_z[0] = 1;
    code->logical_operators_z[1] = 1;
    
    return code;
}

// Free memory allocated for toric code
void free_toric_code(ToricCode *code) {
    if (code) {
        if (code->star_operators) {
            for (int i = 0; i < code->size_x * code->size_y; i++) {
                if (code->star_operators[i]) {
                    free(code->star_operators[i]);
                }
            }
            free(code->star_operators);
        }
        
        if (code->plaquette_operators) {
            for (int i = 0; i < code->size_x * code->size_y; i++) {
                if (code->plaquette_operators[i]) {
                    free(code->plaquette_operators[i]);
                }
            }
            free(code->plaquette_operators);
        }
        
        if (code->logical_operators_x) {
            free(code->logical_operators_x);
        }
        
        if (code->logical_operators_z) {
            free(code->logical_operators_z);
        }
        
        free(code);
    }
}

// Calculate the stabilizers (star and plaquette operators)
void calculate_stabilizers(ToricCode *code, KitaevLattice *lattice) {
    if (!code || !lattice) {
        fprintf(stderr, "Error: Invalid parameters for calculate_stabilizers\n");
        return;
    }
    
    // Check if the lattice is compatible with the toric code
    if (lattice->size_x < code->size_x || lattice->size_y < code->size_y) {
        fprintf(stderr, "Error: Lattice size is smaller than toric code size\n");
        return;
    }
    
    // Calculate star operators (A_v)
    // Star operator = Πσ_x for all links touching vertex v
    for (int i = 0; i < code->size_x; i++) {
        for (int j = 0; j < code->size_y; j++) {
            int vertex_index = i * code->size_y + j;
            int product = 1;
            
            // North link
            int x_north = i;
            int y_north = (j - 1 + code->size_y) % code->size_y;
            product *= lattice->spins[x_north][y_north][0];
            
            // East link
            int x_east = i;
            int y_east = j;
            product *= lattice->spins[x_east][y_east][0];
            
            // South link
            int x_south = i;
            int y_south = j;
            product *= lattice->spins[x_south][y_south][0];
            
            // West link
            int x_west = (i - 1 + code->size_x) % code->size_x;
            int y_west = j;
            product *= lattice->spins[x_west][y_west][0];
            
            // Set the star operator value
            code->star_operators[vertex_index][0] = product;
        }
    }
    
    // Calculate plaquette operators (B_p)
    // Plaquette operator = Πσ_z for all links around plaquette p
    for (int i = 0; i < code->size_x; i++) {
        for (int j = 0; j < code->size_y; j++) {
            int plaquette_index = i * code->size_y + j;
            int product = 1;
            
            // North link
            int x_north = i;
            int y_north = j;
            product *= lattice->spins[x_north][y_north][0];
            
            // East link
            int x_east = (i + 1) % code->size_x;
            int y_east = j;
            product *= lattice->spins[x_east][y_east][0];
            
            // South link
            int x_south = i;
            int y_south = (j + 1) % code->size_y;
            product *= lattice->spins[x_south][y_south][0];
            
            // West link
            int x_west = i;
            int y_west = j;
            product *= lattice->spins[x_west][y_west][0];
            
            // Set the plaquette operator value
            code->plaquette_operators[plaquette_index][0] = product;
        }
    }
    
    // Calculate logical operators
    // Logical X operators are products of σ_x along non-contractible loops
    // Logical Z operators are products of σ_z along non-contractible loops
    
    // First logical X operator (along y direction)
    int x1_product = 1;
    for (int j = 0; j < code->size_y; j++) {
        x1_product *= lattice->spins[0][j][0];
    }
    code->logical_operators_x[0] = x1_product;
    
    // Second logical X operator (along x direction)
    int x2_product = 1;
    for (int i = 0; i < code->size_x; i++) {
        x2_product *= lattice->spins[i][0][0];
    }
    code->logical_operators_x[1] = x2_product;
    
    // First logical Z operator (along y direction)
    int z1_product = 1;
    for (int j = 0; j < code->size_y; j++) {
        z1_product *= lattice->spins[0][j][0];
    }
    code->logical_operators_z[0] = z1_product;
    
    // Second logical Z operator (along x direction)
    int z2_product = 1;
    for (int i = 0; i < code->size_x; i++) {
        z2_product *= lattice->spins[i][0][0];
    }
    code->logical_operators_z[1] = z2_product;
}

// Apply a random error to the toric code
void apply_random_errors(ToricCode *code, double error_rate) {
    if (!code || error_rate < 0.0 || error_rate > 1.0) {
        fprintf(stderr, "Error: Invalid parameters for apply_random_errors\n");
        return;
    }
    
    // Apply random bit-flip errors to the stabilizers
    for (int i = 0; i < code->size_x * code->size_y; i++) {
        for (int j = 0; j < 4; j++) {
            if ((double)rand() / RAND_MAX < error_rate) {
                // Flip the spin by multiplying by -1
                code->star_operators[i][j] *= -1;
            }
        }
    }
    
    // Apply random phase-flip errors to the stabilizers
    for (int i = 0; i < code->size_x * code->size_y; i++) {
        for (int j = 0; j < 4; j++) {
            if ((double)rand() / RAND_MAX < error_rate) {
                // Flip the spin by multiplying by -1
                code->plaquette_operators[i][j] *= -1;
            }
        }
    }
}

// Measure the error syndrome
ErrorSyndrome* measure_error_syndrome(ToricCode *code) {
    if (!code) {
        fprintf(stderr, "Error: Invalid parameter for measure_error_syndrome\n");
        return NULL;
    }
    
    ErrorSyndrome *syndrome = (ErrorSyndrome *)malloc(sizeof(ErrorSyndrome));
    if (!syndrome) {
        fprintf(stderr, "Error: Memory allocation failed for ErrorSyndrome\n");
        return NULL;
    }
    
    // Count errors
    int num_errors = 0;
    
    // Check star operators
    for (int i = 0; i < code->size_x * code->size_y; i++) {
        int product = 1;
        for (int j = 0; j < 4; j++) {
            product *= code->star_operators[i][j];
        }
        
        if (product < 0) {
            num_errors++;
        }
    }
    
    // Check plaquette operators
    for (int i = 0; i < code->size_x * code->size_y; i++) {
        int product = 1;
        for (int j = 0; j < 4; j++) {
            product *= code->plaquette_operators[i][j];
        }
        
        if (product < 0) {
            num_errors++;
        }
    }
    
    syndrome->num_errors = num_errors;
    syndrome->error_type = 0;  // Default to bit-flip
    
    // Allocate memory for error positions
    syndrome->error_positions = (int *)malloc(num_errors * sizeof(int));
    if (!syndrome->error_positions && num_errors > 0) {
        fprintf(stderr, "Error: Memory allocation failed for error_positions\n");
        free(syndrome);
        return NULL;
    }
    
    // Record error positions
    int error_index = 0;
    
    // Record star operator errors
    for (int i = 0; i < code->size_x * code->size_y; i++) {
        int product = 1;
        for (int j = 0; j < 4; j++) {
            product *= code->star_operators[i][j];
        }
        
        if (product < 0 && error_index < num_errors) {
            syndrome->error_positions[error_index++] = i;
            syndrome->error_type = 0;  // Bit-flip error
        }
    }
    
    // Record plaquette operator errors
    for (int i = 0; i < code->size_x * code->size_y; i++) {
        int product = 1;
        for (int j = 0; j < 4; j++) {
            product *= code->plaquette_operators[i][j];
        }
        
        if (product < 0 && error_index < num_errors) {
            syndrome->error_positions[error_index++] = i + code->size_x * code->size_y;
            syndrome->error_type = 1;  // Phase-flip error
        }
    }
    
    return syndrome;
}

// Free memory allocated for error syndrome
void free_error_syndrome(ErrorSyndrome *syndrome) {
    if (syndrome) {
        if (syndrome->error_positions) {
            free(syndrome->error_positions);
        }
        free(syndrome);
    }
}

// Perform error correction
void perform_error_correction(ToricCode *code, ErrorSyndrome *syndrome) {
    if (!code || !syndrome) {
        fprintf(stderr, "Error: Invalid parameters for perform_error_correction\n");
        return;
    }
    
    // In a proper implementation, this would use minimum weight perfect matching
    // to correct errors. For simplicity, we'll just flip the affected stabilizers.
    
    if (syndrome->error_type == 0) {  // Bit-flip errors
        for (int i = 0; i < syndrome->num_errors; i++) {
            int position = syndrome->error_positions[i];
            if (position < code->size_x * code->size_y) {
                // Flip the affected star operator
                for (int j = 0; j < 4; j++) {
                    code->star_operators[position][j] *= -1;
                }
            }
        }
    } else {  // Phase-flip errors
        for (int i = 0; i < syndrome->num_errors; i++) {
            int position = syndrome->error_positions[i] - code->size_x * code->size_y;
            if (position >= 0 && position < code->size_x * code->size_y) {
                // Flip the affected plaquette operator
                for (int j = 0; j < 4; j++) {
                    code->plaquette_operators[position][j] *= -1;
                }
            }
        }
    }
}

// Calculate the ground state degeneracy
int calculate_ground_state_degeneracy(ToricCode *code) {
    if (!code) {
        fprintf(stderr, "Error: Invalid parameter for calculate_ground_state_degeneracy\n");
        return 0;
    }
    
    // For a toric code on a torus, the ground state degeneracy is 4
    // This is because there are 2 logical X operators and 2 logical Z operators
    // which commute with the Hamiltonian but anti-commute with each other
    
    return 4;
}

// Map toric code onto a Kitaev lattice
void map_toric_code_to_lattice(ToricCode *code, KitaevLattice *lattice) {
    if (!code || !lattice) {
        fprintf(stderr, "Error: Invalid parameters for map_toric_code_to_lattice\n");
        return;
    }
    
    // Check if the lattice is compatible with the toric code
    if (lattice->size_x < code->size_x || lattice->size_y < code->size_y) {
        fprintf(stderr, "Error: Lattice size is smaller than toric code size\n");
        return;
    }
    
    // Map star operators to the lattice
    for (int i = 0; i < code->size_x; i++) {
        for (int j = 0; j < code->size_y; j++) {
            int vertex_index = i * code->size_y + j;
            
            // North link
            int x_north = i;
            int y_north = (j - 1 + code->size_y) % code->size_y;
            lattice->spins[x_north][y_north][0] = code->star_operators[vertex_index][0];
            
            // East link
            int x_east = i;
            int y_east = j;
            lattice->spins[x_east][y_east][0] = code->star_operators[vertex_index][1];
            
            // South link
            int x_south = i;
            int y_south = j;
            lattice->spins[x_south][y_south][0] = code->star_operators[vertex_index][2];
            
            // West link
            int x_west = (i - 1 + code->size_x) % code->size_x;
            int y_west = j;
            lattice->spins[x_west][y_west][0] = code->star_operators[vertex_index][3];
        }
    }
}

// Check if the toric code is in a ground state
int is_ground_state(ToricCode *code) {
    if (!code) {
        fprintf(stderr, "Error: Invalid parameter for is_ground_state\n");
        return 0;
    }
    
    // Check all stabilizers
    for (int i = 0; i < code->size_x * code->size_y; i++) {
        // Check star operators
        int star_product = 1;
        for (int j = 0; j < 4; j++) {
            star_product *= code->star_operators[i][j];
        }
        
        if (star_product < 0) {
            return 0;  // Not in ground state
        }
        
        // Check plaquette operators
        int plaquette_product = 1;
        for (int j = 0; j < 4; j++) {
            plaquette_product *= code->plaquette_operators[i][j];
        }
        
        if (plaquette_product < 0) {
            return 0;  // Not in ground state
        }
    }
    
    return 1;  // In ground state
}
