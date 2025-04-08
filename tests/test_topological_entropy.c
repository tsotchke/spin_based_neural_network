#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../include/topological_entropy.h"
#include "../include/kitaev_model.h"

// Test harness for topological entropy calculations
// This file provides tests to ensure numerical stability of entropy calculations

// Test if boundary length is calculated correctly
int test_boundary_length() {
    printf("Testing boundary length calculation...\n");
    
    // Create a Kitaev lattice for testing
    KitaevLattice *lattice = (KitaevLattice*)malloc(sizeof(KitaevLattice));
    if (!lattice) {
        fprintf(stderr, "Error: Memory allocation failed for lattice\n");
        return 0;
    }
    
    // Initialize lattice with basic parameters
    lattice->size_x = 10;
    lattice->size_y = 10;
    lattice->size_z = 10;
    lattice->jx = 1.0;
    lattice->jy = 1.0;
    lattice->jz = -1.0;
    
    // Create entanglement data structure
    EntanglementData *data = (EntanglementData*)malloc(sizeof(EntanglementData));
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed for entanglement data\n");
        free(lattice);
        return 0;
    }
    
    // Clear the data structure
    for (int i = 0; i < 3; i++) {
        data->subsystem_a_coords[i] = 0;
        data->subsystem_a_size[i] = 0;
        data->subsystem_b_coords[i] = 0;
        data->subsystem_b_size[i] = 0;
    }
    data->alpha = 0.0;
    data->gamma = 0.0;
    data->boundary_length = 0.0;
    
    // Define regions to force boundary
    data->subsystem_a_coords[0] = 2;
    data->subsystem_a_coords[1] = 2;
    data->subsystem_a_coords[2] = 0;
    data->subsystem_a_size[0] = 3;
    data->subsystem_a_size[1] = 3;
    data->subsystem_a_size[2] = 1;
    
    data->subsystem_b_coords[0] = 5;
    data->subsystem_b_coords[1] = 2;
    data->subsystem_b_coords[2] = 0;
    data->subsystem_b_size[0] = 3;
    data->subsystem_b_size[1] = 3;
    data->subsystem_b_size[2] = 1;
    
    // Calculate boundary length using the partition function
    partition_regions(lattice, data);
    
    // Print the result
    printf("  Boundary length: %f\n", data->boundary_length);
    
    // Check if boundary length is positive
    int success = (data->boundary_length > 0);
    
    // Print status
    if (success) {
        printf("  PASS: Boundary length is positive\n");
    } else {
        printf("  FAIL: Boundary length should be positive\n");
    }
    
    // Clean up
    free(data);
    free(lattice);
    
    return success;
}

// Test entropy calculation with a large lattice
int test_large_lattice_entropy() {
    printf("Testing entropy calculation with large lattice...\n");
    
    // Create a Kitaev lattice for testing
    KitaevLattice *lattice = (KitaevLattice*)malloc(sizeof(KitaevLattice));
    if (!lattice) {
        fprintf(stderr, "Error: Memory allocation failed for lattice\n");
        return 0;
    }
    
    // Initialize lattice with large dimensions
    lattice->size_x = 10;
    lattice->size_y = 10;
    lattice->size_z = 10;
    lattice->jx = 1.0;
    lattice->jy = 1.0;
    lattice->jz = -1.0;
    
    // Create entanglement data structure
    EntanglementData *data = (EntanglementData*)malloc(sizeof(EntanglementData));
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed for entanglement data\n");
        free(lattice);
        return 0;
    }
    
    // Clear the data structure
    for (int i = 0; i < 3; i++) {
        data->subsystem_a_coords[i] = 0;
        data->subsystem_a_size[i] = 0;
        data->subsystem_b_coords[i] = 0;
        data->subsystem_b_size[i] = 0;
    }
    data->alpha = 0.0;
    data->gamma = 0.0;
    data->boundary_length = 0.0;
    
    // Calculate topological entropy
    double entropy = calculate_topological_entropy(lattice, data);
    
    printf("  Topological entropy: %f\n", entropy);
    printf("  Boundary length after calculation: %f\n", data->boundary_length);
    
    // Check if the entropy calculation didn't produce -inf or NaN
    int success = !isnan(entropy) && !isinf(entropy);
    
    // Print status
    if (success) {
        printf("  PASS: Entropy calculation produced valid result\n");
    } else {
        printf("  FAIL: Entropy calculation produced invalid result\n");
    }
    
    // Clean up
    free(data);
    free(lattice);
    
    return success;
}

// Test if area law term is calculated correctly
int test_area_law_term() {
    printf("Testing area law term calculation...\n");
    
    // Create a Kitaev lattice for testing
    KitaevLattice *lattice = (KitaevLattice*)malloc(sizeof(KitaevLattice));
    if (!lattice) {
        fprintf(stderr, "Error: Memory allocation failed for lattice\n");
        return 0;
    }
    
    // Initialize lattice with basic parameters
    lattice->size_x = 10;
    lattice->size_y = 10;
    lattice->size_z = 10;
    lattice->jx = 1.0;
    lattice->jy = 1.0;
    lattice->jz = -1.0;
    
    // Create entanglement data structure
    EntanglementData *data = (EntanglementData*)malloc(sizeof(EntanglementData));
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed for entanglement data\n");
        free(lattice);
        return 0;
    }
    
    // Clear the data structure
    for (int i = 0; i < 3; i++) {
        data->subsystem_a_coords[i] = 0;
        data->subsystem_a_size[i] = 0;
        data->subsystem_b_coords[i] = 0;
        data->subsystem_b_size[i] = 0;
    }
    data->alpha = 0.0;
    data->gamma = 0.0;
    data->boundary_length = 0.0;
    
    // Calculate topological entropy, which will calculate the area law term internally
    calculate_topological_entropy(lattice, data);
    
    // The area law term is log(boundary_length)
    double area_law_term = log(data->boundary_length);
    
    printf("  Boundary length: %f\n", data->boundary_length);
    printf("  Area law term: %f\n", area_law_term);
    
    // Check if the area law term is a valid number (not -inf or NaN)
    int success = !isnan(area_law_term) && !isinf(area_law_term);
    
    // Print status
    if (success) {
        printf("  PASS: Area law term is valid\n");
    } else {
        printf("  FAIL: Area law term is invalid\n");
    }
    
    // Clean up
    free(data);
    free(lattice);
    
    return success;
}

// Main function to run all tests
int main() {
    printf("==== Topological Entropy Test Suite ====\n\n");
    
    // Seed random number generator
    srand(time(NULL));
    
    // Track test results
    int success_count = 0;
    int total_tests = 3;
    
    // Run tests
    printf("\nTest 1: Boundary Length Calculation\n");
    printf("------------------------------------\n");
    if (test_boundary_length()) success_count++;
    
    printf("\nTest 2: Large Lattice Entropy\n");
    printf("-----------------------------\n");
    if (test_large_lattice_entropy()) success_count++;
    
    printf("\nTest 3: Area Law Term\n");
    printf("--------------------\n");
    if (test_area_law_term()) success_count++;
    
    // Print summary
    printf("\n==== Test Summary ====\n");
    printf("Passed: %d/%d\n", success_count, total_tests);
    
    return (success_count == total_tests) ? 0 : 1;
}
