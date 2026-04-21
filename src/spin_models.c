#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "spin_models.h"

// Function to create a new spin lattice with given dimensions.
// Returns NULL on invalid dimensions or allocation failure.
SpinLattice* initialize_spin_lattice(int size_x, int size_y, int size_z, const char* initial_state) {
    if (size_x <= 0 || size_y <= 0 || size_z <= 0) {
        fprintf(stderr, "Error: Spin lattice dimensions must be positive (got %d x %d x %d)\n",
                size_x, size_y, size_z);
        return NULL;
    }

    SpinLattice *lattice = malloc(sizeof(SpinLattice));
    if (!lattice) {
        fprintf(stderr, "Error: Memory allocation failed for SpinLattice\n");
        return NULL;
    }
    lattice->size_x = size_x;
    lattice->size_y = size_y;
    lattice->size_z = size_z;

    lattice->spins = malloc((size_t)size_x * sizeof(Spin**));
    if (!lattice->spins) {
        fprintf(stderr, "Error: Memory allocation failed for SpinLattice plane array\n");
        free(lattice);
        return NULL;
    }
    for (int i = 0; i < size_x; i++) {
        lattice->spins[i] = malloc((size_t)size_y * sizeof(Spin*));
        if (!lattice->spins[i]) {
            fprintf(stderr, "Error: Memory allocation failed for SpinLattice row at i=%d\n", i);
            for (int a = 0; a < i; a++) {
                for (int b = 0; b < size_y; b++) free(lattice->spins[a][b]);
                free(lattice->spins[a]);
            }
            free(lattice->spins);
            free(lattice);
            return NULL;
        }
        for (int j = 0; j < size_y; j++) {
            lattice->spins[i][j] = malloc((size_t)size_z * sizeof(Spin));
            if (!lattice->spins[i][j]) {
                fprintf(stderr, "Error: Memory allocation failed for SpinLattice column at (%d,%d)\n", i, j);
                for (int b = 0; b < j; b++) free(lattice->spins[i][b]);
                free(lattice->spins[i]);
                for (int a = 0; a < i; a++) {
                    for (int b = 0; b < size_y; b++) free(lattice->spins[a][b]);
                    free(lattice->spins[a]);
                }
                free(lattice->spins);
                free(lattice);
                return NULL;
            }
            for (int k = 0; k < size_z; k++) {
                if (strcmp(initial_state, "all-up") == 0) {
                    lattice->spins[i][j][k].sx = lattice->spins[i][j][k].sy = lattice->spins[i][j][k].sz = 0.5;
                } else if (strcmp(initial_state, "all-down") == 0) {
                    lattice->spins[i][j][k].sx = lattice->spins[i][j][k].sy = lattice->spins[i][j][k].sz = -0.5;
                } else {
                    // Default to random for "random" or any invalid state
                    lattice->spins[i][j][k].sx = (rand() % 2 == 0 ? 0.5 : -0.5);
                    lattice->spins[i][j][k].sy = (rand() % 2 == 0 ? 0.5 : -0.5);
                    lattice->spins[i][j][k].sz = (rand() % 2 == 0 ? 0.5 : -0.5);
                }
            }
        }
    }
    return lattice;
}

// Function to compute the energy of the spin lattice
double compute_spin_energy(SpinLattice *lattice) {
    double energy = 0.0;
    for (int i = 0; i < lattice->size_x; i++) {
        for (int j = 0; j < lattice->size_y; j++) {
            for (int k = 0; k < lattice->size_z; k++) {
                energy += lattice->spins[i][j][k].sx * lattice->spins[i][j][k].sy; 
            }
        }
    }
    return energy;
}

// Function to free the allocated memory for the spin lattice
void free_spin_lattice(SpinLattice *lattice) {
    for (int i = 0; i < lattice->size_x; i++) {
        for (int j = 0; j < lattice->size_y; j++) {
            free(lattice->spins[i][j]);
        }
        free(lattice->spins[i]);
    }
    free(lattice->spins);
    free(lattice);
}
