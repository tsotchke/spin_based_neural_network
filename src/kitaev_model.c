#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kitaev_model.h"

// Function to initialize the 3D Kitaev lattice.
// Returns NULL on invalid dimensions or allocation failure.
KitaevLattice* initialize_kitaev_lattice(int size_x, int size_y, int size_z, double jx, double jy, double jz, const char* initial_state) {
    if (size_x <= 0 || size_y <= 0 || size_z <= 0) {
        fprintf(stderr, "Error: Kitaev lattice dimensions must be positive (got %d x %d x %d)\n",
                size_x, size_y, size_z);
        return NULL;
    }

    KitaevLattice *lattice = malloc(sizeof(KitaevLattice));
    if (!lattice) {
        fprintf(stderr, "Error: Memory allocation failed for KitaevLattice\n");
        return NULL;
    }
    lattice->size_x = size_x;
    lattice->size_y = size_y;
    lattice->size_z = size_z;
    lattice->jx = jx;
    lattice->jy = jy;
    lattice->jz = jz;

    lattice->spins = malloc((size_t)size_x * sizeof(int**));
    if (!lattice->spins) {
        fprintf(stderr, "Error: Memory allocation failed for Kitaev spin plane array\n");
        free(lattice);
        return NULL;
    }
    for (int i = 0; i < size_x; i++) {
        lattice->spins[i] = malloc((size_t)size_y * sizeof(int*));
        if (!lattice->spins[i]) {
            fprintf(stderr, "Error: Memory allocation failed for Kitaev row at i=%d\n", i);
            for (int a = 0; a < i; a++) {
                for (int b = 0; b < size_y; b++) free(lattice->spins[a][b]);
                free(lattice->spins[a]);
            }
            free(lattice->spins);
            free(lattice);
            return NULL;
        }
        for (int j = 0; j < size_y; j++) {
            lattice->spins[i][j] = malloc((size_t)size_z * sizeof(int));
            if (!lattice->spins[i][j]) {
                fprintf(stderr, "Error: Memory allocation failed for Kitaev column at (%d,%d)\n", i, j);
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
                if (strcmp(initial_state, "random") == 0) {
                    lattice->spins[i][j][k] = (rand() % 2) * 2 - 1; // Random +1 or -1
                } else if (strcmp(initial_state, "all-up") == 0) {
                    lattice->spins[i][j][k] = 1;
                } else if (strcmp(initial_state, "all-down") == 0) {
                    lattice->spins[i][j][k] = -1;
                } else {
                    // Default to random if invalid state is provided
                    lattice->spins[i][j][k] = (rand() % 2) * 2 - 1;
                }
            }
        }
    }

    return lattice;
}

// Function to compute the energy of the 3D Kitaev lattice
double compute_kitaev_energy(KitaevLattice *kitaev_lattice) {
    double energy = 0.0;

    for (int i = 0; i < kitaev_lattice->size_x; ++i) {
        for (int j = 0; j < kitaev_lattice->size_y; ++j) {
            for (int k = 0; k < kitaev_lattice->size_z; ++k) {
                int spin = kitaev_lattice->spins[i][j][k];
                // Interactions with neighbors in x, y, and z directions
                if (i < kitaev_lattice->size_x - 1) {
                    energy += kitaev_lattice->jx * spin * kitaev_lattice->spins[i + 1][j][k];
                }
                if (j < kitaev_lattice->size_y - 1) {
                    energy += kitaev_lattice->jy * spin * kitaev_lattice->spins[i][j + 1][k];
                }
                if (k < kitaev_lattice->size_z - 1) {
                    energy += kitaev_lattice->jz * spin * kitaev_lattice->spins[i][j][k + 1];
                }
            }
        }
    }

    return energy;
}

// Function to compute the interaction energy at a specific site (x, y, z)
double compute_kitaev_interaction(KitaevLattice *kitaev_lattice, int x, int y, int z) {
    double interaction_energy = 0.0;

    // Get the spin at the current site
    int spin = kitaev_lattice->spins[x][y][z];

    // Interaction in the x-direction
    if (x < kitaev_lattice->size_x - 1) {
        interaction_energy += kitaev_lattice->jx * spin * kitaev_lattice->spins[x + 1][y][z];
    }
    if (x > 0) {
        interaction_energy += kitaev_lattice->jx * spin * kitaev_lattice->spins[x - 1][y][z];
    }

    // Interaction in the y-direction
    if (y < kitaev_lattice->size_y - 1) {
        interaction_energy += kitaev_lattice->jy * spin * kitaev_lattice->spins[x][y + 1][z];
    }
    if (y > 0) {
        interaction_energy += kitaev_lattice->jy * spin * kitaev_lattice->spins[x][y - 1][z];
    }

    // Interaction in the z-direction
    if (z < kitaev_lattice->size_z - 1) {
        interaction_energy += kitaev_lattice->jz * spin * kitaev_lattice->spins[x][y][z + 1];
    }
    if (z > 0) {
        interaction_energy += kitaev_lattice->jz * spin * kitaev_lattice->spins[x][y][z - 1];
    }

    return interaction_energy;
}

// Function to flip a random spin in the 3D Kitaev lattice
void flip_random_spin_kitaev(KitaevLattice *lattice) {
    int x = rand() % lattice->size_x;
    int y = rand() % lattice->size_y;
    int z = rand() % lattice->size_z;
    lattice->spins[x][y][z] *= -1;  // Flip the spin
}

// Function to free the allocated memory for the 3D Kitaev lattice
void free_kitaev_lattice(KitaevLattice *lattice) {
    for (int i = 0; i < lattice->size_x; i++) {
        for (int j = 0; j < lattice->size_y; j++) {
            free(lattice->spins[i][j]);
        }
        free(lattice->spins[i]);
    }
    free(lattice->spins);
    free(lattice);
}
