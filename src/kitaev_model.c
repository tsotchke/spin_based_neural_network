#include <stdlib.h>
#include <string.h>
#include "kitaev_model.h"

// Function to initialize the 3D Kitaev lattice
KitaevLattice* initialize_kitaev_lattice(int size_x, int size_y, int size_z, double jx, double jy, double jz, const char* initial_state) {
    KitaevLattice *lattice = malloc(sizeof(KitaevLattice));
    lattice->size_x = size_x;
    lattice->size_y = size_y;
    lattice->size_z = size_z;
    lattice->jx = jx;
    lattice->jy = jy;
    lattice->jz = jz;

    lattice->spins = malloc(size_x * sizeof(int**));
    for (int i = 0; i < size_x; i++) {
        lattice->spins[i] = malloc(size_y * sizeof(int*));
        for (int j = 0; j < size_y; j++) {
            lattice->spins[i][j] = malloc(size_z * sizeof(int));
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
