#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ising_model.h"

// Function to initialize the 3D Ising lattice
IsingLattice* initialize_ising_lattice(int size_x, int size_y, int size_z, const char* initial_state) {
    IsingLattice *lattice = malloc(sizeof(IsingLattice));
    lattice->size_x = size_x;
    lattice->size_y = size_y;
    lattice->size_z = size_z;
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

// Helper function to get a spin with periodic boundary conditions
inline int get_spin(IsingLattice *lattice, int x, int y, int z) {
    int mod_x = (x + lattice->size_x) % lattice->size_x;
    int mod_y = (y + lattice->size_y) % lattice->size_y;
    int mod_z = (z + lattice->size_z) % lattice->size_z;
    return lattice->spins[mod_x][mod_y][mod_z];
}

// Function to compute the energy of the 3D Ising lattice
double compute_ising_energy(IsingLattice *lattice) {
    double energy = 0.0;
    for (int x = 0; x < lattice->size_x; x++) {
        for (int y = 0; y < lattice->size_y; y++) {
            for (int z = 0; z < lattice->size_z; z++) {
                int spin = lattice->spins[x][y][z];
                // Interactions with neighbors in x, y, and z directions (periodic boundary)
                energy -= spin * get_spin(lattice, x + 1, y, z);
                energy -= spin * get_spin(lattice, x, y + 1, z);
                energy -= spin * get_spin(lattice, x, y, z + 1);
            }
        }
    }
    return energy;
}

// Function to compute the interaction energy at a specific site (x, y, z)
double compute_ising_interaction(IsingLattice *ising_lattice, int x, int y, int z) {
    double interaction_energy = 0.0;

    // Get the spin at the current site
    int spin = ising_lattice->spins[x][y][z];

    // Interaction in the x-direction
    if (x < ising_lattice->size_x - 1) {
        interaction_energy += spin * ising_lattice->spins[x + 1][y][z];
    }
    if (x > 0) {
        interaction_energy += spin * ising_lattice->spins[x - 1][y][z];
    }

    // Interaction in the y-direction
    if (y < ising_lattice->size_y - 1) {
        interaction_energy += spin * ising_lattice->spins[x][y + 1][z];
    }
    if (y > 0) {
        interaction_energy += spin * ising_lattice->spins[x][y - 1][z];
    }

    // Interaction in the z-direction
    if (z < ising_lattice->size_z - 1) {
        interaction_energy += spin * ising_lattice->spins[x][y][z + 1];
    }
    if (z > 0) {
        interaction_energy += spin * ising_lattice->spins[x][y][z - 1];
    }

    return interaction_energy;
}

// Function to flip a random spin in the 3D Ising lattice based on the Metropolis algorithm
void flip_random_spin_ising(IsingLattice *lattice) {
    int x = rand() % lattice->size_x;
    int y = rand() % lattice->size_y;
    int z = rand() % lattice->size_z;

    int old_spin = lattice->spins[x][y][z];
    int new_spin = -old_spin;

    double delta_energy = 2 * old_spin * (
        get_spin(lattice, x + 1, y, z) + get_spin(lattice, x - 1, y, z) +
        get_spin(lattice, x, y + 1, z) + get_spin(lattice, x, y - 1, z) +
        get_spin(lattice, x, y, z + 1) + get_spin(lattice, x, y, z - 1)
    );

    if (delta_energy < 0 || (rand() / (double)RAND_MAX) < exp(-delta_energy)) {
        lattice->spins[x][y][z] = new_spin;
    }
}

// Function to free the allocated memory for the 3D Ising lattice
void free_ising_lattice(IsingLattice *lattice) {
    for (int i = 0; i < lattice->size_x; i++) {
        for (int j = 0; j < lattice->size_y; j++) {
            free(lattice->spins[i][j]);
        }
        free(lattice->spins[i]);
    }
    free(lattice->spins);
    free(lattice);
}

// Optional: Print the state of the 3D Ising lattice (one slice at a time)
void print_ising_state(IsingLattice *lattice) {
    for (int k = 0; k < lattice->size_z; k++) {
        printf("Slice z = %d:\n", k);
        for (int i = 0; i < lattice->size_x; i++) {
            for (int j = 0; j < lattice->size_y; j++) {
                printf("%d ", lattice->spins[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
