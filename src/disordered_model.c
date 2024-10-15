#include <stdlib.h>
#include "disordered_model.h"

// Function to add disorder to the Ising lattice
void add_disorder_to_ising_lattice(IsingLattice *lattice, double disorder_strength) {
    for (int x = 0; x < lattice->size_x; x++) {
        for (int y = 0; y < lattice->size_y; y++) {
            for (int z = 0; z < lattice->size_z; z++) {
                if ((rand() % 100) < (int)(disorder_strength * 100)) {
                    lattice->spins[x][y][z] *= -1; // Flip the spin
                }
            }
        }
    }
}

// Function to add disorder to the Kitaev lattice
void add_disorder_to_kitaev_lattice(KitaevLattice *lattice, double disorder_strength) {
    for (int x = 0; x < lattice->size_x; x++) {
        for (int y = 0; y < lattice->size_y; y++) {
            for (int z = 0; z < lattice->size_z; z++) {
                if ((rand() % 100) < (int)(disorder_strength * 100)) {
                    lattice->spins[x][y][z] *= -1; // Flip the spin
                }
            }
        }
    }
}

