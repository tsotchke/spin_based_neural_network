#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include "quantum_mechanics.h"

#define PI 3.14159265358979323846

// Quantum superposition function
static void apply_superposition(int *spin, double alpha) {
    double random = (double)rand() / RAND_MAX;
    if (random < alpha * alpha) {
        *spin = 1;
    } else {
        *spin = -1;
    }
}

// Quantum tunneling function
static void apply_tunneling(int *spin, double tunnel_prob) {
    if ((double)rand() / RAND_MAX < tunnel_prob) {
        *spin *= -1;
    }
}

// Quantum decoherence function
static void apply_decoherence(double complex *quantum_state, double decoherence_rate) {
    double random_phase = 2 * PI * ((double)rand() / RAND_MAX);
    *quantum_state *= cexp(I * random_phase * decoherence_rate);
}

void apply_quantum_effects(IsingLattice *ising_lattice, KitaevLattice *kitaev_lattice, SpinLattice *spin_lattice, double noise_level) {
    if (noise_level < 0 || noise_level > 1) {
        fprintf(stderr, "Error: noise_level must be between 0 and 1.\n");
        return;
    }

    double tunnel_prob = noise_level * 0.1;
    double decoherence_rate = noise_level * 0.05;

    for (int x = 0; x < ising_lattice->size_x; x++) {
        for (int y = 0; y < ising_lattice->size_y; y++) {
            for (int z = 0; z < ising_lattice->size_z; z++) {
                // Apply quantum superposition
                apply_superposition(&ising_lattice->spins[x][y][z], sqrt(0.5 + noise_level * 0.5));
                apply_superposition(&kitaev_lattice->spins[x][y][z], sqrt(0.5 + noise_level * 0.5));

                // Apply quantum tunneling
                apply_tunneling(&ising_lattice->spins[x][y][z], tunnel_prob);
                apply_tunneling(&kitaev_lattice->spins[x][y][z], tunnel_prob);

                // Apply quantum decoherence to spin lattice
                double complex quantum_state = spin_lattice->spins[x][y][z].sx + I * spin_lattice->spins[x][y][z].sy;
                apply_decoherence(&quantum_state, decoherence_rate);
                spin_lattice->spins[x][y][z].sx = creal(quantum_state);
                spin_lattice->spins[x][y][z].sy = cimag(quantum_state);

                // Add random fluctuations to spin z-component
                spin_lattice->spins[x][y][z].sz += noise_level * ((double)rand() / RAND_MAX - 0.5);
            }
        }
    }
}

void simulate_entanglement(IsingLattice *ising_lattice, KitaevLattice *kitaev_lattice, double entanglement_prob) {
    if (entanglement_prob < 0 || entanglement_prob > 1) {
        fprintf(stderr, "Error: entanglement_prob must be between 0 and 1.\n");
        return;
    }

    for (int x = 0; x < ising_lattice->size_x; x++) {
        for (int y = 0; y < ising_lattice->size_y; y++) {
            for (int z = 0; z < ising_lattice->size_z; z++) {
                if ((double)rand() / RAND_MAX < entanglement_prob) {
                    // Create Bell state
                    int bell_state = rand() % 4;
                    int target_x = rand() % kitaev_lattice->size_x;
                    int target_y = rand() % kitaev_lattice->size_y;
                    int target_z = rand() % kitaev_lattice->size_z;

                    switch (bell_state) {
                        case 0: // |Φ+⟩ = (|00⟩ + |11⟩) / √2
                            ising_lattice->spins[x][y][z] = 1;
                            kitaev_lattice->spins[target_x][target_y][target_z] = 1;
                            break;
                        case 1: // |Φ-⟩ = (|00⟩ - |11⟩) / √2
                            ising_lattice->spins[x][y][z] = 1;
                            kitaev_lattice->spins[target_x][target_y][target_z] = -1;
                            break;
                        case 2: // |Ψ+⟩ = (|01⟩ + |10⟩) / √2
                            ising_lattice->spins[x][y][z] = 1;
                            kitaev_lattice->spins[target_x][target_y][target_z] = -1;
                            break;
                        case 3: // |Ψ-⟩ = (|01⟩ - |10⟩) / √2
                            ising_lattice->spins[x][y][z] = -1;
                            kitaev_lattice->spins[target_x][target_y][target_z] = 1;
                            break;
                    }
                }
            }
        }
    }
}