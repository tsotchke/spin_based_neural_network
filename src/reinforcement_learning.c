#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "reinforcement_learning.h"
#include "ising_model.h"
#include "kitaev_model.h"
#include "spin_models.h"

double epsilon = INITIAL_EPSILON;

double reinforce_learning(IsingLattice *ising_lattice, KitaevLattice *kitaev_lattice, double current_energy, double previous_energy) {
    if (fabs(previous_energy) < 1e-10) {
        fprintf(stderr, "Previous energy too close to zero, cannot normalize reward.\n");
        return 0.0;
    }

    double ising_energy = compute_ising_energy(ising_lattice);
    double kitaev_energy = compute_kitaev_energy(kitaev_lattice);
    double total_energy = ising_energy + kitaev_energy;

    double energy_change = previous_energy - current_energy;
    double normalized_reward = fmax((energy_change / fabs(previous_energy)) * 100, 0);

    double random_adjustment = ((double)rand() / RAND_MAX) * 0.01 - 0.005;
    double reward = normalized_reward + random_adjustment;

    if (fabs(energy_change) < THRESHOLD) {
        reward = 0;
    }

    epsilon = fmax(INITIAL_EPSILON * exp(-0.01 * fabs(normalized_reward)), 0.01);

    printf("Current Energy: %f, Previous Energy: %f, Total Energy: %f, Epsilon: %f, Random Adjustment: %f, Reward: %f\n",
           current_energy, previous_energy, fabs(total_energy), epsilon, random_adjustment, reward);

    return reward;
}

char *get_ising_state_string(IsingLattice *ising_lattice) {
    int size_x = ising_lattice->size_x;
    int size_y = ising_lattice->size_y;
    int size_z = ising_lattice->size_z;
    char *state_str = (char *)malloc(size_x * size_y * size_z + size_x + 1);

    if (state_str == NULL) {
        fprintf(stderr, "Memory allocation failed for Ising state string.\n");
        return NULL;
    }

    char *ptr = state_str;
    for (int i = 0; i < size_x; i++) {
        for (int j = 0; j < size_y; j++) {
            for (int k = 0; k < size_z; k++) {
                *ptr++ = (ising_lattice->spins[i][j][k] == 1) ? '1' : '0';
            }
        }
        *ptr++ = '\n';
    }
    *ptr = '\0';
    
    return state_str;
}

char *get_kitaev_state_string(KitaevLattice *kitaev_lattice) {
    int size_x = kitaev_lattice->size_x;
    int size_y = kitaev_lattice->size_y;
    int size_z = kitaev_lattice->size_z;
    char *state_str = (char *)malloc(size_x * size_y * size_z + size_x + 1);

    if (state_str == NULL) {
        fprintf(stderr, "Memory allocation failed for Kitaev state string.\n");
        return NULL;
    }

    char *ptr = state_str;
    for (int i = 0; i < size_x; i++) {
        for (int j = 0; j < size_y; j++) {
            for (int k = 0; k < size_z; k++) {
                *ptr++ = (kitaev_lattice->spins[i][j][k] == 1) ? '1' : '0';
            }
        }
        *ptr++ = '\n';
    }
    *ptr = '\0';

    return state_str;
}

void optimize_spins_with_rl(IsingLattice *ising_lattice, KitaevLattice *kitaev_lattice, double reward) {
    for (int i = 0; i < ising_lattice->size_x; i++) {
        for (int j = 0; j < ising_lattice->size_y; j++) {
            for (int k = 0; k < ising_lattice->size_z; k++) {
                if (reward > 0.5 && fabs(reward) > THRESHOLD) {
                    if (should_flip_spin(ising_lattice, i, j, reward)) {
                        ising_lattice->spins[i][j][k] *= -1; // Flip spin
                    }
                } else if (((double)rand() / RAND_MAX) < epsilon) {
                    ising_lattice->spins[i][j][k] *= -1; // Flip spin randomly
                }
            }
        }
    }

    // Apply similar logic to Kitaev lattice spins
    for (int i = 0; i < kitaev_lattice->size_x; i++) {
        for (int j = 0; j < kitaev_lattice->size_y; j++) {
            for (int k = 0; k < kitaev_lattice->size_z; k++) {
                if (reward > 0.5 && fabs(reward) > THRESHOLD) {
                    if (should_flip_spin(kitaev_lattice, i, j, reward)) {
                        kitaev_lattice->spins[i][j][k] *= -1; // Flip spin
                    }
                } else if (((double)rand() / RAND_MAX) < epsilon) {
                    kitaev_lattice->spins[i][j][k] *= -1; // Flip spin randomly
                }
            }
        }
    }
}

// Helper function to decide if a spin should be flipped
int should_flip_spin(void *lattice, int x, int y, double reward) {
    return (((double)rand() / RAND_MAX) < reward * RL_LEARNING_RATE);
}