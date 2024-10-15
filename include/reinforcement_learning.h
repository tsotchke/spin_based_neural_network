#ifndef REINFORCEMENT_LEARNING_H
#define REINFORCEMENT_LEARNING_H
#include "ising_model.h"
#include "kitaev_model.h"

#define RL_LEARNING_RATE 0.1
#define DISCOUNT_FACTOR 0.9
#define INITIAL_EPSILON 0.1
#define THRESHOLD 0.01

double reinforce_learning(IsingLattice *ising_lattice, KitaevLattice *kitaev_lattice, double current_energy, double previous_energy);
void optimize_spins_with_rl(IsingLattice *ising_lattice, KitaevLattice *kitaev_lattice, double reward);
char *get_ising_state_string(IsingLattice *ising_lattice);
char *get_kitaev_state_string(KitaevLattice *kitaev_lattice);
int should_flip_spin(void *lattice, int x, int y, double reward);

#endif // REINFORCEMENT_LEARNING_H