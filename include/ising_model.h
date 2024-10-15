#ifndef ISING_MODEL_H
#define ISING_MODEL_H

typedef struct {
    int size_x;
    int size_y;
    int size_z;
    int ***spins; // 2D array of spins
} IsingLattice;

IsingLattice* initialize_ising_lattice(int size_x, int size_y, int size_z, const char* initial_state);
double compute_ising_energy(IsingLattice *lattice);
double compute_ising_interaction(IsingLattice *ising_lattice, int x, int y, int z);
void flip_random_spin_ising(IsingLattice *lattice);
void free_ising_lattice(IsingLattice *lattice);

#endif // ISING_MODEL_H
