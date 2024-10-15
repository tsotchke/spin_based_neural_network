#ifndef SPIN_MODELS_H
#define SPIN_MODELS_H

typedef struct {
    double sx;
    double sy;
    double sz;
} Spin;

typedef struct {
    int size_x;
    int size_y;
    int size_z;
    Spin ***spins;
} SpinLattice;

SpinLattice* initialize_spin_lattice(int size_x, int size_y, int size_z, const char* initial_state);
double compute_spin_energy(SpinLattice *lattice);
void free_spin_lattice(SpinLattice *lattice);

#endif // SPIN_MODELS_H
