#ifndef KITAEV_MODEL_H
#define KITAEV_MODEL_H

typedef struct {
    int size_x;
    int size_y;
    int size_z;
    double jx, jy, jz;
    int ***spins;
} KitaevLattice;

KitaevLattice* initialize_kitaev_lattice(int size_x, int size_y, int size_z, double jx, double jy, double jz, const char* initial_state);
double compute_kitaev_energy(KitaevLattice *lattice);
double compute_kitaev_interaction(KitaevLattice *kitaev_lattice, int x, int y, int z);
void flip_random_spin_kitaev(KitaevLattice *lattice);
void free_kitaev_lattice(KitaevLattice *lattice);

#endif // KITAEV_MODEL_H
