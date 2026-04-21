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

/* Swendsen–Wang cluster update (Swendsen, Wang 1987). One full sweep:
 *   1. For every nearest-neighbour bond (i, j) with s_i = s_j,
 *      activate the bond with probability p = 1 - exp(-2βJ).
 *      (β is the inverse temperature, J the coupling strength.)
 *   2. Build connected clusters of sites joined by active bonds using
 *      a union-find data structure.
 *   3. Flip each cluster with independent probability 1/2.
 *
 * This nonlocal move eliminates the critical slowing down of the
 * single-spin Metropolis updates near T_c: the autocorrelation time
 * scales as N^z_SW with z_SW ≈ 0 in 2D (vs z ≈ 2 for Metropolis).
 *
 * Operates on the full 3D lattice (size_x × size_y × size_z) with
 * periodic boundary conditions. J = 1 is assumed (the existing
 * single-spin routines use the same convention). Returns 0 on
 * success, -1 on allocation failure. */
int ising_swendsen_wang_step(IsingLattice *lattice, double beta);

#endif // ISING_MODEL_H
