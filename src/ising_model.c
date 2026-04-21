#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ising_model.h"

// Function to initialize the 3D Ising lattice.
// Returns NULL on invalid dimensions or allocation failure, matching the
// idiom used by other `initialize_*` constructors in the codebase
// (neural_network.c, toric_code.c, majorana_modes.c).
IsingLattice* initialize_ising_lattice(int size_x, int size_y, int size_z, const char* initial_state) {
    if (size_x <= 0 || size_y <= 0 || size_z <= 0) {
        fprintf(stderr, "Error: Ising lattice dimensions must be positive (got %d x %d x %d)\n",
                size_x, size_y, size_z);
        return NULL;
    }

    IsingLattice *lattice = malloc(sizeof(IsingLattice));
    if (!lattice) {
        fprintf(stderr, "Error: Memory allocation failed for IsingLattice\n");
        return NULL;
    }
    lattice->size_x = size_x;
    lattice->size_y = size_y;
    lattice->size_z = size_z;
    lattice->spins = malloc((size_t)size_x * sizeof(int**));
    if (!lattice->spins) {
        fprintf(stderr, "Error: Memory allocation failed for Ising spin plane array\n");
        free(lattice);
        return NULL;
    }

    for (int i = 0; i < size_x; i++) {
        lattice->spins[i] = malloc((size_t)size_y * sizeof(int*));
        if (!lattice->spins[i]) {
            fprintf(stderr, "Error: Memory allocation failed for Ising spin row array at i=%d\n", i);
            for (int a = 0; a < i; a++) {
                for (int b = 0; b < size_y; b++) free(lattice->spins[a][b]);
                free(lattice->spins[a]);
            }
            free(lattice->spins);
            free(lattice);
            return NULL;
        }
        for (int j = 0; j < size_y; j++) {
            lattice->spins[i][j] = malloc((size_t)size_z * sizeof(int));
            if (!lattice->spins[i][j]) {
                fprintf(stderr, "Error: Memory allocation failed for Ising spin column at (%d,%d)\n", i, j);
                for (int b = 0; b < j; b++) free(lattice->spins[i][b]);
                free(lattice->spins[i]);
                for (int a = 0; a < i; a++) {
                    for (int b = 0; b < size_y; b++) free(lattice->spins[a][b]);
                    free(lattice->spins[a]);
                }
                free(lattice->spins);
                free(lattice);
                return NULL;
            }
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

// Helper function to get a spin with periodic boundary conditions.
// Use `static inline` so the definition is always emitted per-TU when the
// compiler chooses not to inline (e.g. under -O1 with sanitizers).
static inline int get_spin(IsingLattice *lattice, int x, int y, int z) {
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

/* ===================== Swendsen–Wang cluster update ================== */

typedef struct { int *parent; int *rank; int n; } uf_t;

static int uf_init(uf_t *u, int n) {
    u->n = n;
    u->parent = malloc((size_t)n * sizeof(int));
    u->rank   = calloc((size_t)n, sizeof(int));
    if (!u->parent || !u->rank) { free(u->parent); free(u->rank); return -1; }
    for (int i = 0; i < n; i++) u->parent[i] = i;
    return 0;
}
static int uf_find(uf_t *u, int x) {
    while (u->parent[x] != x) { u->parent[x] = u->parent[u->parent[x]]; x = u->parent[x]; }
    return x;
}
static void uf_union(uf_t *u, int a, int b) {
    int ra = uf_find(u, a), rb = uf_find(u, b);
    if (ra == rb) return;
    if (u->rank[ra] < u->rank[rb]) { int t = ra; ra = rb; rb = t; }
    u->parent[rb] = ra;
    if (u->rank[ra] == u->rank[rb]) u->rank[ra]++;
}
static void uf_free(uf_t *u) { free(u->parent); free(u->rank); }

static inline int sw_site_index(int x, int y, int z, int Ly, int Lz) {
    return ((x * Ly) + y) * Lz + z;
}

int ising_swendsen_wang_step(IsingLattice *lattice, double beta) {
    if (!lattice) return -1;
    int Lx = lattice->size_x, Ly = lattice->size_y, Lz = lattice->size_z;
    int N = Lx * Ly * Lz;

    uf_t u;
    if (uf_init(&u, N) != 0) return -1;

    /* Bond activation probability p = 1 - exp(-2βJ), with J = 1 matching
     * the single-spin routines' convention. */
    double p = 1.0 - exp(-2.0 * beta);

    /* Iterate over the three axis-aligned bond families with PBC. */
    for (int x = 0; x < Lx; x++) {
        for (int y = 0; y < Ly; y++) {
            for (int z = 0; z < Lz; z++) {
                int a = sw_site_index(x, y, z, Ly, Lz);
                int s = lattice->spins[x][y][z];
                int xn = (x + 1) % Lx;
                if (Lx > 1 && lattice->spins[xn][y][z] == s
                    && (rand() / (double)RAND_MAX) < p)
                    uf_union(&u, a, sw_site_index(xn, y, z, Ly, Lz));
                int yn = (y + 1) % Ly;
                if (Ly > 1 && lattice->spins[x][yn][z] == s
                    && (rand() / (double)RAND_MAX) < p)
                    uf_union(&u, a, sw_site_index(x, yn, z, Ly, Lz));
                int zn = (z + 1) % Lz;
                if (Lz > 1 && lattice->spins[x][y][zn] == s
                    && (rand() / (double)RAND_MAX) < p)
                    uf_union(&u, a, sw_site_index(x, y, zn, Ly, Lz));
            }
        }
    }

    /* Decide per-cluster flip. */
    int *flip = malloc((size_t)N * sizeof(int));
    if (!flip) { uf_free(&u); return -1; }
    for (int i = 0; i < N; i++) flip[i] = -1;
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++) for (int z = 0; z < Lz; z++) {
        int a = sw_site_index(x, y, z, Ly, Lz);
        int r = uf_find(&u, a);
        if (flip[r] == -1) flip[r] = (rand() & 1) ? 1 : 0;
        if (flip[r]) lattice->spins[x][y][z] = -lattice->spins[x][y][z];
    }
    free(flip);
    uf_free(&u);
    return 0;
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
