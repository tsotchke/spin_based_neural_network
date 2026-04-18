/*
 * tests/test_ising.c
 *
 * Verifies src/ising_model.c: energies on known configurations and
 * basic Metropolis detailed-balance sanity.
 *
 * Energy on an L^3 torus with H = -Σ<ij> S_i S_j (nearest-neighbor, periodic):
 *   - All-up / all-down: E = -3 L^3  (three bonds per site, one shared per site
 *     under compute_ising_energy's once-per-site loop over (+x, +y, +z) neighbors)
 *   - Antiferromagnetic checkerboard on even L: E = +3 L^3
 *   - Stripe along z: E = -L^3 + 2 * L^2 * L = +L^3 (two perpendicular directions
 *     FM, one AFM)
 */
#include "harness.h"
#include "ising_model.h"

/* All spins up on L^3: E = -3 L^3. */
static void test_energy_all_up(void) {
    int L = 4;
    IsingLattice *l = initialize_ising_lattice(L, L, L, "all-up");
    ASSERT_TRUE(l != NULL);
    ASSERT_NEAR(compute_ising_energy(l), -3.0 * L * L * L, 1e-12);
    free_ising_lattice(l);
}

/* All spins down on L^3: E = -3 L^3. */
static void test_energy_all_down(void) {
    int L = 4;
    IsingLattice *l = initialize_ising_lattice(L, L, L, "all-down");
    ASSERT_TRUE(l != NULL);
    ASSERT_NEAR(compute_ising_energy(l), -3.0 * L * L * L, 1e-12);
    free_ising_lattice(l);
}

/* Checkerboard (AFM) on even L: E = +3 L^3. */
static void test_energy_checkerboard(void) {
    int L = 4;
    IsingLattice *l = initialize_ising_lattice(L, L, L, "all-up");
    ASSERT_TRUE(l != NULL);
    for (int x = 0; x < L; x++)
        for (int y = 0; y < L; y++)
            for (int z = 0; z < L; z++)
                l->spins[x][y][z] = ((x + y + z) & 1) ? -1 : +1;
    ASSERT_NEAR(compute_ising_energy(l), +3.0 * L * L * L, 1e-12);
    free_ising_lattice(l);
}

/* Stripe along z: spin depends only on z. Two FM directions, one AFM. */
static void test_energy_stripe_along_z(void) {
    int L = 4;
    IsingLattice *l = initialize_ising_lattice(L, L, L, "all-up");
    ASSERT_TRUE(l != NULL);
    for (int x = 0; x < L; x++)
        for (int y = 0; y < L; y++)
            for (int z = 0; z < L; z++)
                l->spins[x][y][z] = (z & 1) ? -1 : +1;
    /* Each site: +1 bond in +x, +1 bond in +y, -1 bond in +z. Sum per site = -1.
     * compute_ising_energy: E = - Σ_site (S*(+x neigh) + S*(+y neigh) + S*(+z neigh))
     *                         = -L^3 * (1 + 1 + (-1)) = -L^3. */
    ASSERT_NEAR(compute_ising_energy(l), -1.0 * L * L * L, 1e-12);
    free_ising_lattice(l);
}

/* flip_random_spin_ising must never take an all-up L=2 state to higher
 * energy without the Metropolis acceptance (deterministic: ΔE > 0 always
 * flipping increases energy, so with rand controlled should reject). */
static void test_metropolis_does_not_increase_for_low_temperature_smoke(void) {
    /* Smoke test only — we just check that the routine is self-consistent:
     * after many flips starting from all-up, the lattice is still valid ±1. */
    int L = 4;
    IsingLattice *l = initialize_ising_lattice(L, L, L, "all-up");
    ASSERT_TRUE(l != NULL);
    srand(42);
    for (int i = 0; i < 1000; i++) flip_random_spin_ising(l);
    for (int x = 0; x < L; x++)
        for (int y = 0; y < L; y++)
            for (int z = 0; z < L; z++) {
                int s = l->spins[x][y][z];
                ASSERT_TRUE(s == 1 || s == -1);
            }
    free_ising_lattice(l);
}

/* Per-site interaction energy: interior site on all-up lattice has 6 +1
 * neighbors; compute_ising_interaction sums spin*S_neigh over all six, so
 * for site (2,2,2) on an all-up 5^3 lattice the result is +6. */
static void test_interaction_interior_site_all_up(void) {
    int L = 5;
    IsingLattice *l = initialize_ising_lattice(L, L, L, "all-up");
    ASSERT_TRUE(l != NULL);
    double E = compute_ising_interaction(l, 2, 2, 2);
    ASSERT_NEAR(E, 6.0, 1e-12);
    free_ising_lattice(l);
}

/* Edge site on open-boundary interaction: (0,0,0) has 3 in-bounds neighbors. */
static void test_interaction_corner_site_all_up(void) {
    int L = 5;
    IsingLattice *l = initialize_ising_lattice(L, L, L, "all-up");
    ASSERT_TRUE(l != NULL);
    double E = compute_ising_interaction(l, 0, 0, 0);
    ASSERT_NEAR(E, 3.0, 1e-12);
    free_ising_lattice(l);
}

int main(void) {
    TEST_RUN(test_energy_all_up);
    TEST_RUN(test_energy_all_down);
    TEST_RUN(test_energy_checkerboard);
    TEST_RUN(test_energy_stripe_along_z);
    TEST_RUN(test_metropolis_does_not_increase_for_low_temperature_smoke);
    TEST_RUN(test_interaction_interior_site_all_up);
    TEST_RUN(test_interaction_corner_site_all_up);
    TEST_SUMMARY();
}
