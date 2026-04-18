/*
 * tests/test_kitaev.c
 *
 * Verifies src/kitaev_model.c energies on known configurations.
 * compute_kitaev_energy() uses OPEN boundary conditions, summing
 *   E = Σ_i<L-1 jx * S_i S_{i+1}  (and y, z analogues),
 * so for an L^3 all-up lattice E = (jx + jy + jz) * (L-1) * L^2.
 */
#include "harness.h"
#include "kitaev_model.h"

static void test_energy_all_up_isotropic(void) {
    int L = 4;
    double jx = 1.0, jy = 1.0, jz = 1.0;
    KitaevLattice *l = initialize_kitaev_lattice(L, L, L, jx, jy, jz, "all-up");
    ASSERT_TRUE(l != NULL);
    double expected = (jx + jy + jz) * (L - 1) * L * L;
    ASSERT_NEAR(compute_kitaev_energy(l), expected, 1e-12);
    free_kitaev_lattice(l);
}

static void test_energy_all_up_anisotropic(void) {
    int L = 3;
    double jx = 2.0, jy = -1.0, jz = 0.5;
    KitaevLattice *l = initialize_kitaev_lattice(L, L, L, jx, jy, jz, "all-up");
    ASSERT_TRUE(l != NULL);
    double expected = (jx + jy + jz) * (L - 1) * L * L;
    ASSERT_NEAR(compute_kitaev_energy(l), expected, 1e-12);
    free_kitaev_lattice(l);
}

/* Flipping one spin from all-up: the energy of bonds touching that spin
 * inverts sign. In 3D interior site: 6 bonds. Interior energy change:
 * 2 * 6 * <j per bond>. For all three couplings equal to 1 and an interior
 * site on L=5, flipping reduces the energy of those bonds by 2*6 = 12. */
static void test_flip_interior_spin_anisotropy(void) {
    int L = 5;
    KitaevLattice *l = initialize_kitaev_lattice(L, L, L, 1.0, 1.0, 1.0, "all-up");
    ASSERT_TRUE(l != NULL);
    double e_before = compute_kitaev_energy(l);
    l->spins[2][2][2] = -1;
    double e_after = compute_kitaev_energy(l);
    ASSERT_NEAR(e_after - e_before, -12.0, 1e-12);
    free_kitaev_lattice(l);
}

static void test_flip_random_spin_preserves_pm1(void) {
    int L = 4;
    KitaevLattice *l = initialize_kitaev_lattice(L, L, L, 1.0, 1.0, 1.0, "all-up");
    ASSERT_TRUE(l != NULL);
    srand(7);
    for (int i = 0; i < 200; i++) flip_random_spin_kitaev(l);
    for (int x = 0; x < L; x++)
        for (int y = 0; y < L; y++)
            for (int z = 0; z < L; z++) {
                int s = l->spins[x][y][z];
                ASSERT_TRUE(s == 1 || s == -1);
            }
    free_kitaev_lattice(l);
}

/* Per-site interaction for Kitaev interior: on all-up isotropic lattice,
 * site (2,2,2) sees 6 neighbors × (jx+jy+jz)/3 contribution — actually the
 * function sums jx * S*Sx_neighbors + jy * S*Sy_neighbors + jz * S*Sz_neighbors
 * with 2 neighbors per axis → total = 2*(jx+jy+jz) = 2*(1+1+1) = 6. */
static void test_kitaev_interaction_interior(void) {
    int L = 5;
    KitaevLattice *l = initialize_kitaev_lattice(L, L, L, 1.0, 1.0, 1.0, "all-up");
    ASSERT_TRUE(l != NULL);
    double E = compute_kitaev_interaction(l, 2, 2, 2);
    ASSERT_NEAR(E, 6.0, 1e-12);
    free_kitaev_lattice(l);
}

/* At the corner (0,0,0), only +x, +y, +z neighbors exist; interaction =
 * 1*(jx + jy + jz) = 3. */
static void test_kitaev_interaction_corner(void) {
    int L = 5;
    KitaevLattice *l = initialize_kitaev_lattice(L, L, L, 1.0, 1.0, 1.0, "all-up");
    ASSERT_TRUE(l != NULL);
    double E = compute_kitaev_interaction(l, 0, 0, 0);
    ASSERT_NEAR(E, 3.0, 1e-12);
    free_kitaev_lattice(l);
}

int main(void) {
    TEST_RUN(test_energy_all_up_isotropic);
    TEST_RUN(test_energy_all_up_anisotropic);
    TEST_RUN(test_flip_interior_spin_anisotropy);
    TEST_RUN(test_flip_random_spin_preserves_pm1);
    TEST_RUN(test_kitaev_interaction_interior);
    TEST_RUN(test_kitaev_interaction_corner);
    TEST_SUMMARY();
}
