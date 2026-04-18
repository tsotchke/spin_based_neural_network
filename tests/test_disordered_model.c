/*
 * tests/test_disordered_model.c
 *
 * Covers the disorder-application routines. Disorder is modeled as a
 * probability-gated sign flip on each spin. We test boundary cases
 * (rate=0 is a no-op, rate=1 flips everything) and that the spin
 * magnitude is preserved regardless.
 */
#include "harness.h"
#include "disordered_model.h"

static void test_zero_disorder_is_noop_ising(void) {
    IsingLattice *l = initialize_ising_lattice(3, 3, 3, "all-up");
    ASSERT_TRUE(l != NULL);
    add_disorder_to_ising_lattice(l, 0.0);
    for (int x = 0; x < 3; x++)
        for (int y = 0; y < 3; y++)
            for (int z = 0; z < 3; z++)
                ASSERT_EQ_INT(l->spins[x][y][z], 1);
    free_ising_lattice(l);
}

static void test_unit_disorder_flips_all_ising(void) {
    IsingLattice *l = initialize_ising_lattice(3, 3, 3, "all-up");
    ASSERT_TRUE(l != NULL);
    /* The implementation uses (rand() % 100) < 100 which is always true,
     * so every spin flips exactly once. */
    add_disorder_to_ising_lattice(l, 1.0);
    for (int x = 0; x < 3; x++)
        for (int y = 0; y < 3; y++)
            for (int z = 0; z < 3; z++)
                ASSERT_EQ_INT(l->spins[x][y][z], -1);
    free_ising_lattice(l);
}

static void test_zero_disorder_is_noop_kitaev(void) {
    KitaevLattice *l = initialize_kitaev_lattice(3, 3, 3, 1.0, 1.0, 1.0, "all-up");
    ASSERT_TRUE(l != NULL);
    add_disorder_to_kitaev_lattice(l, 0.0);
    for (int x = 0; x < 3; x++)
        for (int y = 0; y < 3; y++)
            for (int z = 0; z < 3; z++)
                ASSERT_EQ_INT(l->spins[x][y][z], 1);
    free_kitaev_lattice(l);
}

static void test_disorder_preserves_spin_magnitude(void) {
    IsingLattice *l = initialize_ising_lattice(4, 4, 4, "random");
    ASSERT_TRUE(l != NULL);
    srand(17);
    add_disorder_to_ising_lattice(l, 0.3);
    for (int x = 0; x < 4; x++)
        for (int y = 0; y < 4; y++)
            for (int z = 0; z < 4; z++) {
                int s = l->spins[x][y][z];
                ASSERT_TRUE(s == 1 || s == -1);
            }
    free_ising_lattice(l);
}

int main(void) {
    TEST_RUN(test_zero_disorder_is_noop_ising);
    TEST_RUN(test_unit_disorder_flips_all_ising);
    TEST_RUN(test_zero_disorder_is_noop_kitaev);
    TEST_RUN(test_disorder_preserves_spin_magnitude);
    TEST_SUMMARY();
}
