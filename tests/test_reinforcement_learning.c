/*
 * tests/test_reinforcement_learning.c
 *
 * Covers the reward function and spin-optimizer in src/reinforcement_learning.c.
 * v0.3's RL is a reactive heuristic; these tests document its actual
 * behavior rather than asserting "optimal" outcomes.
 */
#include "harness.h"
#include "reinforcement_learning.h"
static void test_reward_positive_when_energy_drops(void) {
    IsingLattice *i = initialize_ising_lattice(3, 3, 3, "all-up");
    KitaevLattice *k = initialize_kitaev_lattice(3, 3, 3, 1.0, 1.0, 1.0, "all-up");
    ASSERT_TRUE(i && k);
    double r = reinforce_learning(i, k, -10.0, -5.0); /* new < old, energy dropped */
    ASSERT_TRUE(r >= 0.0);
    free_ising_lattice(i);
    free_kitaev_lattice(k);
}
static void test_reward_near_zero_when_energy_rises(void) {
    IsingLattice *i = initialize_ising_lattice(3, 3, 3, "all-up");
    KitaevLattice *k = initialize_kitaev_lattice(3, 3, 3, 1.0, 1.0, 1.0, "all-up");
    ASSERT_TRUE(i && k);
    srand(1);
    double r = reinforce_learning(i, k, -5.0, -10.0); /* energy rose: -10 -> -5 */
    /* normalized_reward = fmax(negative, 0) = 0; a tiny random_adjustment
     * in [-0.005, +0.005] is then added, so reward stays near zero. */
    ASSERT_TRUE(r > -0.01 && r < 0.01);
    free_ising_lattice(i);
    free_kitaev_lattice(k);
}
static void test_optimize_spins_preserves_pm1(void) {
    IsingLattice *i = initialize_ising_lattice(3, 3, 3, "random");
    KitaevLattice *k = initialize_kitaev_lattice(3, 3, 3, 1.0, 1.0, 1.0, "random");
    ASSERT_TRUE(i && k);
    srand(1);
    optimize_spins_with_rl(i, k, 5.0);
    for (int x = 0; x < 3; x++)
        for (int y = 0; y < 3; y++)
            for (int z = 0; z < 3; z++) {
                ASSERT_TRUE(i->spins[x][y][z] == 1 || i->spins[x][y][z] == -1);
                ASSERT_TRUE(k->spins[x][y][z] == 1 || k->spins[x][y][z] == -1);
            }
    free_ising_lattice(i);
    free_kitaev_lattice(k);
}
static void test_state_strings_non_null_and_freeable(void) {
    IsingLattice *i = initialize_ising_lattice(2, 2, 2, "all-up");
    KitaevLattice *k = initialize_kitaev_lattice(2, 2, 2, 1.0, 1.0, 1.0, "all-down");
    char *si = get_ising_state_string(i);
    char *sk = get_kitaev_state_string(k);
    ASSERT_TRUE(si != NULL && sk != NULL);
    free(si);
    free(sk);
    free_ising_lattice(i);
    free_kitaev_lattice(k);
}
int main(void) {
    TEST_RUN(test_reward_positive_when_energy_drops);
    TEST_RUN(test_reward_near_zero_when_energy_rises);
    TEST_RUN(test_optimize_spins_preserves_pm1);
    TEST_RUN(test_state_strings_non_null_and_freeable);
    TEST_SUMMARY();
}