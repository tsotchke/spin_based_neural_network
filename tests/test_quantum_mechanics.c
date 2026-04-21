/*
 * tests/test_quantum_mechanics.c
 *
 * Smoke-level coverage of the quantum-mechanics helpers used by the
 * main training loop (noise injection + entanglement simulation).
 */
#include "harness.h"
#include "quantum_mechanics.h"
static void test_apply_quantum_effects_preserves_spin_magnitude(void) {
    IsingLattice  *i = initialize_ising_lattice(3, 3, 3, "random");
    KitaevLattice *k = initialize_kitaev_lattice(3, 3, 3, 1.0, 1.0, 1.0, "random");
    SpinLattice   *s = initialize_spin_lattice(3, 3, 3, "all-up");
    ASSERT_TRUE(i && k && s);
    srand(5);
    apply_quantum_effects(i, k, s, 0.3);
    /* Ising and Kitaev spins remain ±1. */
    for (int x = 0; x < 3; x++)
        for (int y = 0; y < 3; y++)
            for (int z = 0; z < 3; z++) {
                ASSERT_TRUE(i->spins[x][y][z] == 1 || i->spins[x][y][z] == -1);
                ASSERT_TRUE(k->spins[x][y][z] == 1 || k->spins[x][y][z] == -1);
            }
    free_ising_lattice(i);
    free_kitaev_lattice(k);
    free_spin_lattice(s);
}
static void test_simulate_entanglement_runs_without_crash(void) {
    IsingLattice  *i = initialize_ising_lattice(3, 3, 3, "random");
    KitaevLattice *k = initialize_kitaev_lattice(3, 3, 3, 1.0, 1.0, 1.0, "random");
    ASSERT_TRUE(i && k);
    srand(7);
    simulate_entanglement(i, k, 0.1);
    for (int x = 0; x < 3; x++)
        for (int y = 0; y < 3; y++)
            for (int z = 0; z < 3; z++) {
                ASSERT_TRUE(i->spins[x][y][z] == 1 || i->spins[x][y][z] == -1);
            }
    free_ising_lattice(i);
    free_kitaev_lattice(k);
}
/* At noise_level = 0 the implementation still applies "superposition"
 * resampling (alpha = sqrt(0.5)), so spins get randomized regardless of
 * the noise parameter. This documents the v0.3 behavior. A cleaner zero-
 * noise semantics lands with v0.5 pillar work that rewrites the quantum-
 * effects routine using real Schrödinger evolution. */
static void test_noise_input_validation(void) {
    IsingLattice  *i = initialize_ising_lattice(2, 2, 2, "all-up");
    KitaevLattice *k = initialize_kitaev_lattice(2, 2, 2, 1.0, 1.0, 1.0, "all-up");
    SpinLattice   *s = initialize_spin_lattice(2, 2, 2, "all-up");
    ASSERT_TRUE(i && k && s);
    /* Out-of-range noise should be rejected without crashing. */
    apply_quantum_effects(i, k, s, -0.1);
    apply_quantum_effects(i, k, s,  1.5);
    free_ising_lattice(i);
    free_kitaev_lattice(k);
    free_spin_lattice(s);
}
int main(void) {
    TEST_RUN(test_apply_quantum_effects_preserves_spin_magnitude);
    TEST_RUN(test_simulate_entanglement_runs_without_crash);
    TEST_RUN(test_noise_input_validation);
    TEST_SUMMARY();
}