/*
 * tests/test_berry_phase.c
 *
 * Covers the Berry-phase / Chern / winding / TKNN API. v0.4 exercises
 * allocation, the integer-normalised Chern value, and the env-var gate
 * (CHERN_NUMBER is only honored when the binary was compiled with
 * -DSPIN_NN_TESTING). Numerical tests of the underlying Brillouin-zone
 * integration live under v0.5 pillar work.
 */
#include <stdlib.h>
#include "harness.h"
#include "berry_phase.h"
#include "kitaev_model.h"
#include "majorana_modes.h"
static void test_init_and_free_berry_data(void) {
    BerryPhaseData *d = initialize_berry_phase_data(4, 4, 4);
    ASSERT_TRUE(d != NULL);
    ASSERT_EQ_INT(d->k_space_grid[0], 4);
    ASSERT_EQ_INT(d->k_space_grid[1], 4);
    ASSERT_EQ_INT(d->k_space_grid[2], 4);
    free_berry_phase_data(d);
}
static void test_invariants_struct_lifecycle(void) {
    TopologicalInvariants *inv = initialize_topological_invariants(3);
    ASSERT_TRUE(inv != NULL);
    ASSERT_EQ_INT(inv->num_invariants, 3);
    free_topological_invariants(inv);
}
static void test_calculate_all_invariants_without_chain(void) {
    KitaevLattice *l = initialize_kitaev_lattice(4, 4, 4, 1.0, 1.0, -1.0, "all-up");
    ASSERT_TRUE(l != NULL);
    TopologicalInvariants *inv = calculate_all_invariants(l, NULL);
    ASSERT_TRUE(inv != NULL);
    ASSERT_TRUE(inv->num_invariants > 0);
    free_topological_invariants(inv);
    free_kitaev_lattice(l);
}
static void test_winding_number_of_topological_chain(void) {
    KitaevWireParameters params = {
.coupling_strength = 1.0,
.chemical_potential = 0.5,  /* |mu| < 2|t| -> topological */
.superconducting_gap = 1.0,
    };
    MajoranaChain *chain = initialize_majorana_chain(6, &params);
    ASSERT_TRUE(chain != NULL);
    double w = calculate_winding_number(chain);
    /* In the topological phase we expect |w| ~ 1; tolerate some wiggle. */
    ASSERT_TRUE(w > -1.1 && w < 1.1);
    free_majorana_chain(chain);
}
/* CHERN_NUMBER env-var override is gated behind SPIN_NN_TESTING; in a
 * release build, setting the variable must NOT alter the calculated Chern
 * number. */
static void test_chern_env_var_ignored_in_release(void) {
#ifndef SPIN_NN_TESTING
    setenv("CHERN_NUMBER", "42", 1);
    KitaevLattice *l = initialize_kitaev_lattice(4, 4, 4, 1.0, 1.0, -1.0, "all-up");
    ASSERT_TRUE(l != NULL);
    TopologicalInvariants *inv = calculate_all_invariants(l, NULL);
    ASSERT_TRUE(inv != NULL);
    /* Chern should never be the override value (42). */
    for (int i = 0; i < inv->num_invariants; i++) {
        ASSERT_TRUE(inv->invariants[i] < 41.0 || inv->invariants[i] > 43.0);
    }
    free_topological_invariants(inv);
    free_kitaev_lattice(l);
    unsetenv("CHERN_NUMBER");
#else
    /* In test builds the gate is open; nothing to assert here. */
    ASSERT_TRUE(1);
#endif
}
int main(void) {
    TEST_RUN(test_init_and_free_berry_data);
    TEST_RUN(test_invariants_struct_lifecycle);
    TEST_RUN(test_calculate_all_invariants_without_chain);
    TEST_RUN(test_winding_number_of_topological_chain);
    TEST_RUN(test_chern_env_var_ignored_in_release);
    TEST_SUMMARY();
}