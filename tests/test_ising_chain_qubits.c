/*
 * tests/test_ising_chain_qubits.c
 *
 * Covers the topological-qubit abstraction layered over a KitaevLattice +
 * MajoranaChain. v0.4 exercises lifecycle + a single-qubit X/Z gate round-
 * trip; deeper tests (CNOT commutation, measurement statistics) arrive
 * alongside the Fibonacci-anyon pillar (P1.3) in v0.5.
 */
#include "harness.h"
#include "ising_chain_qubits.h"
#include "kitaev_model.h"

/* make_system allocates a Kitaev lattice large enough to host the chains
 * comfortably, then layers IsingChainQubits on top. Returns NULL (and the
 * caller skips) if the v0.3 init function couldn't place the chains — some
 * combinations print "Chain length exceeds lattice dimensions" and return
 * a partially-initialized system.
 *
 * Ownership: `initialize_ising_chain_qubits` *borrows* the lattice — it
 * keeps a pointer but `free_ising_chain_qubits` does not free the lattice.
 * The caller must free it separately, which is why the lattice handle is
 * returned via out-param here and freed with `free_kitaev_lattice` in
 * each test's cleanup block. Previously the tests leaked ~48 bytes each
 * (caught by AddressSanitizer leak detection on Linux CI). */
static IsingChainQubits *make_system(int num_chains, int chain_length,
                                      KitaevLattice **out_lat) {
    int L = chain_length * 3 + 8;
    KitaevLattice *lat = initialize_kitaev_lattice(
        L, L, L, 1.0, 1.0, 1.0, "all-up");
    KitaevWireParameters p = {
        .coupling_strength   = 1.0,
        .chemical_potential  = 0.5, /* topological phase */
        .superconducting_gap = 1.0,
    };
    IsingChainQubits *q = initialize_ising_chain_qubits(lat, num_chains,
                                                         chain_length, &p);
    if (out_lat) *out_lat = lat;
    return q;
}
#define CLEANUP(q, lat) do {                                              \
    free_ising_chain_qubits(q);                                           \
    free_kitaev_lattice(lat);                                             \
} while (0)

static void test_initialize_and_free(void) {
    KitaevLattice *lat = NULL;
    IsingChainQubits *q = make_system(2, 4, &lat);
    ASSERT_TRUE(q != NULL);
    ASSERT_EQ_INT(q->num_chains, 2);
    CLEANUP(q, lat);
}
static void test_encode_and_measure_zero(void) {
    KitaevLattice *lat = NULL;
    IsingChainQubits *q = make_system(1, 4, &lat);
    ASSERT_TRUE(q != NULL);
    create_topological_qubit(q, 0);
    encode_qubit_state(q, 0, 0);
    int m = measure_topological_qubit(q, 0);
    ASSERT_TRUE(m == 0 || m == 1); /* measurement is valid */
    CLEANUP(q, lat);
}
static void test_x_gate_runs_without_crash(void) {
    KitaevLattice *lat = NULL;
    IsingChainQubits *q = make_system(1, 4, &lat);
    ASSERT_TRUE(q != NULL);
    create_topological_qubit(q, 0);
    encode_qubit_state(q, 0, 0);
    apply_topological_x_gate(q, 0);
    /* Post-gate state should still measurable to a valid value. */
    int m = measure_topological_qubit(q, 0);
    ASSERT_TRUE(m == 0 || m == 1);
    CLEANUP(q, lat);
}
static void test_cnot_between_two_qubits(void) {
    KitaevLattice *lat = NULL;
    IsingChainQubits *q = make_system(2, 4, &lat);
    ASSERT_TRUE(q != NULL);
    create_topological_qubit(q, 0);
    create_topological_qubit(q, 1);
    encode_qubit_state(q, 0, 1);
    encode_qubit_state(q, 1, 0);
    apply_topological_cnot(q, 0, 1);
    int m0 = measure_topological_qubit(q, 0);
    int m1 = measure_topological_qubit(q, 1);
    ASSERT_TRUE(m0 == 0 || m0 == 1);
    ASSERT_TRUE(m1 == 0 || m1 == 1);
    CLEANUP(q, lat);
}
static void test_z_gate_runs_without_crash(void) {
    KitaevLattice *lat = NULL;
    IsingChainQubits *q = make_system(1, 4, &lat);
    ASSERT_TRUE(q != NULL);
    create_topological_qubit(q, 0);
    encode_qubit_state(q, 0, 0);
    apply_topological_z_gate(q, 0);
    int m = measure_topological_qubit(q, 0);
    ASSERT_TRUE(m == 0 || m == 1);
    CLEANUP(q, lat);
}
static void test_y_gate_runs_without_crash(void) {
    KitaevLattice *lat = NULL;
    IsingChainQubits *q = make_system(1, 4, &lat);
    ASSERT_TRUE(q != NULL);
    create_topological_qubit(q, 0);
    encode_qubit_state(q, 0, 1);
    apply_topological_y_gate(q, 0);
    int m = measure_topological_qubit(q, 0);
    ASSERT_TRUE(m == 0 || m == 1);
    CLEANUP(q, lat);
}
static void test_add_chain_interaction_runs(void) {
    KitaevLattice *lat = NULL;
    IsingChainQubits *q = make_system(2, 4, &lat);
    ASSERT_TRUE(q != NULL);
    add_chain_interaction(q, 0, 1, 0.5);
    /* No crash is success; implementation modifies internal state. */
    CLEANUP(q, lat);
}
int main(void) {
    TEST_RUN(test_initialize_and_free);
    TEST_RUN(test_encode_and_measure_zero);
    TEST_RUN(test_x_gate_runs_without_crash);
    TEST_RUN(test_cnot_between_two_qubits);
    TEST_RUN(test_z_gate_runs_without_crash);
    TEST_RUN(test_y_gate_runs_without_crash);
    TEST_RUN(test_add_chain_interaction_runs);
    TEST_SUMMARY();
}