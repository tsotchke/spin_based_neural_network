/*
 * tests/test_engine_adapter.c
 *
 * Covers the engine-neutral adapter API. Without an engine linked in
 * (SPIN_NN_HAS_ENGINE undefined), init/shutdown/version return the
 * "disabled" status codes; the flatteners are live regardless and are
 * exercised against all three lattice types.
 */
#include "harness.h"
#include "engine_adapter.h"
static void test_disabled_mode_reports_not_available(void) {
    ASSERT_EQ_INT(engine_adapter_is_available(), 0);
    ASSERT_EQ_INT(engine_adapter_init(), ENGINE_ADAPTER_EDISABLED);
    ASSERT_EQ_INT(engine_adapter_shutdown(), ENGINE_ADAPTER_EDISABLED);
    ASSERT_TRUE(engine_adapter_engine_version() == NULL);
}
static void test_build_version_always_non_null(void) {
    const char *v = engine_adapter_build_version();
    ASSERT_TRUE(v != NULL);
    ASSERT_TRUE(v[0] != '\0');
}
static void test_flatten_ising_size_probe_and_write(void) {
    IsingLattice *l = initialize_ising_lattice(2, 3, 4, "all-up");
    ASSERT_TRUE(l != NULL);
    long need = engine_adapter_flatten_ising(l, NULL, 0);
    ASSERT_EQ_INT(need, 2 * 3 * 4);
    float buf[24];
    long wrote = engine_adapter_flatten_ising(l, buf, sizeof(buf) / sizeof(buf[0]));
    ASSERT_EQ_INT(wrote, 24);
    for (long i = 0; i < 24; i++) ASSERT_NEAR(buf[i], 1.0f, 1e-12);
    free_ising_lattice(l);
}
static void test_flatten_ising_rejects_small_buffer(void) {
    IsingLattice *l = initialize_ising_lattice(2, 2, 2, "all-up");
    ASSERT_TRUE(l != NULL);
    float buf[4];
    long rc = engine_adapter_flatten_ising(l, buf, 4);
    ASSERT_TRUE(rc < 0);
    free_ising_lattice(l);
}
static void test_flatten_ising_null_lattice_is_error(void) {
    float buf[1];
    ASSERT_TRUE(engine_adapter_flatten_ising(NULL, buf, 1) == ENGINE_ADAPTER_EARG);
}
static void test_flatten_kitaev_all_down(void) {
    KitaevLattice *l = initialize_kitaev_lattice(3, 3, 3, 1.0, 1.0, 1.0, "all-down");
    ASSERT_TRUE(l != NULL);
    long need = engine_adapter_flatten_kitaev(l, NULL, 0);
    ASSERT_EQ_INT(need, 27);
    float buf[27];
    ASSERT_EQ_INT(engine_adapter_flatten_kitaev(l, buf, 27), 27);
    for (long i = 0; i < 27; i++) ASSERT_NEAR(buf[i], -1.0f, 1e-12);
    free_kitaev_lattice(l);
}
static void test_flatten_spin_triples_element_count(void) {
    SpinLattice *l = initialize_spin_lattice(2, 2, 2, "all-up");
    ASSERT_TRUE(l != NULL);
    long need = engine_adapter_flatten_spin(l, NULL, 0);
    ASSERT_EQ_INT(need, 2 * 2 * 2 * 3);
    float buf[24];
    long wrote = engine_adapter_flatten_spin(l, buf, 24);
    ASSERT_EQ_INT(wrote, 24);
    free_spin_lattice(l);
}
static void test_flatten_ising_row_major_ordering(void) {
    IsingLattice *l = initialize_ising_lattice(2, 2, 2, "all-up");
    ASSERT_TRUE(l != NULL);
    /* Put distinctive values so ordering is testable. */
    l->spins[0][0][0] = -1;
    l->spins[1][1][1] = -1;
    float buf[8];
    ASSERT_EQ_INT(engine_adapter_flatten_ising(l, buf, 8), 8);
    /* x=0,y=0,z=0 is index 0; x=1,y=1,z=1 is index 7 (row-major). */
    ASSERT_NEAR(buf[0], -1.0f, 1e-12);
    ASSERT_NEAR(buf[7], -1.0f, 1e-12);
    ASSERT_NEAR(buf[3], +1.0f, 1e-12); /* x=0,y=1,z=1 */
    free_ising_lattice(l);
}
int main(void) {
    TEST_RUN(test_disabled_mode_reports_not_available);
    TEST_RUN(test_build_version_always_non_null);
    TEST_RUN(test_flatten_ising_size_probe_and_write);
    TEST_RUN(test_flatten_ising_rejects_small_buffer);
    TEST_RUN(test_flatten_ising_null_lattice_is_error);
    TEST_RUN(test_flatten_kitaev_all_down);
    TEST_RUN(test_flatten_spin_triples_element_count);
    TEST_RUN(test_flatten_ising_row_major_ordering);
    TEST_SUMMARY();
}