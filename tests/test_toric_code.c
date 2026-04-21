/*
 * tests/test_toric_code.c
 *
 * Verifies the data-qubit toric-code model in src/toric_code.c.
 *
 *   - single X error flags exactly 2 plaquette syndromes (adjacent ones)
 *   - single Z error flags exactly 2 vertex syndromes (adjacent ones)
 *   - two X errors on a shared plaquette cancel syndromes on that plaquette
 *   - greedy decoder clears all syndromes for low-weight error patterns
 *   - non-contractible loop of X errors creates a logical error, detected by
 *     toric_code_has_logical_error
 */
#include "harness.h"
#include "toric_code.h"
static int count_ones(const int *arr, int n) {
    int c = 0; for (int i = 0; i < n; i++) c += arr[i]; return c;
}
/* A single X error on a horizontal link flags exactly the two plaquettes
 * that share that link. */
static void test_single_x_error_flags_two_plaquettes(void) {
    ToricCode *c = initialize_toric_code(5, 5);
    ASSERT_TRUE(c != NULL);
    int link = toric_code_link_index(c, 2, 2, 0); /* horizontal link at (2,2) */
    toric_code_apply_x_error(c, link);
    int Ls = c->size_x * c->size_y;
    ASSERT_EQ_INT(count_ones(c->plaquette_syndrome, Ls), 2);
    ASSERT_EQ_INT(count_ones(c->vertex_syndrome, Ls), 0);
    free_toric_code(c);
}
/* A single Z error on a vertical link flags exactly the two vertices
 * that share that link. */
static void test_single_z_error_flags_two_vertices(void) {
    ToricCode *c = initialize_toric_code(5, 5);
    ASSERT_TRUE(c != NULL);
    int link = toric_code_link_index(c, 2, 2, 1);
    toric_code_apply_z_error(c, link);
    int Ls = c->size_x * c->size_y;
    ASSERT_EQ_INT(count_ones(c->vertex_syndrome, Ls), 2);
    ASSERT_EQ_INT(count_ones(c->plaquette_syndrome, Ls), 0);
    free_toric_code(c);
}
/* Two X errors on horizontal links sharing plaquette (2,2) cancel that
 * plaquette's syndrome. Link(2,2,0) is shared by plaq(2,2) and plaq(2,1);
 * link(2,3,0) is shared by plaq(2,3) and plaq(2,2). Net: plaq(2,2) flipped
 * twice, plaq(2,1) and plaq(2,3) each flipped once -> 2 flagged. */
static void test_x_errors_compose_in_gf2(void) {
    ToricCode *c = initialize_toric_code(5, 5);
    ASSERT_TRUE(c != NULL);
    int a = toric_code_link_index(c, 2, 2, 0);
    int b = toric_code_link_index(c, 2, 3, 0);
    toric_code_apply_x_error(c, a);
    toric_code_apply_x_error(c, b);
    int Ls = c->size_x * c->size_y;
    ASSERT_EQ_INT(count_ones(c->plaquette_syndrome, Ls), 2);
    ASSERT_EQ_INT(count_ones(c->vertex_syndrome, Ls), 0);
    free_toric_code(c);
}
/* Greedy decoder clears syndromes for low-weight random error patterns
 * in both X and Z channels. */
static void test_greedy_decoder_clears_syndromes_low_rate(void) {
    srand(42);
    for (int trial = 0; trial < 10; trial++) {
        ToricCode *c = initialize_toric_code(7, 7);
        ASSERT_TRUE(c != NULL);
        apply_random_errors(c, 0.03);
        int rc = toric_code_decode_greedy(c);
        ASSERT_EQ_INT(rc, 0);
        int Ls = c->size_x * c->size_y;
        ASSERT_EQ_INT(count_ones(c->vertex_syndrome, Ls), 0);
        ASSERT_EQ_INT(count_ones(c->plaquette_syndrome, Ls), 0);
        free_toric_code(c);
    }
}
/* A non-contractible X loop: all vertical primal edges in a row y=y0.
 * These form a dual horizontal cycle (wraps around the torus on the dual
 * lattice). No plaquette syndromes (every plaq sees 2 of these links),
 * but toric_code_has_logical_error must detect it. */
static void test_non_contractible_loop_is_logical_error(void) {
    ToricCode *c = initialize_toric_code(5, 5);
    ASSERT_TRUE(c != NULL);
    int y0 = 2;
    for (int x = 0; x < c->size_x; x++) {
        int link = toric_code_link_index(c, x, y0, 1);
        toric_code_apply_x_error(c, link);
    }
    int Ls = c->size_x * c->size_y;
    ASSERT_EQ_INT(count_ones(c->plaquette_syndrome, Ls), 0);
    ASSERT_EQ_INT(count_ones(c->vertex_syndrome, Ls), 0);
    ASSERT_EQ_INT(toric_code_has_logical_error(c), 1);
    free_toric_code(c);
}
/* A contractible X loop: the 4 primal edges adjacent to a single VERTEX.
 * (The 4 edges of a plaquette's boundary would instead flag 4 outer
 * plaquettes; only the vertex-boundary pattern is stabilizer-trivial.) */
static void test_contractible_loop_is_not_logical(void) {
    ToricCode *c = initialize_toric_code(5, 5);
    ASSERT_TRUE(c != NULL);
    int links[4];
    toric_code_vertex_links(c, 2, 2, links);
    for (int i = 0; i < 4; i++) toric_code_apply_x_error(c, links[i]);
    int Ls = c->size_x * c->size_y;
    ASSERT_EQ_INT(count_ones(c->plaquette_syndrome, Ls), 0);
    ASSERT_EQ_INT(count_ones(c->vertex_syndrome, Ls), 0);
    ASSERT_EQ_INT(toric_code_has_logical_error(c), 0);
    free_toric_code(c);
}
/* calculate_ground_state_degeneracy = 4 on torus (2 logical qubits). */
static void test_ground_state_degeneracy(void) {
    ToricCode *c = initialize_toric_code(3, 3);
    ASSERT_EQ_INT(calculate_ground_state_degeneracy(c), 4);
    free_toric_code(c);
}
/* perform_error_correction is the legacy alias for decode_greedy. */
static void test_perform_error_correction_clears_low_rate_errors(void) {
    srand(99);
    ToricCode *c = initialize_toric_code(5, 5);
    ASSERT_TRUE(c != NULL);
    apply_random_errors(c, 0.02);
    ErrorSyndrome *s = measure_error_syndrome(c);
    ASSERT_TRUE(s != NULL);
    int rc = perform_error_correction(c, s);
    ASSERT_EQ_INT(rc, 0);
    int Ls = c->size_x * c->size_y;
    int res = 0;
    for (int i = 0; i < Ls; i++) res += c->vertex_syndrome[i] + c->plaquette_syndrome[i];
    ASSERT_EQ_INT(res, 0);
    free_error_syndrome(s);
    free_toric_code(c);
}
/* Separate X-channel and Z-channel syndrome extractors. */
static void test_measure_x_and_z_syndromes_separately(void) {
    ToricCode *c = initialize_toric_code(5, 5);
    ASSERT_TRUE(c != NULL);
    toric_code_apply_x_error(c, toric_code_link_index(c, 1, 1, 0));
    toric_code_apply_z_error(c, toric_code_link_index(c, 2, 2, 1));
    ErrorSyndrome *sx = toric_code_measure_x_syndrome(c);
    ErrorSyndrome *sz = toric_code_measure_z_syndrome(c);
    ASSERT_EQ_INT(sx->num_errors, 2);
    ASSERT_EQ_INT(sz->num_errors, 2);
    ASSERT_EQ_INT(sx->error_type, 0);
    ASSERT_EQ_INT(sz->error_type, 1);
    free_error_syndrome(sx);
    free_error_syndrome(sz);
    free_toric_code(c);
}
/* is_ground_state returns 1 on a clean code, 0 when errors flag any stab. */
static void test_is_ground_state_true_when_clean(void) {
    ToricCode *c = initialize_toric_code(3, 3);
    ASSERT_TRUE(c != NULL);
    ASSERT_EQ_INT(is_ground_state(c), 1);
    toric_code_apply_x_error(c, toric_code_link_index(c, 1, 1, 0));
    ASSERT_EQ_INT(is_ground_state(c), 0);
    free_toric_code(c);
}
/* calculate_stabilizers reads from an attached KitaevLattice and populates
 * the toric code's x_errors from negative spins. */
static void test_calculate_stabilizers_reads_kitaev_lattice(void) {
    KitaevLattice *lat = initialize_kitaev_lattice(4, 4, 2, 1.0, 1.0, 1.0, "all-down");
    ToricCode *c = initialize_toric_code(3, 3);
    ASSERT_TRUE(lat && c);
    calculate_stabilizers(c, lat);
    /* All spins -1 -> all horizontal-link x_errors set to 1. */
    for (int x = 0; x < c->size_x; x++) {
        for (int y = 0; y < c->size_y; y++) {
            int k = toric_code_link_index(c, x, y, 0);
            ASSERT_EQ_INT(c->x_errors[k], 1);
        }
    }
    free_toric_code(c);
    free_kitaev_lattice(lat);
}
/* map_toric_code_to_lattice writes x_errors back as ±1 spin values. */
static void test_map_toric_code_to_lattice_writes_spins(void) {
    KitaevLattice *lat = initialize_kitaev_lattice(4, 4, 2, 1.0, 1.0, 1.0, "all-up");
    ToricCode *c = initialize_toric_code(3, 3);
    ASSERT_TRUE(lat && c);
    toric_code_apply_x_error(c, toric_code_link_index(c, 1, 1, 0));
    map_toric_code_to_lattice(c, lat);
    ASSERT_EQ_INT(lat->spins[1][1][0], -1);
    /* Unaffected horizontal link at (0,0) should remain +1. */
    ASSERT_EQ_INT(lat->spins[0][0][0],  1);
    free_toric_code(c);
    free_kitaev_lattice(lat);
}
/* Named correction API is functionally identical to the error API but
 * reads better in decoder code. */
static void test_correction_api_mirrors_error_api(void) {
    ToricCode *c = initialize_toric_code(3, 3);
    ASSERT_TRUE(c != NULL);
    int link = toric_code_link_index(c, 1, 1, 0);
    toric_code_apply_x_correction(c, link);
    ASSERT_EQ_INT(c->x_errors[link], 1);
    toric_code_apply_x_correction(c, link); /* flip back */
    ASSERT_EQ_INT(c->x_errors[link], 0);
    toric_code_apply_z_correction(c, toric_code_link_index(c, 2, 2, 1));
    ASSERT_EQ_INT(c->z_errors[toric_code_link_index(c, 2, 2, 1)], 1);
    free_toric_code(c);
}
/* Vertex and plaquette link accessors return 4 distinct link indices. */
static void test_vertex_links_are_distinct(void) {
    ToricCode *c = initialize_toric_code(4, 4);
    int out[4];
    toric_code_vertex_links(c, 1, 1, out);
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++)
            ASSERT_TRUE(out[i] != out[j]);
    toric_code_plaquette_links(c, 1, 1, out);
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++)
            ASSERT_TRUE(out[i] != out[j]);
    free_toric_code(c);
}
int main(void) {
    TEST_RUN(test_single_x_error_flags_two_plaquettes);
    TEST_RUN(test_single_z_error_flags_two_vertices);
    TEST_RUN(test_x_errors_compose_in_gf2);
    TEST_RUN(test_greedy_decoder_clears_syndromes_low_rate);
    TEST_RUN(test_non_contractible_loop_is_logical_error);
    TEST_RUN(test_contractible_loop_is_not_logical);
    TEST_RUN(test_ground_state_degeneracy);
    TEST_RUN(test_perform_error_correction_clears_low_rate_errors);
    TEST_RUN(test_measure_x_and_z_syndromes_separately);
    TEST_RUN(test_is_ground_state_true_when_clean);
    TEST_RUN(test_calculate_stabilizers_reads_kitaev_lattice);
    TEST_RUN(test_map_toric_code_to_lattice_writes_spins);
    TEST_RUN(test_correction_api_mirrors_error_api);
    TEST_RUN(test_vertex_links_are_distinct);
    TEST_SUMMARY();
}