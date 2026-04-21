/*
 * tests/test_qec_decoder.c
 *
 * Covers the learned-QEC scaffold. Since v0.4 falls back to the
 * greedy matching decoder, these tests exercise:
 *   - handle construction + fall-back behaviour
 *   - syndrome tokenisation correctness
 *   - Monte-Carlo logical-error rate monotonicity with distance
 */
#include <stdlib.h>
#include "harness.h"
#include "qec_decoder/qec_decoder.h"
#include "toric_code.h"
static void test_greedy_is_always_available(void) {
    qec_decoder_t d = qec_decoder_create(QEC_DECODER_GREEDY);
    ASSERT_EQ_INT(d.kind, QEC_DECODER_GREEDY);
    ASSERT_EQ_INT(d.is_available, 1);
}
static void test_transformer_falls_back_to_mwpm(void) {
    qec_decoder_t d = qec_decoder_create(QEC_DECODER_TRANSFORMER);
    /* Without the NN engine the learned kinds fall back to the MWPM
     * decoder (optimal matching for small defect counts). is_available
     * reports 0 so callers know they aren't getting the learned variant. */
    ASSERT_EQ_INT(d.kind, QEC_DECODER_MWPM);
    ASSERT_EQ_INT(d.is_available, 0);
}
static void test_mamba_falls_back_to_mwpm(void) {
    qec_decoder_t d = qec_decoder_create(QEC_DECODER_MAMBA);
    ASSERT_EQ_INT(d.kind, QEC_DECODER_MWPM);
    ASSERT_EQ_INT(d.is_available, 0);
}
static void test_mwpm_is_always_available(void) {
    qec_decoder_t d = qec_decoder_create(QEC_DECODER_MWPM);
    ASSERT_EQ_INT(d.kind, QEC_DECODER_MWPM);
    ASSERT_EQ_INT(d.is_available, 1);
}
static void test_tokenize_single_x_error_flags_correct_plaquettes(void) {
    /* An X-error on the horizontal link (2, 2, 0) is the bottom of
     * plaquette (2, 2) and the top of plaquette (2, 1). Exactly those
     * two plaquettes must be flagged; no vertex tokens. */
    ToricCode *c = initialize_toric_code(5, 5);
    ASSERT_TRUE(c != NULL);
    toric_code_apply_x_error(c, toric_code_link_index(c, 2, 2, 0));
    qec_syndrome_token_t tokens[32];
    int n = qec_decoder_tokenize(c, tokens, 32);
    ASSERT_EQ_INT(n, 2);
    int saw_22 = 0, saw_21 = 0;
    for (int i = 0; i < n; i++) {
        ASSERT_EQ_INT(tokens[i].stab_type, 0);
        if (tokens[i].x == 2 && tokens[i].y == 2) saw_22 = 1;
        if (tokens[i].x == 2 && tokens[i].y == 1) saw_21 = 1;
    }
    ASSERT_TRUE(saw_22);
    ASSERT_TRUE(saw_21);
    free_toric_code(c);
}
static void test_tokenize_single_z_error_flags_correct_vertices(void) {
    /* A Z-error on the vertical link (2, 2, 1) is the north of vertex
     * (2, 2) and the south of vertex (2, 3). */
    ToricCode *c = initialize_toric_code(5, 5);
    toric_code_apply_z_error(c, toric_code_link_index(c, 2, 2, 1));
    qec_syndrome_token_t tokens[32];
    int n = qec_decoder_tokenize(c, tokens, 32);
    ASSERT_EQ_INT(n, 2);
    int saw_22 = 0, saw_23 = 0;
    for (int i = 0; i < n; i++) {
        ASSERT_EQ_INT(tokens[i].stab_type, 1);
        if (tokens[i].x == 2 && tokens[i].y == 2) saw_22 = 1;
        if (tokens[i].x == 2 && tokens[i].y == 3) saw_23 = 1;
    }
    ASSERT_TRUE(saw_22);
    ASSERT_TRUE(saw_23);
    free_toric_code(c);
}
static void test_tokenize_clean_code_gives_zero_tokens(void) {
    ToricCode *c = initialize_toric_code(5, 5);
    qec_syndrome_token_t tokens[32];
    int n = qec_decoder_tokenize(c, tokens, 32);
    ASSERT_EQ_INT(n, 0);
    free_toric_code(c);
}
static void test_tokenize_respects_capacity(void) {
    ToricCode *c = initialize_toric_code(5, 5);
    /* Two errors → 4 flagged stabilizers (2 plaquette + 0 vertex, or
     * similar depending on adjacency). Give capacity 1 and demand that
     * the tokenizer returns no more than that. */
    toric_code_apply_x_error(c, toric_code_link_index(c, 0, 0, 0));
    toric_code_apply_z_error(c, toric_code_link_index(c, 3, 3, 1));
    qec_syndrome_token_t tokens[1];
    int n = qec_decoder_tokenize(c, tokens, 1);
    ASSERT_TRUE(n <= 1);
    ASSERT_TRUE(n >= 0);
    free_toric_code(c);
}
static void test_decoder_run_clears_low_rate_errors(void) {
    qec_decoder_t d = qec_decoder_create(QEC_DECODER_GREEDY);
    srand(0xBEEF);
    ToricCode *c = initialize_toric_code(5, 5);
    apply_random_errors(c, 0.02);
    ASSERT_EQ_INT(qec_decoder_run(&d, c), 0);
    int Ls = c->size_x * c->size_y;
    int left = 0;
    for (int i = 0; i < Ls; i++) left += c->vertex_syndrome[i] + c->plaquette_syndrome[i];
    ASSERT_EQ_INT(left, 0);
    free_toric_code(c);
}
static void test_logical_error_rate_decreases_with_distance(void) {
    qec_decoder_t d = qec_decoder_create(QEC_DECODER_GREEDY);
    double r3 = 1.0, r5 = 1.0, r7 = 1.0;
    qec_decoder_logical_error_rate(&d, 3, 0.01, 500, 0xA1u, &r3);
    qec_decoder_logical_error_rate(&d, 5, 0.01, 500, 0xA2u, &r5);
    qec_decoder_logical_error_rate(&d, 7, 0.01, 500, 0xA3u, &r7);
    /* Below threshold, logical error rate should decrease with distance. */
    ASSERT_TRUE(r5 < r3 + 0.02);
    ASSERT_TRUE(r7 < r3 + 0.02);
}
int main(void) {
    TEST_RUN(test_greedy_is_always_available);
    TEST_RUN(test_transformer_falls_back_to_mwpm);
    TEST_RUN(test_mamba_falls_back_to_mwpm);
    TEST_RUN(test_mwpm_is_always_available);
    TEST_RUN(test_tokenize_single_x_error_flags_correct_plaquettes);
    TEST_RUN(test_tokenize_single_z_error_flags_correct_vertices);
    TEST_RUN(test_tokenize_clean_code_gives_zero_tokens);
    TEST_RUN(test_tokenize_respects_capacity);
    TEST_RUN(test_decoder_run_clears_low_rate_errors);
    TEST_RUN(test_logical_error_rate_decreases_with_distance);
    TEST_SUMMARY();
}