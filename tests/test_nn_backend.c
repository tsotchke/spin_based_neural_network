/*
 * tests/test_nn_backend.c
 *
 * Covers the polymorphic NN-handle wrapper. Without an engine wired in,
 * creating a network with NN_BACKEND_ENGINE must transparently fall back
 * to legacy (with a diagnostic printed to stderr). The legacy path must
 * always succeed and round-trip through forward/train.
 */
#include <string.h>
#include "harness.h"
#include "nn_backend.h"
static void test_parse_legacy(void) {
    int ok = 0;
    ASSERT_EQ_INT(nn_backend_parse("legacy", &ok), NN_BACKEND_LEGACY);
    ASSERT_EQ_INT(ok, 1);
}
static void test_parse_engine(void) {
    int ok = 0;
    ASSERT_EQ_INT(nn_backend_parse("engine", &ok), NN_BACKEND_ENGINE);
    ASSERT_EQ_INT(ok, 1);
}
static void test_parse_case_insensitive(void) {
    int ok = 0;
    ASSERT_EQ_INT(nn_backend_parse("LEGACY", &ok), NN_BACKEND_LEGACY);
    ASSERT_EQ_INT(ok, 1);
    ASSERT_EQ_INT(nn_backend_parse("Engine", &ok), NN_BACKEND_ENGINE);
    ASSERT_EQ_INT(ok, 1);
}
static void test_parse_unknown_sets_ok_zero(void) {
    int ok = 1;
    nn_backend_parse("llama", &ok);
    ASSERT_EQ_INT(ok, 0);
}
static void test_parse_null_is_error(void) {
    int ok = 1;
    nn_backend_parse(NULL, &ok);
    ASSERT_EQ_INT(ok, 0);
}
static void test_backend_name_roundtrip(void) {
    ASSERT_TRUE(strcmp(nn_backend_name(NN_BACKEND_LEGACY), "legacy") == 0);
    ASSERT_TRUE(strcmp(nn_backend_name(NN_BACKEND_ENGINE), "engine") == 0);
}
static void test_legacy_create_and_free(void) {
    spin_nn_t *nn = spin_nn_create(NN_BACKEND_LEGACY, 8, 1, 4, 2, 0 /* ReLU */);
    ASSERT_TRUE(nn != NULL);
    ASSERT_EQ_INT(spin_nn_backend(nn), NN_BACKEND_LEGACY);
    ASSERT_TRUE(spin_nn_legacy_handle(nn) != NULL);
    spin_nn_free(nn);
}
static void test_legacy_forward_returns_pointer(void) {
    spin_nn_t *nn = spin_nn_create(NN_BACKEND_LEGACY, 4, 1, 4, 2, 0);
    ASSERT_TRUE(nn != NULL);
    double input[4] = {0.1, 0.2, 0.3, 0.4};
    double *out = spin_nn_forward(nn, input);
    ASSERT_TRUE(out != NULL);
    spin_nn_free(nn);
}
static void test_legacy_train_returns_zero(void) {
    spin_nn_t *nn = spin_nn_create(NN_BACKEND_LEGACY, 4, 1, 4, 2, 0);
    ASSERT_TRUE(nn != NULL);
    double input[4]  = {0.1, 0.2, 0.3, 0.4};
    double target[2] = {0.5, -0.5};
    ASSERT_EQ_INT(spin_nn_train(nn, input, target, 1e-3), 0);
    spin_nn_free(nn);
}
static void test_engine_backend_falls_back_to_legacy(void) {
    /* stderr gets a warning — we're not asserting on it here, just that
     * the network is actually created and usable as legacy. */
    spin_nn_t *nn = spin_nn_create(NN_BACKEND_ENGINE, 4, 1, 4, 2, 0);
    ASSERT_TRUE(nn != NULL);
    ASSERT_EQ_INT(spin_nn_backend(nn), NN_BACKEND_LEGACY);
    spin_nn_free(nn);
}
static void test_free_null_is_safe(void) {
    spin_nn_free(NULL);
    ASSERT_TRUE(1);
}
int main(void) {
    TEST_RUN(test_parse_legacy);
    TEST_RUN(test_parse_engine);
    TEST_RUN(test_parse_case_insensitive);
    TEST_RUN(test_parse_unknown_sets_ok_zero);
    TEST_RUN(test_parse_null_is_error);
    TEST_RUN(test_backend_name_roundtrip);
    TEST_RUN(test_legacy_create_and_free);
    TEST_RUN(test_legacy_forward_returns_pointer);
    TEST_RUN(test_legacy_train_returns_zero);
    TEST_RUN(test_engine_backend_falls_back_to_legacy);
    TEST_RUN(test_free_null_is_safe);
    TEST_SUMMARY();
}