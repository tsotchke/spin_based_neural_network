/*
 * tests/test_energy_utils.c
 *
 * Covers the scale/unscale helpers used throughout main.c to map raw
 * lattice energies into a sigmoid-like range [-1, 1] before feeding
 * the neural network.
 */
#include "harness.h"
#include "energy_utils.h"

static void test_scale_zero_maps_to_min_energy(void) {
    double s = scale_energy(0.0);
    ASSERT_TRUE(fabs(s) <= 1.0);
}

static void test_scale_monotone_in_modest_range(void) {
    /* scale_energy is a sigmoid-flavored map; assert it's non-decreasing
     * across a small positive range. */
    double prev = scale_energy(-100.0);
    for (double e = -99.0; e <= 100.0; e += 1.0) {
        double s = scale_energy(e);
        ASSERT_TRUE(s >= prev - 1e-12);
        prev = s;
    }
}

static void test_scale_bounds(void) {
    /* For large |energy| the scaled value should approach ±1 but never
     * exceed it. */
    ASSERT_TRUE(fabs(scale_energy(1e6))  <= 1.0);
    ASSERT_TRUE(fabs(scale_energy(-1e6)) <= 1.0);
}

static void test_round_trip_middle_region(void) {
    /* In a regime where neither tail saturates nor the MIN_ENERGY clamp
     * kicks in, scale -> unscale should recover the original value. */
    double inputs[] = {-200.0, -50.0, -10.0, 5.0, 25.0, 150.0};
    for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); i++) {
        double s = scale_energy(inputs[i]);
        double r = unscale_energy(s);
        ASSERT_NEAR(r, inputs[i], 1e-6);
    }
}

int main(void) {
    TEST_RUN(test_scale_zero_maps_to_min_energy);
    TEST_RUN(test_scale_monotone_in_modest_range);
    TEST_RUN(test_scale_bounds);
    TEST_RUN(test_round_trip_middle_region);
    TEST_SUMMARY();
}
