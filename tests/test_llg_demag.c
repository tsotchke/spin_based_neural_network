/*
 * tests/test_llg_demag.c
 *
 * Demag field verification. The FFT-convolution path must agree with
 * a direct O(N²) real-space sum for the same kernel, and must give
 * the expected sign (demag opposes magnetisation) on simple
 * configurations.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "llg/demag.h"
static double direct_demag_z(const double *m, int Lx, int Ly, int sx, int sy) {
    /* Direct O(N²) sum of -Σ_{r'} G(r-r') m_z(r') using minimum-image
     * distances. Cross-check reference for the FFT path. */
    double acc = 0.0;
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++) {
        int dx = sx - x; if (dx < 0) dx += Lx; if (dx > Lx/2) dx -= Lx;
        int dy = sy - y; if (dy < 0) dy += Ly; if (dy > Ly/2) dy -= Ly;
        double r2 = (double)dx * dx + (double)dy * dy;
        if (r2 == 0.0) continue;
        double r = sqrt(r2);
        double G = 1.0 / (r2 * r);
        acc += G * m[3*(x * Ly + y) + 2];
    }
    return -acc;   /* H_dz = -Σ G · m_z */
}
static void test_demag_uniform_up_gives_negative_field(void) {
    /* Uniformly magnetised +z film: demag opposes, H_dz < 0 everywhere. */
    int Lx = 8, Ly = 8;
    llg_demag_2d_t *d = llg_demag_2d_create(Lx, Ly);
    ASSERT_TRUE(d != NULL);
    double *m = calloc((size_t)Lx * Ly * 3, sizeof(double));
    for (long k = 0; k < (long)Lx * Ly; k++) m[3*k + 2] = 1.0;
    double *Hdz = calloc((size_t)Lx * Ly, sizeof(double));
    ASSERT_EQ_INT(llg_demag_2d_apply_z(d, m, Hdz), 0);
    for (long k = 0; k < (long)Lx * Ly; k++) {
        ASSERT_TRUE(Hdz[k] < 0.0);    /* opposes magnetisation */
    }
    free(m); free(Hdz);
    llg_demag_2d_free(d);
}
static void test_demag_fft_matches_direct_sum(void) {
    /* FFT path and direct O(N²) sum must agree on every site. */
    int Lx = 8, Ly = 8;
    llg_demag_2d_t *d = llg_demag_2d_create(Lx, Ly);
    double *m = calloc((size_t)Lx * Ly * 3, sizeof(double));
    /* Random magnetisation pattern. */
    unsigned long long rng = 0x4242ULL;
    for (long k = 0; k < (long)Lx * Ly; k++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        m[3*k + 2] = (double)(rng >> 11) / 9007199254740992.0 - 0.5;
    }
    double *Hdz = calloc((size_t)Lx * Ly, sizeof(double));
    ASSERT_EQ_INT(llg_demag_2d_apply_z(d, m, Hdz), 0);
    double max_err = 0.0;
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++) {
        double H_ref = direct_demag_z(m, Lx, Ly, x, y);
        double diff = fabs(Hdz[x * Ly + y] - H_ref);
        if (diff > max_err) max_err = diff;
    }
    printf("# demag FFT vs direct-sum max error = %.3e\n", max_err);
    ASSERT_TRUE(max_err < 1e-8);
    free(m); free(Hdz);
    llg_demag_2d_free(d);
}
static void test_demag_zero_m_gives_zero_field(void) {
    int Lx = 16, Ly = 16;
    llg_demag_2d_t *d = llg_demag_2d_create(Lx, Ly);
    double *m = calloc((size_t)Lx * Ly * 3, sizeof(double));
    double *Hdz = calloc((size_t)Lx * Ly, sizeof(double));
    llg_demag_2d_apply_z(d, m, Hdz);
    for (long k = 0; k < (long)Lx * Ly; k++)
        ASSERT_NEAR(Hdz[k], 0.0, 1e-12);
    free(m); free(Hdz);
    llg_demag_2d_free(d);
}
int main(void) {
    TEST_RUN(test_demag_uniform_up_gives_negative_field);
    TEST_RUN(test_demag_fft_matches_direct_sum);
    TEST_RUN(test_demag_zero_m_gives_zero_field);
    TEST_SUMMARY();
}