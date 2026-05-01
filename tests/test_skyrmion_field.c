/*
 * tests/test_skyrmion_field.c
 *
 * Smoke + correctness tests for the Belavin-Polyakov skyrmion field
 * generator and the Berg-Lüscher lattice charge integrator.
 *
 * Tests:
 *   1. centre is the north pole (m_z = 1)
 *   2. far field is the south pole (m_z → -1 as r → ∞)
 *   3. unit-vector constraint: |m(r)| = 1 everywhere
 *   4. Q = +1 skyrmion integrates to 1 within lattice tolerance
 *   5. Q = +2 skyrmion integrates to 2
 *   6. Q = -1 antiskyrmion integrates to -1
 *   7. Q = 3 skyrmion integrates to 3
 *   8. helicity sign-flip swaps m_y → -m_y (Néel ↔ Néel-flipped)
 *
 * Charge integration tolerance: 5% on Lx = Ly = 64, R = 8.  The
 * Berg-Lüscher discretisation converges as 1/L²; for tighter
 * tolerances bump L.
 */

#include "harness.h"
#include "skyrmion/skyrmion_field.h"

#include <math.h>
#include <stdlib.h>

static double *alloc_field(int Lx, int Ly) {
    return (double *)calloc((size_t)3 * Lx * Ly, sizeof(double));
}

static double mag2(const double *m, int ix, int iy, int Ly) {
    const double *p = &m[3 * ((size_t)ix * Ly + iy)];
    return p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
}

static void test_centre_is_north_pole(void) {
    int Lx = 32, Ly = 32;
    double *m = alloc_field(Lx, Ly);
    skyrmion_field_params_t p = {
        .Q = 1, .R = 4.0, .cx = 16.0, .cy = 16.0, .helicity = 0.0
    };
    skyrmion_field_compute(&p, Lx, Ly, m);
    const double *centre = &m[3 * (16 * Ly + 16)];
    /* (cx, cy) = (16, 16) → r = 0 → m = (0, 0, 1). */
    ASSERT_NEAR(centre[0], 0.0, 1e-12);
    ASSERT_NEAR(centre[1], 0.0, 1e-12);
    ASSERT_NEAR(centre[2], 1.0, 1e-12);
    free(m);
}

static void test_far_field_is_south_pole(void) {
    int Lx = 64, Ly = 64;
    double *m = alloc_field(Lx, Ly);
    skyrmion_field_params_t p = {
        .Q = 1, .R = 2.0, .cx = 32.0, .cy = 32.0, .helicity = 0.0
    };
    skyrmion_field_compute(&p, Lx, Ly, m);
    /* Corner of a 64×64 lattice with skyrmion of R=2 at centre:
     * r = 32√2 ≈ 45.25, so m_z ≈ (4 - 2048) / (4 + 2048) ≈ -0.996. */
    const double *corner = &m[3 * (0 * Ly + 0)];
    ASSERT_TRUE(corner[2] < -0.99);
    free(m);
}

static void test_unit_vector_constraint(void) {
    int Lx = 32, Ly = 32;
    double *m = alloc_field(Lx, Ly);
    skyrmion_field_params_t p = {
        .Q = 1, .R = 6.0, .cx = 16.0, .cy = 16.0, .helicity = 0.0
    };
    skyrmion_field_compute(&p, Lx, Ly, m);
    double max_err = 0.0;
    for (int ix = 0; ix < Lx; ix++) {
        for (int iy = 0; iy < Ly; iy++) {
            double err = fabs(mag2(m, ix, iy, Ly) - 1.0);
            if (err > max_err) max_err = err;
        }
    }
    ASSERT_NEAR(max_err, 0.0, 1e-12);
    free(m);
}

static void test_charge_Q1(void) {
    /* Tolerance comment: the Berg-Lüscher integrator on OBC has a
     * finite-volume cap-correction of O((R/L)²); we choose L = 128,
     * R = 6 so that the missing-cap fraction is ~0.05 % and the
     * residual lattice discretisation dominates at ~0.5 %. */
    int Lx = 128, Ly = 128;
    double *m = alloc_field(Lx, Ly);
    skyrmion_field_params_t p = {
        .Q = 1, .R = 6.0, .cx = 64.0, .cy = 64.0, .helicity = 0.0
    };
    skyrmion_field_compute(&p, Lx, Ly, m);
    double Q = skyrmion_topological_charge(Lx, Ly, m);
    ASSERT_NEAR(Q, 1.0, 0.01);
    free(m);
}

static void test_charge_Q2(void) {
    int Lx = 128, Ly = 128;
    double *m = alloc_field(Lx, Ly);
    skyrmion_field_params_t p = {
        .Q = 2, .R = 6.0, .cx = 64.0, .cy = 64.0, .helicity = 0.0
    };
    skyrmion_field_compute(&p, Lx, Ly, m);
    double Q = skyrmion_topological_charge(Lx, Ly, m);
    ASSERT_NEAR(Q, 2.0, 0.02);
    free(m);
}

static void test_charge_Q_neg1_antiskyrmion(void) {
    int Lx = 128, Ly = 128;
    double *m = alloc_field(Lx, Ly);
    skyrmion_field_params_t p = {
        .Q = -1, .R = 6.0, .cx = 64.0, .cy = 64.0, .helicity = 0.0
    };
    skyrmion_field_compute(&p, Lx, Ly, m);
    double Q = skyrmion_topological_charge(Lx, Ly, m);
    ASSERT_NEAR(Q, -1.0, 0.01);
    free(m);
}

static void test_charge_Q3(void) {
    int Lx = 128, Ly = 128;
    double *m = alloc_field(Lx, Ly);
    skyrmion_field_params_t p = {
        .Q = 3, .R = 6.0, .cx = 64.0, .cy = 64.0, .helicity = 0.0
    };
    skyrmion_field_compute(&p, Lx, Ly, m);
    double Q = skyrmion_topological_charge(Lx, Ly, m);
    ASSERT_NEAR(Q, 3.0, 0.03);
    free(m);
}

static void test_helicity_flips_my(void) {
    int Lx = 32, Ly = 32;
    double *m_neel = alloc_field(Lx, Ly);
    double *m_antineel = alloc_field(Lx, Ly);
    skyrmion_field_params_t neel = {
        .Q = 1, .R = 4.0, .cx = 16.0, .cy = 16.0, .helicity = 0.0
    };
    skyrmion_field_params_t antineel = {
        .Q = 1, .R = 4.0, .cx = 16.0, .cy = 16.0, .helicity = M_PI
    };
    skyrmion_field_compute(&neel, Lx, Ly, m_neel);
    skyrmion_field_compute(&antineel, Lx, Ly, m_antineel);
    /* Anti-Néel is the Néel field with (m_x, m_y) → (-m_x, -m_y).
     * m_z is unchanged. */
    double max_err_xy = 0.0, max_err_z = 0.0;
    for (int ix = 0; ix < Lx; ix++) {
        for (int iy = 0; iy < Ly; iy++) {
            const double *a = &m_neel[3 * (ix * Ly + iy)];
            const double *b = &m_antineel[3 * (ix * Ly + iy)];
            double ex = fabs(a[0] + b[0]);
            double ey = fabs(a[1] + b[1]);
            double ez = fabs(a[2] - b[2]);
            if (ex > max_err_xy) max_err_xy = ex;
            if (ey > max_err_xy) max_err_xy = ey;
            if (ez > max_err_z) max_err_z = ez;
        }
    }
    ASSERT_NEAR(max_err_xy, 0.0, 1e-12);
    ASSERT_NEAR(max_err_z, 0.0, 1e-12);
    free(m_neel);
    free(m_antineel);
}

int main(void) {
    TEST_RUN(test_centre_is_north_pole);
    TEST_RUN(test_far_field_is_south_pole);
    TEST_RUN(test_unit_vector_constraint);
    TEST_RUN(test_charge_Q1);
    TEST_RUN(test_charge_Q2);
    TEST_RUN(test_charge_Q_neg1_antiskyrmion);
    TEST_RUN(test_charge_Q3);
    TEST_RUN(test_helicity_flips_my);
    TEST_SUMMARY();
}
