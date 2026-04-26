/*
 * tests/test_physics_loss.c
 *
 * Covers the five physics-informed PDE residual losses in
 * src/physics_loss.c. v0.4 is a smoke-level check: each loss returns
 * a finite number for a reasonable lattice. v0.5 pillar P2.7 upgrades
 * the numerics (SIREN, Fourier features, variational form); tests
 * there will assert actual residual decay.
 */
#include "harness.h"
#include "physics_loss.h"
static double run_loss(const char *type) {
    IsingLattice  *ising  = initialize_ising_lattice(4, 4, 4, "random");
    KitaevLattice *kitaev = initialize_kitaev_lattice(4, 4, 4, 1.0, 1.0, -1.0, "random");
    SpinLattice   *spin   = initialize_spin_lattice(4, 4, 4, "random");
    double L = compute_physics_loss(-1.0, 0.5, 0.2,
                                    ising, kitaev, spin,
                                    0.1 /*dt*/, 0.1 /*dx*/, type);
    free_ising_lattice(ising);
    free_kitaev_lattice(kitaev);
    free_spin_lattice(spin);
    return L;
}
static void test_heat_loss_finite(void) {
    double L = run_loss("heat");
    ASSERT_TRUE(L == L); /* not NaN */
    ASSERT_TRUE(L < 1e300 && L > -1e300);
}
static void test_schrodinger_loss_finite(void) {
    double L = run_loss("schrodinger");
    ASSERT_TRUE(L == L);
    ASSERT_TRUE(L < 1e300 && L > -1e300);
}
static void test_maxwell_loss_finite(void) {
    double L = run_loss("maxwell");
    ASSERT_TRUE(L == L);
    ASSERT_TRUE(L < 1e300 && L > -1e300);
}
static void test_navier_stokes_loss_finite(void) {
    double L = run_loss("navier_stokes");
    ASSERT_TRUE(L == L);
    ASSERT_TRUE(L < 1e300 && L > -1e300);
}
static void test_wave_loss_finite(void) {
    double L = run_loss("wave");
    ASSERT_TRUE(L == L);
    ASSERT_TRUE(L < 1e300 && L > -1e300);
}
/* Laplacian of a constant field should vanish. */
static void test_laplacian_of_constant_is_zero(void) {
    IsingLattice *l = initialize_ising_lattice(5, 5, 5, "all-up");
    ASSERT_TRUE(l != NULL);
    double lap = laplacian_3d(l->spins, 2, 2, 2,
                              l->size_x, l->size_y, l->size_z, 0.1);
    ASSERT_NEAR(lap, 0.0, 1e-12);
    free_ising_lattice(l);
}
/* Micromagnetic loss vanishes for a uniform configuration aligned with the
 * applied field (no exchange cost, no anisotropy cost, optimal Zeeman). */
static void test_micromagnetic_loss_minimum_at_aligned_state(void) {
    SpinLattice *s = initialize_spin_lattice(4, 4, 4, "all-up");
    ASSERT_TRUE(s != NULL);
    /* "all-up" sets sx=sy=sz=+0.5; force the configuration to (0,0,1) along z */
    for (int x = 0; x < s->size_x; x++)
    for (int y = 0; y < s->size_y; y++)
    for (int z = 0; z < s->size_z; z++) {
        s->spins[x][y][z].sx = 0.0;
        s->spins[x][y][z].sy = 0.0;
        s->spins[x][y][z].sz = 1.0;
    }
    /* Exchange = 0 (uniform), anisotropy = 0 (m||z), Zeeman = -μ₀·1·B_z */
    double L = micromagnetic_loss(s, 1.0, 0.1, 0.0, 0.0, 0.5);
    /* Exchange and anisotropy contributions are zero; only Zeeman remains:
     * E/N = -μ₀ · 0.5 ≈ -6.28e-7 */
    ASSERT_TRUE(L < 0.0);
    ASSERT_TRUE(L > -1e-5);
    free_spin_lattice(s);
}
/* Hard-constraint projection: rescaled vectors all have unit norm. */
static void test_project_to_unit_sphere(void) {
    SpinLattice *s = initialize_spin_lattice(3, 3, 3, "random");
    ASSERT_TRUE(s != NULL);
    /* Force a non-unit configuration */
    for (int x = 0; x < 3; x++)
    for (int y = 0; y < 3; y++)
    for (int z = 0; z < 3; z++) {
        s->spins[x][y][z].sx = 1.0;
        s->spins[x][y][z].sy = 2.0;
        s->spins[x][y][z].sz = 2.0;
    }
    double dev = project_spin_lattice_to_unit_sphere(s);
    ASSERT_TRUE(dev > 0.0);  /* before-projection deviation is large */
    for (int x = 0; x < 3; x++)
    for (int y = 0; y < 3; y++)
    for (int z = 0; z < 3; z++) {
        Spin m = s->spins[x][y][z];
        double n = sqrt(m.sx*m.sx + m.sy*m.sy + m.sz*m.sz);
        ASSERT_NEAR(n, 1.0, 1e-12);
    }
    free_spin_lattice(s);
}
/* Fourier features at a single coordinate produce alternating sin/cos pairs. */
static void test_fourier_features_basic_shape(void) {
    double coord[1] = {0.25};
    int n_freqs = 3;
    double out[6] = {0};
    fourier_features(coord, 1, n_freqs, out);
    /* k=0: freq=2π, sin(2π·0.25)=1, cos(2π·0.25)=0 */
    ASSERT_NEAR(out[0],  1.0, 1e-12);
    ASSERT_NEAR(out[1],  0.0, 1e-12);
    /* k=1: freq=4π, sin(4π·0.25)=0, cos(4π·0.25)=-1 */
    ASSERT_NEAR(out[2],  0.0, 1e-12);
    ASSERT_NEAR(out[3], -1.0, 1e-12);
    /* k=2: freq=8π, sin(8π·0.25)=0, cos(8π·0.25)=1 */
    ASSERT_NEAR(out[4],  0.0, 1e-12);
    ASSERT_NEAR(out[5],  1.0, 1e-12);
}
int main(void) {
    srand(13);
    TEST_RUN(test_heat_loss_finite);
    TEST_RUN(test_schrodinger_loss_finite);
    TEST_RUN(test_maxwell_loss_finite);
    TEST_RUN(test_navier_stokes_loss_finite);
    TEST_RUN(test_wave_loss_finite);
    TEST_RUN(test_laplacian_of_constant_is_zero);
    TEST_RUN(test_micromagnetic_loss_minimum_at_aligned_state);
    TEST_RUN(test_project_to_unit_sphere);
    TEST_RUN(test_fourier_features_basic_shape);
    TEST_SUMMARY();
}