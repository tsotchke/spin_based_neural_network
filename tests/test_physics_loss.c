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
int main(void) {
    srand(13);
    TEST_RUN(test_heat_loss_finite);
    TEST_RUN(test_schrodinger_loss_finite);
    TEST_RUN(test_maxwell_loss_finite);
    TEST_RUN(test_navier_stokes_loss_finite);
    TEST_RUN(test_wave_loss_finite);
    TEST_RUN(test_laplacian_of_constant_is_zero);
    TEST_SUMMARY();
}