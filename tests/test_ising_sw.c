/*
 * tests/test_ising_sw.c
 *
 * Swendsen–Wang cluster updates for the 3D Ising model.
 *   (1) A ferromagnetic lattice at T = 0 (β → ∞) must stay ferromagnetic
 *       under SW: all-aligned is the fixed point of the algorithm up to
 *       a global spin flip.
 *   (2) At very high temperature (β → 0), the bond-activation
 *       probability p = 1 - exp(-2β) ≈ 0, so no bonds fuse and each
 *       site is its own cluster; the per-site flip probability is 1/2,
 *       so magnetisation should be ~0 after many sweeps.
 *   (3) Near the 3D Ising critical β_c ≈ 0.2216, equilibrium energy
 *       per site should be finite and different from the T=0 and
 *       T=∞ limits — a smoke test that the update is mixing but not
 *       trivial. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "ising_model.h"
static double magnetisation(const IsingLattice *L) {
    double m = 0.0;
    int n = 0;
    for (int x = 0; x < L->size_x; x++)
    for (int y = 0; y < L->size_y; y++)
    for (int z = 0; z < L->size_z; z++) { m += L->spins[x][y][z]; n++; }
    return m / (double)n;
}
static void test_sw_preserves_ferromagnet_at_zero_temperature(void) {
    /* All +1 spins; high β means p ≈ 1, so every ferromagnetic bond
     * is active and the entire lattice forms one cluster. That
     * cluster flips with prob 1/2, so m is either +1 or -1 — |m| = 1. */
    IsingLattice *L = initialize_ising_lattice(4, 4, 4, "up");
    srand(0x4242);
    for (int s = 0; s < 50; s++) ising_swendsen_wang_step(L, 10.0);
    double m = magnetisation(L);
    printf("# SW T=0 |m| = %.3f (expected 1)\n", fabs(m));
    ASSERT_NEAR(fabs(m), 1.0, 1e-10);
    free_ising_lattice(L);
}
static void test_sw_demagnetises_at_high_temperature(void) {
    /* β = 0.01, p ≈ 0.02. Most bonds inactive; clusters are tiny and
     * each flips independently. After enough sweeps, ⟨|m|⟩ should
     * drop close to zero for a moderately-sized lattice. */
    IsingLattice *L = initialize_ising_lattice(6, 6, 6, "up");
    srand(0xDEAD);
    for (int s = 0; s < 100; s++) ising_swendsen_wang_step(L, 0.01);
    /* Average |m| over a further batch of 50 sweeps to smooth. */
    double avg = 0.0;
    for (int s = 0; s < 50; s++) {
        ising_swendsen_wang_step(L, 0.01);
        avg += fabs(magnetisation(L));
    }
    avg /= 50.0;
    printf("# SW T=∞: ⟨|m|⟩ = %.3f (expected near 0)\n", avg);
    /* For N=216 sites at β=0.01 uncorrelated m ~ 1/√N ≈ 0.07. */
    ASSERT_TRUE(avg < 0.2);
    free_ising_lattice(L);
}
static void test_sw_near_critical_temperature(void) {
    /* 3D Ising T_c: β_c ≈ 0.2216. Energy per site at T_c is roughly
     * -0.99 (from Talapov–Blöte Monte Carlo). Our smaller lattice
     * (6×6×6) gives finite-size shifts but the qualitative
     * ordering — i.e., not stuck at either limit — must hold. */
    IsingLattice *L = initialize_ising_lattice(6, 6, 6, "random");
    srand(0xBEEF);
    for (int s = 0; s < 200; s++) ising_swendsen_wang_step(L, 0.22);
    double E = compute_ising_energy(L);
    int N = L->size_x * L->size_y * L->size_z;
    double epsite = E / (double)N;
    printf("# SW near T_c: E/N = %.3f (between -3 and -0.5 expected)\n", epsite);
    /* Must be between the T=0 limit (-3 for 6-NN 3D Ising / 2 since
     * we're computing a bond-sum) and T=∞ (-0 roughly). Use loose
     * bounds. */
    ASSERT_TRUE(epsite < -0.2);
    ASSERT_TRUE(epsite > -3.5);
    free_ising_lattice(L);
}
int main(void) {
    TEST_RUN(test_sw_preserves_ferromagnet_at_zero_temperature);
    TEST_RUN(test_sw_demagnetises_at_high_temperature);
    TEST_RUN(test_sw_near_critical_temperature);
    TEST_SUMMARY();
}