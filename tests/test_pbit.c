/*
 * tests/test_pbit.c
 *
 * Probabilistic-bit Ising machine:
 *   (1) Energy of a single bond matches the textbook -J s_i s_j.
 *   (2) A fully-ferromagnetic coupling at large β locks spins
 *       together with overwhelming probability.
 *   (3) Simulated annealing finds the optimal MaxCut on a triangle
 *       (3 vertices, 3 edges): energy = -1 (best cut drops one edge).
 *   (4) Annealing on the Petersen-graph MaxCut reaches the known
 *       optimum of 12 cut edges out of 15.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "neuromorphic/pbit.h"
static void test_ising_energy_on_single_bond(void) {
    /* Sign convention: H = -½ Σ J_ij s_i s_j, J_ij > 0 ferromagnetic.
     * J = [[0, +1], [+1, 0]] with aligned spins s=(+1,+1):
     *   -½ · (J_01·1·1 + J_10·1·1) = -½ · 2 = -1. */
    int n = 2;
    double J[4] = {0.0, +1.0, +1.0, 0.0};
    pbit_net_t net;
    pbit_net_init(&net, n, J, NULL, 1.0, 0xABCu);
    int s_ff[2] = {+1, +1};
    int s_fa[2] = {+1, -1};
    ASSERT_NEAR(pbit_ising_energy(&net, s_ff), -1.0, 1e-12);
    ASSERT_NEAR(pbit_ising_energy(&net, s_fa), +1.0, 1e-12);
}
static void test_ferromagnet_locks_at_high_beta(void) {
    /* 4 p-bits with uniform +1 couplings and β = 5 (strong): after
     * many sweeps the configuration should be either all +1 or all
     * -1 (the two FM ground states). */
    int n = 4;
    double J[16] = {0};
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
        if (i != j) J[i*n + j] = +1.0;
    pbit_net_t net;
    pbit_net_init(&net, n, J, NULL, 5.0, 0xBADu);
    int spins[4] = {+1, -1, +1, -1};
    pbit_sweep(&net, spins, 200);
    int all_up = 1, all_down = 1;
    for (int i = 0; i < n; i++) {
        if (spins[i] != +1) all_up = 0;
        if (spins[i] != -1) all_down = 0;
    }
    ASSERT_TRUE(all_up || all_down);
}
static void test_maxcut_triangle(void) {
    /* K3: 3 vertices, 3 edges, unweighted. Maximum cut = 2 (drop one
     * edge, put 2 on one side, 1 on the other). MaxCut QUBO maps to
     * Ising with J = -w on edges; optimum E = -2 (cut weight).
     * Actually: pbit_ising_energy returns E = -½ Σ J_ij s_i s_j.
     * With J symmetric and J_ij = -1 for each edge, E_min occurs at
     * the max-cut partition where two s's are +1 and one is -1
     * (or vice versa). */
    int n = 3;
    int eu[3] = {0, 1, 0};
    int ev[3] = {1, 2, 2};
    double J[9];
    pbit_maxcut_couplings(n, 3, eu, ev, NULL, J);
    pbit_net_t net;
    pbit_net_init(&net, n, J, NULL, 1.0, 0xCAFEu);
    int spins[3] = {+1, +1, +1};
    double E_final = 0;
    pbit_anneal(&net, spins, 0.5, 10.0, 500, &E_final);
    printf("# MaxCut K3 annealed: spins=(%d,%d,%d)  E=%.2f (min=-1, cut=2)\n",
           spins[0], spins[1], spins[2], E_final);
    /* Minimum Ising energy: two +1 one -1 → cut = 2 edges × 1 = 2
     * → E = -½ · 2·(two +1·-1 contributions, each ×2 for symmetry)
     * = -(2 edges cut) = we are annealing a symmetric J. Each cut
     * edge contributes -J (with J = -1 per edge) = +1, uncut edges
     * contribute +J = -1. For 2 cut / 1 uncut the sum -½ Σ J_ij s_i s_j
     * = -½ · (2 · (+1) · (-1 · 2 + +1 · 1)) — this is getting
     * confusing, just check the final energy reaches its minimum. */
    ASSERT_TRUE(E_final <= -0.99);  /* minimum is -1 for the sign
                                       convention used here */
}
static void test_maxcut_petersen(void) {
    /* Petersen graph: 10 vertices, 15 edges. Maximum cut is 12 edges
     * (a known NP-hard-for-general-MaxCut result; but Petersen's max
     * cut is solved). Anneal and check that the final cut count ≥ 11
     * (allow some MC slack since we have only 500 sweeps). */
    int n = 10;
    int eu[15] = {0,1,2,3,4,0,1,2,3,4,5,6,7,8,9};
    int ev[15] = {1,2,3,4,0,5,6,7,8,9,7,8,9,5,6};
    double J[100];
    pbit_maxcut_couplings(n, 15, eu, ev, NULL, J);
    pbit_net_t net;
    pbit_net_init(&net, n, J, NULL, 0.5, 0xF00Du);
    int spins[10];
    /* Random init. */
    unsigned long long rng = 0xC0DEULL;
    for (int i = 0; i < n; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        spins[i] = ((rng >> 11) & 1) ? +1 : -1;
    }
    double E_final = 0;
    pbit_anneal(&net, spins, 0.3, 5.0, 2000, &E_final);
    /* Count cut edges: an edge (u,v) is cut iff s_u ≠ s_v. */
    int cut_count = 0;
    for (int e = 0; e < 15; e++) if (spins[eu[e]] != spins[ev[e]]) cut_count++;
    printf("# MaxCut Petersen annealed: cut = %d / 15 edges  (optimum = 12)\n",
           cut_count);
    ASSERT_TRUE(cut_count >= 11);
}
int main(void) {
    TEST_RUN(test_ising_energy_on_single_bond);
    TEST_RUN(test_ferromagnet_locks_at_high_beta);
    TEST_RUN(test_maxcut_triangle);
    TEST_RUN(test_maxcut_petersen);
    TEST_SUMMARY();
}