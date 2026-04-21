/*
 * tests/test_majorana.c
 *
 * Verifies the Hilbert-space Majorana braiding implementation in
 * src/majorana_modes.c. Tests:
 *   - γ_i^2 = I
 *   - {γ_i, γ_j} = 0 for i != j
 *   - ||B_{ij} ψ||^2 = ||ψ||^2 (unitarity)
 *   - B_{ij}^4 = -I, B_{ij}^8 = +I (Ising-anyon order-8 statistics)
 *   - Parity conservation under both γ_i γ_j and B_{ij}
 */
#include "harness.h"
#include "majorana_modes.h"
static void randomize_state(MajoranaHilbertState *state, unsigned seed) {
    srand(seed);
    double norm_sq = 0.0;
    for (int n = 0; n < state->hilbert_dim; n++) {
        double re = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        double im = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        state->amplitudes[n] = re + im * _Complex_I;
        norm_sq += re * re + im * im;
    }
    double inv_norm = 1.0 / sqrt(norm_sq);
    for (int n = 0; n < state->hilbert_dim; n++) {
        state->amplitudes[n] *= inv_norm;
    }
}
/* γ_i^2 = I: apply the same Majorana operator twice, recover original state. */
static void test_majorana_square_is_identity(void) {
    int num_majoranas = 6; /* 3 fermion modes -> 8-dim Hilbert space */
    for (int op = 0; op < num_majoranas; op++) {
        MajoranaHilbertState *psi = initialize_majorana_hilbert_state(num_majoranas);
        MajoranaHilbertState *original = initialize_majorana_hilbert_state(num_majoranas);
        ASSERT_TRUE(psi && original);
        randomize_state(psi, 42u + (unsigned)op);
        majorana_state_copy(psi, original);
        apply_majorana_op_to_state(op, psi);
        apply_majorana_op_to_state(op, psi);
        for (int n = 0; n < psi->hilbert_dim; n++) {
            ASSERT_NEAR_COMPLEX(psi->amplitudes[n], original->amplitudes[n], 1e-12);
        }
        free_majorana_hilbert_state(psi);
        free_majorana_hilbert_state(original);
    }
}
/* {γ_i, γ_j} = 0 for i != j: (γ_i γ_j + γ_j γ_i) |ψ⟩ = 0. */
static void test_majorana_anticommutation(void) {
    int num_majoranas = 6;
    for (int i = 0; i < num_majoranas; i++) {
        for (int j = i + 1; j < num_majoranas; j++) {
            MajoranaHilbertState *a = initialize_majorana_hilbert_state(num_majoranas);
            MajoranaHilbertState *b = initialize_majorana_hilbert_state(num_majoranas);
            ASSERT_TRUE(a && b);
            randomize_state(a, 7u + (unsigned)(i * num_majoranas + j));
            majorana_state_copy(a, b);
            /* a <- γ_i γ_j |ψ⟩; b <- γ_j γ_i |ψ⟩ */
            apply_majorana_op_to_state(j, a);
            apply_majorana_op_to_state(i, a);
            apply_majorana_op_to_state(i, b);
            apply_majorana_op_to_state(j, b);
            for (int n = 0; n < a->hilbert_dim; n++) {
                ASSERT_NEAR_COMPLEX(a->amplitudes[n] + b->amplitudes[n],
                                    0.0 + 0.0 * _Complex_I, 1e-12);
            }
            free_majorana_hilbert_state(a);
            free_majorana_hilbert_state(b);
        }
    }
}
/* Braiding is unitary: ||Bψ||^2 = ||ψ||^2. */
static void test_braid_unitarity(void) {
    int num_majoranas = 6;
    MajoranaHilbertState *psi = initialize_majorana_hilbert_state(num_majoranas);
    ASSERT_TRUE(psi);
    randomize_state(psi, 99u);
    double before = majorana_state_norm_squared(psi);
    apply_braid_unitary(psi, 0, 1);
    apply_braid_unitary(psi, 2, 3);
    apply_braid_unitary(psi, 1, 4);
    apply_braid_unitary(psi, 0, 5);
    double after = majorana_state_norm_squared(psi);
    ASSERT_NEAR(before, after, 1e-12);
    free_majorana_hilbert_state(psi);
}
/* B^4 = -I and B^8 = +I (Ising-anyon statistics). */
static void test_braid_order_8(void) {
    int num_majoranas = 4; /* 2 fermion modes -> 4-dim Hilbert */
    for (int i = 0; i < num_majoranas; i++) {
        for (int j = i + 1; j < num_majoranas; j++) {
            MajoranaHilbertState *psi = initialize_majorana_hilbert_state(num_majoranas);
            MajoranaHilbertState *original = initialize_majorana_hilbert_state(num_majoranas);
            ASSERT_TRUE(psi && original);
            randomize_state(psi, 123u + (unsigned)(i * 10 + j));
            majorana_state_copy(psi, original);
            /* Apply B four times: should equal -I */
            for (int k = 0; k < 4; k++) apply_braid_unitary(psi, i, j);
            for (int n = 0; n < psi->hilbert_dim; n++) {
                ASSERT_NEAR_COMPLEX(psi->amplitudes[n],
                                    -original->amplitudes[n], 1e-10);
            }
            /* Four more applications: total 8 -> back to +I */
            for (int k = 0; k < 4; k++) apply_braid_unitary(psi, i, j);
            for (int n = 0; n < psi->hilbert_dim; n++) {
                ASSERT_NEAR_COMPLEX(psi->amplitudes[n],
                                    original->amplitudes[n], 1e-10);
            }
            free_majorana_hilbert_state(psi);
            free_majorana_hilbert_state(original);
        }
    }
}
/* Fermion-number parity P = ∏ (1-2n_k) is preserved by γ_i γ_j and by B_{ij}.
 * Individually γ_i flips parity, but products of two do not. */
static void test_braid_parity_conservation(void) {
    int num_majoranas = 6;
    MajoranaHilbertState *psi = initialize_majorana_hilbert_state(num_majoranas);
    ASSERT_TRUE(psi);
    /* Start in an even-parity basis state: |000000> has parity +1. */
    majorana_hilbert_state_set_vacuum(psi);
    apply_braid_unitary(psi, 0, 3);
    apply_braid_unitary(psi, 1, 2);
    /* Check only even-parity basis states carry amplitude. */
    for (int b = 0; b < psi->hilbert_dim; b++) {
        int popcount = __builtin_popcount(b);
        if (popcount & 1) {
            ASSERT_NEAR(cabs(psi->amplitudes[b]), 0.0, 1e-12);
        }
    }
    free_majorana_hilbert_state(psi);
}
/* Legacy operator-permutation braiding still exists for back-compat. */
static void test_legacy_braid_swaps_operators(void) {
    KitaevWireParameters params = {
.coupling_strength = 1.0,.chemical_potential = 0.5,.superconducting_gap = 1.0
    };
    MajoranaChain *chain = initialize_majorana_chain(3, &params);
    ASSERT_TRUE(chain != NULL);
    double _Complex op0 = chain->operators[0];
    double _Complex op1 = chain->operators[1];
    braid_majorana_operators_legacy(chain, 0, 1);
    ASSERT_NEAR_COMPLEX(chain->operators[0],  op1, 1e-12);
    ASSERT_NEAR_COMPLEX(chain->operators[1], -op0, 1e-12);
    free_majorana_chain(chain);
}
/* calculate_majorana_parity returns ±1. */
static void test_parity_returns_plus_or_minus_one(void) {
    KitaevWireParameters params = {1.0, 0.5, 1.0};
    MajoranaChain *chain = initialize_majorana_chain(4, &params);
    ASSERT_TRUE(chain != NULL);
    int p = calculate_majorana_parity(chain);
    ASSERT_TRUE(p == 1 || p == -1);
    free_majorana_chain(chain);
}
/* detect_majorana_zero_modes returns non-zero in the topological phase
 * (|mu| < 2|t|) and zero otherwise. */
static void test_zero_modes_detected_in_topological_phase(void) {
    KitaevWireParameters topo = {1.0, 0.5, 1.0};
    MajoranaChain *chain = initialize_majorana_chain(5, &topo);
    ASSERT_TRUE(chain != NULL);
    double s = detect_majorana_zero_modes(chain, &topo);
    ASSERT_TRUE(s > 0.0);
    free_majorana_chain(chain);
}
static void test_zero_modes_absent_in_trivial_phase(void) {
    KitaevWireParameters trivial = {1.0, 3.0 /* |mu| > 2|t| */, 1.0};
    MajoranaChain *chain = initialize_majorana_chain(5, &trivial);
    ASSERT_TRUE(chain != NULL);
    double s = detect_majorana_zero_modes(chain, &trivial);
    ASSERT_NEAR(s, 0.0, 1e-12);
    free_majorana_chain(chain);
}
static void test_compute_kitaev_wire_energy_finite(void) {
    KitaevWireParameters params = {1.0, 0.5, 1.0};
    MajoranaChain *chain = initialize_majorana_chain(6, &params);
    ASSERT_TRUE(chain != NULL);
    double E = compute_kitaev_wire_energy(chain, &params);
    ASSERT_TRUE(E == E); /* not NaN */
    free_majorana_chain(chain);
}
/* apply_majorana_operator flips a single site on a lattice. */
static void test_apply_majorana_operator_flips_site(void) {
    KitaevWireParameters params = {1.0, 0.5, 1.0};
    MajoranaChain *chain = initialize_majorana_chain(3, &params);
    KitaevLattice *lat = initialize_kitaev_lattice(3, 3, 3, 1.0, 1.0, 1.0, "all-up");
    ASSERT_TRUE(chain && lat);
    int before = lat->spins[0][0][0];
    apply_majorana_operator(chain, 0, lat);
    int after  = lat->spins[0][0][0];
    ASSERT_EQ_INT(after, -before);
    free_majorana_chain(chain);
    free_kitaev_lattice(lat);
}
/* map_chain_to_lattice populates a row of alternating spins. */
static void test_map_chain_sets_alternating_spins(void) {
    KitaevWireParameters params = {1.0, 0.5, 1.0};
    MajoranaChain *chain = initialize_majorana_chain(4, &params);
    KitaevLattice *lat = initialize_kitaev_lattice(8, 8, 8, 1.0, 1.0, 1.0, "all-up");
    ASSERT_TRUE(chain && lat);
    map_chain_to_lattice(chain, lat, 0, 0, 0, 0 /* x-direction */);
    /* Expected pattern after the map: +1 at even sites, -1 at odd. */
    ASSERT_EQ_INT(lat->spins[0][0][0],  1);
    ASSERT_EQ_INT(lat->spins[1][0][0], -1);
    ASSERT_EQ_INT(lat->spins[2][0][0],  1);
    ASSERT_EQ_INT(lat->spins[3][0][0], -1);
    free_majorana_chain(chain);
    free_kitaev_lattice(lat);
}
static void test_hilbert_state_copy_and_inner_product(void) {
    MajoranaHilbertState *a = initialize_majorana_hilbert_state(4);
    MajoranaHilbertState *b = initialize_majorana_hilbert_state(4);
    ASSERT_TRUE(a && b);
    majorana_state_copy(a, b);
    double _Complex ip = majorana_states_inner_product(a, b);
    ASSERT_NEAR(creal(ip), 1.0, 1e-12); /* |<0|0>|^2 = 1 */
    ASSERT_NEAR(cimag(ip), 0.0, 1e-12);
    free_majorana_hilbert_state(a);
    free_majorana_hilbert_state(b);
}
int main(void) {
    TEST_RUN(test_majorana_square_is_identity);
    TEST_RUN(test_majorana_anticommutation);
    TEST_RUN(test_braid_unitarity);
    TEST_RUN(test_braid_order_8);
    TEST_RUN(test_braid_parity_conservation);
    TEST_RUN(test_legacy_braid_swaps_operators);
    TEST_RUN(test_parity_returns_plus_or_minus_one);
    TEST_RUN(test_zero_modes_detected_in_topological_phase);
    TEST_RUN(test_zero_modes_absent_in_trivial_phase);
    TEST_RUN(test_compute_kitaev_wire_energy_finite);
    TEST_RUN(test_apply_majorana_operator_flips_site);
    TEST_RUN(test_map_chain_sets_alternating_spins);
    TEST_RUN(test_hilbert_state_copy_and_inner_product);
    TEST_SUMMARY();
}