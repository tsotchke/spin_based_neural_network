# Testing

v0.4 introduces a lightweight TAP-style test harness at `tests/harness.h`
and one suite per library module, plus per-pillar suites that land with
v0.5+ work. Every `tests/test_*.c` is itself a `main()` binary built by
the Makefile. This document describes how to run, extend, and interpret
the suite. For the live list of suites and their Makefile targets, run
`grep -E '^test_[a-z_]+:' Makefile | sort` — the table below covers the
v0.4 core modules but does not track the pillar-specific additions that
get appended over time.

## Running the suite

```sh
make test
```

Builds every suite under `build/test_*` and runs them in sequence. Any
failure propagates the exit code. Typical wall-clock on an M-series Mac:
≈ 2 seconds full clean rebuild + run.

Individual suites can also be run on their own:

```sh
make test_majorana && ./build/test_majorana
```

## Output format

Each suite emits TAP 13 output:

```
ok 1 - test_majorana_square_is_identity
ok 2 - test_majorana_anticommutation
not ok 3 - test_some_broken_thing
    # test_some_broken_thing: expected X, got Y (tests/test_majorana.c:42)
1..N
# passed: M / N
```

The legacy `test_topological_entropy` suite (pre-v0.4) emits a slightly
different format but is included in `make test` for historical
continuity.

## Test coverage summary

The table below enumerates v0.4 core-module suites. Pillar tests
(`test_nqs*`, `test_mps*`, `test_llg*`, `test_qec_*`,
`test_torque_net*`, `test_fibonacci_anyons`, `test_hopfield`,
`test_rbm_cd`, `test_thqcp_coupling`, etc.) land alongside their
pillar code and are driven by `make test` through the same harness;
see `tests/` for the complete list.

As of v0.4.1, two Hamiltonian kernels were added to the NQS pillar
(`NQS_HAM_KITAEV_HEISENBERG`, `NQS_HAM_KAGOME_HEISENBERG`); their
coverage lives in `tests/test_nqs_kitaev.c` (9 tests including a
legacy-vs-KH cross-check), `tests/test_nqs_kagome.c` (new, 7 tests
with an independently-derived 2×2 PBC bond-list cross-check), and
end-to-end SR convergence tests in `tests/test_nqs_holomorphic_sr.c`
(2 complex-RBM + 1 real-MLP paths).

The v0.4.1 follow-up also ships three sample-based diagnostics and
an exact-reference solver for the kagome ground-state question:

- `tests/test_nqs_chi_F.c` (6 cases) — χ_F finiteness +
  non-negativity on complex-RBM and legacy-MLP ansätze, bad-args
  rejection, Monte-Carlo consistency across batch sizes, per-bond-
  class phase probe output, and a rejection check for the kagome-
  only bond-phase helper on non-kagome Hamiltonians.
- `tests/test_nqs_excited.c` (4 cases) — excited-state SR via the
  Choo–Neupert–Carleo orthogonal-ansatz penalty. μ=0 equivalence
  with holomorphic SR, null-reference rejection, 2-site Heisenberg
  triplet recovery to four decimal places, and a 60-iter kagome
  N=12 smoke exercising the multi-sublattice kernel.
- `tests/test_nqs_lanczos.c` gains
  `test_kagome_lanczos_k_lowest_gives_exact_gap`: k-Ritz
  extraction must return ascending eigenvalues whose smallest
  matches the rank-1 refine to 10⁻⁸ and whose gap is positive
  (regression guard for a sort-order bug).

Research driver (NOT part of `make test`):
`make research_kagome_N12_diagnostics` chains GS SR → χ_F →
per-bond-class phase → excited-state SR → Lanczos-exact E₀/E₁/gap
in one run; takes O(20 min) on an M-series Mac and emits a self-
contained TAP-style report for longitudinal comparison.

| Suite | Tests | Covers | Notes |
|---|---|---|---|
| `test_majorana` | 13 | `src/majorana_modes.c` | Hilbert-space braiding (`B⁴=-I`, `B⁸=I`), anticommutation, parity, legacy braid, zero-modes detection, chain-to-lattice mapping |
| `test_toric_code` | 14 | `src/toric_code.c` | data-qubit model, primal/dual paths, greedy decoder, homology-based logical error, stabilizer/Kitaev-lattice integration |
| `test_ising` | 7 | `src/ising_model.c` | energies on FM / AFM / stripe configurations, interior + corner interaction, Metropolis smoke |
| `test_kitaev` | 6 | `src/kitaev_model.c` | isotropic / anisotropic energies, interior / corner interaction, flip preserves ±1 |
| `test_topological_entropy` | 3 | `src/topological_entropy.c` | legacy entropy suite (pre-v0.4) |
| `test_engine_adapter` | 8 | `src/engine_adapter.c` (v0.4) | disabled-mode status codes, all three flatteners, row-major ordering |
| `test_nn_backend` | 11 | `src/nn_backend.c` (v0.4) | polymorphic handle, parse/name round-trip, legacy create/forward/train, engine fall-back |
| `test_spin_models` | 4 | `src/spin_models.c` | SpinLattice init, all-up / all-down, energy finiteness |
| `test_energy_utils` | 4 | `src/energy_utils.c` | scale/unscale round-trip, monotonicity, bounds |
| `test_disordered_model` | 4 | `src/disordered_model.c` | zero/unit disorder bounds, magnitude preservation |
| `test_eshkol_bridge` | 5 | `src/eshkol_bridge.c` (v0.4) | disabled-mode lifecycle, EARG validation |
| `test_physics_loss` | 6 | `src/physics_loss.c` | all 5 PDE losses run finitely, Laplacian of constant vanishes |
| `test_berry_phase` | 5 | `src/berry_phase.c` | lifecycle, invariants struct, winding number in topological phase, `CHERN_NUMBER` gating |
| `test_reinforcement_learning` | 4 | `src/reinforcement_learning.c` | reward signs, optimiser preserves ±1 spins, state-string round-trip |
| `test_quantum_mechanics` | 3 | `src/quantum_mechanics.c` | quantum effects preserve ±1, entanglement runs cleanly, input validation |
| `test_ising_chain_qubits` | 7 | `src/ising_chain_qubits.c` | init/free, encode/measure, X/Y/Z/CNOT gates, chain interaction |
| `test_matrix_neon` | 5 | `src/matrix_neon.c` | NEON probe, matvec identity/diagonal/zero, eigenvalue finiteness |

## Harness API

`tests/harness.h` exposes these macros:

```c
TEST_RUN(fn_name);                 /* run a test function, tracks pass/fail */
TEST_FAIL(fmt, ...);               /* record failure with message */
ASSERT_TRUE(cond);
ASSERT_EQ_INT(a, b);
ASSERT_NEAR(a, b, eps);            /* double comparison */
ASSERT_NEAR_COMPLEX(a, b, eps);    /* complex-double comparison */
TEST_SUMMARY();                    /* emits "1..N" + pass count; returns exit code */
```

Pattern for a test file:

```c
#include "harness.h"
#include "my_module.h"

static void test_thing_one(void) {
    ASSERT_EQ_INT(add(1, 2), 3);
}

int main(void) {
    TEST_RUN(test_thing_one);
    TEST_SUMMARY();
}
```

## Adding a new test suite

1. Create `tests/test_<module>.c` using the pattern above.
2. Add a target to the `Makefile`:

```make
test_<module>: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_<module> \
	    tests/test_<module>.c src/<module>.c $(extra_sources) $(LDFLAGS)
```

3. Add `test_<module>` to the dependency list of the aggregate `test:`
   target, and append `@$(BIN_DIR)/test_<module>` to its recipe.
4. `make clean && make test` to verify.

## What counts as "100% coverage" here

Every library module (`src/*.c` or `src/<subdir>/*.c` that compiles
into the main binary) has at least one test suite. Per-function
coverage is high but not exhaustive — the current criterion is that
every public function declared under `include/` is exercised by at
least one test.

The `main.c` driver, `topological_example.c` standalone demo, and the
SDL2 `visualization*.c` files are excluded: they are not library units
but binaries, and the math / data-flow portions of their logic live in
the library modules that *are* tested. Pillar modules (`src/nqs/`,
`src/mps/`, and any future pillar subdirectory) each carry their own
`tests/test_<pillar>*.c` files alongside this baseline.

## When to add to an existing suite vs. create a new one

- **Append** to an existing suite when adding cases to a function that
  already has tests, or covering a new function in an already-tested
  module.
- **Create** a new suite when introducing a new module, or when an
  existing module's test file would grow beyond ~400 lines.

## Test philosophy

Tests in this repo lean toward **physics-meaningful** assertions rather
than interface-shape checks. Examples:

- Braiding: `B⁴ = -I` and `B⁸ = +I` (order-8 Ising statistics)
- Toric code: a non-contractible loop of X errors flags no plaquettes
  but trips `toric_code_has_logical_error()`
- Ising: all-up energy on L³ torus = `-3L³`
- Berry phase: winding number is positive in the topological phase
  (|μ| < 2|t|) and zero in the trivial phase

Smoke tests (e.g. "runs without crashing") are acceptable for
stochastic / heuristic routines where the v0.3 behavior is best
documented rather than asserted; they are explicitly marked as such.

## References

For the TAP (Test Anything Protocol) format, see https://testanything.org.
