# Spin-Based Neural Computation Framework v0.4.0

v0.4.0 is the **foundation release** for the v0.5+ research pillars. It
significantly advances the topological-quantum-computing physics beyond
v0.3, adds engine-neutral scaffolding for the upcoming neural-network
research pillars, and ships a comprehensive 109-test verification suite
and 4-suite benchmark harness.

## Physics improvements

### Majorana braiding — Hilbert-space unitaries

v0.4 extends Majorana braiding with a full Hilbert-space implementation
of the Ising-anyon braiding unitary:

```
B_{ij} = exp(π γ_i γ_j / 4) = (1 + γ_i γ_j) / √2
```

acting on the `2^(N/2)`-dim fermion-parity Fock basis, constructed via
Jordan-Wigner. Verified properties (see `tests/test_majorana.c`):

- `γ_i² = I` for all i
- `{γ_i, γ_j} = 0` for i ≠ j
- `B^4 = −I` (half-period)
- `B^8 = +I` (order-8 non-Abelian statistics)
- Braiding preserves the `||ψ||²` norm (unitary)
- Braiding preserves fermion parity

The v0.3 operator-space braiding path remains available as
`braid_majorana_operators_legacy()`; `braid_majorana_modes()` continues
to work unchanged. The new `MajoranaHilbertState` API is additive.

New public API in `include/majorana_modes.h`:

- `MajoranaHilbertState *initialize_majorana_hilbert_state(int num_majoranas)`
- `void free_majorana_hilbert_state(MajoranaHilbertState *state)`
- `void majorana_hilbert_state_set_vacuum(MajoranaHilbertState *state)`
- `void apply_majorana_op_to_state(int op_index, MajoranaHilbertState *state)`
- `void apply_braid_unitary(MajoranaHilbertState *state, int i, int j)`
- `double majorana_state_norm_squared(const MajoranaHilbertState *state)`
- `double _Complex majorana_states_inner_product(const MajoranaHilbertState *a, const MajoranaHilbertState *b)`
- `void majorana_state_copy(const MajoranaHilbertState *src, MajoranaHilbertState *dst)`

### Toric-code error correction — data-qubit model

v0.4 improves the toric-code error-correction implementation with:

- An explicit **data-qubit model**: 2·L_x·L_y data qubits, one per link,
  with independent `x_errors[]` and `z_errors[]` GF(2) accumulators.
  Corrections now flip data qubits directly and re-derive the syndromes
  from them, so repeated error-and-correction cycles remain consistent.
- **Primal/dual-lattice-aware path walks**: Z-error corrections walk
  the primal lattice (vertex-to-vertex); X-error corrections walk the
  dual (plaquette-to-plaquette). The walk direction determines which
  link type is flipped at each step.
- **Homology-based logical-error detection**:
  `toric_code_has_logical_error()` computes the error chain's class in
  H₁(dual, Z₂) by intersection with primal 1-cycles, which is invariant
  under stabilizer action and therefore robust to any sequence of
  corrections.
- **Greedy matching decoder** (`toric_code_decode_greedy()`): pairs
  flagged syndromes by toroidal taxicab distance and applies corrections
  along shortest paths. Not optimal (proper MWPM comes in v0.5), but
  effective at physically-reasonable error rates.

New public API in `include/toric_code.h`:

- Data-qubit accessors: `toric_code_link_index`, `toric_code_vertex_links`,
  `toric_code_plaquette_links`
- Error / correction: `toric_code_apply_{x,z}_{error,correction}`,
  `toric_code_refresh_syndromes`
- Channel-specific syndromes: `toric_code_measure_x_syndrome`,
  `toric_code_measure_z_syndrome`
- Decoding: `toric_code_decode_greedy`
- Queries: `toric_code_has_logical_error`, `is_ground_state` (now
  also verifies no accumulated logical error)

Baseline toric-decoder logical-error rate, measured on M-series Mac
(see `benchmarks/results/toric_decoder/`):

| distance | p=1% | p=3% | p=5% |
|---|---|---|---|
| 3 | 0.8% | 6.8% | 15.4% |
| 5 | 0.2% | 5.6% | 14.0% |
| 7 | 0.0% | 2.4% | 9.2% |

Logical error rate decreases with distance at p=1% (below threshold). At
p=5% all distances show high rates (above threshold). The learned
surface-code decoder in v0.5 pillar P1.3 must beat these numbers.

### Berry-phase `CHERN_NUMBER` test override

The `CHERN_NUMBER` environment-variable hook is now a test-only feature,
gated behind `#ifdef SPIN_NN_TESTING`. Release builds run the actual
Chern-number calculation unconditionally; test builds opt in with
`-DSPIN_NN_TESTING` when they need to force specific values for
regression testing.

## Engine-neutral scaffolding

Three new modules support the external-engine integration planned for
v0.5+. All compile in the default build as dormant stubs (enabled with
`make ENGINE_ENABLE=1 ENGINE_ROOT=...`):

- **`engine_adapter`** — single translation unit that will bridge to an
  external NN / tensor / reasoning engine. Engine-neutral: any engine
  providing `engine_backend_init`, `engine_backend_shutdown`,
  `engine_backend_version` as weak hooks can plug in. Planned backends
  for v0.5+: an Eshkol-native NN engine (working title
  `eshkol-transformers`, built on
  https://github.com/tsotchke/eshkol) and Noesis (reasoning engine,
  in development, not yet publicly released).
- **`eshkol_bridge`** — lazy wrapper over the Eshkol FFI for
  Scheme-orchestrated training tapes. `ESHKOL_BRIDGE_EDISABLED` until
  built with `-DSPIN_NN_HAS_ESHKOL=1`.
- **`nn_backend`** — polymorphic neural-network handle.
  `NN_BACKEND_LEGACY` uses the in-tree MLP; `NN_BACKEND_ENGINE` delegates
  to whichever engine is wired in via `engine_adapter`. New CLI flag
  `--nn-backend={legacy,engine}` toggles at runtime; engine transparently
  falls back to legacy with a diagnostic when no engine is available.

## Training-loop integration

New CLI flags on the main binary drive in-loop topological feedback:

- `--cadence-decoder N` — run the greedy toric-code decoder every N
  training iterations against a syndrome sampled at
  `--decoder-error-rate P` (default 0.03). Logical-error flag folds
  into `physics_loss` with coefficient `--lambda-logical L`.
- `--cadence-invariants N` — compute topological invariants every N
  iterations (advisory in v0.4; pillar P1.2 in v0.5 folds them into the
  loss).

See `include/training_config.h` for the full option set.

## Test and benchmark harnesses

### Test suite: 109 tests across 17 suites

Every library module has at least one test. Run `make test` for the
full suite. TAP-style output; any failure propagates the exit code.

| Suite | Tests | Module(s) covered |
|---|---|---|
| test_majorana | 13 | majorana_modes.c (incl. legacy + Hilbert paths) |
| test_toric_code | 14 | toric_code.c (all v0.4 APIs) |
| test_ising | 7 | ising_model.c |
| test_kitaev | 6 | kitaev_model.c |
| test_topological_entropy | 3 | topological_entropy.c (legacy) |
| test_engine_adapter | 8 | engine_adapter.c |
| test_nn_backend | 11 | nn_backend.c |
| test_spin_models | 4 | spin_models.c |
| test_energy_utils | 4 | energy_utils.c |
| test_disordered_model | 4 | disordered_model.c |
| test_eshkol_bridge | 5 | eshkol_bridge.c |
| test_physics_loss | 6 | physics_loss.c (5 PDE losses + Laplacian) |
| test_berry_phase | 5 | berry_phase.c (incl. CHERN_NUMBER gating) |
| test_reinforcement_learning | 4 | reinforcement_learning.c |
| test_quantum_mechanics | 3 | quantum_mechanics.c |
| test_ising_chain_qubits | 7 | ising_chain_qubits.c |
| test_matrix_neon | 5 | matrix_neon.c |

Full rebuild + test run on an M-series Mac completes in ≈ 2 seconds.

### Benchmark harness: 4 suites, 19 JSON records

Run `./scripts/run_benchmarks.sh` to build and execute all benchmarks;
results land under `benchmarks/results/<suite>/<name>.json`.

- `bench_ising` — Metropolis sweeps/sec at L = 8, 16, 32
- `bench_kitaev` — Metropolis sweeps/sec at L = 8, 16, 32
- `bench_majorana_braid` — braids/sec at N = 8, 12, 16, 20 Majoranas
- `bench_toric_decoder` — greedy-decoder logical-error-rate curves at
  d = 3, 5, 7 × p = 1%, 3%, 5%

See `docs/benchmarks.md` for the JSON schema and provenance notes.

## Build system and stack probe

- `make test` runs all 17 test suites.
- `make bench` builds all 4 benchmark suites;
  `./scripts/run_benchmarks.sh` runs them and writes JSON.
- `./scripts/check_stack.sh` probes for optional external dependencies
  (engine, libirrep, Eshkol runtime). All advisory in v0.4.
- `make ENGINE_ENABLE=1 ENGINE_ROOT=...` (v0.5+) lights up the
  engine-adapter build path.

## Backward compatibility

- The `arm`, `non_arm`, and `universal` targets still emit compatible
  executables under `build/` and accept every v0.3 CLI flag unchanged.
- `braid_majorana_modes()` remains available and delegates to the legacy
  operator-permutation path.
- All existing example shell scripts (`run_topological_examples.sh`,
  `different_topo_values.sh`) run without modification.

## Upgrade notes

- The `CHERN_NUMBER` environment variable is now a test-only feature.
  If you relied on it to force specific values during experimentation,
  rebuild with `-DSPIN_NN_TESTING`.
- `include/toric_code.h` adds many new public symbols but removes none;
  existing code compiles unchanged.
- `include/majorana_modes.h` adds the `MajoranaHilbertState` type and
  associated functions; `braid_majorana_modes()` is preserved.

## Forward roadmap: v0.5 research pillars

See [docs/architecture_v0.4.md](docs/architecture_v0.4.md). Headline pillars:

- **P1.1 Neural Network Quantum States** — ViT / factored-ViT
  wavefunctions for frustrated magnets (J1-J2 square, kagome, Kitaev
  honeycomb)
- **P1.2 Equivariant GNN torques + real Landau-Lifshitz-Gilbert dynamics**
  — E(3)-equivariant message passing (uses forthcoming `libirrep`)
- **P1.3 Learned surface-code decoders** — transformer and Mamba
  variants beating the v0.4 MWPM baseline
- **P1.3b Fibonacci anyons** — universal-gate counterpart to the
  non-universal Majorana module via Solovay-Kitaev
- **P1.4 Neural-operator LLG accelerators + discrete flow-matching
  Boltzmann samplers**
- **P2.x** Time-dependent NQS, MPS warm-start, foundation NQS, p-bit
  neuromorphic, KAN-NQS, Lanczos, PINN upgrades, gauge-invariant
  sampling, thermodynamic computing

## Getting started

```sh
git clone https://github.com/tsotchke/spin_based_neural_network.git
cd spin_based_neural_network
make arm test bench
./scripts/run_benchmarks.sh
./build/spin_based_neural_computation_arm --help
```

## Full changelog

See [CHANGELOG.md](CHANGELOG.md).

---
# Spin-Based Neural Computation Framework v0.3.0

We're excited to announce the release of version 0.3.0 of the Spin-Based Neural Computation Framework, featuring comprehensive topological quantum computing capabilities.

## Key Features

### Topological Quantum Computing
- **Berry Phase and Topological Invariants**: Calculate Berry phase, curvature, and derived invariants (Chern numbers, TKNN invariant, winding numbers)
- **Majorana Zero Modes**: Simulate Majorana fermions with braiding operations and non-Abelian statistics
- **Topological Entanglement Entropy**: Measure topological order through entanglement entropy
- **Toric Code Error Correction**: Implement error correction using Kitaev's toric code
- **Phase Identification**: Distinguish between Z2, Non-Abelian, and Trivial topological phases

### Interactive Visualization
A new visualization tool provides interactive exploration of topological quantum phenomena with four different views:
- Berry curvature visualization with Chern number calculation
- Toric code error correction with plaquette and vertex operators
- Majorana zero modes in a circular chain configuration
- Topological entanglement entropy with Kitaev-Preskill construction

### Performance Enhancements
- NEON SIMD optimizations for matrix operations
- Multiple build targets for different hardware architectures:
  - ARM-specific build with NEON acceleration
  - Generic build for maximum compatibility
  - Universal build with runtime detection

### New Command-Line Options
- `--calculate-entropy`: Calculate topological entanglement entropy
- `--calculate-invariants`: Calculate topological invariants
- `--use-error-correction`: Enable toric code error correction
- `--majorana-chain-length N`: Set the length of the Majorana chain
- `--toric-code-size X Y`: Set the dimensions of the toric code lattice

## Documentation
- Comprehensive documentation for all topological features
- Detailed examples demonstrating different topological phases
- Interactive visualization guide

## Technical Improvements
- Enhanced matrix operations with optimized algorithms for eigenvalue calculations
- Improved memory management for large quantum simulations
- Optimized cache utilization in matrix operations
- Enhanced numerical stability in Berry phase calculations

## Example Scripts
- `run_topological_examples.sh`: Run examples for different topological phases
- `different_topo_values.sh`: Demonstrate varying topological invariant values

## Getting Started
See the [README.md](README.md) for installation instructions and usage examples.

## Full Changelog
For a complete list of changes, see the [CHANGELOG.md](CHANGELOG.md).
