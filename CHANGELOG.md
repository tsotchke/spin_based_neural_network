# Changelog

All notable changes to the Spin-Based Neural Computation Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - Foundation for v0.5 research pillars

### Added
- **Hilbert-space Majorana braiding** — new `MajoranaHilbertState` over the
  `2^(N/2)`-dim fermion-parity Fock basis plus `apply_braid_unitary()`
  implementing `B_{ij} = (1 + γ_i γ_j)/√2`. Verified `B^4 = -I`,
  `B^8 = I`, anticommutation, unitarity, parity conservation. The v0.3
  operator-space braiding path is retained as
  `braid_majorana_operators_legacy()` and `braid_majorana_modes()` still
  works unchanged.
- **Toric-code data-qubit model** — per-link `x_errors` / `z_errors`
  accumulators, `toric_code_apply_{x,z}_{error,correction}`,
  homology-based `toric_code_has_logical_error()`, greedy
  matching-baseline decoder `toric_code_decode_greedy()` with primal /
  dual path walks. Serves as the MWPM baseline for the learned decoder
  coming in v0.5 pillar P1.3.
- **`engine_adapter` scaffolding** (`src/engine_adapter.c`,
  `include/engine_adapter.h`) — engine-neutral bridge between the spin
  framework and an external NN / tensor / reasoning engine. Planned
  backends: an Eshkol-native NN engine (working title
  `eshkol-transformers`, built on https://github.com/tsotchke/eshkol)
  and Noesis (reasoning engine, in development, not yet publicly
  released). All entry points compile today behind
  `#ifdef SPIN_NN_HAS_ENGINE`; enable with
  `make ENGINE_ENABLE=1 ENGINE_ROOT=...` once a chosen engine is
  available.
- **`eshkol_bridge`** (`src/eshkol_bridge.c`, `include/eshkol_bridge.h`) —
  lazy wrapper over the Eshkol FFI; compiles without Eshkol present,
  wires up once `-DSPIN_NN_HAS_ESHKOL=1` is enabled in v0.5.
- **`nn_backend`** (`src/nn_backend.c`, `include/nn_backend.h`) —
  polymorphic neural-network handle with `NN_BACKEND_LEGACY` and
  `NN_BACKEND_ENGINE` variants, and `--nn-backend={legacy,engine}` CLI flag.
- **Training-loop cadences** — `--cadence-decoder N`, `--decoder-error-rate P`,
  `--cadence-invariants N`, `--lambda-logical L`. Decoder logical-error
  flag folds into `physics_loss` as a soft penalty during training.
- **Test harness** — TAP-style `tests/harness.h` plus 17 suites covering
  18 of 18 library modules (109 tests total). `make test` runs them all.
  See `docs/testing.md`.
- **Benchmark harness** — 4 suites (`bench_ising`, `bench_kitaev`,
  `bench_majorana_braid`, `bench_toric_decoder`) emitting JSON under
  `benchmarks/results/<suite>/`. `scripts/run_benchmarks.sh` orchestrates.
- **`scripts/check_stack.sh`** — advisory probe for optional stack
  components: external engine, libirrep, Eshkol runtime. All missing
  dependencies are informational in v0.4.

### Improved
- Toric-code `perform_error_correction` now delegates to the greedy
  data-qubit decoder, flipping data qubits directly and re-deriving
  syndromes so iterated error / correction cycles stay consistent.
- Majorana braiding is now available as a Hilbert-space unitary (the v0.3
  operator-space path is retained as
  `braid_majorana_operators_legacy`).
- `toric_code_has_logical_error()` uses homology-class winding numbers
  against primal / dual basis cycles, which are invariant under stabilizer
  action.

### Changed
- `CHERN_NUMBER` environment-variable override in `src/berry_phase.c` is now
  gated behind `#ifdef SPIN_NN_TESTING`; release builds do not ship the
  back-door.
- Neural-network creation in `main.c` goes through `spin_nn_create()`;
  legacy behavior is preserved by default.
## [0.3.0] - 2025-04-08

### Added
- **Topological Quantum Computing Features**:
  - Berry phase and curvature calculations
  - Topological invariant determination (Chern numbers, TKNN invariant, winding numbers)
  - Majorana zero mode simulation with braiding operations
  - Topological entanglement entropy measurements
  - Toric code implementation with basic error correction
  - Example program demonstrating different topological phases (Z2, Non-Abelian, Trivial)
- **Visualization Tool**:
  - Interactive visualization of Berry curvature with Chern number calculation
  - Toric code error correction visualization with plaquette and vertex operators
  - Majorana zero modes visualization in a circular chain configuration
  - Topological entanglement entropy visualization with Kitaev-Preskill construction
- **Command-Line Options**:
  - `--calculate-entropy`: Calculate topological entanglement entropy
  - `--calculate-invariants`: Calculate topological invariants
  - `--use-error-correction`: Enable toric code error correction
  - `--majorana-chain-length N`: Set the length of the Majorana chain
  - `--toric-code-size X Y`: Set the dimensions of the toric code lattice
- **Performance Optimizations**:
  - NEON SIMD optimizations for matrix operations
  - Multiple build targets for different hardware capabilities
  - Improved von Neumann entropy calculation with vectorized operations
- **Documentation**:
  - Comprehensive documentation for all topological features
  - Example scripts showcasing different topological phases
  - Installation and usage instructions for all features

### Changed
- Enhanced matrix operations with optimized algorithms for eigenvalue calculations
- Improved memory management for large quantum simulations
- Optimized cache utilization in matrix operations
- Refined command-line interface for better usability

### Fixed
- Numerical stability in Berry phase calculations near phase transitions
- Memory leaks in large lattice simulations
- Improved error handling in critical computational paths
- Edge cases in topological invariant calculations

## [0.2.0] - 2024-11-15

### Added
- Neural network integration with physics-based loss functions
- Energy-based learning for spin system optimization
- Reinforcement learning capabilities for state optimization
- Physics-informed loss functions (heat, Schrödinger, Maxwell, Navier-Stokes, wave)
- Multiple activation functions (ReLU, tanh, sigmoid)
- Extended command-line interface with more configuration options

### Changed
- Improved matrix operation performance
- Enhanced logging for better reproducibility
- Restructured code for better modularity

### Fixed
- Memory management issues in spin model simulations
- Numerical stability in quantum calculations
- Parameter validation in command-line interface

## [0.1.0] - 2024-06-22

### Added
- Initial release with core spin model simulation capabilities
- Comprehensive implementation of spin models:
  - Ising model
  - Kitaev model
  - Disordered model
- Basic command-line interface
- Logging and reproducibility features
