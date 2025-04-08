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
