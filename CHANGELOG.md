# Changelog

All notable changes to the Spin-Based Neural Computation Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
