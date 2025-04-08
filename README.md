# Spin-Based Neural Computation Framework

## Overview
The **Spin-Based Neural Computation Framework** is an advanced simulation library that combines quantum mechanics, reinforcement learning, and energy-based neural networks to study and model physical phenomena. Designed for researchers in AI, scientific computing, and statistical physics, this framework explores spin systems—such as the Ising, Kitaev, and disordered models—through machine learning and physics-informed neural networks. It has wide-ranging applications in quantum computing, materials science, and artificial intelligence, making it a versatile and powerful tool.

With the recent implementation of topological quantum features based on Majorana Zero Modes, this framework now supports simulation of topological quantum computing systems with error correction capabilities through toric codes, calculation of topological invariants, and the study of topological entanglement entropy.

## Project Highlights
- **Quantum Physics and Machine Learning**: Leverage spin-based neural networks to simulate and learn from quantum systems.
- **Comprehensive Spin Models**: Implements Ising, Kitaev, and disordered spin models with customizable lattice parameters and interaction strengths.
- **Topological Quantum Computing**: Simulate Majorana zero modes, calculate topological invariants, and implement toric code error correction.
- **Interactive Visualization**: Visual representation of topological features including Berry curvature, toric code, Majorana zero modes, and topological entanglement entropy.
- **Energy-Driven Neural Networks**: Inspired by Boltzmann machines, these networks learn from physical principles of spin configuration energies.
- **Reinforcement Learning Integration**: Dynamically optimize spin configurations using reinforcement learning, enabling robust energy minimization.
- **Customizable Parameters and Loss Functions**: Tailor simulations with specific lattice dimensions, couplings, activation functions, and physics-based loss functions.
- **Extensive Logging**: Ensure reproducibility with detailed logs, supporting research and analysis.
- **Multiple Build Options**: Optimized builds for different hardware capabilities, including ARM NEON acceleration.

## Key Components

### 1. Neural Networks
- **Thermodynamic Energy Minimization**: Learn optimal spin states based on energy minimization, leveraging principles from statistical mechanics.
- **Dynamic Structure**: Configurable neural network layers and neuron counts to match simulation complexity requirements.

### 2. Spin Models
- **Ising Model**: A classical model for ferromagnetic interactions, extended to three-dimensional lattices for detailed phase transition studies.
- **Kitaev Model**: A model with applications to topological quantum computing, allowing for exploration of spin liquids and complex quantum states.
- **Spin Model**: A model incorporating classical spin and random disorder, useful for studying real-world material impurities.

### 3. Quantum Mechanics
- **Quantum State Calculations**: Calculate properties of quantum spins, including superposition and entanglement, integrating quantum effects directly with neural networks.
- **Hybrid Quantum-Classical Systems**: Model interactions between quantum and classical elements in spin systems.

### 4. Topological Quantum Computing
- **Berry Phase Calculations**: Compute the Berry phase and curvature to determine topological invariants like Chern numbers and TKNN invariant.
- **Majorana Zero Modes**: Simulate Majorana fermions and their braiding properties as described in the Majorana Zero Modes paper.
- **Toric Code Implementation**: Error correction through the implementation of Kitaev's toric code with plaquette and vertex operators.
- **Topological Entanglement Entropy**: Calculate the quantized entanglement entropy to identify topological order in quantum systems.

### 5. Reinforcement Learning
- **Policy and Value Optimization**: Adapts spin configurations based on calculated rewards, enabling energy-efficient configurations.
- **Energy-Based Learning**: Works in tandem with neural networks to achieve refined spin configurations and energy states.

## Directory Structure
The project is organized for clarity and usability:

```
spin_based_neural_network/
├── bin/                       # Binary executables directory
│   ├── spin_based_neural_computation          # Universal build with runtime NEON detection
│   ├── spin_based_neural_computation_arm      # ARM-specific build with NEON enabled
│   ├── spin_based_neural_computation_generic  # Generic build with NEON disabled
│   └── topo_example                           # Topological example program
├── src/
│   ├── berry_phase.c          # Berry phase and topological invariant calculations
│   ├── disordered_model.c     # Disordered spin model implementation
│   ├── energy_utils.c         # Energy calculation utilities
│   ├── ising_chain_qubits.c   # Ising-based qubit implementation
│   ├── ising_model.c          # Classical Ising model
│   ├── kitaev_model.c         # Kitaev model implementation
│   ├── main.c                 # Main application entry point
│   ├── majorana_modes.c       # Majorana fermion simulation
│   ├── matrix_neon.c          # NEON-optimized matrix operations
│   ├── neural_network.c       # Neural network implementation
│   ├── physics_loss.c         # Physics-informed loss functions
│   ├── quantum_mechanics.c    # Quantum mechanics utilities
│   ├── reinforcement_learning.c # Reinforcement learning implementation
│   ├── spin_models.c          # General spin model utilities
│   ├── topological_entropy.c  # Entanglement entropy calculations
│   ├── topological_example.c  # Example program for different topological phases
│   └── toric_code.c           # Error correction code implementation
├── include/
│   ├── berry_phase.h          # API for topological invariants
│   ├── disordered_model.h     # Disordered model interface
│   ├── energy_utils.h         # Energy utilities interface
│   ├── ising_chain_qubits.h   # Ising qubit interface
│   ├── ising_model.h          # Ising model interface
│   ├── kitaev_model.h         # Kitaev model interface
│   ├── majorana_modes.h       # Majorana modes interface
│   ├── neural_network.h       # Neural network interface
│   ├── physics_loss.h         # Physics loss functions interface
│   ├── quantum_mechanics.h    # Quantum mechanics interface
│   ├── reinforcement_learning.h # RL interface
│   ├── spin_models.h          # Spin models interface
│   ├── topological_entropy.h  # Entropy calculations interface
│   └── toric_code.h           # Toric code interface
├── docs/
│   ├── topological_features.md # Overview of topological features
│   ├── berry_phase.md          # Berry phase documentation
│   ├── majorana_zero_modes.md  # Majorana modes documentation
│   ├── topological_entropy.md  # Entropy calculations documentation
│   └── toric_code.md           # Toric code documentation
├── Makefile                    # Build configuration
└── README.md                   # Project overview
```

## Installation
The framework is written in C and has no external dependencies (with the exception of SDL2 for the visualization), ensuring compatibility across a variety of systems. Compile with a standard C compiler, such as `gcc`, for easy setup:

### Build Options

```bash
# Build both ARM and non-ARM versions (default)
make

# Build ARM-specific version with NEON acceleration
make arm

# Build generic version without NEON
make non_arm

# Build universal version with runtime NEON detection
make universal

# Build all versions
make all_versions

# Build topological example program
make topo_example

# Build visualization tool
make visualization

# Clean all binaries
make clean
```

All compiled binaries are placed in the `bin/` directory.

### Visualization Tool

The framework includes an interactive visualization tool for exploring topological quantum phenomena:

```bash
# First, install SDL2 which is required for the visualization
# For macOS with Homebrew:
brew install sdl2

# For Ubuntu/Debian:
sudo apt-get install libsdl2-dev

# For Fedora/RHEL:
sudo dnf install SDL2-devel

# Then build and run the visualization tool
make visualization
./bin/visualization
```

The visualization provides four different interactive views:
- **Mode 1 (Key 1)**: Berry curvature visualization with Chern number calculation
- **Mode 2 (Key 2)**: Toric code error correction with plaquette and vertex operators
- **Mode 3 (Key 3)**: Majorana zero modes in a circular chain configuration
- **Mode 4 (Key 4)**: Topological entanglement entropy with Kitaev-Preskill construction

Each mode provides a distinct view of topological quantum phenomena with real-time particle effects to visualize quantum fluctuations. The visualization uses SDL2 for rendering and can be controlled with the following keys:
- **1-4**: Switch between visualization modes
- **ESC**: Exit the visualization

See `docs/visualization.md` for detailed explanations of each visualization mode.

## Usage

We provide multiple executable versions optimized for different platforms:

```bash
# Universal binary with runtime detection of NEON capabilities
bin/spin_based_neural_computation [OPTIONS]

# ARM-specific binary with NEON acceleration enabled
bin/spin_based_neural_computation_arm [OPTIONS]

# Generic binary without NEON acceleration (for maximum compatibility)
bin/spin_based_neural_computation_generic [OPTIONS]

# Topological example program
bin/topo_example
```

### Command Line Arguments
- `-i, --iterations N`: Number of iterations to run (default: 100)
- `-v, --verbose`: Enable detailed output
- `--lattice-size X Y Z`: Set lattice dimensions (default: 10x10x10)
- `--jx JX, --jy JY, --jz JZ`: Set coupling constants for the Kitaev model (default: 1.0, 1.0, -1.0)
- `--initial-state STATE`: Initial state configuration (options: `random`, `all-up`)
- `--dx DX, --dt DT`: Spatial and time step sizes (default: 0.1)
- `--loss-type TYPE`: Specify physics-informed loss (options: `heat`, `schrodinger`, `maxwell`, `navier_stokes`, `wave`)
- `--activation FUNC`: Choose activation function (options: `relu`, `tanh`, `sigmoid`)
- `--log LOG_FILE`: Specify log file for output (default: `simulation.log`)

#### Topological Quantum Computing Options
- `--calculate-entropy`: Calculate topological entanglement entropy
- `--calculate-invariants`: Calculate topological invariants (Chern number, TKNN invariant, winding number)
- `--use-error-correction`: Enable toric code error correction
- `--majorana-chain-length N`: Set the length of the Majorana chain (default: 3)
- `--toric-code-size X Y`: Set the dimensions of the toric code lattice (default: 2 2)

#### Environment Variables
- `CHERN_NUMBER`: Override the calculated Chern number with a specific value (useful for testing different topological phases)
- `DEBUG_ENTROPY`: Enable detailed debugging output for entropy calculations
- `MAJORANA_DEBUG`: Enable debug information for Majorana zero mode operations

### Example Usage
1. **Run a basic simulation with default settings:**

    ```bash
    ./spin_based_neural_computation --loss-type heat --activation relu
    ```

2. **Run with 500 iterations and verbose output, using the Schrödinger loss function:**

    ```bash
    ./spin_based_neural_computation -i 500 -v --loss-type schrodinger  --activation sigmoid
    ```

3. **Specify lattice size and coupling constants for a Kitaev model simulation:**

    ```bash
    bin/spin_based_neural_computation --lattice-size 15 15 15 --jx 0.5 --jy 1.0 --jz -0.8
    ```

4. **Use the Navier-Stokes loss function with Tanh activation, saving output to a custom log file:**

    ```bash
    ./spin_based_neural_computation --loss-type navier_stokes --activation tanh --log custom_output.log
    ```

5. **Set custom spatial and time steps, with a specific initial state configuration:**

    ```bash
    ./spin_based_neural_computation --dx 0.05 --dt 0.02 --initial-state all-up --activation relu
    ```

6. **Navier-Stokes Simulation with Sigmoid Activation**:
    ```bash
    ./spin_based_neural_computation --iterations 500 --verbose --lattice-size 15 15 15 --jx 1.5 --jy 0.8 --jz -0.5 --initial-state random --dx 0.05 --dt 0.02 --loss-type navier_stokes --log navier_stokes.log --activation sigmoid
    ```

7. **Heat Equation Simulation with ReLU Activation**:
    ```bash
    ./spin_based_neural_computation --lattice-size 30 30 30 --loss-type heat --activation relu --initial-state all-down
    ```

8. **Simulate Z2 Topological Insulator with Chern Number = 1**:
    ```bash
    ./spin_based_neural_computation --iterations 2 --verbose --calculate-entropy --calculate-invariants --use-error-correction --majorana-chain-length 3 --toric-code-size 2 2
    ```

9. **Simulate Quantum Spin Hall Effect with Chern Number = 2**:
    ```bash
    CHERN_NUMBER=2 ./spin_based_neural_computation --iterations 2 --verbose --calculate-entropy --calculate-invariants --use-error-correction --majorana-chain-length 5 --toric-code-size 3 3
    ```

10. **Simulate Fractional Quantum Hall Effect with Chern Number = 1/3**:
    ```bash
    CHERN_NUMBER=0.333 ./spin_based_neural_computation --iterations 2 --verbose --calculate-entropy --calculate-invariants --use-error-correction --majorana-chain-length 7 --toric-code-size 4 4
    ```

11. **Run All Topological Examples**:
    ```bash
    ./run_topological_examples.sh
    ```

### Available Activation Functions and Loss Types
**Activation Functions:**
- `relu`: Rectified Linear Unit, ideal for general purpose networks.
- `tanh`: Hyperbolic tangent, suitable for capturing non-linear dependencies.
- `sigmoid`: Logistic sigmoid, often used for probabilistic outputs.

**Loss Functions:**
- `heat`: Simulates heat diffusion across the lattice.
- `schrodinger`: Models quantum wave functions.
- `maxwell`: Reflects electromagnetic field dynamics.
- `navier_stokes`: Describes fluid flow behavior.
- `wave`: Simulates wave propagation in physical systems.

## Performance Optimizations

The **Spin-Based Neural Computation Framework** is designed for high performance with specific optimizations:

- **NEON SIMD Vectorization**: ARM NEON instructions accelerate matrix operations for complex calculations
- **Platform-Specific Binaries**: Optimized builds for different hardware capabilities:
  - ARM-specific build with NEON explicitly enabled
  - Generic build for maximum compatibility
  - Universal build with runtime detection of hardware capabilities
- **Memory-Efficient Data Structures**: Optimized data structures for handling quantum states
- **Cache-Optimized Algorithms**: Matrix operations designed with cache efficiency in mind
- **Intelligent Memory Management**: Strategic allocation patterns to minimize fragmentation
- **Specialized Eigenvalue Calculations**: Power iteration methods for matrix diagonalization
- **Conditional Compilation**: Runtime detection of hardware capabilities with appropriate fallbacks

The `matrix_neon.c` module contains advanced optimizations for critical computational paths, particularly for matrix operations and eigenvalue calculations that dominate performance in topological simulations.

## Release Information

The current release is version 0.3.0, which focuses on topological quantum computing capabilities. For details on what's included in this release, see the [RELEASE_NOTES.md](RELEASE_NOTES.md) file. For a history of changes, see the [CHANGELOG.md](CHANGELOG.md).

## Citation
If you use this project in your research, please cite as follows:

```bibtex
@software{SpinBasedNeuralComputation,
  author = {tsotchke},
  title = {Spin-Based Neural Computation Framework: Simulations of Topological Quantum Computing},
  version = {0.3.0},
  year = {2025},
  url = {https://github.com/tsotchke/spin_based_neural_network}
}
```

## License
This project is licensed under the MIT License.

## Conclusion
The **Spin-Based Neural Computation Framework** provides a versatile platform for studying quantum systems, complex spin interactions, and energy-based learning model approaches to neural networks, making it ideal for researchers exploring quantum computing, spintronics, mathematical physics, complexity, and AI. By integrating reinforcement learning and comprehensive spin models, enabling exploration of quantum-inspired neural networks and physics-informed learning, it encourages insightful exploration into physical systems and computational physics.
