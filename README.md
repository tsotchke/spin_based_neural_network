# Spin-Based Neural Computation Framework

## Overview
The **Spin-Based Neural Computation Framework** is an advanced simulation library that combines quantum mechanics, reinforcement learning, and energy-based neural networks to study and model physical phenomena. Designed for researchers in AI, scientific computing, and statistical physics, this framework explores spin systems—such as the Ising, Kitaev, and disordered models—through machine learning and physics-informed neural networks. It has wide-ranging applications in quantum computing, materials science, and artificial intelligence, making it a versatile and powerful tool.

With the recent implementation of topological quantum features based on Majorana Zero Modes, this framework now supports simulation of topological quantum computing systems with error correction capabilities through toric codes, calculation of topological invariants, and the study of topological entanglement entropy.

> **v0.4 foundation note.** v0.4 significantly advances the
> topological-QC physics: Majorana braiding now has a full Hilbert-space
> unitary implementation with verified non-Abelian statistics, and the
> toric code gains an explicit data-qubit model with a greedy MWPM
> baseline decoder. See [docs/architecture_v0.4.md](docs/architecture_v0.4.md)
> for the v0.5 roadmap (neural-network quantum states, equivariant
> Landau-Lifshitz dynamics, learned surface-code decoders,
> Fibonacci-anyon gates).
>
> **v0.4.1 update.** Two new Hamiltonian kernels in the NQS pillar:
> `NQS_HAM_KITAEV_HEISENBERG` (brick-wall honeycomb KH, real +
> complex amplitude paths) and `NQS_HAM_KAGOME_HEISENBERG` (kagome
> Heisenberg on a three-sublattice geometry). The kagome kernel is
> *infrastructure* for the open kagome-S=½ ground-state question
> (gapped Z₂ vs gapless Dirac spin liquid) — it evaluates the
> Hamiltonian on kagome bonds; the scientific decision still depends
> on ansatz choice, symmetry projection, and finite-size scaling.
> `LIBIRREP_MIN` bumped to 1.3.0-alpha for the incoming p6mm +
> batched-RDM primitives. Full suite 343 / 343, AddressSanitizer +
> UndefinedBehaviorSanitizer clean.
>
> **v0.4.1 follow-up (kagome diagnostics pipeline).** Three new
> sample-based diagnostics and a spin-gap solver land on top of the
> kagome kernel:
> - `nqs_compute_chi_F` — Tr(S)/2 of the quantum geometric tensor,
>   the fidelity-susceptibility convergence / transition probe.
> - `nqs_compute_kagome_bond_phase` — per-bond-class ⟨ψ(s_{ij})/ψ(s)⟩
>   for Marshall-like vs Dirac-compatible phase structure.
> - `nqs_sr_{step,run}_excited` — excited-state VMC via the Choo–
>   Neupert–Carleo 2018 orthogonal-ansatz penalty; validated to
>   4-decimal agreement with the exact triplet on 2-site Heisenberg.
> - `nqs_lanczos_k_lowest_kagome_heisenberg` — multi-Ritz Lanczos on
>   the full 2^N Hilbert space (dim=4096 at N=12). On our 2×2 PBC
>   cluster: E₀ = **−5.44487522 J**, E₁ = −5.32839240 J, exact
>   spin gap **Δ = 0.116483 J**. Anchors every VMC energy estimate
>   against a machine-precision reference and gives the 5-diagnostic
>   protocol's spin-gap probe exact targets at N=12, 18, 24.
>
> End-to-end driver: `make research_kagome_N12_diagnostics` chains
> GS SR → χ_F → bond-phase → excited-SR → Lanczos-exact E₀/E₁/gap
> in one research artefact. See
> [docs/research/kagome_KH_plan.md](docs/research/kagome_KH_plan.md).

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
├── build/                       # Generated locally; not tracked
│   ├── spin_based_neural_computation          # Universal build with runtime NEON detection
│   ├── spin_based_neural_computation_arm      # ARM-specific build with NEON enabled
│   ├── spin_based_neural_computation_generic  # Generic build with NEON disabled
│   ├── topo_example                           # Topological phases example
│   ├── test_*                                 # one binary per tests/test_*.c (see Makefile)
│   └── bench_*                                # one binary per benchmarks/bench_*.c (see Makefile)
├── src/
│   ├── main.c                   # Driver / CLI entry point
│   ├── ising_model.c            # 3D Ising model
│   ├── kitaev_model.c           # Anisotropic Kitaev model
│   ├── disordered_model.c       # Quenched disorder injection
│   ├── spin_models.c            # Continuous SpinLattice type
│   ├── energy_utils.c           # Sigmoid energy scale / unscale
│   ├── neural_network.c         # Legacy MLP (Adam, batch norm)
│   ├── nn_backend.c             # v0.4 polymorphic NN handle
│   ├── engine_adapter.c         # v0.4 engine-neutral bridge (dormant by default)
│   ├── eshkol_bridge.c          # v0.4 Eshkol FFI bridge (dormant by default)
│   ├── physics_loss.c           # 5 physics-informed PDE residual losses
│   ├── reinforcement_learning.c # Reactive reward-based RL
│   ├── quantum_mechanics.c      # Noise + entanglement helpers
│   ├── majorana_modes.c         # Majorana fermions + Hilbert-space braiding (v0.4)
│   ├── toric_code.c             # Data-qubit model + greedy decoder (v0.4)
│   ├── berry_phase.c            # Berry phase / Chern / winding / TKNN
│   ├── topological_entropy.c    # Von Neumann + topological entanglement entropy
│   ├── ising_chain_qubits.c     # Topological qubits from Majorana chains
│   ├── matrix_neon.c            # ARM NEON SIMD kernels
│   ├── topological_example.c    # Standalone topological-phases demo
│   ├── visualization.c          # SDL2 viewer (4 modes)
│   └── visualization_main.c     # Visualization entry point
├── include/                     # public headers (one per src module; count tracks src/)
├── tests/                       # test_*.c suites (run all via `make test`)
│   ├── harness.h
│   └── test_*.c
├── benchmarks/                  # bench_*.c suites, JSON results
│   ├── bench_common.h
│   ├── bench_*.c
│   └── results/                 # Per-suite JSON output
├── scripts/
│   ├── run_benchmarks.sh        # Builds and runs every benchmark suite
│   └── check_stack.sh           # Probes for optional external dependencies
├── docs/
│   ├── architecture_v0.4.md     # v0.4 architecture + v0.5+ research roadmap
│   ├── testing.md               # Test harness reference + per-suite coverage
│   ├── benchmarks.md            # JSON schema + reference numbers
│   ├── classical_models.md      # Ising / Kitaev / disordered / spin models (v0.4)
│   ├── training.md              # nn_backend, cadence flags, physics losses (v0.4)
│   ├── engine_integration.md    # engine_adapter + eshkol_bridge integration (v0.4)
│   ├── topological_features.md  # Overview of topological features
│   ├── berry_phase.md           # Berry phase + topological invariants
│   ├── majorana_zero_modes.md   # Majorana modes + braiding
│   ├── topological_entropy.md   # Entanglement entropy calculations
│   ├── toric_code.md            # Toric code QEC
│   └── visualization.md         # SDL2 viewer
├── Makefile
├── README.md
├── RELEASE_NOTES.md
├── CHANGELOG.md
└── CONTRIBUTING.md
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

# Clean all local build outputs
make clean
```

All local build outputs are generated under `build/` and are not shipped
in the repository.

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
./build/visualization
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

The Makefile can generate multiple local executables optimized for
different platforms:

```bash
# Universal binary with runtime detection of NEON capabilities
./build/spin_based_neural_computation [OPTIONS]

# ARM-specific binary with NEON acceleration enabled
./build/spin_based_neural_computation_arm [OPTIONS]

# Generic binary without NEON acceleration (for maximum compatibility)
./build/spin_based_neural_computation_generic [OPTIONS]

# Topological example program
./build/topo_example
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
    ./build/spin_based_neural_computation --loss-type heat --activation relu
    ```

2. **Run with 500 iterations and verbose output, using the Schrödinger loss function:**

    ```bash
    ./build/spin_based_neural_computation -i 500 -v --loss-type schrodinger  --activation sigmoid
    ```

3. **Specify lattice size and coupling constants for a Kitaev model simulation:**

    ```bash
    ./build/spin_based_neural_computation --lattice-size 15 15 15 --jx 0.5 --jy 1.0 --jz -0.8
    ```

4. **Use the Navier-Stokes loss function with Tanh activation, saving output to a custom log file:**

    ```bash
    ./build/spin_based_neural_computation --loss-type navier_stokes --activation tanh --log custom_output.log
    ```

5. **Set custom spatial and time steps, with a specific initial state configuration:**

    ```bash
    ./build/spin_based_neural_computation --dx 0.05 --dt 0.02 --initial-state all-up --activation relu
    ```

6. **Navier-Stokes Simulation with Sigmoid Activation**:
    ```bash
    ./build/spin_based_neural_computation --iterations 500 --verbose --lattice-size 15 15 15 --jx 1.5 --jy 0.8 --jz -0.5 --initial-state random --dx 0.05 --dt 0.02 --loss-type navier_stokes --log navier_stokes.log --activation sigmoid
    ```

7. **Heat Equation Simulation with ReLU Activation**:
    ```bash
    ./build/spin_based_neural_computation --lattice-size 30 30 30 --loss-type heat --activation relu --initial-state all-down
    ```

8. **Simulate Z2 Topological Insulator with Chern Number = 1**:
    ```bash
    ./build/spin_based_neural_computation --iterations 2 --verbose --calculate-entropy --calculate-invariants --use-error-correction --majorana-chain-length 3 --toric-code-size 2 2
    ```

9. **Simulate Quantum Spin Hall Effect with Chern Number = 2**:
    ```bash
    CHERN_NUMBER=2 ./build/spin_based_neural_computation --iterations 2 --verbose --calculate-entropy --calculate-invariants --use-error-correction --majorana-chain-length 5 --toric-code-size 3 3
    ```

10. **Simulate Fractional Quantum Hall Effect with Chern Number = 1/3**:
    ```bash
    CHERN_NUMBER=0.333 ./build/spin_based_neural_computation --iterations 2 --verbose --calculate-entropy --calculate-invariants --use-error-correction --majorana-chain-length 7 --toric-code-size 4 4
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


The current release is **v0.4.1** — a capability-addition release
layered on top of the v0.4.0 foundation.

**v0.4.1 additions (on top of v0.4.0):**

- **Kitaev-Heisenberg kernel on brick-wall honeycomb**
  (`NQS_HAM_KITAEV_HEISENBERG`). Real + complex-amplitude paths,
  Chaloupka–Jackeli–Khaliullin sign convention, reduces to
  Heisenberg at `K=0` and to pure Kitaev at `J=0`. Config fields
  `cfg.kh_K`, `cfg.kh_J`.
- **Kagome Heisenberg kernel**
  (`NQS_HAM_KAGOME_HEISENBERG`). Three-sublattice kagome geometry,
  PBC (default) or OBC, N = 3·Lx·Ly. The target Hamiltonian for the
  open kagome-S=½ ground-state problem.
- **Nine new analytical checkpoints + one SR-loop convergence test**
  for the two kernels.
- **`LIBIRREP_MIN`** bumped 1.2 → 1.3.0-alpha.
- **Full suite**: 343 / 343 passing, zero warnings under
  `-Wall -Wextra`, AddressSanitizer + UndefinedBehaviorSanitizer
  clean.

See [RELEASE_NOTES.md](RELEASE_NOTES.md) for the full v0.4.1 notes.

**v0.4.0 highlights:**

- **Hilbert-space Majorana-anyon braiding.** The new
  `apply_braid_unitary()` operates as a genuine unitary
  `B = (1 + γ_i γ_j) / √2` on the `2^(N/2)`-dim fermion-parity Fock
  basis, with `B⁴ = −I` and `B⁸ = I` (order-8 Ising-anyon statistics)
  verified by test. The v0.3 operator-space braiding path is retained
  as `braid_majorana_operators_legacy()`.
- **Toric-code error correction with explicit data qubits.** Per-link
  `x_errors` / `z_errors` accumulators, greedy matching decoder with
  primal-vs-dual path walks, and homology-based logical-error
  detection. Serves as the MWPM baseline for the learned decoder
  coming in v0.5 (pillar P1.3).
- **Engine-neutral scaffolding.** New `engine_adapter`,
  `eshkol_bridge`, and `nn_backend` modules. All compile dormant today
  (`#ifdef SPIN_NN_HAS_ENGINE` guards); plug in an external NN / tensor /
  reasoning engine (an Eshkol-native NN engine is the primary planned
  target; Noesis is a candidate once released) in v0.5+.
- **Training-loop cadences.** `--cadence-decoder`, `--cadence-invariants`,
  `--lambda-logical` flags fold decoder logical-error signals into the
  physics loss as soft penalties during training.
- **Assert-based test suite** (`tests/test_*.c`, one binary per file,
  driven by `harness.h`) covering every library module. Run all via
  `make test`; see `docs/testing.md` for the per-module coverage map.
- **Benchmark harness** (`benchmarks/bench_*.c`) emitting JSON under
  `benchmarks/results/`. Run `./scripts/run_benchmarks.sh`.

See [RELEASE_NOTES.md](RELEASE_NOTES.md) for the full details,
[CHANGELOG.md](CHANGELOG.md) for history, and
[docs/architecture_v0.4.md](docs/architecture_v0.4.md) for the forward
roadmap (v0.5 research pillars: ViT wavefunctions for frustrated magnets,
equivariant Landau-Lifshitz-Gilbert dynamics, learned surface-code
decoders, Fibonacci-anyon universal gates, neural-operator Boltzmann
samplers, and more).

## Citation
If you use this project in your research, please cite as follows:

```bibtex
@software{SpinBasedNeuralComputation,
  author = {tsotchke},
  title = {Spin-Based Neural Computation Framework: Simulations of Topological Quantum Computing},
  version = {0.4.0},
  year = {2026},
  url = {https://github.com/tsotchke/spin_based_neural_network}
}
```

## References

Foundational physics and algorithms cited across the framework. Per-topic
references are also carried in the relevant files under `docs/`.

### Topological quantum computing
1. A. Y. Kitaev, "Fault-tolerant quantum computation by anyons," *Annals of Physics*, vol. 303, pp. 2–30, 2003.
2. A. Y. Kitaev, "Unpaired Majorana fermions in quantum wires," *Physics-Uspekhi*, vol. 44, pp. 131–136, 2001.
3. A. Y. Kitaev and J. Preskill, "Topological Entanglement Entropy," *Physical Review Letters*, vol. 96, p. 110404, 2006.
4. E. Dennis, A. Kitaev, A. Landahl, and J. Preskill, "Topological quantum memory," *Journal of Mathematical Physics*, vol. 43, pp. 4452–4505, 2002.
5. C. Nayak, S. H. Simon, A. Stern, M. Freedman, and S. Das Sarma, "Non-Abelian anyons and topological quantum computation," *Reviews of Modern Physics*, vol. 80, pp. 1083–1159, 2008.
6. S. Das Sarma, M. Freedman, and C. Nayak, "Majorana zero modes and topological quantum computation," *npj Quantum Information*, vol. 1, p. 15001, 2015.
7. D. A. Ivanov, "Non-Abelian statistics of half-quantum vortices in p-wave superconductors," *Physical Review Letters*, vol. 86, pp. 268–271, 2001.
8. P. Jordan and E. Wigner, "Über das Paulische Äquivalenzverbot," *Zeitschrift für Physik*, vol. 47, pp. 631–651, 1928.

### Berry phase, Chern numbers, and topological invariants
9. M. V. Berry, "Quantal phase factors accompanying adiabatic changes," *Proceedings of the Royal Society of London A*, vol. 392, pp. 45–57, 1984.
10. D. J. Thouless, M. Kohmoto, M. P. Nightingale, and M. den Nijs, "Quantized Hall Conductance in a Two-Dimensional Periodic Potential," *Physical Review Letters*, vol. 49, pp. 405–408, 1982.
11. F. D. M. Haldane, "Model for a quantum Hall effect without Landau levels," *Physical Review Letters*, vol. 61, pp. 2015–2018, 1988.
12. X.-L. Qi and S.-C. Zhang, "Topological insulators and superconductors," *Reviews of Modern Physics*, vol. 83, pp. 1057–1110, 2011.

### Graph-theoretic algorithms
13. J. Edmonds, "Paths, trees, and flowers," *Canadian Journal of Mathematics*, vol. 17, pp. 449–467, 1965.
14. A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. N. Cleland, "Surface codes: Towards practical large-scale quantum computation," *Physical Review A*, vol. 86, p. 032324, 2012.

### Learned quantum error correction (v0.5 pillar P1.3)
15. J. Bausch, A. Senior, F. Heras, T. Edlich, A. Davies, M. Newman, C. Jones, K. Satzinger, M. Y. Niu, S. Blackwell, G. Holland, D. Kafri, J. Atalaya, C. Gidney, D. Hassabis, S. Boixo, H. Neven, and P. Kohli, "Learning high-accuracy error decoding for quantum processors," *Nature*, vol. 635, pp. 834–840, 2024. DOI: 10.1038/s41586-024-08148-8.
16. V. Ninkovic, O. Kundacina, D. Vukobratovic, and C. Häger, "Scalable Neural Decoders for Practical Real-Time Quantum Error Correction," arXiv:2510.22724, 2025.

### Neural Network Quantum States (v0.5 pillar P1.1)
17. G. Carleo and M. Troyer, "Solving the quantum many-body problem with artificial neural networks," *Science*, vol. 355, pp. 602–606, 2017.
18. S. Sorella, "Green Function Monte Carlo with Stochastic Reconfiguration," *Physical Review Letters*, vol. 80, pp. 4558–4561, 1998.
19. R. Rende, L. Viteritti, L. Bardone, F. Becca, and S. Goldt, "A simple linear algebra identity to optimize large-scale neural network quantum states," *Communications Physics*, 2024. arXiv:2310.05715.
20. A. Chen and M. Heyl, "Empowering deep neural quantum states through efficient optimization," *Nature Physics*, vol. 20, pp. 1476–1481, 2024.

### Equivariant neural networks for spin dynamics (v0.5 pillar P1.2)
21. S. Batzner, A. Musaelian, L. Sun, M. Geiger, J. P. Mailoa, M. Kornbluth, N. Molinari, T. E. Smidt, and B. Kozinsky, "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials," *Nature Communications*, vol. 13, p. 2453, 2022.
22. I. Batatia, D. P. Kovacs, G. N. C. Simm, C. Ortner, and G. Csanyi, "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields," *NeurIPS 2022*. arXiv:2206.07697.

### Fibonacci anyons (v0.5 pillar P1.3b)
23. A. Y. Kitaev, A. H. Shen, and M. N. Vyalyi, *Classical and Quantum Computation*, AMS Graduate Studies in Mathematics, vol. 47, 2002 (Solovay–Kitaev algorithm).
24. S. Xu *et al.*, "Non-Abelian braiding of Fibonacci anyons with a superconducting processor," *Nature Physics*, vol. 20, pp. 1469–1475, 2024. arXiv:2404.00091.
25. Z. K. Minev, K. Najafi, S. Majumder, J. Wang, A. Stern, E.-A. Kim, C.-M. Jian, and G. Zhu, "Realizing string-net condensation: Fibonacci anyon braiding for universal gates and sampling chromatic polynomials," *Nature Communications*, vol. 16, article 6225, 2025. arXiv:2406.12820.

### Neural operators and flow matching (v0.5 pillar P1.4)
26. Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. M. Stuart, and A. Anandkumar, "Fourier Neural Operator for Parametric Partial Differential Equations," *ICLR 2021*. arXiv:2010.08895.
27. Y. Cai, J. Li, and D. Wang, "NeuralMAG: Fast and Generalizable Micromagnetic Simulation with Deep Neural Nets," arXiv:2410.14986, 2024.
28. Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, "Flow Matching for Generative Modeling," *ICLR 2023*. arXiv:2210.02747.

### Original framework reference
29. tsotchke, "Majorana Zero Modes in Topological Quantum Computing: Error-Resistant Codes Through Dynamical Symmetries," 2022.

See `docs/architecture_v0.4.md` §7 for a combined bibliography that also
indexes references used in the v0.5+ research pillars.

## License
This project is licensed under the MIT License.

## Conclusion
The **Spin-Based Neural Computation Framework** provides a versatile platform for studying quantum systems, complex spin interactions, and energy-based learning model approaches to neural networks, making it ideal for researchers exploring quantum computing, spintronics, mathematical physics, complexity, and AI. By integrating reinforcement learning and comprehensive spin models, enabling exploration of quantum-inspired neural networks and physics-informed learning, it encourages insightful exploration into physical systems and computational physics.
