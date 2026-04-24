# Spin-Based Neural Computation Framework

A pure-C research framework for quantum spin systems, built around four
pillars:

1. **Neural Network Quantum States (NQS)** — variational wavefunctions
   (MLP, RBM, complex RBM) trained by stochastic reconfiguration on
   TFIM, Heisenberg, XXZ, J1-J2, Kitaev-Heisenberg (brick-wall
   honeycomb), and kagome Heisenberg Hamiltonians. Sample-based
   diagnostics (χ_F, kagome bond-phase), excited-state VMC, and
   full-basis Lanczos refinement for machine-precision exact
   references at N ≤ 24.
2. **Topological quantum computing** — Berry phase, Chern / TKNN /
   winding numbers, Hilbert-space Majorana braiding (order-8
   non-Abelian statistics), toric-code QEC with greedy MWPM baseline,
   Kitaev-Preskill topological entanglement entropy.
3. **Classical spin models** — 3D Ising, anisotropic Kitaev, disordered
   / continuous-spin models; used as substrates for statistical
   mechanics, Monte Carlo training, and as reference Hamiltonians for
   the NQS pillar.
4. **Physics-informed neural networks** — PDE residual losses (heat,
   Schrödinger, Maxwell, Navier-Stokes, wave) and energy-driven
   reinforcement learning on spin lattices.

Target audiences: researchers in strongly-correlated quantum magnetism,
topological phases, quantum error correction, and physics-informed ML.
Pure C11, ARM NEON SIMD kernels, SDL2 for live visualisation, no Python
runtime dependency.

> **Current release: v0.4.2** (2026-04-24). A diagnostic-stack
> addition on top of the v0.4.1 Kitaev-Heisenberg + kagome
> Heisenberg local-energy kernels. Full suite **359 / 359**
> passing, zero warnings under `-Wall -Wextra`, AddressSanitizer +
> UndefinedBehaviorSanitizer clean. See
> [RELEASE_NOTES.md](RELEASE_NOTES.md) for the complete notes.
>
> **What landed (v0.4.2 research capability)**
> - `nqs_compute_chi_F` — fidelity susceptibility χ_F = Tr(S)/2
>   from a sampled batch (Zanardi–Paunković 2006).
> - `nqs_compute_kagome_bond_phase` — per-bond-class amplitude
>   ratios ⟨ψ(s_{ij})/ψ(s)⟩ for Marshall vs Dirac-compatible
>   phase structure.
> - `nqs_sr_{step,run}_excited` — excited-state VMC via the
>   Choo–Neupert–Carleo 2018 (arXiv:1810.10196) orthogonal-
>   ansatz penalty; 4-decimal agreement with the exact 2-site
>   Heisenberg triplet.
> - `nqs_lanczos_k_lowest_kagome_heisenberg` — multi-Ritz
>   Lanczos on the full 2^N Hilbert space (dim = 4096 at
>   N = 12). On our 2×2 PBC cluster: **E₀ = −5.44487522 J**,
>   **E₁ = −5.32839240 J**, **spin gap Δ = 0.116483 J** to
>   machine precision. Anchors every VMC estimate against an
>   exact reference.
> - End-to-end driver `make research_kagome_N12_diagnostics`
>   chains GS SR → χ_F → bond-phase → excited-SR →
>   Lanczos-exact E₀/E₁/gap in one O(20 min) artefact.
>
> See
> [docs/research/kagome_KH_plan.md](docs/research/kagome_KH_plan.md)
> for the research program this stack supports (5-diagnostic
> protocol for the kagome Z₂ vs Dirac spin-liquid question,
> coordinated with [libirrep](https://github.com/tsotchke/libirrep)).
>
> **Cumulative since v0.4.0** — the `NQS_HAM_KITAEV_HEISENBERG`
> and `NQS_HAM_KAGOME_HEISENBERG` local-energy kernels (v0.4.1)
> plus the foundation upgrades from v0.4.0 (Hilbert-space
> Majorana braiding with order-8 non-Abelian statistics, toric
> code with explicit data qubits + greedy MWPM decoder,
> engine-neutral scaffolding for external NN / tensor engines).
> See [CHANGELOG.md](CHANGELOG.md) for the full history and
> [docs/architecture_v0.4.md](docs/architecture_v0.4.md) for the
> v0.5+ roadmap (ViT wavefunctions, equivariant LLG, learned
> surface-code decoders, Fibonacci-anyon gates, neural-operator
> Boltzmann samplers).

## Key Components

### 1. Neural Network Quantum States (NQS)
Variational Monte Carlo ansätze for quantum spin Hamiltonians, with
stochastic reconfiguration (SR) training and Lanczos post-processing.

- **Hamiltonian kernels**: TFIM, Heisenberg, XXZ, J1-J2,
  `NQS_HAM_KITAEV_HEISENBERG` (brick-wall honeycomb, CJK sign
  convention), `NQS_HAM_KAGOME_HEISENBERG` (three-sublattice
  kagome, PBC/OBC). Real + complex-amplitude paths.
- **Ansätze**: legacy MLP, real / complex RBM, with Marshall-sign
  wrapper and translation symmetry wrapper available.
- **Optimizers**: real-projected SR, holomorphic SR for complex
  ansätze, real-time tVMC (forward-Euler + Heun).
- **Diagnostics (v0.4.2)**: sample-based χ_F = Tr(S)/2
  (`nqs_compute_chi_F`), kagome per-bond-class phase probe
  (`nqs_compute_kagome_bond_phase`), excited-state SR via
  orthogonal-ansatz penalty (`nqs_sr_{step,run}_excited`).
- **Exact reference (v0.4.2)**: full-basis Lanczos refinement for
  TFIM, Heisenberg, and kagome Heisenberg
  (`nqs_lanczos_refine_*`), with multi-Ritz extraction for spin
  gaps (`nqs_lanczos_k_lowest_kagome_heisenberg`). Machine-
  precision anchor at N ≤ 24.

See [docs/nqs.md](docs/nqs.md) and
[docs/research/kagome_KH_plan.md](docs/research/kagome_KH_plan.md).

### 2. Spin Models
Classical lattice Hamiltonians used for statistical mechanics,
Monte Carlo training, and as substrates for the NQS pillar.

- **Ising model**: 3D lattice, ferromagnetic / antiferromagnetic
  couplings, Metropolis dynamics.
- **Kitaev model**: Anisotropic honeycomb model with topological
  spin-liquid behaviour; baseline for the Kitaev-Heisenberg kernel.
- **Disordered / continuous-spin model**: Quenched disorder
  injection for spin-glass studies.

See [docs/classical_models.md](docs/classical_models.md).

### 3. Quantum Mechanics
- **State calculations**: Quantum spin expectation values,
  superposition, entanglement entropy.
- **Hybrid quantum–classical systems**: Quantum effects driven by
  classical spin backgrounds.

See [docs/quantum_mechanics.md](docs/quantum_mechanics.md).

### 4. Topological Quantum Computing
- **Berry phase / Chern number / TKNN invariant / winding number**
  calculations on 2D spin systems.
- **Hilbert-space Majorana braiding** (v0.4): genuine unitaries
  `B = (1 + γ_i γ_j)/√2` acting on the 2^{N/2}-dim fermion-parity
  Fock basis, with `B⁴ = −I` and `B⁸ = I` (order-8 non-Abelian
  statistics) verified.
- **Toric-code QEC** (v0.4): explicit data-qubit model +
  greedy-matching decoder with primal/dual path walks and
  homology-based logical-error detection. MWPM baseline for the
  learned decoder in v0.5.
- **Topological entanglement entropy**: Kitaev-Preskill
  construction; quantized γ as a topological-order indicator.

See [docs/topological_features.md](docs/topological_features.md),
[docs/berry_phase.md](docs/berry_phase.md),
[docs/majorana_zero_modes.md](docs/majorana_zero_modes.md),
[docs/toric_code.md](docs/toric_code.md),
[docs/topological_entropy.md](docs/topological_entropy.md).

### 5. Physics-Informed Neural Networks
- **PDE residual losses**: heat, Schrödinger, Maxwell,
  Navier–Stokes, wave equations as training signals on the
  legacy MLP or (optionally) the engine-backed network.

See [docs/physics_loss.md](docs/physics_loss.md).

### 6. Reinforcement Learning
- **Policy/value optimisation**: reward-based spin-configuration
  tuning with energy-driven rewards.

See [docs/reinforcement_learning.md](docs/reinforcement_learning.md).

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
│   ├── neural_network.c         # Legacy MLP (Adam, batch norm, SIREN)
│   ├── nn_backend.c             # Polymorphic NN handle (v0.4)
│   ├── engine_adapter.c         # Engine-neutral bridge (dormant by default)
│   ├── eshkol_bridge.c          # Eshkol FFI bridge (dormant by default)
│   ├── libirrep_bridge.c        # libirrep bridge (gated on SPIN_NN_HAS_IRREP)
│   ├── moonlab_bridge.c         # Moonlab bridge (gated on SPIN_NN_HAS_MOONLAB)
│   ├── qllm_bridge.c            # qllm bridge (opaque)
│   ├── qgtl_bridge.c            # QGTL bridge (opaque)
│   ├── noesis_bridge.c          # Noesis reasoning-engine bridge
│   ├── physics_loss.c           # Physics-informed PDE residual losses
│   ├── reinforcement_learning.c # Reactive reward-based RL
│   ├── quantum_mechanics.c      # Noise + entanglement helpers
│   ├── majorana_modes.c         # Majorana fermions + Hilbert-space braiding (v0.4)
│   ├── toric_code.c             # Data-qubit model + greedy decoder (v0.4)
│   ├── berry_phase.c            # Berry phase / Chern / winding / TKNN
│   ├── topological_entropy.c    # Von Neumann + topological entanglement entropy
│   ├── ising_chain_qubits.c     # Topological qubits from Majorana chains
│   ├── matrix_neon.c            # ARM NEON SIMD kernels
│   ├── nqs/                     # NQS pillar (P1.1)
│   │   ├── nqs_ansatz.c         #   MLP / RBM / complex RBM ansätze
│   │   ├── nqs_sampler.c        #   Metropolis sampler
│   │   ├── nqs_gradient.c       #   Per-Hamiltonian local-energy kernels
│   │   ├── nqs_optimizer.c      #   Real + holomorphic SR, tVMC, excited-SR
│   │   ├── nqs_marshall.c       #   Marshall sign wrapper
│   │   ├── nqs_translation.c    #   Translation symmetry projection
│   │   ├── nqs_diagnostics.c    #   χ_F + kagome bond-phase (v0.4.2)
│   │   └── nqs_lanczos.c        #   Full-basis Lanczos post-processing (v0.4.2)
│   ├── mps/                     # MPS + DMRG + TEBD + Lanczos substrate (P2.2 / P2.6)
│   ├── llg/                     # Landau-Lifshitz-Gilbert dynamics (P1.2)
│   ├── equivariant_gnn/         # SO(3)-equivariant torque predictor (P1.2)
│   ├── qec_decoder/             # Learned surface-code decoder scaffold (P1.3)
│   ├── fibonacci_anyons/        # Fibonacci anyons + braiding (P1.3b)
│   ├── neural_operator/         # Fourier Neural Operator + FFT (P1.4)
│   ├── flow_matching/           # Discrete flow matching (P1.4)
│   ├── thermodynamic/           # Hopfield + RBM-CD (P2.9)
│   ├── thqcp/                   # THQCP scaffold
│   ├── neuromorphic/            # p-bit / p-dit mode (P2.4)
│   ├── topological_example.c    # Standalone topological-phases demo
│   ├── visualization.c          # SDL2 viewer (4 modes)
│   └── visualization_main.c     # Visualization entry point
├── include/                     # Public headers mirroring src/ layout
├── tests/                       # test_*.c suites (run all via `make test`)
│   ├── harness.h
│   └── test_*.c                 # 359 tests across 50+ suites
├── benchmarks/                  # bench_*.c suites + JSON results
│   ├── bench_common.h
│   ├── bench_*.c
│   └── results/                 # Per-suite JSON output
├── scripts/
│   ├── run_benchmarks.sh        # Builds and runs every benchmark suite
│   ├── check_stack.sh           # Probes for optional external dependencies
│   ├── research_kagome_N12_convergence.c   # v0.4.2 MC convergence probe
│   └── research_kagome_N12_diagnostics.c   # v0.4.2 end-to-end diagnostic driver
├── docs/
│   ├── architecture_v0.4.md     # Architecture + v0.5+ research roadmap
│   ├── testing.md               # Test harness reference + per-suite coverage
│   ├── benchmarks.md            # JSON schema + reference numbers
│   ├── nqs.md                   # NQS pillar — kernels, ansätze, SR, diagnostics
│   ├── training.md              # nn_backend, cadence flags, physics losses
│   ├── engine_integration.md    # engine_adapter + eshkol_bridge integration
│   ├── cross_project_integration.md # Cross-project vendoring / coordination
│   ├── libirrep_1_2_coordination.md # libirrep coordination record
│   ├── classical_models.md      # Ising / Kitaev / disordered / spin models
│   ├── physics_loss.md          # PDE residual losses
│   ├── quantum_mechanics.md     # State calculations + entanglement helpers
│   ├── reinforcement_learning.md# Reward-based RL on spin lattices
│   ├── topological_features.md  # Overview of topological features
│   ├── berry_phase.md           # Berry phase + topological invariants
│   ├── majorana_zero_modes.md   # Majorana modes + braiding
│   ├── topological_entropy.md   # Entanglement entropy calculations
│   ├── toric_code.md            # Toric code QEC
│   ├── visualization.md         # SDL2 viewer
│   └── research/
│       └── kagome_KH_plan.md    # v0.4.2 — kagome Z₂ vs Dirac open program
├── eshkol/                      # Scheme training scripts (dormant; engine-integrated)
├── vendor/                      # Vendored external binaries (qllm, libirrep)
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

Current release: **v0.4.2** (2026-04-24). A kagome-diagnostics +
Lanczos reference addition on top of the v0.4.1 Kitaev-Heisenberg
and kagome Heisenberg local-energy kernels.

**v0.4.2 headline capabilities**

- Sample-based χ_F diagnostic (`nqs_compute_chi_F`).
- Per-bond-class phase probe on kagome
  (`nqs_compute_kagome_bond_phase`).
- Excited-state VMC via the Choo–Neupert–Carleo orthogonal-
  ansatz penalty (`nqs_sr_{step,run}_excited`).
- Full-basis Lanczos refinement + multi-Ritz gap extraction for
  kagome Heisenberg (`nqs_lanczos_{refine,k_lowest}_kagome_heisenberg`).
- End-to-end research driver
  (`make research_kagome_N12_diagnostics`).
- Full suite **359 / 359** passing, AddressSanitizer +
  UndefinedBehaviorSanitizer clean.

**v0.4.1 (2026-04-23)** — Kitaev-Heisenberg and kagome Heisenberg
local-energy kernels (`NQS_HAM_KITAEV_HEISENBERG`,
`NQS_HAM_KAGOME_HEISENBERG`); `LIBIRREP_MIN` bumped to 1.3.0-alpha.

**v0.4.0 foundation highlights** — Hilbert-space Majorana braiding
with order-8 non-Abelian statistics, toric code with explicit data
qubits + greedy MWPM decoder, engine-neutral scaffolding (`engine_adapter`,
`eshkol_bridge`, `nn_backend`), training-loop decoder cadence flags,
assert-based test suite + benchmark harness.

See [RELEASE_NOTES.md](RELEASE_NOTES.md) for the full notes,
[CHANGELOG.md](CHANGELOG.md) for history, and
[docs/architecture_v0.4.md](docs/architecture_v0.4.md) for the v0.5+
research roadmap (ViT wavefunctions for frustrated magnets,
equivariant Landau-Lifshitz-Gilbert dynamics, learned surface-code
decoders, Fibonacci-anyon universal gates, neural-operator Boltzmann
samplers).

## Citation
If you use this project in your research, please cite as follows:

```bibtex
@software{SpinBasedNeuralComputation,
  author = {tsotchke},
  title = {Spin-Based Neural Computation Framework: Simulations of Topological Quantum Computing},
  version = {0.4.2},
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

### Neural Network Quantum States (pillar P1.1, diagnostics pipeline in v0.4.2)
17. G. Carleo and M. Troyer, "Solving the quantum many-body problem with artificial neural networks," *Science*, vol. 355, pp. 602–606, 2017.
18. S. Sorella, "Green Function Monte Carlo with Stochastic Reconfiguration," *Physical Review Letters*, vol. 80, pp. 4558–4561, 1998.
19. R. Rende, L. Viteritti, L. Bardone, F. Becca, and S. Goldt, "A simple linear algebra identity to optimize large-scale neural network quantum states," *Communications Physics*, 2024. arXiv:2310.05715.
20. A. Chen and M. Heyl, "Empowering deep neural quantum states through efficient optimization," *Nature Physics*, vol. 20, pp. 1476–1481, 2024.

### Quantum geometric tensor, fidelity susceptibility, excited-state VMC, Lanczos (v0.4.2)
20a. J. P. Provost and G. Vallée, "Riemannian structure on manifolds of quantum states," *Communications in Mathematical Physics*, vol. 76, pp. 289–301, 1980.  *(QGT / Fubini–Study metric on the variational manifold.)*
20b. P. Zanardi and N. Paunković, "Ground state overlap and quantum phase transitions," *Physical Review E*, vol. 74, p. 031123, 2006.  *(χ_F = Tr(S)/2 convention used in `nqs_compute_chi_F`.)*
20c. K. Choo, T. Neupert, and G. Carleo, "Two-dimensional frustrated J1-J2 model studied with neural network quantum states," *Physical Review B*, vol. 100, p. 125124, 2019. arXiv:1810.10196.  *(Orthogonal-ansatz penalty for excited states — `nqs_sr_step_excited`.)*
20d. C. Lanczos, "An iteration method for the solution of the eigenvalue problem of linear differential and integral operators," *Journal of Research of the National Bureau of Standards*, vol. 45, pp. 255–282, 1950.  *(Krylov eigensolver underpinning `lanczos_smallest_with_init` and `lanczos_k_smallest_with_init`.)*

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

