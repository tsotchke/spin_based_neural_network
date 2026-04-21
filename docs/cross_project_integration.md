# Cross-Project Integration

`spin_based_neural_network` is one project in a larger open-source
ecosystem of pure-C / Scheme libraries for scientific computing. This
document describes the technical bridges between them so that external
collaborators can see how the pieces fit together.

All projects in this document are MIT / Apache-2.0 licensed and live
under [github.com/tsotchke](https://github.com/tsotchke). Every
project has a stable C ABI.

## The ecosystem

| Project | Role | Language | Public header |
|---|---|---|---|
| [`eshkol`](https://github.com/tsotchke/eshkol) | Scheme-on-LLVM with compile-time automatic differentiation | Scheme + C++ LLVM backend | `eshkol.h` |
| [`libirrep`](https://github.com/tsotchke/libirrep) | SO(3) / SU(2) / O(3) irrep math (CG, spherical harmonics, Wigner-D, NequIP-style tensor-product layers) | C | `irrep/irrep.h` |
| [`moonlab`](https://github.com/tsotchke/moonlab) (aka `quantum_simulator`) | Dense state-vector quantum simulator + surface-code + Fibonacci/Ising anyons + MBL + skyrmions | C (+ Metal on macOS) | `libquantumsim.h` |
| [`quantum_geometric_tensor`](https://github.com/tsotchke/quantum_geometric_tensor) | Quantum geometric tensor (Fubini-Study metric, Berry curvature, natural gradient) + hardware backends (IBM, Rigetti, D-Wave) + distributed training | C | `qgt/qgt.h` |
| [`semiclassical_qllm`](https://github.com/tsotchke/semiclassical_qllm) | Mixed-curvature product-manifold transformer inference + KAN + 6 attention types + flash / paged / ring / hull KV cache + SafeTensors / HuggingFace loader + LoRA + Riemannian optimizers | C11 + C++17 + CUDA + Objective-C Metal | `semiclassical_qllm.h`, `eshkol_ffi.h` |
| `spin_based_neural_network` (this repo) | Disordered-magnet physics: Ising / Kitaev / XXZ + NQS + MPS / DMRG / TEBD + LLG micromagnetics + topological codes + thermodynamic computing + equivariant torque network + THQCP coupling scheduler | C | per-module headers |
| [`noesis`](https://github.com/tsotchke/noesis) | Neuro-symbolic cognitive architecture on Eshkol: proof trees, factor graphs, belief propagation, workspace arbitration, free-energy inference, synthesizer decoding — all on one AD tape | Eshkol | `libeshkol.h` |

## Dependency direction (bottom is depended upon by top)

```
              +--------+
              | noesis |
              +---+----+
                  | AD tape + symbolic primitives
              +---v----+
              | eshkol |
              +---+----+
                  | FFI
              +---v----------------+
              | semiclassical_qllm |
              +---+-------+--------+
                  |       |
       +----------+       +----------+
       |                             |
 +-----v----+                 +------v-----+
 |  QGTL    |                 |  moonlab   |
 +-----+----+                 +------+-----+
       |                             |
       | uses                        | uses
       +------------+----------------+
                    |
                    v
        +-----------+-------------+
        | spin_based_neural_network |
        +-----------+-------------+
                    |
                    | uses
                    v
               +----+-----+
               | libirrep |
               +----------+
```

Every arrow is a C-ABI boundary. No Python, no other runtime
dependency.

## Bridges this repo provides

All bridges are dormant by default — they compile to `EDISABLED` stubs.
Enable with specific Makefile flags.

### `src/libirrep_bridge.c` — SO(3) / SU(2) math primitives

- Enable: `make IRREP_ENABLE=1 IRREP_ROOT=/path/to/libirrep/install`
- Exposes: spherical-harmonic evaluation, Clebsch-Gordan coefficients,
  Wigner-d elements, opaque NequIP layer handles (gated further behind
  `SPIN_NN_HAS_IRREP_NEQUIP` since `nequip.h` lands in libirrep ≥ 1.1).
- Tests: `tests/test_libirrep_bridge.c` (5 disabled-mode, 6 live-mode),
  `tests/test_torque_net_irrep.c` (SH addition theorem agreement with
  hand-rolled cartesian dot to 6.1e-17).

### `src/moonlab_bridge.c` — quantum-simulator ground truth

- Enable: `make MOONLAB_ENABLE=1 MOONLAB_ROOT=/path/to/quantum_simulator`
- Exposes: surface-code distance-d creation, error application, MWPM
  decode, logical-error-rate Monte Carlo.
- Use case: ground-truth reference for learned-decoder work and the
  THQCP qubit-plane validation.
- Tests: `tests/test_moonlab_bridge.c` (4 dormant, 4 live).

### `include/engine_adapter.h` + `src/engine_adapter.c` — generic NN engine

- Enable: `make ENGINE_ENABLE=1 ENGINE_ROOT=/path/to/engine`
- Intended backends: `semiclassical_qllm` for transformer /
  KAN-based learned QEC decoders and transformer-NQS ansätze.

### `include/eshkol_bridge.h` + `src/eshkol_bridge.c` — autodiff orchestration

- Lazy wrapper over Eshkol FFI for training-loop orchestration.
- Enable: build-time `-DSPIN_NN_HAS_ESHKOL=1` + link against `libeshkol`.

## Who uses whom for what

**Cross-validation axis**: every physics claim in this repo is
regression-tested against either DMRG (internal), exact diagonalization
(internal), or `moonlab` (external). NQS ↔ DMRG ↔ TEBD ↔ ED ↔ moonlab
agreement to machine precision is the baseline quality bar.

**Equivariance axis**: for physics claims involving SO(3) symmetry
(equivariant torque net, NequIP-style layers), `libirrep` is the
ground-truth source. SH addition theorem matches the torque net's
hand-rolled contractions to machine precision.

**Geometry axis**: `QGTL` is used where the quantum geometric tensor
or Fubini-Study metric appears. The holomorphic stochastic
reconfiguration path here uses the same mathematical structure as
QGTL's natural-gradient descent; both projects share the underlying
geometry.

**Neural-execution axis**: the roadmap for transformer-based NQS
ansätze (pillar P1.1 in `docs/architecture_v0.4.md`) delegates to
`semiclassical_qllm` via the engine adapter. Its Riemannian optimizers
and mixed-curvature attention are relevant to quantum-state
representation.

**Reasoning axis**: the long-term plan for the THQCP coupling
scheduler (pillar P3.0 in `src/thqcp/`) is to let `noesis`'s proof-
trace reasoning decide when quantum windows open and which qubits
are activated — replacing the current deterministic / stagnation
policies with a learned classifier that carries calibrated uncertainty
and verifiable proof traces.

## Getting started as an external collaborator

The minimum setup to reproduce a cross-validated result from this repo:

1. Clone this repo. Install a C11 compiler + `make`. `make test` runs
   the full 309+-test regression suite without any external dependency.
2. Optional: install `libirrep` from source. Run `make IRREP_ENABLE=1
   IRREP_ROOT=<path> test_libirrep_bridge test_torque_net_irrep` to
   enable the 13 live-mode irrep-dependent tests.
3. Optional: install `moonlab` from source and run `make
   MOONLAB_ENABLE=1 MOONLAB_LIBDIR=<moonlab>/build test_moonlab_bridge`
   for the 4 live-mode surface-code QEC tests.

Deeper integrations (`engine_adapter` + `eshkol_bridge`) are reserved
for contributors actively collaborating on the NN-executor or autodiff
paths.

## Versioning policy

Each project maintains its own semantic-version tag. This repo tracks
the specific versions it validates against in `VERSION_PINS` (planned
for v0.5). Current pinned compatibility:

- `libirrep` ≥ 1.0.0 (primitive math), ≥ 1.1.0 recommended (NequIP).
- `moonlab` ≥ 0.1.2.
- `eshkol` ≥ 1.1.0.
- `semiclassical_qllm` ≥ 0.1.0.
- `QGTL` ≥ 0.7.7 beta.
- `noesis` ≥ 1.0.0.

## Asking questions / contributing

- File issues on the relevant project's GitHub page.
- For cross-project architectural discussions, any of the project repos
  is a reasonable entry point; issues are triaged between them.
