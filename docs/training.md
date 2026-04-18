# Training Loop — Configuration and Cadence Flags

This document covers how the main training binary drives its neural
network: the polymorphic `nn_backend` handle, the in-loop cadences that
fold topological observables back into the physics loss, and the
command-line flags that control them.

For the topological / quantum-mechanical observables themselves, see
`majorana_zero_modes.md`, `toric_code.md`, `berry_phase.md`, and
`topological_entropy.md`. For the classical lattice models the loop
operates on, see `classical_models.md`.

## 1. The `spin_nn_t` polymorphic handle

`include/nn_backend.h` defines a polymorphic handle that abstracts over
two concrete neural-network implementations:

| Backend | Enum | Source | Status |
|---|---|---|---|
| In-tree MLP | `NN_BACKEND_LEGACY` | `src/neural_network.c` | Active — default |
| External engine | `NN_BACKEND_ENGINE` | `src/engine_adapter.c` | Dormant in v0.4; falls back to legacy |

```c
#include "nn_backend.h"

int input_size  = L*L*L*3;     /* e.g. flattened SpinLattice */
int hidden      = 3;
int neurons     = 256;
int output_size = 1;

spin_nn_t *nn = spin_nn_create(NN_BACKEND_LEGACY,
                               input_size, hidden, neurons, output_size,
                               ACTIVATION_RELU);
double *y = spin_nn_forward(nn, x);           /* one inference */
spin_nn_train(nn, x, target, learning_rate);  /* one gradient step */
spin_nn_free(nn);
```

**Behavior contract:**

- `spin_nn_create(NN_BACKEND_ENGINE, ...)` prints a diagnostic to
  `stderr` and falls back to `NN_BACKEND_LEGACY` when the binary was
  built without `SPIN_NN_HAS_ENGINE` — the returned handle is still
  usable.
- `spin_nn_forward` returns a pointer into the network's internal
  buffer; the data is valid until the next `spin_nn_forward` or
  `spin_nn_train` call on the same handle.
- `spin_nn_backend(nn)` reports the *actual* backend of a handle, which
  may differ from the requested backend if a fall-back happened.
- `spin_nn_legacy_handle(nn)` returns the underlying `NeuralNetwork *`
  when backend is `NN_BACKEND_LEGACY`, else `NULL`. Useful for code
  paths not yet ported to the polymorphic API.
- `spin_nn_free(NULL)` is a no-op.

**Parser:**

```c
int ok = 0;
nn_backend_kind_t b = nn_backend_parse("engine", &ok);  /* case-insensitive */
/* accepts "legacy" and "engine"; anything else → NN_BACKEND_LEGACY with *ok = 0 */
```

## 2. In-loop topological feedback

`include/training_config.h` carries the cadence knobs that determine
how often the training loop pulls topological observables into the
physics loss:

```c
typedef struct {
    int    cadence_entropy;       /* 0 = off (reserved, not yet CLI-exposed) */
    int    cadence_invariants;    /* run Chern/winding/TKNN every N iters */
    int    cadence_decoder;       /* run toric-code greedy decoder every N iters */
    double decoder_error_rate;    /* per-qubit X/Z error rate for decoder feedback */
    double lambda_topological;    /* reserved for γ_topo penalty (not yet wired) */
    double lambda_logical;        /* physics_loss weight on logical-error flag */
    int    verbose;
} training_config_t;

training_config_t cfg = training_config_defaults();
```

Defaults (see `include/training_config.h`): all cadences = 0 (off),
`decoder_error_rate = 0.03`, `lambda_topological = 0.1`,
`lambda_logical = 1.0`.

**Inside the training loop** (`src/main.c`, near the physics-loss
computation):

1. Every `cadence_decoder` iterations, a toric code is seeded from the
   current Kitaev lattice via `calculate_stabilizers`, random errors
   are stamped at rate `decoder_error_rate` with `apply_random_errors`,
   the greedy decoder `toric_code_decode_greedy` runs, and
   `toric_code_has_logical_error` produces a 0/1 flag. That flag
   multiplied by `lambda_logical` is added to `physics_loss`.
2. Every `cadence_invariants` iterations, `calculate_all_invariants`
   runs on the Kitaev lattice as a probe. In v0.4 the result is
   advisory (not folded into the loss); the v0.5 equivariant-LLG
   pillar tightens this.
3. `cadence_entropy` and `lambda_topological` are reserved in the
   struct for the pillar P1.2 work that folds topological entanglement
   entropy into the loss directly; v0.4 does not wire them to CLI.

## 3. Command-line mapping

| Flag | Type | Default | Wires to |
|---|---|---|---|
| `--nn-backend {legacy,engine}` | string | `legacy` | `nn_backend_parse` + `spin_nn_create` |
| `--cadence-decoder N` | int | 0 (off) | `tcfg.cadence_decoder` |
| `--decoder-error-rate P` | double | 0.03 | `tcfg.decoder_error_rate` |
| `--cadence-invariants N` | int | 0 (off) | `tcfg.cadence_invariants` |
| `--lambda-logical L` | double | 1.0 | `tcfg.lambda_logical` |
| `--loss-type TYPE` | string | `heat` | `compute_physics_loss` dispatch |
| `--activation FUNC` | string | `relu` | `ACTIVATION_RELU / TANH / SIGMOID` |
| `--debug-entropy` | flag | off | Sets `DEBUG_ENTROPY=1` env var |
| `--debug-quantum` | flag | off | Sets `DEBUG_QUANTUM=1` env var |

See `src/main.c` at the top of `main()` for the complete option-parser.
Missing-value flags follow `getopt_long` conventions.

## 4. End-to-end example

```sh
./bin/spin_based_neural_computation_arm \
    --iterations 200                     \
    --lattice-size "6 6 6"               \
    --jx 1.0 --jy 1.0 --jz -1.0          \
    --loss-type heat                     \
    --activation relu                    \
    --nn-backend legacy                  \
    --cadence-decoder 20                 \
    --decoder-error-rate 0.05            \
    --cadence-invariants 50              \
    --lambda-logical 0.5                 \
    --log run.log                        \
    --verbose
```

This runs 200 training iterations on a 6³ Kitaev lattice; every 20
iterations the toric-code decoder samples a syndrome at 5% error rate
and folds the logical-error flag into the physics loss with weight
0.5; every 50 iterations the topological invariants are probed.

## 5. Physics-informed losses

`compute_physics_loss(ising_E, kitaev_E, spin_E, lattices..., dt, dx, type)`
in `include/physics_loss.h` dispatches on the `type` string:

| `type` | Residual | Physical motivation |
|---|---|---|
| `heat` | `∂u/∂t - α ∇²u` | thermal diffusion, α=1e-7 m²/s |
| `schrodinger` | `iℏ ∂ψ/∂t + (ℏ²/2m) ∇²ψ` | electronic wavefunction |
| `maxwell` | curl / divergence form | EM field coupling |
| `navier_stokes` | `∂u/∂t + u·∇u - ν∇²u + ∇p` | fluid-like dynamics |
| `wave` | `∂²u/∂t² - c² ∇²u` | classical wave propagation |

The module also exposes discretized operators used internally by the
five residuals — `divergence`, `laplacian_3d`, `gradient_{x,y,z}`
(int-valued Ising/Kitaev lattices) and `*_spin` variants for
continuous `Spin***` lattices. These are third-order-accurate finite
differences with open boundary conditions.

Full v0.5 expansion (SIREN activations, Fourier features, hard-constraint
projection, variational form) is scheduled for pillar P2.7 — see
`architecture_v0.4.md` §4.

## 6. Tests

- `tests/test_nn_backend.c` (11 tests) — parse / create / forward /
  train / engine fall-back.
- `tests/test_physics_loss.c` (6 tests) — every PDE loss returns a
  finite value; Laplacian of a constant is zero.
- `tests/test_energy_utils.c` (4 tests) — scale/unscale round-trip.

Run `make test` for the full suite.
