# Quantum Mechanics

`src/quantum_mechanics.c` / `include/quantum_mechanics.h` inject quantum
effects into the classical Ising/Kitaev/SpinLattice trio each training
iteration: superposition resampling, tunneling, decoherence, and simple
Bell-state entanglement. The module is a semiclassical placeholder —
real Schrödinger evolution arrives alongside the time-dependent NQS
pillar P2.1 (see `architecture_v0.4.md` §4).

## 1. Public API

Only two entry points, both driven by probabilities in `[0, 1]`:

```c
void apply_quantum_effects(IsingLattice   *ising,
                           KitaevLattice  *kitaev,
                           SpinLattice    *spin,
                           double          noise_level);

void simulate_entanglement(IsingLattice   *ising,
                           KitaevLattice  *kitaev,
                           double          entanglement_prob);
```

Both validate their probability argument and log to `stderr` + return
early if it falls outside `[0, 1]`. Otherwise they visit every site of
each lattice and mutate it according to the operations below.

## 2. `apply_quantum_effects`

For every site `(x, y, z)` in the Ising, Kitaev, and Spin lattices,
applies three quantum operations in sequence.

### 2.1 Superposition resampling

Implements a simple Born-rule-style collapse:

```
P(spin = +1) = α²,     where α = √(0.5 + noise_level · 0.5)
```

At `noise_level = 0`, `α = √0.5 ≈ 0.707` — spins are resampled to ±1
with equal probability on every call. At `noise_level = 1`, `α = 1`
and every spin is forced to `+1`. Intermediate `noise_level` biases
the distribution toward `+1`.

This behavior is tested in `tests/test_quantum_mechanics.c`:

> A `noise_level = 0` call is **not** a no-op — every spin is
> resampled. The module deliberately does not treat zero noise as
> pass-through; v0.5 pillar P2.1 replaces this with proper Schrödinger
> evolution where zero noise is the identity map.

### 2.2 Tunneling

After superposition resampling, each site is flipped with probability:

```
tunnel_prob = noise_level · 0.1
```

So at `noise_level = 0` tunneling is off; at `noise_level = 1` it
flips 10% of sites per call.

### 2.3 Decoherence on the continuous spin lattice

For each site in the `SpinLattice`, the complex amplitude
`z = sx + i · sy` is rotated by a random phase:

```
z → z · exp(i · 2π · u · decoherence_rate),
decoherence_rate = noise_level · 0.05,
u ~ Uniform[0, 1]
```

and `sx`, `sy` are overwritten with the new real/imaginary parts. The
`sz` component then receives an additive noise term:

```
sz += noise_level · (u − 0.5)
```

(Note this can drift `sz` outside the nominal `[-0.5, +0.5]` range the
`spin_models.c` initializer uses; the v0.3 behaviour is preserved for
back-compat but the pillar P1.2 equivariant-LLG work will project
back onto the unit sphere.)

## 3. `simulate_entanglement`

For each Ising site, with probability `entanglement_prob` selects one
of the four Bell states at random and writes correlated values into
the Ising site and a *random* Kitaev site:

| Bell state | Ising spin | Kitaev spin (random target) |
|---|---|---|
| `\|Φ⁺⟩` | +1 | +1 |
| `\|Φ⁻⟩` | +1 | −1 |
| `\|Ψ⁺⟩` | +1 | −1 |
| `\|Ψ⁻⟩` | −1 | +1 |

Probability `entanglement_prob` outside `[0, 1]` → logs to `stderr`
and returns without mutation.

This is a *model* of entanglement, not a full density-matrix
simulation — it produces the classical projection that would result
from measuring one half of each Bell pair, which is the signal the
downstream neural network consumes.

## 4. Integration in `main.c`

The main training loop invokes both functions once per iteration:

```c
apply_quantum_effects(ising_lattice, kitaev_lattice, spin_lattice, noise_level);
simulate_entanglement(ising_lattice, kitaev_lattice, entanglement_prob);
```

with `noise_level = 0.2` and `entanglement_prob = 0.1` by default. No
CLI flags control these in v0.4; the v0.5 pillar P2.1 work will
expose them alongside the time-dependent NQS configuration.

## 5. Tests

`tests/test_quantum_mechanics.c` (3 tests):

1. `apply_quantum_effects` preserves the spin magnitude (± 1) on Ising
   and Kitaev lattices across a randomized call.
2. `simulate_entanglement` runs to completion without crashing on a
   small random lattice.
3. Input validation: out-of-range `noise_level` (`-0.1`, `1.5`) is
   rejected without a crash.

Run `make test_quantum_mechanics && ./build/test_quantum_mechanics`.

## 6. Known limitations

- **Not unitary.** The "Born rule" sampling is a classical stochastic
  projection, not a unitary quantum operation. True unitary evolution
  arrives with P2.1.
- **Decoherence only touches `sx`/`sy`.** The `SpinLattice` structure
  has three components but decoherence treats it as a complex scalar.
- **Random Kitaev target in entanglement.** The partner site is picked
  uniformly at random per call; there is no persistent pairing.

## 7. v0.5 roadmap

Pillar P2.1 (time-dependent NQS / p-tVMC) replaces this module with
real Schrödinger evolution:

- `src/nqs_dynamics/tdse_integrator.c` — solves `i ℏ ∂ψ/∂t = Ĥ ψ` via
  the tVMC equation `S(θ) θ̇ = -i F(θ)`.
- Unitary dynamics, norm preservation, and proper Bell-state
  generation via physical gate sequences rather than stochastic
  selection.

See `architecture_v0.4.md` §P2.1 for the detailed roadmap.
