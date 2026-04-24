# Classical Spin Models

This document covers the classical spin-system building blocks that
underlie the topological and neural-network layers in the framework:
the 3D Ising, Kitaev, and disordered models; the continuous-spin
`SpinLattice`; the energy-scaling helpers; and the shared lattice
conventions.

All four models share the `(size_x, size_y, size_z)` 3D-lattice shape
and an `initial_state` string (`"random"`, `"all-up"`, `"all-down"`).
Random number generation uses the C standard library `rand()` and can
be seeded via `srand()` at the start of `main()` — reproduce runs with
a fixed seed.

## 1. Ising model (`include/ising_model.h`)

3D nearest-neighbor Ising model on a cubic lattice. Each site carries
a spin `s_i ∈ {+1, -1}`. The energy functional, implemented with
periodic boundary conditions:

```
H = -Σ_{<ij>} s_i s_j
```

where `<ij>` ranges over nearest-neighbor pairs in all three axes.
`compute_ising_energy()` returns this value; `compute_ising_interaction()`
returns the contribution from a single site's six neighbors (with open
boundary conditions for local updates).

```c
IsingLattice *l = initialize_ising_lattice(L, L, L, "random");
double E = compute_ising_energy(l);          /* total, periodic BC */
double E_i = compute_ising_interaction(l, x, y, z); /* site, open BC */
flip_random_spin_ising(l);                    /* Metropolis proposal */
free_ising_lattice(l);
```

`flip_random_spin_ising()` proposes a single-site flip, accepting via
the Metropolis rule `min(1, exp(-ΔE))` against a uniform random draw.

**Known reference energies** (verified by `tests/test_ising.c`):

| configuration (L³ torus) | `compute_ising_energy` |
|---|---|
| all-up / all-down | `-3 L³` |
| checkerboard antiferromagnet (L even) | `+3 L³` |
| stripe along z (spin depends only on z) | `-L³` (2 FM axes, 1 AFM) |

## 2. Kitaev model (`include/kitaev_model.h`)

Classical anisotropic Ising model with axis-dependent coupling constants.
This is **not** the quantum Kitaev honeycomb model (Kitaev 2006, Ann. Phys.
321:2–111); it is a classical spin lattice with the same coupling structure
whose quantum analogue is the honeycomb model.  The quantum topological
invariants (Chern number via FHS, TEE via Kitaev-Preskill) computed in
`berry_phase.c` and `topological_entropy.c` use these coupling constants to
parameterise the quantum phase diagram.

`compute_kitaev_energy()` uses open boundary conditions:

```
H = Σ_i (J_x s_i s_{i+x̂} + J_y s_i s_{i+ŷ} + J_z s_i s_{i+ẑ})
```

where the sum is taken only over in-bounds pairs (no wrap-around). The
sign convention is additive (not the `-Σ` used by Ising); chosen so that
`J_x, J_y, J_z` enter the energy directly.

```c
KitaevLattice *l = initialize_kitaev_lattice(L, L, L,
                                             /*jx*/ 1.0, /*jy*/ 1.0,
                                             /*jz*/ -1.0, "random");
double E = compute_kitaev_energy(l);
double E_i = compute_kitaev_interaction(l, x, y, z);
flip_random_spin_kitaev(l);
free_kitaev_lattice(l);
```

For all-up isotropic (`J_x = J_y = J_z = 1`) on an L³ lattice,
`compute_kitaev_energy()` returns `(J_x + J_y + J_z)(L-1)L²` — i.e.
`3(L-1)L²` at unit couplings.

The Kitaev model is the substrate on top of which Majorana wires,
toric-code qubits, and Berry-phase calculations are built: see
`docs/majorana_zero_modes.md`, `docs/toric_code.md`, `docs/berry_phase.md`.

## 3. Disordered model (`include/disordered_model.h`)

Adds quenched disorder to an existing Ising or Kitaev lattice by
flipping each spin independently with probability `disorder_strength`:

```c
add_disorder_to_ising_lattice(ising, 0.1);   /* 10% of sites flip */
add_disorder_to_kitaev_lattice(kitaev, 0.1);
```

`disorder_strength` must lie in `[0, 1]`:

- `0.0` — no-op
- `1.0` — every spin flips
- intermediate — random subset flips, driven by `rand()`

The helpers are used in `main.c` to inject background disorder during
each training iteration (see `noise_level`, `disorder_strength`
parameters at the top of `src/main.c:main()`).

## 4. Continuous-spin lattice (`include/spin_models.h`)

`SpinLattice` is a 3D array of `Spin { sx, sy, sz }` triples used by the
continuous-vector physics-loss path and as the target type for the
equivariant LLG pillar scheduled for v0.5 (pillar P1.2).

```c
SpinLattice *l = initialize_spin_lattice(L, L, L, "all-up");
/* "all-up"   → sx = sy = sz = +0.5 at every site
 * "all-down" → sx = sy = sz = -0.5
 * "random"   → each component sampled from {+0.5, -0.5} independently */
double E = compute_spin_energy(l);   /* simple Σ sx*sy placeholder */
free_spin_lattice(l);
```

v0.4 ships this type with a toy energy functional (`Σ_i s^x_i s^y_i`)
pending the real micromagnetic energy that lands with P1.2 — exchange,
DMI, anisotropy, Zeeman, and demagnetization terms.

## 5. Energy scaling (`include/energy_utils.h`)

Two helpers map raw lattice energies into a bounded range for the
neural network's numerical stability:

```c
double s = scale_energy(E);     /* map ℝ → (-1, 1) via sigmoid */
double r = unscale_energy(s);   /* inverse, up to a clamp near 0 */
```

The map is `s = 2 / (1 + exp(-α E)) - 1` with
`α = ENERGY_SCALE_FACTOR = 1e-2` (declared in `energy_utils.h`). The
inverse clamps |E| < `MIN_ENERGY = 1e-10` to a ±`MIN_ENERGY /α` floor to
avoid division by zero at the origin.

Round-tripping `scale ∘ unscale` is exact to machine precision in the
unsaturated region (|E| modest); outside, the sigmoid saturates and
`unscale` diverges logarithmically. `tests/test_energy_utils.c` asserts
the round-trip recovers ±{5, 10, 25, 50, 150, 200} to 1e-6.

## 6. Conventions

- **Lattice indexing**: all four models use `spins[x][y][z]`, row-major
  in `x`.
- **Boundary conditions**: Ising's `compute_ising_energy` uses periodic
  BC; Kitaev's `compute_kitaev_energy` uses open BC. Local interaction
  helpers (`compute_*_interaction`) use open BC in both cases.
- **Initial states**: `"random"`, `"all-up"`, `"all-down"`. Unknown
  strings fall back to `"random"`.
- **Lifecycle**: every `initialize_*` pairs with `free_*`. Do not free
  individual spin arrays — the free routines walk the full 3D structure.

## 7. Tests

The v0.4 test suite covers all four classical-model modules:

- `tests/test_ising.c` — 7 tests: known-config energies, stripe
  geometries, interior/corner interactions, Metropolis smoke.
- `tests/test_kitaev.c` — 6 tests: isotropic and anisotropic energies,
  interior/corner interactions, flip preserves ±1.
- `tests/test_spin_models.c` — 4 tests: initialisation, component
  values, energy finiteness.
- `tests/test_disordered_model.c` — 4 tests: boundary rates, magnitude
  preservation.
- `tests/test_energy_utils.c` — 4 tests: monotonicity, bounds,
  round-trip accuracy in the linear region.

Run `make test` to execute them all, or
`make test_ising && ./bin/test_ising` for a single suite.
