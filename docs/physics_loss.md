# Physics-Informed Loss Functions

`src/physics_loss.c` / `include/physics_loss.h` implement five PDE
residual losses that augment the neural network's supervised objective
with a physics prior. The design follows the physics-informed
neural-network pattern: the network's prediction is constrained to
satisfy a differential equation residual on the lattice, with the
residual computed from the instantaneous (Ising, Kitaev, continuous)
spin state.

v0.4 ships these as proof-of-concept residuals; v0.5 pillar P2.7 adds
SIREN/Fourier-feature activations, hard-constraint projection, and a
variational / weak-form micromagnetic upgrade.

## 1. Dispatch

All five residuals share a common signature via
`compute_physics_loss()`:

```c
double L = compute_physics_loss(ising_E, kitaev_E, spin_E,
                                ising_lattice, kitaev_lattice,
                                spin_lattice, dt, dx, loss_type);
```

| `loss_type` string | Residual function | Physical quantity |
|---|---|---|
| `"heat"` | `heat_loss` | `∂u/∂t − α ∇²u` |
| `"schrodinger"` | `schrodinger_loss` | `i ℏ ∂ψ/∂t + (ℏ² / 2m) ∇² ψ` |
| `"maxwell"` | `maxwell_loss` | `∇·E, ∇×B` style coupling |
| `"navier_stokes"` | `navier_stokes_loss` | `∂u/∂t + (u·∇)u − ν ∇² u + ∇p` |
| `"wave"` | `wave_loss` | `∂²u/∂t² − c² ∇² u` |

Unknown `loss_type` falls back to `"heat"`. Constants (`ALPHA`, `C`,
`HBAR`, `M`, `EPSILON0`, `MU0`) live in `physics_loss.h`; values are
the SI-MKS numerical constants (`α = 10⁻⁷ m²/s`, `c = 299 792 458 m/s`,
`ℏ = 1.054 571 8 × 10⁻³⁴ J·s`, etc.).

## 2. Finite-difference operators

The residuals are assembled from a small set of 3D lattice operators.
Every operator uses **periodic boundary conditions** (wrap-around with
modular indexing) and **central differences** on a uniform grid with
spacing `dx`. Two API flavors exist:

- `int***` variant — for integer-valued Ising / Kitaev lattice data.
- `_spin` variant — for `Spin***` (continuous `SpinLattice`) data,
  dispatching on the `sx` field of each site.

### Laplacian (3D, second-order)

```
(∇² u)_ijk ≈ [ u(i±1,j,k) + u(i,j±1,k) + u(i,j,k±1) − 6 u(i,j,k) ] / dx²
```

```c
double lap = laplacian_3d(lattice, x, y, z, size_x, size_y, size_z, dx);
double lap_s = laplacian_3d_spin(spin_lattice_3d_array, x, y, z, size_x, size_y, size_z, dx);
```

The Laplacian of a constant field is identically zero — verified by
`tests/test_physics_loss.c`.

### Gradient (per-axis, second-order central)

```
(∂u/∂x)_ijk ≈ [ u(i+1,j,k) − u(i−1,j,k) ] / (2 dx)
```

```c
double gx = gradient_x(field, x, y, z, size_x, dx);
double gy = gradient_y(field, x, y, z, size_y, dx);
double gz = gradient_z(field, x, y, z, size_z, dx);
/* _spin variants dispatch on Spin::sx */
```

### Divergence (vector field split across three scalar fields)

```
(∇·u)_ijk = ∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z
```

The signature takes three scalar fields — two integer lattices (Ising,
Kitaev) plus one `Spin***` — to emulate a vector field laid out across
the three available lattice types:

```c
double div = divergence(u_x, u_y, u_z_spin, x, y, z,
                        size_x, size_y, size_z, dx);
```

For a homogeneous vector-field treatment on `SpinLattice` alone use
`divergence_spin`, which reads `sx`, `sy`, `sz` components of each
site:

```c
double div_s = divergence_spin(spin_arr, spin_arr, spin_arr, x, y, z,
                               size_x, size_y, size_z, dx);
```

## 3. Residual definitions

Each residual is evaluated at every lattice site and squared-summed to
give the scalar loss. The implementation uses `ising_lattice` as the
dominant scalar field; the other two arguments modulate the residual.

### Heat (`heat_loss`)

Classical 3D heat equation with diffusivity `α`:

```
r = ∂u/∂t − α ∇² u
```

Time derivative is approximated as `(kitaev_energy − ising_energy) / dt`;
spatial Laplacian is computed site-by-site on the Ising lattice. The
loss is `Σ r² / N`.

### Schrödinger (`schrodinger_loss`)

Time-dependent Schrödinger equation for a single-particle wavefunction
approximation:

```
r = i ℏ ∂ψ/∂t + (ℏ² / 2m) ∇² ψ
```

Real and imaginary parts are both evaluated from the Ising-lattice
Laplacian with `ℏ = HBAR` and `m = M` baked in.

### Maxwell (`maxwell_loss`)

Simplified Maxwell coupling that exercises both curl and divergence:

```
r = ∇·E  +  (∇×B)·ê
```

Curl components come from `discrete_curl_ising` and
`discrete_curl_kitaev`; divergence from the cross-lattice `divergence`
helper.

### Navier-Stokes (`navier_stokes_loss`)

Incompressible fluid-like residual:

```
r = ∂u/∂t + (u·∇)u − ν ∇² u + ∇p
```

with `ν` and `p` stand-ins approximated from the three energy inputs.
Primarily a diagnostic — v0.4 does not evolve a consistent pressure
field. Use `heat` or `schrodinger` for serious training.

### Wave (`wave_loss`)

Classical scalar wave equation at speed `c`:

```
r = ∂² u / ∂t² − c² ∇² u
```

Second time derivative is approximated as
`(spin_energy − 2 kitaev_energy + ising_energy) / dt²`.

## 4. In-loop integration

The training loop in `src/main.c` calls
`compute_physics_loss(...)` once per iteration with:

- `ising_energy = compute_ising_energy(ising_lattice)`
- `kitaev_energy = compute_kitaev_energy(kitaev_lattice)`
- `spin_energy = compute_spin_energy(spin_lattice)`
- `dt = 0.1`, `dx = 0.1` by default (CLI `--dt`, `--dx`)
- `loss_type = "heat"` by default (CLI `--loss-type`)

The result is added to the neural network's supervised loss and used as
a physics-informed regularizer. In v0.4, `--cadence-decoder` also folds
a toric-code logical-error flag into the same `physics_loss` scalar
(see `training.md`).

## 5. Numerical conventions

- **Boundary conditions**: periodic (modular indexing).
- **Stencils**: 7-point Laplacian, 2-point central gradient. Both are
  second-order accurate in `dx`.
- **Time derivative approximation**: finite-difference between
  consecutive energy samples; `dt` sets the virtual time scale.
- **Constants**: SI-MKS. Losses are *unnormalised* — absolute
  magnitude depends on lattice size, field values, and the constants
  in `physics_loss.h`. Use `scale_energy` / `unscale_energy` from
  `energy_utils.h` if you need a bounded comparison across loss types.

## 6. Tests

`tests/test_physics_loss.c` (6 tests):

- All five residuals return a finite (non-NaN, non-inf) value on a
  random 4³ lattice.
- Laplacian of an all-up Ising lattice vanishes to machine precision.

Run `make test_physics_loss && ./build/test_physics_loss`.

## 7. v0.5 roadmap — pillar P2.7

Planned upgrades (see `architecture_v0.4.md` §P2.7):

- **SIREN** (sinusoidal) and **Fourier-feature** activations at the
  network's input layer.
- **Hard-constraint projection** for unit-norm magnetisation
  (`|m| = 1` preserved to machine precision).
- **Variational / weak-form** micromagnetic loss: exchange + DMI +
  anisotropy + Zeeman + demag as an energy functional, minimised
  directly instead of residual-minimised.
- **Parametric / conditional PINN**: Hamiltonian coefficients
  (`J_x, J_y, J_z`) become network inputs so one model covers a phase
  diagram.
- **Causal / curriculum** training schedules for time-dependent PDEs.

## 8. References

The residuals in this module are standard in the physics-informed
neural network literature. For theoretical background and recent
technique surveys, see the "Neural operators and flow matching"
section of `README.md §References` and `architecture_v0.4.md §7`.
