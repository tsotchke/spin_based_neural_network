# Spin-Based Neural Computation Framework v0.5.0

v0.5.0 is a **research-driven release** layered on top of v0.4.3.
Builds out the empirical side of the kagome AFM Heisenberg
ground-state question on PBC clusters at machine precision,
culminating in five independent observables that all reject simple
Z₂ Toric Code at N=27 in favour of the U(1) Dirac scenario.  No
public-API breaks; v0.4.3 contracts retain their signatures and
semantics.

## Highlights

- **Sector-projecting Lanczos infrastructure** — `lanczos_smallest_-
  projected`, `lanczos_smallest_projected_lean` (3-term recurrence,
  no Krylov-basis storage; required for N ≥ 24 where full
  reorthogonalisation is infeasible), `lanczos_smallest_projected_-
  lean_eigvec` (two-pass with eigenvector reconstruction),
  `lanczos_continued_fraction` for spectral functions S(q,ω).
  In-loop sector projection inside the Krylov build kills the
  power-method amplification of machine-precision sector leakage.
- **Full kagome p6m projector as a vector op** — `nqs_kagome_p6m_-
  project_inplace`. 3-bit-chunk lookup table optimisation: 62 KB
  cache footprint, ~5× faster than per-bit-shift loops.
- **Empirical p6m representation extraction** — full 6×6×12 = 432
  matrix elements ⟨ψ_α | σ_g | ψ_β⟩ on the L=3 PBC kagome AFM
  6-state low-energy manifold (1D + 2D irreps). Matches the C₆ᵥ
  character-table prediction to **1.835·10⁻¹¹** on the diagonal
  and **3.331·10⁻¹⁶** (machine ε) on the off-diagonal — an end-to-end
  empirical-symbolic bridge to the synthetic-symbolic verification
  framework (Drinfeld centre Z(Vec_{Z₂}), F/R-symbols, pentagon and
  hexagon, Verlinde, Lagrangian algebras, Witt class, RT lens-space
  invariants).
- **Empirical lattice modular S via Zhang-Grover-Vishwanath MES** —
  three progressively-more-complete runs; the methodologically
  clean K=5 run with both E₂ doublet partners (obtained via
  orthogonal-projection-penalty Lanczos at machine-precision
  doublet degeneracy) gives singular-value spectrum
  (0.92, 0.80, 0.20, 0.013, ≈0). Numerical rank 4 but NOT the
  flat-1/2 spectrum Z₂ TC requires.
- **Memory-lean projecting Lanczos at N=27** — eigenvector
  reconstruction via two-pass (α/β save → tridiagonal diagonalise
  → Lanczos replay) with O(few · dim) memory, unblocking
  post-processing (TEE, partial trace, S(q,ω)) at full cluster size.
- **Parallel real-input partial trace** — 8× wall-time speedup over
  the libirrep complex-cast baseline (85 s/eval → 11 s/eval at 14
  OpenMP threads); cross-validated against `libirrep_partial_trace`
  on N=12 random ψ at 1.4·10⁻¹⁷ max element-wise difference.

## Five-observable rejection of simple Z₂ TC at N=27

Three independent ED-side findings + two new modular-S diagnostics
all point away from the gapped Z₂ Toric Code reading:

1. **Global GS in E₂ doublet** at -11.7795 J (0.17 J below A₁).
2. **7 quasi-degenerate S=½ states** in 0.222 J spread (Z₂ predicts
   exactly 4; Ising 3; U(1) Dirac unbounded).
3. **Cross-sector gap → 0** under linear-in-1/N extrapolation
   across the full 6-irrep spectrum.
4. **C(d) decay η ≈ 1.5** over 9 distance shells at N=27 —
   algebraic, not exponential.
5. **Lattice modular S NOT (1/2)·Hadamard₄** across three MES
   variants. The doublet-symmetric K=5 run finds rank 4 but
   non-flat singular values, and closest 4-of-5 sub-matrix
   Frobenius distance from (1/2)·H₄ is 1.17 — comparable to the
   lowest-4 K=4 result (1.07).

The U(1) Dirac scenario (gapless, continuum of low-energy states)
is FAVOURED at L=3 PBC, revising the Z₂-favourable reading that
came from probing only the 4 1D-irrep sectors. Cleaner identification
still requires larger N (e.g. N=30 OBC) or a finite-temperature
thermal Hall κ_xy probe.

## Predictive observables and pipeline tools

`scripts/research_kagome_*.c` ship as a complete diagnostic stack:

| Tool | Purpose |
| ---- | ------- |
| `research_kagome_full_analysis` | (Γ, irrep) projected Lanczos with eigenvector reconstruction; emits S(q), Renyi spectrum, sum-rule checks |
| `research_kagome_sz_spatial` | Joint Sz + spatial-irrep projected Lanczos (incl. 2D irreps E₁, E₂) |
| `research_kagome_p6m_rep` / `_6state` | Empirical extraction of ⟨ψ_α \| σ_g \| ψ_β⟩ for all 12 C₆ᵥ elements; 4-state and 6-state variants |
| `research_kagome_modular` | C₆ matrix-element extraction with finer-grained reporting |
| `research_kagome_mes` | Empirical lattice modular S via Zhang-Grover-Vishwanath MES; runtime K ≤ MAX_K = 8 |
| `research_kagome_e2_p2` | Orthogonal-projection-penalty Lanczos for the second partner of the (Γ, E₂, Sz=1/2) doublet |
| `analyze_mes_result.sh` | Post-processor: K=4 Frobenius fit to (1/2)·H₄, K≥5 SV-spectrum + closest-sub-fit + rank-gap |
| `build_master_synthesis.py` | Aggregates per-sector JSONs into `master_synthesis.json` |

## What's not in this release

- **Tier-1 v0.5+ pillars** (factored-attention ViT NQS, foundation
  NQS, real µMAG #1/#3/#4, learned QEC decoders) remain out of
  scope. Honest fallback documentation is in place; full
  implementations track against forthcoming infrastructure.

## No breaking changes

All v0.4.3 public symbols retain their signatures and semantics.
The changes since v0.4.3 are all additive (new scripts, new
infrastructure functions in `mps/lanczos.h` and `nqs/nqs_symproj.h`,
new sector enum values).

---

# Spin-Based Neural Computation Framework v0.4.3

v0.4.3 is a **correctness + capability release** layered on top of
v0.4.2.  An end-to-end audit upgraded four legacy modules from
order-of-magnitude analytical proxies to first-principles
implementations, and four new literature-grade capabilities landed
alongside.  No public-API breaks.

## Highlights

- **MinSR optimiser** (Chen-Heyl 2024 / Rende-Goldt 2024) — solves
  the SR system in the N_s × N_s sample-space Gram matrix instead
  of the N_p × N_p quantum geometric tensor.  Drops linear-system
  memory from O(N_p²) to O(N_s²); same δθ as the CG-SR path on the
  same RNG seed.  Required for ansätze with N_p ≫ N_s (deep ResNet,
  ViT, …).
- **Full kagome wallpaper-group symmetry projection** —
  `nqs_kagome_{translation,p2,p3,p6,p6m}_perm` build permutation
  tables for every standard subgroup of p6m.  6-fold centre at
  (a₁ + a₂)/2 identified numerically; all builders verified by
  group-closure tests + end-to-end ψ-invariance under the full
  orbit.
- **9-term torque_net** with L=2 quadrupolar features and
  time-reversal classification — basis splits cleanly into t-odd
  {w1, w3, w4, w6, w8} (B_eff-compatible) and t-even {w0, w2, w5, w7};
  `torque_net_zero_t_even_weights` enforces the strict-t-odd contract
  for conservative LLG.
- **µMAG-lite trajectory benchmark** — first end-to-end physical
  validation of the LLG pillar.  Reference Heisenberg+Zeeman LLG
  trajectory → torque_net fit → torque_net-driven LLG.  Trajectory
  L_∞ agreement: 1.1e-16 over 40 RK4 steps (machine precision).

## First-principles upgrades (audit pass)

- **Berry phase / Chern number**: now uses QWZ Bloch states plus
  the Fukui-Hatsugai-Suzuki gauge-invariant lattice plaquette sum,
  replacing the earlier order-of-magnitude analytical proxy.
  Yields exact integer Chern numbers to machine precision.
- **Topological entanglement entropy**: now computes the Shannon
  entropy of the marginal P(s_A) directly via exact Boltzmann
  enumeration (N≤20) or Metropolis MC, replacing the earlier
  scalar correction term used for subsystem > 10 sites.
- **Majorana zero-mode detection + parity**: now uses BdG
  diagonalisation with the deterministic Kitaev-2001 parity result,
  replacing the earlier parameter-driven proxy.
- **NQS optimiser CG breakdown** is now signalled with iteration
  and residual rather than reported as `converged=1`.
- **NQS Lanczos materialisation** emits a stoquasticity warning
  when the `Re(ψ)`-only projection drops significant phase content
  — flags non-stoquastic kagome / J1-J2 / KH cases.
- **`matrix_neon` complex matvec** rewritten with `vld2q_f64` +
  `vfma{q,sq}` for actual 2-wide SIMD (the earlier
  `vsetq_lane_f64`-by-lane construction compiled to scalar SISD).
- **`qec_decoder`** transformer / Mamba kinds now emit a one-shot
  per-kind stderr warning at `qec_decoder_create()` time so the MWPM
  fallback is visible (was: silent fallback with `is_available=0`
  the only signal).

## libirrep linkage

[libirrep](https://github.com/tsotchke/libirrep) is now vendored as
a git submodule at `vendor/libirrep`.  After cloning, fetch it with

```
git submodule update --init --recursive
```

then `make IRREP_ENABLE=1 test_libirrep_bridge test_torque_net_irrep`
builds libirrep from the submodule and runs the bridge tests in one
shot — no external paths or pre-installed dependency required.  When
enabled, 9 additional tests run (Clebsch-Gordan unit elements,
Wigner-D identity at zero, spherical-harmonic addition theorem to
~6e-17).  System installs are still supported via
`IRREP_ROOT=/some/install`.  The default build (without
`IRREP_ENABLE=1`) remains libirrep-free.

## What's not in this release

- **Factored-attention ViT NQS** (Rende et al. 2024): 3-4 weeks of
  dedicated work.  MinSR + p6m projection are the two pieces it
  needs from this release.
- **Foundation NQS** (Hamiltonian-conditioned across the KH phase
  diagram): long-horizon.
- **Real µMAG #1, #3, #4**: µMAG-lite covers the synthetic-target
  slice; NIST-spec problems require dedicated parameter
  conversion + reference-data work.
- **Learned QEC decoders** (transformer, Mamba): the literature
  ceiling vs MWPM in the depolarising-only regime is ~10–25 % LER,
  not the 1.5× margin seen with hardware-specific (leakage / soft
  I-Q) wins.  Honest fallback documentation lands here; the
  implementation stays in scope for v0.5+ if the hardware modelling
  catches up.

## No breaking changes

All v0.4.2 public symbols retain their signatures and semantics.
`torque_net_params_t` gained four new weight fields (w5..w8)
inserted before `r_cut`; downstream code using designated
initialisers is unaffected.  Test files that used positional
initialisers were updated.

---

# Spin-Based Neural Computation Framework v0.4.2

v0.4.2 is a **research-capability release** layered on top of v0.4.1.
It adds the full kagome-Heisenberg diagnostic stack and an exact
reference solver, taking the NQS pillar from "scaffold + two kernels"
to "diagnostics complete + machine-precision anchor." No breaking
changes.

## What's new

### Sample-based diagnostics on trained NQS wavefunctions

Three new entry points in `include/nqs/nqs_diagnostics.h`, all
consuming a freshly sampled batch and returning scalars without
mutating the ansatz:

- **`nqs_compute_chi_F(cfg, Lx, Ly, ansatz, sampler, &trace_S, &per_param)`**
  — trace of the quantum geometric tensor
  `S_{k,l} = ⟨O_k* O_l⟩ − ⟨O_k*⟩⟨O_l⟩` with
  `O_k = ∂ log ψ / ∂θ_k`. Returns `Tr(S) = Σ_k(⟨|O_k|²⟩ − |⟨O_k⟩|²)`;
  χ_F = Tr(S)/2 in the Zanardi–Paunković 2006 convention. Works
  transparently on real and complex-amplitude ansätze via the
  holomorphic-gradient path.

- **`nqs_compute_kagome_bond_phase(cfg, Lx, Ly, ansatz, sampler, re[3], im[3], cnt[3])`**
  — per-bond-class circular mean of the amplitude ratio
  `ψ(s_{ij})/ψ(s)` on kagome. Classes are the three sublattice
  pairs {A-B, A-C, B-C}. Marshall-like sign structure shows as
  |⟨r⟩| ≈ 1 with arg ≈ π across all classes; frustrated / Dirac-
  compatible phases show mixed magnitudes and phases.

- **`nqs_sr_{step,run}_excited(..., ref_log_amp_fn, ref_log_amp_user, penalty_mu, ...)`**
  — excited-state stochastic reconfiguration via the Choo–Neupert–
  Carleo 2018 (arXiv:1810.10196) orthogonal-ansatz penalty.
  Augments the holomorphic-SR local energy by
  `μ · r(s) · conj(⟨r⟩)` with `r(s) = ψ_ref(s)/ψ(s)`, reusing the
  same QGT metric + CG solve as the ground-state SR path.
  `out_info->mean_energy` reports the physical ⟨H⟩, not the
  augmented loss. Log-ratio clamped at `exp(±10)` for tail-event
  safety. Validated on 2-site Heisenberg: reference cRBM reaches
  E = −0.7528, excited run with μ = 5 reaches E = +0.2498 —
  **4-decimal agreement with the exact triplet** E₁ = +0.25.

### Full-basis Lanczos — exact reference at small N

- **`nqs_lanczos_refine_kagome_heisenberg`** and its generic
  precursor `nqs_lanczos_refine_heisenberg` build the full 2^N-dim
  Heisenberg matvec matching the VMC local-energy kernel bond-for-
  bond (up-triangle + down-triangle on kagome), seed Lanczos from
  the trained state's `Re(ψ)`, and return the refined ground-state
  energy to machine precision.

- **`lanczos_k_smallest_with_init`** (in `include/mps/lanczos.h`) —
  k smallest Ritz values extracted from one Krylov pass; the spin
  gap E₁ − E₀ drops out as a single subtraction. NQS-level wrapper:
  **`nqs_lanczos_k_lowest_kagome_heisenberg`**.

**Research numbers on our 2×2 PBC kagome N=12 cluster** (dim = 4096):

| Quantity             | Value (Lanczos, machine precision) |
|----------------------|-----------------------------------|
| E₀                   | **−5.44487522 J**                 |
| E₁                   | −5.32839240 J                     |
| E₂                   | −5.29823654 J                     |
| **Spin gap Δ**       | **0.116483 J**                    |

The Leung-Elser 1993 literature value (−5.238 J) is 3.8 % off our
cluster's true GS — different PBC-wrap convention. Lanczos anchors
at any N ≤ 24 (dim 2²⁴ = 16M ≈ 128 MB at one double per basis
state) without an external ED code.

### End-to-end research driver

`make research_kagome_N12_diagnostics` chains:

1. Stage A  — complex-RBM holomorphic SR training
2. Stage A' — exact ⟨ψ|H|ψ⟩ + Lanczos refinement (E₀ to machine precision)
3. Stage B  — χ_F = Tr(S)/2
4. Stage C  — per-bond-class amplitude ratios on kagome
5. Stage D  — excited-state SR + k-lowest Lanczos (exact E₁ and Δ)

Output is a self-contained TAP-style report suitable for
longitudinal archiving under `benchmarks/results/`. Typical run
(500 GS + 300 excited iters): **~22 min on an M-series Mac**.

A simpler MC-only convergence probe is at
`make research_kagome_N12` (`scripts/research_kagome_N12_convergence.c`).

## Benchmark additions

`benchmarks/bench_nqs.c` gains sampler + full-SR-step throughput
rows for the KH and kagome kernels, so per-Hamiltonian drift
across releases surfaces in `benchmarks/results/`.

## Test suite

**359 / 359 passing**, up from 343 at v0.4.1. Zero warnings under
`-Wall -Wextra`. `make test_asan` clean (AddressSanitizer +
UndefinedBehaviorSanitizer).

New suites:

- `tests/test_nqs_chi_F.c` (6 cases): χ_F finiteness + non-
  negativity on complex-RBM and legacy-MLP, bad-args rejection,
  MC consistency across batch sizes, per-bond-class phase output,
  and a rejection check for the kagome-only bond-phase helper on
  non-kagome Hamiltonians.
- `tests/test_nqs_excited.c` (4 cases): μ=0 equivalence with
  holomorphic SR, null-reference rejection, 2-site Heisenberg
  triplet recovery to four decimal places, and a kagome N=12
  pipeline smoke.
- `tests/test_nqs_lanczos.c` gains
  `test_kagome_lanczos_k_lowest_gives_exact_gap`: k-Ritz ascending
  order, E₀ matches rank-1 refine to 10⁻⁸, positive spin gap.

## No breaking changes

All prior v0.4.1 public symbols retain their signatures and
semantics. New capability is opt-in via new entry points.

---

# Spin-Based Neural Computation Framework v0.4.1

v0.4.1 is a **capability-addition release** layered on top of v0.4.0:
two new Hamiltonian kernels in the NQS pillar plus a small amount of
build-system housekeeping. No breaking changes.

## New local-energy kernels

### Kitaev-Heisenberg on brick-wall honeycomb

`NQS_HAM_KITAEV_HEISENBERG` adds a unified Kitaev + Heisenberg
Hamiltonian:

```
H = K · Σ_⟨ij⟩ σ^{γ_ij}_i σ^{γ_ij}_j  +  J · Σ_⟨ij⟩ σ_i · σ_j
```

on the brick-wall honeycomb, with `γ_ij ∈ {x, y, z}` set by the
usual parity colouring (horizontal `(x+y)` even → x-bond, else
y-bond; vertical → z-bond). Positive K / J follow the
Chaloupka–Jackeli–Khaliullin antiferromagnetic convention. Config
fields: `cfg.kh_K`, `cfg.kh_J`.

Reduces cleanly:
- K = 0 → stock Heisenberg on the honeycomb bond list.
- J = 0 → pure Kitaev (up to the overall sign of K — this kernel
  uses `H = +K σ^γ σ^γ` whereas the legacy `local_energy_kitaev`
  uses `H = −J σ^γ σ^γ`).

Both real and complex-amplitude paths are shipped
(`local_energy_kh`, `local_energy_kh_complex`). Kitaev-dominated
regimes are non-stoquastic and require `NQS_ANSATZ_COMPLEX_RBM`.
See `docs/nqs.md §4` for the full per-bond matrix-element table.

### Heisenberg on kagome lattice

`NQS_HAM_KAGOME_HEISENBERG` adds nearest-neighbour Heisenberg on a
three-sublattice kagome geometry. Caller passes `(size_x, size_y) =
(Lx_cells, Ly_cells)` and sizes the sampler with
`num_sites = 3 · Lx · Ly`. Flat index
`i = 3·(cx·Ly + cy) + s`, `s ∈ {A, B, C}`. Each cell contributes an
up-triangle plus a down-triangle anchored at `A(cx, cy)`. A 2×2 PBC
cluster has `N = 12` sites and 24 bonds with coord 4 everywhere.

Config fields: `cfg.j_coupling` (J), `cfg.kagome_pbc` (1 = PBC
default, 0 = OBC).

This is the kernel for the open kagome Heisenberg S=½ ground-state
problem (gapped Z₂ spin liquid vs gapless Dirac spin liquid). Full
diagnostic coverage (γ, S_VN, k-point spectrum) pairs with
libirrep ≥ 1.3.0-alpha's batched RDM, entropy, and point-group
projection primitives — gated behind `SPIN_NN_HAS_IRREP` and dormant
in the default tree until libirrep is vendored.

## Housekeeping

- `VERSION_PINS`: `LIBIRREP_MIN` bumped from 1.2 to 1.3.0-alpha to
  match the incoming libirrep release with kagome geometry, p6mm
  wallpaper group, and config-projection helpers.
- `VERSION_PINS`: removed five stale per-test tolerance keys that
  drifted from their source. Per-test tolerances live at the top of
  each `tests/test_*.c`; general policy in `REPRODUCIBILITY.md`.
- `Makefile`: `test_asan` now picks ASAN's `detect_leaks` flag by
  OS (Linux = 1, Darwin = 0). macOS's ASan runtime aborts at startup
  with `detect_leaks=1`; this keeps the target runnable on both.

## Tests added

- `tests/test_nqs_kitaev.c` gains 4 KH analytical checkpoints on a
  `2×2` brick-wall cluster: `K=0, J=1`; `K=1, J=0`; `K=J=1`;
  antiparallel-z-bond off-diagonal trigger.
- `tests/test_nqs_kagome.c` ships 5 kagome checkpoints: 2×2 PBC
  all-up, uniform-ψ invariance (`E_loc = (J/4)·|bonds|`), J scaling,
  2×2 OBC all-up, 1×1 PBC degenerate regression guard.
- `tests/test_nqs_holomorphic_sr.c` gains an end-to-end KH
  convergence test (complex RBM + holomorphic SR on 2×2, 60 iters,
  head-to-tail descent assertion).

## Test suite + sanitizer

- Full `make test` at v0.4.1: **343 / 343 passing**, up from 333 in
  v0.4.0 (now 359 / 359 at v0.4.2). Zero warnings under
  `-Wall -Wextra`.
- `make test_asan` (AddressSanitizer + UndefinedBehaviorSanitizer):
  clean across the full suite. No UB, no memory errors.

## Compatibility

- Fully backwards-compatible with v0.4.0. No existing API signatures
  changed; no existing dispatch case reassigned.
- Two new `nqs_hamiltonian_kind_t` enum values (5, 6) appended.
- Three new `nqs_config_t` fields (`kh_K`, `kh_J`, `kagome_pbc`)
  appended at the end of the struct; `nqs_config_defaults()`
  initialises them.
- `nqs_local_energy` / `nqs_local_energy_complex` now compute the
  site count per-Hamiltonian (previously `size_x × size_y` uniformly)
  — necessary for multi-sublattice lattices. For every existing
  Hamiltonian this yields the same value as before.

## Next (superseded by v0.4.2)

- ~~`χ_F`-from-samples helper (trace of the QGT)~~ — **landed in
  v0.4.2** as `nqs_compute_chi_F`.
- Point-group projection (`NQS_SYM_POINT_GROUP` ↔ libirrep's
  `irrep_pg_project`) wires up when libirrep `v1.3.0-alpha.1` lands.
- Open kagome-S=½ ground-state research continues publicly under
  `docs/research/kagome_KH_plan.md` (v0.4.2+).

---

# Spin-Based Neural Computation Framework v0.4.0

v0.4.0 is the **foundation release** for the v0.5+ research pillars. It
significantly advances the topological-quantum-computing physics beyond
v0.3, adds engine-neutral scaffolding for the upcoming neural-network
research pillars, and ships a comprehensive 109-test verification suite
and 4-suite benchmark harness.

## Physics improvements

### Majorana braiding — Hilbert-space unitaries

v0.4 extends Majorana braiding with a full Hilbert-space implementation
of the Ising-anyon braiding unitary:

```
B_{ij} = exp(π γ_i γ_j / 4) = (1 + γ_i γ_j) / √2
```

acting on the `2^(N/2)`-dim fermion-parity Fock basis, constructed via
Jordan-Wigner. Verified properties (see `tests/test_majorana.c`):

- `γ_i² = I` for all i
- `{γ_i, γ_j} = 0` for i ≠ j
- `B^4 = −I` (half-period)
- `B^8 = +I` (order-8 non-Abelian statistics)
- Braiding preserves the `||ψ||²` norm (unitary)
- Braiding preserves fermion parity

The v0.3 operator-space braiding path remains available as
`braid_majorana_operators_legacy()`; `braid_majorana_modes()` continues
to work unchanged. The new `MajoranaHilbertState` API is additive.

New public API in `include/majorana_modes.h`:

- `MajoranaHilbertState *initialize_majorana_hilbert_state(int num_majoranas)`
- `void free_majorana_hilbert_state(MajoranaHilbertState *state)`
- `void majorana_hilbert_state_set_vacuum(MajoranaHilbertState *state)`
- `void apply_majorana_op_to_state(int op_index, MajoranaHilbertState *state)`
- `void apply_braid_unitary(MajoranaHilbertState *state, int i, int j)`
- `double majorana_state_norm_squared(const MajoranaHilbertState *state)`
- `double _Complex majorana_states_inner_product(const MajoranaHilbertState *a, const MajoranaHilbertState *b)`
- `void majorana_state_copy(const MajoranaHilbertState *src, MajoranaHilbertState *dst)`

### Toric-code error correction — data-qubit model

v0.4 improves the toric-code error-correction implementation with:

- An explicit **data-qubit model**: 2·L_x·L_y data qubits, one per link,
  with independent `x_errors[]` and `z_errors[]` GF(2) accumulators.
  Corrections now flip data qubits directly and re-derive the syndromes
  from them, so repeated error-and-correction cycles remain consistent.
- **Primal/dual-lattice-aware path walks**: Z-error corrections walk
  the primal lattice (vertex-to-vertex); X-error corrections walk the
  dual (plaquette-to-plaquette). The walk direction determines which
  link type is flipped at each step.
- **Homology-based logical-error detection**:
  `toric_code_has_logical_error()` computes the error chain's class in
  H₁(dual, Z₂) by intersection with primal 1-cycles, which is invariant
  under stabilizer action and therefore robust to any sequence of
  corrections.
- **Greedy matching decoder** (`toric_code_decode_greedy()`): pairs
  flagged syndromes by toroidal taxicab distance and applies corrections
  along shortest paths. Not optimal (proper MWPM comes in v0.5), but
  effective at physically-reasonable error rates.

New public API in `include/toric_code.h`:

- Data-qubit accessors: `toric_code_link_index`, `toric_code_vertex_links`,
  `toric_code_plaquette_links`
- Error / correction: `toric_code_apply_{x,z}_{error,correction}`,
  `toric_code_refresh_syndromes`
- Channel-specific syndromes: `toric_code_measure_x_syndrome`,
  `toric_code_measure_z_syndrome`
- Decoding: `toric_code_decode_greedy`
- Queries: `toric_code_has_logical_error`, `is_ground_state` (now
  also verifies no accumulated logical error)

Baseline toric-decoder logical-error rate, measured on M-series Mac
(see `benchmarks/results/toric_decoder/`):

| distance | p=1% | p=3% | p=5% |
|---|---|---|---|
| 3 | 0.8% | 6.8% | 15.4% |
| 5 | 0.2% | 5.6% | 14.0% |
| 7 | 0.0% | 2.4% | 9.2% |

Logical error rate decreases with distance at p=1% (below threshold). At
p=5% all distances show high rates (above threshold). The learned
surface-code decoder in v0.5 pillar P1.3 must beat these numbers.

### Berry-phase `CHERN_NUMBER` test override

The `CHERN_NUMBER` environment-variable hook is now a test-only feature,
gated behind `#ifdef SPIN_NN_TESTING`. Release builds run the actual
Chern-number calculation unconditionally; test builds opt in with
`-DSPIN_NN_TESTING` when they need to force specific values for
regression testing.

## Engine-neutral scaffolding

Three new modules support the external-engine integration planned for
v0.5+. All compile in the default build as dormant stubs (enabled with
`make ENGINE_ENABLE=1 ENGINE_ROOT=...`):

- **`engine_adapter`** — single translation unit that will bridge to an
  external NN / tensor / reasoning engine. Engine-neutral: any engine
  providing `engine_backend_init`, `engine_backend_shutdown`,
  `engine_backend_version` as weak hooks can plug in. Planned backends
  for v0.5+: an Eshkol-native NN engine (working title
  `eshkol-transformers`, built on
  https://github.com/tsotchke/eshkol) and Noesis (reasoning engine,
  in development, not yet publicly released).
- **`eshkol_bridge`** — lazy wrapper over the Eshkol FFI for
  Scheme-orchestrated training tapes. `ESHKOL_BRIDGE_EDISABLED` until
  built with `-DSPIN_NN_HAS_ESHKOL=1`.
- **`nn_backend`** — polymorphic neural-network handle.
  `NN_BACKEND_LEGACY` uses the in-tree MLP; `NN_BACKEND_ENGINE` delegates
  to whichever engine is wired in via `engine_adapter`. New CLI flag
  `--nn-backend={legacy,engine}` toggles at runtime; engine transparently
  falls back to legacy with a diagnostic when no engine is available.

## Training-loop integration

New CLI flags on the main binary drive in-loop topological feedback:

- `--cadence-decoder N` — run the greedy toric-code decoder every N
  training iterations against a syndrome sampled at
  `--decoder-error-rate P` (default 0.03). Logical-error flag folds
  into `physics_loss` with coefficient `--lambda-logical L`.
- `--cadence-invariants N` — compute topological invariants every N
  iterations (advisory in v0.4; pillar P1.2 in v0.5 folds them into the
  loss).

See `include/training_config.h` for the full option set.

## Test and benchmark harnesses

### Test suite: 109 tests across 17 suites

Every library module has at least one test. Run `make test` for the
full suite. TAP-style output; any failure propagates the exit code.

| Suite | Tests | Module(s) covered |
|---|---|---|
| test_majorana | 13 | majorana_modes.c (incl. legacy + Hilbert paths) |
| test_toric_code | 14 | toric_code.c (all v0.4 APIs) |
| test_ising | 7 | ising_model.c |
| test_kitaev | 6 | kitaev_model.c |
| test_topological_entropy | 3 | topological_entropy.c (legacy) |
| test_engine_adapter | 8 | engine_adapter.c |
| test_nn_backend | 11 | nn_backend.c |
| test_spin_models | 4 | spin_models.c |
| test_energy_utils | 4 | energy_utils.c |
| test_disordered_model | 4 | disordered_model.c |
| test_eshkol_bridge | 5 | eshkol_bridge.c |
| test_physics_loss | 6 | physics_loss.c (5 PDE losses + Laplacian) |
| test_berry_phase | 5 | berry_phase.c (incl. CHERN_NUMBER gating) |
| test_reinforcement_learning | 4 | reinforcement_learning.c |
| test_quantum_mechanics | 3 | quantum_mechanics.c |
| test_ising_chain_qubits | 7 | ising_chain_qubits.c |
| test_matrix_neon | 5 | matrix_neon.c |

Full rebuild + test run on an M-series Mac completes in ≈ 2 seconds.

### Benchmark harness: 4 suites, 19 JSON records

Run `./scripts/run_benchmarks.sh` to build and execute all benchmarks;
results land under `benchmarks/results/<suite>/<name>.json`.

- `bench_ising` — Metropolis sweeps/sec at L = 8, 16, 32
- `bench_kitaev` — Metropolis sweeps/sec at L = 8, 16, 32
- `bench_majorana_braid` — braids/sec at N = 8, 12, 16, 20 Majoranas
- `bench_toric_decoder` — greedy-decoder logical-error-rate curves at
  d = 3, 5, 7 × p = 1%, 3%, 5%

See `docs/benchmarks.md` for the JSON schema and provenance notes.

## Build system and stack probe

- `make test` runs all 17 test suites.
- `make bench` builds all 4 benchmark suites;
  `./scripts/run_benchmarks.sh` runs them and writes JSON.
- `./scripts/check_stack.sh` probes for optional external dependencies
  (engine, libirrep, Eshkol runtime). All advisory in v0.4.
- `make ENGINE_ENABLE=1 ENGINE_ROOT=...` (v0.5+) lights up the
  engine-adapter build path.

## Backward compatibility

- The `arm`, `non_arm`, and `universal` targets still emit compatible
  executables under `build/` and accept every v0.3 CLI flag unchanged.
- `braid_majorana_modes()` remains available and delegates to the legacy
  operator-permutation path.
- All existing example shell scripts (`run_topological_examples.sh`,
  `different_topo_values.sh`) run without modification.

## Upgrade notes

- The `CHERN_NUMBER` environment variable is now a test-only feature.
  If you relied on it to force specific values during experimentation,
  rebuild with `-DSPIN_NN_TESTING`.
- `include/toric_code.h` adds many new public symbols but removes none;
  existing code compiles unchanged.
- `include/majorana_modes.h` adds the `MajoranaHilbertState` type and
  associated functions; `braid_majorana_modes()` is preserved.

## Forward roadmap: v0.5 research pillars

See [docs/architecture_v0.4.md](docs/architecture_v0.4.md). Headline pillars:

- **P1.1 Neural Network Quantum States** — ViT / factored-ViT
  wavefunctions for frustrated magnets (J1-J2 square, kagome, Kitaev
  honeycomb)
- **P1.2 Equivariant GNN torques + real Landau-Lifshitz-Gilbert dynamics**
  — E(3)-equivariant message passing (uses forthcoming `libirrep`)
- **P1.3 Learned surface-code decoders** — transformer and Mamba
  variants beating the v0.4 MWPM baseline
- **P1.3b Fibonacci anyons** — universal-gate counterpart to the
  non-universal Majorana module via Solovay-Kitaev
- **P1.4 Neural-operator LLG accelerators + discrete flow-matching
  Boltzmann samplers**
- **P2.x** Time-dependent NQS, MPS warm-start, foundation NQS, p-bit
  neuromorphic, KAN-NQS, Lanczos, PINN upgrades, gauge-invariant
  sampling, thermodynamic computing

## Getting started

```sh
git clone https://github.com/tsotchke/spin_based_neural_network.git
cd spin_based_neural_network
make arm test bench
./scripts/run_benchmarks.sh
./build/spin_based_neural_computation_arm --help
```

## Full changelog

See [CHANGELOG.md](CHANGELOG.md).

---
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
