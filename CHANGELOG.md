# Changelog

All notable changes to the Spin-Based Neural Computation Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — sector-projected NQS predictive observables on kagome AFM

End-to-end pipeline from random NQS seed through projecting Lanczos
to predictive observables on the kagome 2×2 PBC ground state, plus
infrastructure to extend to N=27 PBC.

### Added — variational stack
- **Sector-projecting Lanczos** (`mps/lanczos.h`):
  `lanczos_smallest_projected`, `lanczos_k_smallest_projected` —
  in-loop sector projection inside the Krylov build kills the
  power-method amplification of machine-precision sector leakage.
  Required for accurate sector spectra at small dim where the gap
  between in-sector and out-of-sector lowest eigenvalues is smaller
  than the leak-amplification factor.
- **Memory-lean projecting Lanczos** (`mps/lanczos.h`):
  `lanczos_smallest_projected_lean` — 3-term recurrence with no
  Krylov-basis storage; O(few · dim) memory.  Required to push to
  N≥24 where full reorthogonalisation is infeasible (300 GB at N=27).
- **Two-pass lean projecting Lanczos with eigenvector reconstruction**
  (`mps/lanczos.h`): `lanczos_smallest_projected_lean_eigvec`.  Pass 1
  saves α/β; pass 2 diagonalises tridiagonal; pass 3 replays Lanczos
  to accumulate Ritz vector.  Wall 2× lean, memory unchanged.
  Unblocks all post-processing (TEE, S(q,ω), partial trace) at large N.
- **Continued-fraction Lanczos** (`mps/lanczos.h`):
  `lanczos_continued_fraction` + `lanczos_cf_evaluate`.  Evaluates
  ⟨φ|(z−H)⁻¹|φ⟩ via Stieltjes continued fraction for spectral
  functions (S(q,ω), Green's functions).
- **Kagome p6m projector as vector op** (`nqs/nqs_symproj.h`):
  `nqs_kagome_p6m_project_inplace`.  Applies P_α = (1/G) Σ_g χ(g) T(g)
  in place to a length-2^N state vector.  3-bit-chunk lookup table
  optimisation: 62 KB cache footprint, ~5× faster than the
  per-bit-shift loop.
- **OpenMP parallelisation** of the kagome H matvec and the projector
  via `make OPENMP=1` (Apple Clang + Homebrew libomp by default;
  fallback to gcc -fopenmp).  M2 Ultra: ~14× speedup on 14 perf
  cores.
- **Sector-projected Heisenberg ED via libirrep**: callback variants
  of nqs_lanczos refine + k-lowest paths
  (`nqs_lanczos_{refine,k_lowest}_kagome_heisenberg_with_cb` /
  `_projected`).  Sector spectrum extraction without sector-built
  sparse Hamiltonian.

### Added — research drivers (predictive observables on N=12 kagome AFM)
- `scripts/research_kagome_b1_train.c` — sector-projected NQS training
  in (Γ, B_1) + Lanczos refinement.  hidden=64/1500-iter run reaches
  E_var = -5.358 (1.6% rel err); Lanczos refinement on the projected
  ψ_sym hits E_0 = -5.4448752170 to 1e-10 % precision in 33 iters.
- `scripts/research_kagome_sector_spectrum.c` — sector-resolved
  low-energy spectrum scan via projecting k-lowest Lanczos.  All four
  1D-irrep ground states match libirrep ED to 8+ digits.
- `scripts/research_kagome_tee.c` — γ_TEE area-law fit on the
  projected (Γ, B_1) ground state.  γ ≈ 1.06 · log 2 (Z₂ spin-liquid
  signature).
- `scripts/research_kagome_correlations.c` — static spin correlations
  ⟨S_i·S_j⟩ + structure factor S(q).  S(q=Γ)=0 (no FM order),
  S(q=K-eq)/N = 0.033 (no Bragg peak → spin liquid).
- `scripts/research_kagome_sqw.c` — dynamic structure factor S(q,ω)
  via continued-fraction Lanczos.  Predicts inelastic-neutron line
  shapes; lowest peak ω = 0.60 J at non-Γ momenta.
- `scripts/research_kagome_3x3_e0.c` — N=27 PBC sector ground states
  via lean projecting Lanczos.  Memory ~6 GB working set.
- `scripts/research_kagome_observables_lean.c` — combined predictive-
  observables driver (E_0 + ψ_0 + correlations + γ_TEE) for any
  (L, irrep) pair.  Supports L=1..3.

### Added — tests + manifests
- `test_nqs_sector_lanczos` (3 cases): regression tests locking in
  the libirrep ED agreement for the projected, lean, and
  eigenvector-reconstruction Lanczos paths.
- Six archived JSON result manifests under `benchmarks/results/nqs/`:
  `kagome_b1_h64_convergence`, `kagome_b1_lanczos_refined`,
  `kagome_sector_spectrum`, `kagome_tee_2x2`,
  `kagome_correlations_2x2`, `kagome_sqw_2x2`,
  `kagome_sector_observables_2x2`.

### Discovered + fixed
- **Sector-leakage power-method amplification in projecting Lanczos**:
  without in-loop projection, machine-precision (~1e-14) numerical
  sector leakage gets amplified by Lanczos dynamics over O(log dim ·
  log eigen-ratio) iterations and converges to the global ground state
  regardless of seed sector.  Discovered via diagnostics on the
  kagome 2×2 PBC cluster (ψ_sym sector-pure to 1e-15, but Lanczos
  returned global B_1 GS from any sector seed).  Fixed by inserting
  P_α projection after every Krylov step.  Likely a silent bug in
  many literature codes that don't explicitly handle this.
- **libirrep_bridge_entropy doc bug**: `n` parameter is the matrix
  *dimension* (= 2^nA for spin-1/2 RDM), not site count.  Header
  docstrings updated to clarify.

## [0.4.3] — 2026-04-26 — MinSR + kagome p6m + audit corrections

A correctness-and-capability release: a top-to-bottom audit replaced
silently-wrong physics in four legacy modules, and four new
literature-grade capabilities landed on top.  No public-API breaks.

### Added — variational stack

- **MinSR optimiser** (`nqs_sr_step_minsr_full`, `nqs_sr_run_minsr` in
  `include/nqs/nqs_optimizer.h`).  Sample-space stochastic
  reconfiguration via the Chen-Heyl 2024 / Rende-Goldt 2024
  push-through identity:
  `(O_c^T O_c / N_s + ε I)^{-1} O_c^T = O_c^T (T + ε I)^{-1}`
  with `T = O_c O_c^T / N_s` the N_s × N_s sample-space Gram matrix.
  Drops the linear-system memory from N_p² to N_s² and bounds the
  Krylov dimension by N_s.  Cholesky-factor + back-solve, no BLAS.
  Same RNG seed → same δθ as the matrix-free CG path.

- **Full kagome wallpaper-group symmetry projection**
  (`include/nqs/nqs_symproj.h`, `src/nqs/nqs_symproj.c`).  Generic
  `nqs_symproj_wrapper_t` takes any `|G| × N` permutation table plus
  a 1-D character vector; ships with builders for p1 (translations),
  p2 (× C₂ at A), p3 (× C₃ at up-triangle centroid), p6
  (× C₆ at hexagon centroid (a₁+a₂)/2), and p6m (12 point ops).
  6-fold rotation centre identified numerically by
  `tools/find_kagome_p6_center.c`.

- **µMAG-lite trajectory benchmark**
  (`tests/test_torque_net_micromagnetic_trajectory.c`).  Reference
  Heisenberg + Zeeman LLG trajectory → torque_net fit on the
  (m_t, B_eff_t) pairs → torque_net-driven LLG from same initial
  state.  Trajectory L_∞ agreement: 1.1e-16 over 40 RK4 steps;
  fit residual 5e-16; w4 recovered exactly (0.800000 vs 0.800000).
  First end-to-end physical validation of the LLG pillar.

### Added — torque_net basis extension

- **L=2 quadrupolar features** (`include/equivariant_gnn/torque_net.h`).
  Four new vector terms built from rank-2 contractions of
  {m_i, m_j, r̂}: `(m_i · r̂) m_j`, `(m_i · m_j) m_j`,
  `(m_i · m_j) r̂`, `(m_j · r̂) r̂`.  Total basis: 5 → 9 weights.
  Fit replaced 5×5 Gauss elim with generic 9×9 partial-pivoting solve.

- **Time-reversal classification + `zero_t_even_weights` helper**.
  Header now documents each term's t-parity (B_eff is t-odd; basis
  splits into t-odd {w1, w3, w4, w6, w8} and t-even
  {w0, w2, w5, w7}).  `torque_net_time_reversal_residual` and
  `torque_net_zero_t_even_weights` enforce the strict-t-odd contract
  for conservative LLG.  Backwards-compatible: the four golden tests
  still produce bit-exact outputs with the new basis.

### Added — physics_loss + main loop

- **Variational `micromagnetic_loss`** (`include/physics_loss.h`):
  exchange + uniaxial anisotropy + Zeeman energy functional.
  Wired as the `"micromagnetic"` `loss_type` in
  `compute_physics_loss`.
- **`project_spin_lattice_to_unit_sphere`**: hard |m|=1 constraint
  with drift diagnostic.
- **`fourier_features`**: NeRF-style sin/cos embedding for PINN
  coordinate inputs.
- **Topological observables fold into `physics_loss`** when the
  invariants and entropy cadences are configured.  New CLI flags:
  `--cadence-entropy`, `--lambda-chern`, `--lambda-topological`,
  `--target-chern`, `--target-gamma`.

### Changed — physics correctness

- **Berry phase / Chern number**.  `get_eigenstate` now returns the
  exact lower-band Bloch state of the Qi-Wu-Zhang square-lattice
  model (was: a fake plane-wave).  `calculate_chern_number` uses the
  Fukui-Hatsugai-Suzuki gauge-invariant lattice-plaquette sum (was:
  a magnetic-monopole heuristic).  Yields exact integer Chern
  numbers to machine precision for gapped systems.  Print labels
  for "FQHE C=1/3" and "Z₂ TI C=1" removed; replaced with correct
  QAH / trivial / higher-Chern classification.
- **Topological entanglement entropy**.
  `calculate_von_neumann_entropy`'s 10%-interaction heuristic for
  >10-site subsystems replaced with real Shannon entropy of the
  marginal P(s_A) computed via exact Boltzmann enumeration (N≤20)
  or Metropolis MC (N>20).  Memory-efficient diagonal-only path
  caps |A| at 24 sites.
- **Majorana physics**.  `calculate_majorana_parity` no longer
  returns `rand()` — deterministic ±1 from the Kitaev-2001 BdG
  ground-state.  `detect_majorana_zero_modes` now diagonalises the
  2N × 2N BdG Hamiltonian and returns an end-localisation measure
  derived from the lowest-|E| eigenvector (was: a parameter-driven
  heuristic).
- **NQS optimiser CG breakdown** in `nqs_sr_step_*` is now signalled
  via `converged=0` plus an iter/residual stderr message (was:
  silent break with `converged=1` left from the previous
  successful step).
- **NQS Lanczos materialisation** emits a stoquasticity warning
  when `‖Im ψ‖² / ‖ψ‖²` exceeds 1e-6.  Kagome / J1-J2 / frustrated
  KH ground states are non-stoquastic; the `Re(ψ) = |ψ| cos(arg ψ)`
  projection silently discarded physical phase content there.
- **`matrix_neon` complex matvec** uses `vld2q_f64` + `vfma{q,sq}`
  for true 2-wide SIMD (was: `vsetq_lane_f64` lane-by-lane build,
  which compiled to scalar SISD).
- **`qec_decoder`** transformer / Mamba kinds were silently falling
  back to MWPM with only `is_available = 0`.  Added one-shot
  per-kind stderr warnings at create time so a production run can
  no longer believe a "learned decoder" is real.
- **`torque_net` header** rewritten to state SO(3) covariance
  honestly (was: claimed "E(3)-equivariant"; only 5 scalar weights,
  no parity gating, no irrep tower).  Upgrade path documented.
- **`noesis_bridge` / `qgtl_bridge`** distinguish "compiled out"
  (`_EDISABLED`) from "compiled in but body is a placeholder"
  (`_ENOT_IMPLEMENTED`).  The live path was previously
  indistinguishable from the disabled path.
- **`mps/dmrg` docstring**: clarified that the bond-dim-5 MPO
  dispatches between TFIM, Heisenberg, and XXZ (was: described as
  "hard-coded XXZ").
- **`flow_matching` docstring**: v0.4 ships a constant rate, v0.5+
  will install a learned schedule; previously the field name and
  docstring suggested the schedule was already there.
- **`llg/demag`** silent NULL return on non-power-of-two grids
  replaced with an actionable stderr message naming the
  next-power-of-two pad size.

### Fixed — documentation

- TEE sign convention in `docs/visualization.md`: `γ = +log(D) > 0`
  (was: `S_topo = -log(2)`, sign error).
- Chern-number classification in `docs/berry_phase.md` §4.2:
  removed false claims that Z₂ topological insulators have integer
  Chern number and that FQHE Chern is 1/3 (Chern is always integer;
  Z₂ TI has Z₂ index, not Chern).
- KitaevLattice naming in `docs/classical_models.md`: clarified
  this is a classical anisotropic Ising lattice, **not** the
  quantum Kitaev honeycomb model — the quantum phase diagram is
  used by `berry_phase.c` and `topological_entropy.c` to
  parameterise the quantum analogue.
- Removed duplicate §3.5 in `docs/topological_entropy.md`.
- Bluestein FFT comment in `src/neural_operator/fft.c`: now
  honestly says "naive O(n²) DFT for non-PoT n; Bluestein deferred".

### Build

- **libirrep is now a git submodule at `vendor/libirrep`**.  After
  cloning, run `git submodule update --init --recursive` once; then
  `make IRREP_ENABLE=1 test_libirrep_bridge test_torque_net_irrep`
  builds libirrep from the submodule (`make -C vendor/libirrep lib`)
  and links the bridge in one shot.  No external path required;
  system installs still work via `IRREP_ROOT=/some/install`.
- Removed pre-existing hardcoded local paths from
  `include/moonlab_bridge.h`, `include/noesis_bridge.h`,
  `tests/test_moonlab_bridge.c`, and `docs/libirrep_1_2_coordination.md`
  (replaced with github URLs and `/path/to/...` placeholders).

### Tests

- New: `test_nqs_minsr` (MinSR vs CG-SR equivalence on TFIM 2×2),
  `test_nqs_symproj` (translation + p2 + p3 + p6 + p6m closure and
  end-to-end ψ-invariance under full orbits),
  `test_torque_net_micromagnetic_trajectory` (LLG trajectory match
  to machine precision after fit).
- Extended: `test_torque_net.c` adds isolated L=2 equivariance, full
  9-weight fit, and t-parity subset assertions.  `test_physics_loss.c`
  covers the new variational micromagnetic loss, hard-constraint
  projection, and Fourier-feature embedding.

### No breaking changes

All v0.4.2 public symbols retain their signatures and semantics.
The `torque_net_params_t` struct gained four new weight fields
(w5..w8) inserted before `r_cut`; positional initialisers in the
test suite were converted to designated initialisers as a result.
External code using designated initialisers is unaffected.

---

## [0.4.2] — 2026-04-24 — kagome diagnostics + Lanczos reference

### Added — sample-based diagnostics
- **χ_F = Tr(S)/2 helper** (`nqs_compute_chi_F` in
  `include/nqs/nqs_diagnostics.h`). Returns the trace of the quantum
  geometric tensor from a freshly sampled batch via the same
  complex-gradient path holomorphic SR uses. Real and complex
  ansätze transparently supported. Convention: Zanardi–Paunković
  2006 (χ_F = Tr(S)/2).
- **Bipartite phase probe on kagome** (`nqs_compute_kagome_bond_phase`).
  Per-bond-class circular mean of the amplitude ratio
  ⟨ψ(s_{ij})/ψ(s)⟩_α for α ∈ {A-B, A-C, B-C}. Distinguishes
  Marshall-like sign structure from frustrated / Dirac-compatible
  phase profiles.

### Added — excited-state VMC
- **Excited-state stochastic reconfiguration** (`nqs_sr_{step,run}_excited`
  in `include/nqs/nqs_optimizer.h`). Implements Choo–Neupert–Carleo
  2018 (arXiv:1810.10196) orthogonal-ansatz penalty VMC. Augments
  the holomorphic-SR local energy by μ·r(s)·conj(⟨r⟩) where
  r(s) = ψ_ref(s)/ψ(s); log-ratio clamped at exp(±10) to contain
  tail events. `out_info->mean_energy` reports the physical ⟨H⟩,
  not the augmented loss. Validated on 2-site Heisenberg: excited-
  SR with μ=5 recovers E₁ = +0.25 to four decimal places against
  an exact reference.

### Added — exact reference via Lanczos
- **Kagome Heisenberg Lanczos refinement**
  (`nqs_exact_energy_kagome_heisenberg`,
  `nqs_lanczos_refine_kagome_heisenberg` in
  `include/nqs/nqs_lanczos.h`). Builds the full 2^N-dim Hamiltonian
  matvec for the 2×2 PBC cluster (N=12, dim=4096) matching the VMC
  local-energy kernel bond-for-bond, and refines the trained cRBM
  state to machine precision. On our specific cluster
  E₀_exact = −5.44487522 J (3.8 % below the Leung-Elser literature
  value — different PBC-wrap convention).
- **Multi-Ritz Lanczos** (`lanczos_k_smallest_with_init` in
  `include/mps/lanczos.h` +
  `nqs_lanczos_k_lowest_kagome_heisenberg`). Extracts the k
  smallest eigenvalues from one Krylov run, so the spin gap
  Δ = E₁ − E₀ = 0.116483 J on N=12 drops out as one subtraction
  alongside E₀.
- Existing `nqs_lanczos_refine_heisenberg` now seeds Lanczos from
  the trained state's Re(ψ) rather than a deterministic xorshift
  fallback, dropping convergence from full-dim to tens of Krylov
  steps on well-trained ansätze.

### Added — end-to-end research driver
- **`scripts/research_kagome_N12_diagnostics.c`** (invoked via
  `make research_kagome_N12_diagnostics`) chains GS SR →
  χ_F → per-bond-class phase → excited-state SR →
  Lanczos-exact E₀/E₁/gap on one N=12 PBC kagome cluster. Output
  is a TAP-style report. Typical run: ~22 min on an M-series Mac.
  Not wired into `make test`.
- `scripts/research_kagome_N12_convergence.c` (via
  `make research_kagome_N12`) is the simpler MC-only convergence
  probe.

### Added — benchmarks
- Sampler + holomorphic-SR-step throughput rows for the KH and
  kagome kernels in `benchmarks/bench_nqs.c`; bench suite now
  covers the three major NQS pipeline stages (local-energy,
  sampler, full-SR-step) per Hamiltonian.

### Tests
- **New `tests/test_nqs_chi_F.c`** (6 cases): χ_F finiteness +
  non-negativity on complex-RBM and legacy-MLP ansätze, bad-args
  rejection, MC consistency across batch sizes, and the kagome
  bond-phase probe with per-class output + rejection on non-kagome
  Hamiltonians.
- **New `tests/test_nqs_excited.c`** (4 cases): μ=0 equivalence
  with holomorphic SR, null-reference rejection, 2-site Heisenberg
  triplet recovery to four decimal places, and a kagome N=12
  pipeline smoke.
- `tests/test_nqs_lanczos.c` gains
  `test_kagome_lanczos_k_lowest_gives_exact_gap`: ascending Ritz
  order, E₀ matches rank-1 refine to 10⁻⁸, positive gap.
- **Total: 359 / 359 passing**, up from 343 at v0.4.1. Zero
  warnings under `-Wall -Wextra`. AddressSanitizer +
  UndefinedBehaviorSanitizer clean.

### No breaking changes
All v0.4.1 public symbols retain their signatures and semantics.
Every new capability is opt-in via new entry points.

---

## [0.4.1] — 2026-04-23 — Hamiltonian kernels: KH + kagome

### Added
- **Kitaev-Heisenberg local-energy kernel** (`NQS_HAM_KITAEV_HEISENBERG`)
  on the brick-wall honeycomb. Convention `H = K · Σ σ^γ σ^γ + J · Σ σ·σ`
  (Chaloupka–Jackeli–Khaliullin sign). Real + complex-amplitude paths.
  Config: `cfg.kh_K`, `cfg.kh_J`. Reduces to Heisenberg at K=0 and to
  pure Kitaev (up to the sign of K) at J=0. Scope: honeycomb KH phase
  diagram capability, *not* the kagome Heisenberg ground-state problem.
- **Heisenberg-on-kagome local-energy kernel** (`NQS_HAM_KAGOME_HEISENBERG`).
  Three-sublattice kagome geometry with PBC (default) or OBC: 2×2 PBC
  cluster → N=12, 24 bonds, coord 4. Real + complex-amplitude paths.
  Config: `cfg.j_coupling`, `cfg.kagome_pbc`. Caller passes
  `(size_x, size_y) = (Lx_cells, Ly_cells)` through the existing
  dispatch; `N_sites = 3·Lx·Ly` is computed internally. Target for
  the kagome Heisenberg S=½ ground-state problem (gapped Z₂ vs
  gapless Dirac spin liquid).
- Eleven new analytical checkpoint tests (`tests/test_nqs_kitaev.c`
  gains 4 KH cases, `tests/test_nqs_kagome.c` ships 7 kagome cases)
  plus one end-to-end complex-RBM + holomorphic SR convergence test
  for KH in `tests/test_nqs_holomorphic_sr.c`.

### Changed
- `VERSION_PINS`: `LIBIRREP_MIN` bumped 1.2 → 1.3.0-alpha to match the
  incoming libirrep release carrying kagome geometry, p6mm wallpaper
  group, and config-projection helpers.
- `VERSION_PINS`: removed five stale tolerance entries that drifted
  from their source (per-test tolerances are declared at the top of
  each `tests/test_*.c` and covered by `REPRODUCIBILITY.md`).

### Changed
- `nqs_local_energy` and `nqs_local_energy_complex` (and the `_batch`
  variants) now compute the site count per-Hamiltonian rather than
  assuming `size_x × size_y` — required for lattices with more than
  one site per unit cell. Every existing Hamiltonian kernel yields
  the same `N` as before; only new multi-sublattice kernels (kagome)
  see a different value.
- `nqs_sr_step` / `nqs_sr_step_holomorphic` / `nqs_sr_step_custom` now
  read `N` from `nqs_sampler_num_sites(sampler)` rather than
  recomputing `size_x * size_y`, for the same reason.
- New accessor `nqs_sampler_num_sites(const nqs_sampler_t *s)` exposes
  the sampler's configured site count to consumers.

### Tests
- Full suite at v0.4.1: 343 / 343 passing (was 333 in v0.4), 0
  warnings under `-Wall -Wextra`, 0 regressions. (See v0.4.2 for
  the follow-up count of 359 / 359.)

## [Unreleased] — v0.5 pillar landings

### Added — pillar P2.1 (time-dependent NQS)
- **Real-time tVMC integrator** (`nqs_tvmc_step_real_time`). For real
  parameters θ, the complex tVMC projection equation `S · θ̇ = -i · F`
  projects to `Re(S) · θ̇ = Im(F)`; the Fubini–Study metric is reused
  from the holomorphic SR path. Forward-Euler conserves ⟨H⟩ to O(dt²).
- **Heun (2nd-order) tVMC integrator** (`nqs_tvmc_step_heun`). One
  extra MC sampling per step; drift drops from 0.052 to 0.013 at
  dt = 0.02, T = 0.3 on TFIM N = 4.
- Raw-parameter accessor `nqs_ansatz_params_raw` for multi-stage
  time-steppers that need to snapshot / restore θ.

### Added — pillar P1.2 (equivariant LLG)
- **SO(3)-equivariant torque predictor** (`src/equivariant_gnn/`).
  Pure-C tensor-product primitives; output τ transforms as a proper
  rank-1 tensor under rotations (max residual 1.55e-15 over random
  SO(3) samples). LLG adapter plugs τ into the integrator's
  `field_fn` slot; 200 RK4 steps keep |m|=1 to machine precision.
- **libirrep bridge NequIP layer** (`libirrep_bridge_nequip_*`).
  Opaque wrappers around libirrep's NequIP layer via e3nn-style
  multiset strings; gated behind `SPIN_NN_HAS_IRREP_NEQUIP` so the
  bridge remains buildable against libirrep 1.0 (which predates
  `nequip.h`). Full tower lands once libirrep ≥ 1.1 is vendored.

### Added — pillar P2.9 (thermodynamic computing)
- **Hopfield associative memory** (`src/thermodynamic/hopfield.c`).
  Hebbian storage + zero-T sync updates + finite-T Metropolis sweep.
  Reliable recall at K/N = 0.1 (below Amit–Gutfreund–Sompolinsky 0.138).
- **CD-1 RBM generative model** (`src/thermodynamic/rbm_cd.c`).
  Block-Gibbs sampling, mean-field statistics inside the gradient.
  After 5000 epochs on a 4-bit 2-pattern dataset: sample hit-rate
  0.996 vs chance 0.125.

### Added — pillar P1.1 (NQS Hamiltonian coverage)
- **XXZ Hamiltonian** (`NQS_HAM_XXZ` + `j_z_coupling`). Generic
  local-energy kernel parametrises Jxy (off-diagonal) and Jz
  (diagonal) separately; the existing Heisenberg path delegates with
  Jxy = Jz. Cross-checked against MPS DMRG on N=4 chains for three
  anisotropy regimes.

### Added — pillar P1.2 (equivariant LLG) — extras
- **Closed-form fitter** for the torque-net's five linear weights
  (`torque_net_fit_weights`). Recovers planted synthetic weights to
  machine precision (1.6e-16 residual) on a 3×3 periodic grid over
  40 random configurations.

### Added — pillar P2.7 (PINN groundwork)
- **SIREN activation** for the legacy MLP (`ACTIVATION_SIREN`),
  including the Sitzmann et al. 2020 weight-init scheme (first
  layer U[−1/fan_in, 1/fan_in]; deeper layers scaled by 1/ω).

### Added — pillar P3.0 (THQCP — thermodynamic hybrid quantum-classical processor)
- **THQCP coupling scheduler** (`src/thqcp/coupling.c` + `include/thqcp/coupling.h`).
  Three-phase state machine PHASE_ANNEAL → PHASE_QUANTUM → PHASE_FEEDBACK
  on a p-bit annealing plane + defect-qubit coherent-window plane.
  Open policies: PERIODIC, STAGNATION, NEVER. Theoretical grounding:
  Sanchez-Forero 2024 adiabatic-response stochastic thermodynamics.
- **Coherent qubit window** (`THQCP_WINDOW_COHERENT`). Exact 2-level
  evolution under `H_q = h_z σ_z + h_x σ_x` with Born-rule projective
  measurement; gives proper transverse-field-quantum-annealing
  tunneling probability `P(flip) = (h_x²/Ω²) sin²(Ωτ)`. Stub model
  retained as `THQCP_WINDOW_STUB` for ablation baselines.
- Ferromagnetic Ising N=16 at β=[0.05, 6.0] over 400 sweeps reaches
  exact ground state E = -120 in the anneal-only branch.

### Added — cross-project integration
- **Moonlab bridge** (`src/moonlab_bridge.c`). Gated behind
  `SPIN_NN_HAS_MOONLAB`; forwards to libquantumsim's surface-code
  + MWPM-decoder API. Provides ground-truth QEC reference for the
  joint-trained neural decoder program.
- **libirrep bridge live-path tests** — 6/6 live-mode tests
  pass via `make IRREP_ENABLE=1` against libirrep 1.0.0. Torque net
  ↔ libirrep SH-addition-theorem cross-check passes at residual
  6.1e-17.
- **Golden-vector suite** (`tests/test_downstream_compat/`) —
  five fixed (h_in, edge_vec, weights) configs + expected TP
  outputs for the libirrep 1.2 torque-net convergence target.
  Both repositories vendor the same JSON files; any convention
  drift fires simultaneously on both CI runs. 5/5 bit-exact
  agreement on current tree.
- **Cross-project integration docs** (`docs/cross_project_integration.md`).
  External-collaborator onboarding: stack inventory, dependency DAG,
  bridges provided, cross-validation axes, version compatibility.

### Added — other
- **Flow-matching per-site rates**
  (`flow_matching_fit_rates_to_magnetisation`,
  `flow_matching_sample_biased_rates`). Closed-form inversion of the
  two-state CTMC relation `m(1) = b · (1 − e^{−c})` gives per-site
  rates that hit a prescribed target magnetisation at t=1.

### Tests
- Suite grew from 277 → 316 tests; all pass. Plus 6 live-mode tests
  with `IRREP_ENABLE=1` and 4 with `MOONLAB_ENABLE=1`.

## [0.4.0] - Foundation for v0.5 research pillars

### Added
- **Hilbert-space Majorana braiding** — new `MajoranaHilbertState` over the
  `2^(N/2)`-dim fermion-parity Fock basis plus `apply_braid_unitary()`
  implementing `B_{ij} = (1 + γ_i γ_j)/√2`. Verified `B^4 = -I`,
  `B^8 = I`, anticommutation, unitarity, parity conservation. The v0.3
  operator-space braiding path is retained as
  `braid_majorana_operators_legacy()` and `braid_majorana_modes()` still
  works unchanged.
- **Toric-code data-qubit model** — per-link `x_errors` / `z_errors`
  accumulators, `toric_code_apply_{x,z}_{error,correction}`,
  homology-based `toric_code_has_logical_error()`, greedy
  matching-baseline decoder `toric_code_decode_greedy()` with primal /
  dual path walks. Serves as the MWPM baseline for the learned decoder
  coming in v0.5 pillar P1.3.
- **`engine_adapter` scaffolding** (`src/engine_adapter.c`,
  `include/engine_adapter.h`) — engine-neutral bridge between the spin
  framework and an external NN / tensor / reasoning engine. Planned
  backends: an Eshkol-native NN engine (working title
  `eshkol-transformers`, built on https://github.com/tsotchke/eshkol)
  and Noesis (reasoning engine, in development, not yet publicly
  released). All entry points compile today behind
  `#ifdef SPIN_NN_HAS_ENGINE`; enable with
  `make ENGINE_ENABLE=1 ENGINE_ROOT=...` once a chosen engine is
  available.
- **`eshkol_bridge`** (`src/eshkol_bridge.c`, `include/eshkol_bridge.h`) —
  lazy wrapper over the Eshkol FFI; compiles without Eshkol present,
  wires up once `-DSPIN_NN_HAS_ESHKOL=1` is enabled in v0.5.
- **`nn_backend`** (`src/nn_backend.c`, `include/nn_backend.h`) —
  polymorphic neural-network handle with `NN_BACKEND_LEGACY` and
  `NN_BACKEND_ENGINE` variants, and `--nn-backend={legacy,engine}` CLI flag.
- **Training-loop cadences** — `--cadence-decoder N`, `--decoder-error-rate P`,
  `--cadence-invariants N`, `--lambda-logical L`. Decoder logical-error
  flag folds into `physics_loss` as a soft penalty during training.
- **Test harness** — TAP-style `tests/harness.h` plus 17 suites covering
  18 of 18 library modules (109 tests total). `make test` runs them all.
  See `docs/testing.md`.
- **Benchmark harness** — 4 suites (`bench_ising`, `bench_kitaev`,
  `bench_majorana_braid`, `bench_toric_decoder`) emitting JSON under
  `benchmarks/results/<suite>/`. `scripts/run_benchmarks.sh` orchestrates.
- **`scripts/check_stack.sh`** — advisory probe for optional stack
  components: external engine, libirrep, Eshkol runtime. All missing
  dependencies are informational in v0.4.

### Improved
- Toric-code `perform_error_correction` now delegates to the greedy
  data-qubit decoder, flipping data qubits directly and re-deriving
  syndromes so iterated error / correction cycles stay consistent.
- Majorana braiding is now available as a Hilbert-space unitary (the v0.3
  operator-space path is retained as
  `braid_majorana_operators_legacy`).
- `toric_code_has_logical_error()` uses homology-class winding numbers
  against primal / dual basis cycles, which are invariant under stabilizer
  action.

### Changed
- `CHERN_NUMBER` environment-variable override in `src/berry_phase.c` is now
  gated behind `#ifdef SPIN_NN_TESTING`; release builds do not ship the
  back-door.
- Neural-network creation in `main.c` goes through `spin_nn_create()`;
  legacy behavior is preserved by default.
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
