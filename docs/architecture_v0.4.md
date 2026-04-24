# Architecture — v0.4 Foundation and v0.5+ Research Roadmap

This document captures the v0.4 architecture and the research pillars
scheduled for v0.5 and beyond. It complements the per-feature docs:

- Physics modules: `berry_phase.md`, `majorana_zero_modes.md`,
  `toric_code.md`, `topological_entropy.md`, `topological_features.md`.
- Classical substrate: `classical_models.md` (Ising, Kitaev,
  disordered, continuous SpinLattice, energy scaling).
- Training loop and NN handle: `training.md` (polymorphic
  `nn_backend`, cadence flags, physics losses).
- Engine integration (v0.5+): `engine_integration.md`
  (`engine_adapter` and `eshkol_bridge`).
- Infrastructure: `testing.md` and `benchmarks.md`.
- UI: `visualization.md`.

## 1. Release cadence

| Release | Scope |
|---|---|
| **v0.3** | Topological quantum computing primitives (Berry / Chern / winding / TKNN, Majorana modes, toric code, topological entanglement entropy, visualization). |
| **v0.4 (foundation release)** | Foundation — physics improvements, engine-neutral scaffolding, assert-based test suite over every library module, benchmark harness. |
| **v0.4.1** | Capability addition: Kitaev-Heisenberg kernel on brick-wall honeycomb; Heisenberg-on-kagome kernel (three-sublattice, target for the open S=½ gapped-Z₂-vs-gapless-Dirac question); `LIBIRREP_MIN` bumped to 1.3.0-alpha. |
| **v0.4.2 (current)** | kagome diagnostics stack + exact-reference solver: χ_F = Tr(S)/2 (`nqs_compute_chi_F`), per-bond-class phase probe (`nqs_compute_kagome_bond_phase`), excited-state VMC via Choo–Neupert–Carleo penalty (`nqs_sr_{step,run}_excited`), full-basis Lanczos refinement + multi-Ritz (`nqs_lanczos_{refine,k_lowest}_kagome_heisenberg`). End-to-end research driver (`make research_kagome_N12_diagnostics`). Exact E₀ / E₁ / spin gap Δ anchors on N ≤ 24. 359 / 359 tests passing. |
| **v0.5** | Tier-1 research pillars: NQS, equivariant LLG, learned QEC decoders, neural-operator Boltzmann samplers, Fibonacci-anyon gates. |
| **v0.6** | Tier-2 pillars: time-dependent NQS, MPS warm-start, foundation NQS, KAN-NQS, p-bit neuromorphic, gauge-invariant sampling, thermodynamic computing. |
| **v0.7** | Observability — full benchmark suite, visualization modes for every pillar, reproducible experiment manifests, dashboard. |
| **v1.0** | API freeze, paper-quality reproduction recipes, declared API stability. |

Each release binds to specific versions of the external toolchain (to be
locked in `VERSION_PINS` when v0.5 lands).

## 2. v0.4 module map

```
spin_based_neural_network/
├── src/
│   ├── ising_model.c, kitaev_model.c, disordered_model.c, spin_models.c
│   │                                    ↑ classical spin models
│   ├── majorana_modes.c                 — Hilbert-space braiding (new, v0.4)
│   ├── toric_code.c                     — data-qubit model + greedy MWPM (new, v0.4)
│   ├── berry_phase.c                    — Chern/TKNN/winding + test-gated override
│   ├── topological_entropy.c            — von Neumann entropy, Kitaev-Preskill
│   ├── ising_chain_qubits.c             — topological qubits from MZMs
│   ├── physics_loss.c                   — 5 PDE residual losses
│   ├── neural_network.c                 — legacy MLP + Adam + batch norm
│   ├── nn_backend.c                     — polymorphic NN handle (new, v0.4)
│   ├── engine_adapter.c                 — engine-neutral bridge (new, v0.4, dormant)
│   ├── eshkol_bridge.c                  — Eshkol FFI wrapper (new, v0.4, dormant)
│   ├── reinforcement_learning.c         — reactive RL heuristic
│   ├── energy_utils.c                   — sigmoid energy scaling
│   ├── quantum_mechanics.c              — noise injection + Bell-state entanglement
│   ├── matrix_neon.c                    — ARM NEON SIMD kernels
│   ├── visualization*.c                 — SDL2 viewer (4 modes)
│   ├── main.c                           — driver
│   ├── topological_example.c            — standalone demo
│   └── <pillar>/                        — v0.5+ pillar scaffolds (nqs/, mps/,
│       equivariant_gnn/, llg/, neural_operator/, flow_matching/, qec_decoder/,
│       fibonacci_anyons/, neuromorphic/, thermodynamic/, thqcp/) — each a
│       self-contained module tree with its own header-and-source pair and
│       matching tests/test_<pillar>*.c files. The `nqs/` tree is
│       diagnostics-complete at v0.4.2 (`nqs_diagnostics.{h,c}`,
│       `nqs_lanczos.{h,c}` for χ_F, kagome bond phase, and exact
│       Lanczos reference). Other pillars are interface-stable in
│       v0.4; full implementations land incrementally in v0.5 and
│       beyond.
├── tests/    — one suite per library module + per-pillar suites (see `docs/testing.md`)
├── benchmarks/   — `bench_*.c` suites, emit JSON under results/
├── scripts/  — run_benchmarks.sh, check_stack.sh
└── docs/     — this file + per-feature deep dives
```

## 3. External toolchain (v0.5+ plug-in points)

The `engine_adapter` module is engine-neutral: any library that exposes
`engine_backend_init / engine_backend_shutdown / engine_backend_version`
as weak hooks can become the NN / tensor / reasoning backend. Planned
targets:

- **Eshkol-native NN engine** (working title `eshkol-transformers`) —
  a transformer / KAN / MoE engine written *in* Eshkol
  (https://github.com/tsotchke/eshkol). Scheduled for v0.6+. Provides
  multi-dtype tensors (FP16 / BF16 / INT8 / INT4), Flash Attention,
  SafeTensors loading, and Riemannian optimizers on hyperbolic /
  spherical product manifolds. See §5 and the `eshkol/README.md` for
  the Scheme-script side.
- **Noesis reasoning engine** — in-development symbolic / program-
  synthesis engine, not yet publicly released. Will plug into
  `engine_adapter` as an alternative backend for pillars that combine
  numerical training with symbolic reasoning.

Python is **not** a runtime dependency of this project.

## 4. v0.5 — Tier-1 research pillars

### P1.1 Neural Network Quantum States

**Goal.** Ground-state variational solver that treats the wavefunction
amplitude ψ(s) as a transformer (or KAN) evaluated on a spin
configuration s. Benchmarks: frustrated magnets — J1-J2 square lattice
at J2/J1 = 0.5, kagome Heisenberg, Shastry-Sutherland, Kitaev honeycomb
at the isotropic point.

**Architecture.** `src/nqs/` with:

- `nqs_ansatz.c` — MLP / RBM / complex-RBM ansätze today; transformer /
  factored-ViT / autoregressive land once the external NN engine is
  vendored. Emits `(log|ψ|, arg ψ)` for a spin configuration.
- `nqs_sampler.c` — Metropolis sampler; exact autoregressive sampler
  lands with the autoregressive ansatz.
- `nqs_gradient.c` — local-energy estimators per Hamiltonian:
  TFIM, Heisenberg, XXZ, J1-J2, Kitaev-Heisenberg (v0.4.1),
  kagome Heisenberg (v0.4.1). Real + complex-amplitude paths.
- `nqs_optimizer.c` — real-projected SR + holomorphic SR [16] with
  QGT preconditioning via conjugate gradient. Also real-time tVMC
  (forward-Euler + Heun) and excited-state SR via the
  Choo–Neupert–Carleo 2018 [18c] orthogonal-ansatz penalty (v0.4.2).
- `nqs_diagnostics.c` (v0.4.2) — sample-based χ_F = Tr(S)/2
  (`nqs_compute_chi_F`) and kagome per-bond-class phase probe
  (`nqs_compute_kagome_bond_phase`). Conventions from
  Provost–Vallée 1980 [18a] and Zanardi–Paunković 2006 [18b].
- `nqs_lanczos.c` (v0.4.2) — full-basis Lanczos refinement with
  the trained state as Krylov seed, on TFIM, Heisenberg, and
  kagome Heisenberg kernels. Multi-Ritz k-smallest extraction for
  spin gaps (`lanczos_k_smallest_with_init` in `mps/lanczos.c` +
  `nqs_lanczos_k_lowest_kagome_heisenberg`). Lanczos eigensolver
  per [18d]. Machine-precision anchor at N ≤ 24.
- `scripts/research_kagome_N12_diagnostics.c` — end-to-end driver
  chaining SR + χ_F + bond-phase + excited-SR + Lanczos on one
  N=12 PBC kagome cluster. Build target: `make
  research_kagome_N12_diagnostics`.
- `eshkol/train_nqs.esk` — Scheme-side tape driver (lands when
  the Eshkol bridge activates).

**References.** Carleo & Troyer [15] — original NQS. Rende et al.
[17] — transformer optimisation for large NQS via a linear-algebra
identity. Sorella [16] — stochastic reconfiguration. Chen & Heyl
[18] — modern large-scale NQS optimisation. Provost & Vallée [18a],
Zanardi & Paunković [18b] — QGT and fidelity susceptibility
conventions. Choo, Neupert & Carleo [18c] — excited-state VMC
penalty. Lanczos [18d] — Krylov eigensolver.

### P1.2 Equivariant GNN torques + real Landau-Lifshitz-Gilbert dynamics

**Goal.** Replace energy-only Monte Carlo with actual time-evolved
magnetisation dynamics ṁ = -γ m × B_eff - αγ m × (m × B_eff), where
B_eff comes from an E(3)-equivariant torque network. Unlocks skyrmions,
domain walls, spin waves.

**Architecture.** `src/equivariant_gnn/` + `src/llg/`:

- Irrep tensor products via the forthcoming `libirrep` library
  (Clebsch-Gordan, spherical harmonics, Wigner-D up to l=8).
- NequIP-style [19] message passing: `h_ij = TP(h_i, Y_l^m(r̂_ij)) * φ(||r_ij||)`.
- MACE-style [20] many-body messages for body-order ≥ 2.
- 4th-order Runge-Kutta or Heun integrator for the LLG equation;
  stochastic-LLG noise for finite temperature.

**Benchmarks.** Skyrmion nucleation under applied current; magnetization
relaxation in a thin film; comparison against established micromagnetic
solvers such as the NIST µMAG Standard Problems.

### P1.3 Learned surface-code decoders + Fibonacci anyons

**Goal.** Replace the v0.4 greedy decoder with transformer- and
Mamba-based learned decoders that beat its threshold. Parallel track:
Fibonacci anyons for universal topological gates.

**Architecture.** `src/qec_decoder/`:

- Syndrome-history tokenisation (per-qubit or per-stabilizer tokens).
- Transformer decoder [13] — conditioned on syndrome history, emits a
  correction Pauli string.
- Mamba decoder [14] — selective SSM scan variant for lower asymptotic
  complexity.
- `src/fibonacci_anyons/` — F-matrix, R-matrix, fusion trees,
  Solovay–Kitaev compiler [21] from arbitrary single-qubit unitaries to
  braid words, benchmarked against experimental demonstrations
  [22, 23].

**References.** Bausch et al. (AlphaQubit) [13], Ninkovic et al. [14],
Solovay–Kitaev [21], Fibonacci anyon experiments [22, 23].

### P1.4 Neural-operator LLG + discrete flow-matching Boltzmann sampler

**Goal.** 10-100× wall-clock speedup on LLG trajectories via learned
Fourier / Tucker neural operators. Parallel: discrete flow-matching as
a drop-in replacement for Metropolis at phase transitions.

**Architecture.**

- `src/neural_operator/` — Fourier Neural Operator spectral
  convolutions [24], following the U-Net-style demag-surrogate pattern
  validated by NeuralMAG [25]. Tucker decomposition of the 3D
  spectral-weight tensor to keep memory tractable on a single GPU.
- `src/flow_matching/` — continuous-time flow matching [26] adapted
  to discrete spin states; Langevin-in-logits sampling head as a
  drop-in replacement for Metropolis at phase transitions.

## 5. v0.5+ external dependencies

| Dependency | Purpose | Status |
|---|---|---|
| `libirrep` (new public library, MIT) | Irrep tensor products, Y_l^m, Wigner-D | Being built in parallel. Required for P1.2. |
| `eshkol` (Scheme-on-LLVM; https://github.com/tsotchke/eshkol) | Autodiff tape orchestration + compiled training scripts | Required for P1.1+. |
| `eshkol-transformers` (working title) | Eshkol-native NN engine (transformer / KAN / MoE) | v0.6+ project. Plugs into the `engine_adapter` once available. |
| Noesis | Reasoning engine (symbolic / program-synthesis) | In development, not yet publicly released. Optional alternative `engine_adapter` backend. |

Cross-platform GPU (Metal + CUDA + ROCm/HIP + Vulkan) is a commitment
for v0.5 via the engine layer; CPU + NEON/AVX2 ship today.

## 6. Testing and benchmarks as the release contract

A v0.5 pillar is considered landed when:

1. A new `tests/test_<pillar>.c` suite exists with at least one test per
   public API function (see the existing `tests/test_*.c` suites as the
   pattern).
2. A new `benchmarks/bench_<pillar>.c` suite emits JSON under
   `benchmarks/results/<pillar>/`.
3. Reproducibility: a manifest records the git commits of this repo +
   `eshkol` + `libirrep` + any external engines, plus CLI flags and
   hardware info.
4. `make clean && make arm && make test && make bench` all succeed on
   a supported host.

## 7. References

All references below have been verified to exist (venue, authors, title,
identifier).

### Topological quantum computing
1. A. Y. Kitaev, "Fault-tolerant quantum computation by anyons," *Annals of Physics*, vol. 303, pp. 2-30, 2003.
2. A. Y. Kitaev, "Unpaired Majorana fermions in quantum wires," *Physics-Uspekhi*, vol. 44, pp. 131-136, 2001.
3. A. Y. Kitaev, "Anyons in an exactly solved model and beyond," *Annals of Physics*, vol. 321, pp. 2-111, 2006.
4. A. Y. Kitaev and J. Preskill, "Topological Entanglement Entropy," *Physical Review Letters*, vol. 96, p. 110404, 2006.
5. E. Dennis, A. Kitaev, A. Landahl, and J. Preskill, "Topological quantum memory," *Journal of Mathematical Physics*, vol. 43, pp. 4452-4505, 2002.
6. A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. N. Cleland, "Surface codes: Towards practical large-scale quantum computation," *Physical Review A*, vol. 86, p. 032324, 2012.
7. D. J. Thouless, M. Kohmoto, M. P. Nightingale, and M. den Nijs, "Quantized Hall Conductance in a Two-Dimensional Periodic Potential," *Physical Review Letters*, vol. 49, pp. 405-408, 1982.
8. M. V. Berry, "Quantal phase factors accompanying adiabatic changes," *Proc. R. Soc. Lond. A*, vol. 392, pp. 45-57, 1984.
9. C. Nayak, S. H. Simon, A. Stern, M. Freedman, and S. Das Sarma, "Non-Abelian anyons and topological quantum computation," *Reviews of Modern Physics*, vol. 80, pp. 1083-1159, 2008.
10. A. Stern, "Non-Abelian states of matter," *Nature*, vol. 464, pp. 187-193, 2010.
11. P. Jordan and E. Wigner, "Über das Paulische Äquivalenzverbot," *Zeitschrift für Physik*, vol. 47, pp. 631-651, 1928.

### Graph-theoretic algorithms
12. J. Edmonds, "Paths, trees, and flowers," *Canadian Journal of Mathematics*, vol. 17, pp. 449-467, 1965.

### Learned QEC decoders (pillar P1.3)
13. J. Bausch, A. Senior, F. Heras, T. Edlich, A. Davies, M. Newman, C. Jones, K. Satzinger, M. Y. Niu, S. Blackwell, G. Holland, D. Kafri, J. Atalaya, C. Gidney, D. Hassabis, S. Boixo, H. Neven, and P. Kohli, "Learning high-accuracy error decoding for quantum processors," *Nature*, vol. 635, pp. 834–840, 2024. DOI: 10.1038/s41586-024-08148-8.
14. V. Ninkovic, O. Kundacina, D. Vukobratovic, and C. Häger, "Scalable Neural Decoders for Practical Real-Time Quantum Error Correction," arXiv:2510.22724, 2025.

### Neural Network Quantum States (pillar P1.1, diagnostics pipeline shipped v0.4.2)
15. G. Carleo and M. Troyer, "Solving the quantum many-body problem with artificial neural networks," *Science*, vol. 355, pp. 602-606, 2017.
16. S. Sorella, "Green Function Monte Carlo with Stochastic Reconfiguration," *Physical Review Letters*, vol. 80, pp. 4558-4561, 1998.
17. R. Rende, L. Viteritti, L. Bardone, F. Becca, and S. Goldt, "A simple linear algebra identity to optimize large-scale neural network quantum states," *Communications Physics*, 2024. arXiv:2310.05715.
18. A. Chen and M. Heyl, "Empowering deep neural quantum states through efficient optimization," *Nature Physics*, vol. 20, pp. 1476-1481, 2024.

### Quantum geometric tensor, fidelity susceptibility, excited-state VMC, Lanczos (v0.4.2 diagnostics stack)
18a. J. P. Provost and G. Vallée, "Riemannian structure on manifolds of quantum states," *Communications in Mathematical Physics*, vol. 76, pp. 289–301, 1980.  *(QGT / Fubini–Study metric underlying `nqs_compute_chi_F`.)*
18b. P. Zanardi and N. Paunković, "Ground state overlap and quantum phase transitions," *Physical Review E*, vol. 74, p. 031123, 2006.  *(χ_F = Tr(S)/2 convention.)*
18c. K. Choo, T. Neupert, and G. Carleo, "Two-dimensional frustrated J1-J2 model studied with neural network quantum states," *Physical Review B*, vol. 100, p. 125124, 2019. arXiv:1810.10196.  *(Orthogonal-ansatz penalty in `nqs_sr_step_excited`.)*
18d. C. Lanczos, "An iteration method for the solution of the eigenvalue problem of linear differential and integral operators," *Journal of Research of the National Bureau of Standards*, vol. 45, pp. 255–282, 1950.  *(Krylov eigensolver underpinning `lanczos_{smallest,k_smallest}_with_init`.)*

### Equivariant GNNs (pillar P1.2)
19. S. Batzner, A. Musaelian, L. Sun, M. Geiger, J. P. Mailoa, M. Kornbluth, N. Molinari, T. E. Smidt, and B. Kozinsky, "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials," *Nature Communications*, vol. 13, p. 2453, 2022.
20. I. Batatia, D. P. Kovacs, G. N. C. Simm, C. Ortner, and G. Csanyi, "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields," *NeurIPS 2022*. arXiv:2206.07697.

### Fibonacci anyons (pillar P1.3b)
21. A. Y. Kitaev, A. H. Shen, and M. N. Vyalyi, *Classical and Quantum Computation*, AMS Graduate Studies in Mathematics, vol. 47, 2002 (Solovay–Kitaev algorithm).
22. S. Xu *et al.*, "Non-Abelian braiding of Fibonacci anyons with a superconducting processor," *Nature Physics*, vol. 20, pp. 1469-1475, 2024. arXiv:2404.00091.
23. Z. K. Minev, K. Najafi, S. Majumder, J. Wang, A. Stern, E.-A. Kim, C.-M. Jian, and G. Zhu, "Realizing string-net condensation: Fibonacci anyon braiding for universal gates and sampling chromatic polynomials," *Nature Communications*, vol. 16, article 6225, 2025. arXiv:2406.12820.

### Neural operators + flow matching (pillar P1.4)
24. Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. M. Stuart, and A. Anandkumar, "Fourier Neural Operator for Parametric Partial Differential Equations," *ICLR 2021*. arXiv:2010.08895.
25. Y. Cai, J. Li, and D. Wang, "NeuralMAG: Fast and Generalizable Micromagnetic Simulation with Deep Neural Nets," arXiv:2410.14986, 2024.
26. Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, "Flow Matching for Generative Modeling," *ICLR 2023*. arXiv:2210.02747.
