# Neural Network Quantum States (NQS) — v0.5 Pillar P1.1

This document covers the NQS scaffold introduced in v0.4 as the base
for pillar P1.1 in v0.5. An NQS represents the quantum wavefunction
`ψ(s)` on spin configurations `s` as a parametric network evaluated on
patch-tokenised lattice input, and minimises the variational energy
`⟨E⟩ = ⟨ψ|Ĥ|ψ⟩ / ⟨ψ|ψ⟩` via stochastic reconfiguration on Monte Carlo
samples drawn from `|ψ(s)|²`.

## 1. Module layout

```
include/nqs/
├── nqs_config.h       # configuration struct + enums
├── nqs_sampler.h      # Metropolis-Hastings sampler
├── nqs_gradient.h     # local-energy estimator + accumulator
├── nqs_ansatz.h       # wavefunction ansatz contract
└── nqs_optimizer.h    # stochastic reconfiguration + CG solve

src/nqs/
├── nqs_sampler.c      # single-flip MH with xorshift64 RNG
├── nqs_gradient.c     # TFIM, Heisenberg, J1-J2 local energies
├── nqs_ansatz.c       # mean-field, real RBM, complex RBM (ViT slot reserved)
└── nqs_optimizer.c    # SR step with matrix-free CG

tests/test_nqs.c       # 14 tests
benchmarks/bench_nqs.c # sampler + SR step throughput
```

## 2. Configuration

`nqs_config_t` in `include/nqs/nqs_config.h` carries every knob the
pipeline consumes:

```c
nqs_config_t cfg = nqs_config_defaults();
cfg.hamiltonian      = NQS_HAM_TFIM;
cfg.transverse_field = 1.0;
cfg.j_coupling       = 1.0;
cfg.num_samples      = 1024;
cfg.num_thermalize   = 256;
cfg.num_iterations   = 200;
cfg.learning_rate    = 1e-2;
cfg.sr_diag_shift    = 1e-3;
```

Fields worth highlighting:

| Field | Purpose |
|---|---|
| `ansatz` | **Shipped in v0.4:** `NQS_ANSATZ_LEGACY_MLP` (mean-field, N params), `NQS_ANSATZ_RBM` (real-amplitude restricted Boltzmann machine, Carleo–Troyer 2017), `NQS_ANSATZ_COMPLEX_RBM` (complex-amplitude RBM for non-stoquastic Hamiltonians). **Not in v0.4 — v0.5 slot only:** `NQS_ANSATZ_VIT`, `NQS_ANSATZ_FACTORED_VIT`, `NQS_ANSATZ_AUTOREGRESSIVE`, `NQS_ANSATZ_KAN`. Requesting one of the v0.5 slots in a v0.4 build makes `nqs_ansatz_create` return `NULL`. |
| `symmetries` | Bitmask (`NQS_SYM_TRANSLATION`, `NQS_SYM_SPIN_FLIP`, `NQS_SYM_U1`, `NQS_SYM_POINT_GROUP`, `NQS_SYM_SU2`). Applied as wavefunction projections in v0.5. |
| `hamiltonian` | `NQS_HAM_TFIM`, `NQS_HAM_HEISENBERG`, `NQS_HAM_J1_J2`, `NQS_HAM_KITAEV_HONEYCOMB`. Selects the local-energy kernel. |
| `num_samples`, `num_thermalize`, `num_decorrelate` | Metropolis batch parameters. |
| `sr_diag_shift` | Tikhonov ε on the quantum geometric tensor. |
| `sr_cg_max_iters`, `sr_cg_tol` | Conjugate-gradient knobs for the QGT solve. |

## 3. Sampler

`nqs_sampler_t` runs a Metropolis-Hastings chain over `{+1, -1}^N`
configurations drawn from `|ψ(s)|²`:

```c
nqs_sampler_t *s = nqs_sampler_create(N, &cfg,
                                      nqs_ansatz_log_amp, ansatz);
nqs_sampler_thermalize(s);            /* cfg.num_thermalize steps */
const int *sample = nqs_sampler_next(s);         /* one sample    */
int batch[N * K];
nqs_sampler_batch(s, K, batch);                  /* bulk sample   */
double acc = nqs_sampler_acceptance_ratio(s);    /* diagnostic    */
nqs_sampler_free(s);
```

- Proposals: single-site spin flips.
- Acceptance: `log(u) < 2 · (log|ψ'| − log|ψ|)` (standard Metropolis).
- RNG: xorshift64 seeded from `cfg.rng_seed`; independent of the global
  `rand()` stream, so multiple samplers can run concurrently with
  distinct seeds.
- Cluster / Swendsen-Wang moves: reserved via `cfg.cluster_moves`;
  enabled in v0.5 for the frustrated-lattice benchmarks.

## 4. Local-energy estimator

`nqs_local_energy(cfg, size_x, size_y, spins, log_amp, user)` computes
`E_loc(s) = ⟨s|Ĥ|ψ⟩/⟨s|ψ⟩` by enumerating the Hamiltonian's
off-diagonal connections and evaluating `ψ(s')/ψ(s)` through the
supplied `log_amp` callback.

v0.4 ships closed-form kernels for:

| Hamiltonian | Diagonal term | Off-diagonal term |
|---|---|---|
| TFIM | `-J Σ_⟨ij⟩ s_i s_j` | `-Γ Σ_i ψ(s⊕e_i)/ψ(s)` |
| Heisenberg | `J/4 Σ_⟨ij⟩ s_i s_j` | `J/2 Σ_⟨ij⟩ [s_i = -s_j] · ψ(s ⊕ {i,j})/ψ(s)` |
| J1-J2 | same + `J_2/4 Σ_⟨⟨ij⟩⟩ s_i s_j` | same + J2 next-nearest offdiag |
| Kitaev honeycomb | reserved — falls back to TFIM in v0.4 | — |

Bulk variant:

```c
double energies[batch_size];
nqs_local_energy_batch(&cfg, L, L, batch, batch_size,
                       nqs_ansatz_log_amp, ansatz, energies);
```

Plus an accumulator (`nqs_energy_accumulator_t`) for streaming mean
and variance, used inside the SR step.

## 5. Ansätze

Three concrete ansatz kinds ship in v0.4, all plugged into the same
`nqs_ansatz_*` contract:

**`NQS_ANSATZ_LEGACY_MLP` — mean-field (default).**
```
log ψ(s) = Σ_i θ_i · s_i
∂ log ψ / ∂θ_i = s_i
```
One parameter per site, real amplitude, zero phase. The simplest ansatz
with a non-trivial parameter-space Jacobian; used as a sanity baseline.

**`NQS_ANSATZ_RBM` — real restricted Boltzmann machine (Carleo–Troyer 2017).**
```
log ψ(s) = Σ_i a_i s_i + Σ_h log(2 cosh(b_h + Σ_i W_hi s_i))
```
`N + M + M·N` real parameters (`M = cfg.rbm_hidden_units`, default `2N`).
Strictly positive amplitudes; handles TFIM directly and bipartite
Heisenberg via the Marshall sign rule.

**`NQS_ANSATZ_COMPLEX_RBM` — complex RBM for non-stoquastic Hamiltonians.**
Same functional form but `a_i, b_h, W_hi ∈ ℂ`; storage is `2·(N + M + M·N)`
doubles (real parts first, then imaginary parts). `log ψ` is complex,
so both `|ψ(s)|` and `arg ψ(s)` depend on the configuration. Required for
frustrated (J1–J2) and Kitaev-type Hamiltonians where the ground state
carries non-trivial phase structure.

All three expose the same public API:

```c
nqs_ansatz_t *nqs_ansatz_create(const nqs_config_t *cfg, int num_sites);
void          nqs_ansatz_free(nqs_ansatz_t *a);
long          nqs_ansatz_num_params(const nqs_ansatz_t *a);
void          nqs_ansatz_log_amp(const int *spins, int num_sites,
                                 void *user,
                                 double *out_log_abs, double *out_arg);
int           nqs_ansatz_logpsi_gradient(nqs_ansatz_t *a,
                                         const int *spins, int num_sites,
                                         double *out_grad);
int           nqs_ansatz_apply_update(nqs_ansatz_t *a,
                                      const double *delta, double step);
```

The v0.5 slots (`NQS_ANSATZ_VIT`, `NQS_ANSATZ_FACTORED_VIT`,
`NQS_ANSATZ_AUTOREGRESSIVE`, `NQS_ANSATZ_KAN`) are reserved but not
implemented in v0.4 — `nqs_ansatz_create` returns `NULL` for them.
They wake up when the external NN engine (transformer / KAN backend)
is wired in alongside pillar P1.1.

## 6. Stochastic reconfiguration

One SR step is:

1. Draw a batch `{s_1, …, s_N}` from `|ψ|²` via `nqs_sampler_batch`.
2. Compute per-sample local energies `E_loc(s_i)` via
   `nqs_local_energy_batch`.
3. Compute per-sample log-psi gradients `O_k(s_i)` via
   `nqs_ansatz_logpsi_gradient`.
4. Form the **force** `F_k = ⟨O_k* E_loc⟩ − ⟨O_k*⟩ ⟨E_loc⟩`.
5. Solve `(S + ε I) δθ = F` via preconditioned CG, where
   `S_kl = ⟨O_k* O_l⟩ − ⟨O_k*⟩ ⟨O_l⟩` is the quantum geometric tensor.
6. Apply the update `θ ← θ − η · δθ`.

The CG solver uses a **matrix-free** `S v` product: given the
`N_s × N_p` matrix `O` of per-sample log-psi gradients, `S v` reduces
to `(1/N_s) O^T (O v) − ⟨O⟩ · (⟨O⟩·v) + ε v`. Cost per CG iteration is
`O(N_s · N_p)` — linear in both batch size and parameter count.

```c
nqs_sr_step_info_t info;
nqs_sr_step(&cfg, Lx, Ly, ansatz, sampler, &info);
/* info.mean_energy, info.variance_energy, info.update_norm,
 * info.acceptance_ratio, info.cg_iterations, info.converged */

/* Or run a full schedule and collect the energy trace: */
double trace[cfg.num_iterations];
nqs_sr_run(&cfg, Lx, Ly, ansatz, sampler, trace);
```

## 7. Tests

`tests/test_nqs.c` (14 tests):

- Ansatz lifecycle + parameter access + gradient vector equals the
  spin vector.
- Sampler thermalisation, batch API, acceptance ratio on a uniform
  reference ansatz (should be 1.0).
- TFIM local energy on all-up lattice equals `-J · bond_count`.
- Heisenberg local energy on Néel antiferromagnet equals `+3`
  (3x3 open-BC lattice, J=1).
- J1-J2 reduces to Heisenberg at `J_2 = 0`.
- Energy accumulator mean and population variance.
- SR step runs to completion and produces a finite mean energy.
- SR run populates an energy trace without numerical failures.

Run `make test_nqs && ./build/test_nqs`.

## 8. Benchmarks

`benchmarks/bench_nqs.c` (5 records emitted per run):

- Metropolis sampler throughput at L ∈ {4, 6, 8}.
- SR step throughput (256 samples/step, 20 steps) at L ∈ {4, 6}.

Reference numbers on an Apple-Silicon M-series Mac:

| Metric | L = 4 | L = 6 | L = 8 |
|---|---|---|---|
| Sampler (samples/s) | ≈ 1.7 × 10⁷ | — (try 2 × 10⁶) | — (try 5 × 10⁵) |
| SR step/s | ≈ 4 × 10³ | — (fewer by batch²) | — |

Numbers scale roughly as O(N²) per SR step for the mean-field ansatz
(sampling + local-energy enumeration dominated). Real-world pillar
P1.1 work should report wall-clock convergence on published
benchmarks (e.g. J1-J2 at J2/J1 = 0.5) rather than raw throughput.

## 9. v0.5 roadmap

The v0.4 scaffold is **complete at the interface level**: every API
called by samplers, local-energy estimators, and the SR step is
stable, tested, and benchmarked. The remaining work for pillar P1.1 is
to supply richer concrete ansatz implementations via the external NN
engine:

- Vision-Transformer wavefunction (NQS_ANSATZ_VIT).
- Factored / translationally-symmetric ViT (NQS_ANSATZ_FACTORED_VIT).
- Gauge-invariant autoregressive sampler (NQS_ANSATZ_AUTOREGRESSIVE).
- KAN wavefunction (NQS_ANSATZ_KAN) — pillar P2.5.

See `architecture_v0.4.md` §P1.1 for the full pillar plan.

## 10. References

- G. Carleo and M. Troyer, "Solving the quantum many-body problem with
  artificial neural networks," *Science* 355 (2017).
- S. Sorella, "Green Function Monte Carlo with Stochastic
  Reconfiguration," *Physical Review Letters* 80 (1998).
- R. Rende, L. Viteritti, L. Bardone, F. Becca, S. Goldt, "A simple
  linear algebra identity to optimize large-scale neural network
  quantum states," *Communications Physics* (2024).
- A. Chen and M. Heyl, "Empowering deep neural quantum states through
  efficient optimization," *Nature Physics* 20:1476-1481 (2024).
