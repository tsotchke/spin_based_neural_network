# Kagome Heisenberg S=½ — research plan

*Open-science research program tracking the kagome Heisenberg S=½
ground-state question: **gapped Z₂ spin liquid** (topological order,
γ = ln 2) vs **gapless Dirac spin liquid** (algebraic correlations,
γ = 0). The infrastructure, diagnostics, and results land publicly on
this repo and the companion
[`libirrep`](https://github.com/tsotchke/libirrep); coordination runs
through the `agent-notes/inbox-{spin,irrep}` channel on the authors'
working machines.*

## Infrastructure shipped (public, v0.4.1 → v0.4.2)

- `NQS_HAM_KAGOME_HEISENBERG` kernel in `src/nqs/nqs_gradient.c`, real +
  complex amplitude, PBC/OBC. 2×2 PBC → N=12, 24 bonds, coord 4.
- Complex-RBM + holomorphic SR end-to-end tested; 60-iter smoke on N=12
  descends head 5.99 → tail ≈ 0.20. Not yet a research number.
- Latent heap-buffer-overflow found and fixed (nqs_sampler_num_sites
  accessor + SR-optimizer N from sampler, not from size_x*size_y).

## Target numbers for N-schedule

Published ED / DMRG references, E₀/N per site (J = 1):

| N  | Geometry      | E₀ reference                          | Source                      |
|---:|---------------|---------------------------------------|-----------------------------|
| 12 | 2×2 PBC torus | ≈ −0.4365 J                           | Leung-Elser (1993) + others |
| 12 | **our cluster** | **−0.4537 J** (E₀_total = −5.4449, exact Lanczos) | this repo, `nqs_lanczos_refine_kagome_heisenberg` |
| 18 | Lanczos torus | ≈ −0.4383 J                           | Lecheminant et al. (1997)   |
| 24 | Lanczos torus | ≈ −0.4385 J                           | Läuchli-Sudan (2011)        |
| 30 | ED            | ≈ −0.4386 J                           | Läuchli-Sudan (2011)        |
| 36 | DMRG cylinder | ≈ −0.4379 J (depends on cylinder)     | Yan-Huse-White (2011)       |
| ∞  | extrapolated  | ≈ −0.4386 J (tight bound from 48+)    | various                     |

Our cluster convention differs from Lanczos; first sanity check is
to reproduce the N=12 PBC torus value to within 2% (E₀ ≈ −5.238 on
our cluster) using complex RBM + holomorphic SR. If we can't hit
−5 J at N=12 with a small complex RBM, the kernel is either wrong
or the ansatz is too weak; in either case we need to find out
before scaling up.

## N-schedule (committed to in `inbox-irrep/2026-04-23-kagome-pivot-ack.md`)

| N    | Role                                | Expected NQS gap vs ED |
|-----:|-------------------------------------|------------------------|
| 12   | handshake, tight cross-check        | ≤ 2% at converged cRBM |
| 18   | ED-anchored                          | 5–10%                  |
| 24   | tightest ED anchor                   | 2–5% with projection   |
| 30   | ED upper bound (irrep: Lanczos w/ full-reorth port) | open |
| 36   | NQS-only territory                   | 10–20% at best         |
| 48+  | out of scope for v0.4.x line         | —                      |

## 5-diagnostic protocol (joint with libirrep)

From `inbox-spin/2026-04-23-kagome-systematic.md` (irrep's proposal).
Bar: 3-of-5 converge in the same direction.

| # | Diagnostic                 | Z₂ prediction | Dirac prediction | Our role     | Their role |
|--:|----------------------------|---------------|------------------|--------------|------------|
| 1 | Δ_S(N→∞) spin gap          | ~0.13 J       | 0                | NQS excited state + penalty | ED anchor ≤ 24 |
| 2 | KP γ on annular geometry   | +ln 2         | 0                | sample producer | irrep rdm.h batched |
| 3 | K-point spectrum           | gapped at K   | Dirac cone       | NQS per-irrep | little-group irreps |
| 4 | ⟨S·S⟩(r) decay             | exponential   | algebraic        | MC accumulator | — |
| 5 | S_VN area-law subleading   | +γ constant   | +c log L         | sample producer | irrep rdm.h batched |

## Sample-producer contract with libirrep

`nqs_sampler_batch(s, n, out)` returns `int` row-major `n × num_sites`,
which is the shape libirrep's forthcoming `irrep_rdm_batch_partial_trace`
and `_entropy` are sized for. Target throughput: ≥ 10k samples/s at
N_region = 6 (their size gate in M15b). Our current sampler on kagome
N=12 runs ~16M samples/s, well above the bar.

## Open items on our side

1. **Run proper convergence at N=12 PBC** — first 1000-iter run
   (hidden=48, samples=2048, seed 0xC0AEED) lands at E = -5.046 ±
   0.267 (tail mean over last 100 iters), per-site -0.4205. This
   is 3.66 % above the Leung-Elser literature value (-5.238 J) but
   **7.3 % above the true GS on this cluster** (E₀ = -5.44487522 J
   from Lanczos; see open item 6). Below the 2 % target but within
   striking distance; longer training and a wider hidden layer
   should close it. Driver: `scripts/research_kagome_N12_convergence.c`,
   invoked via `make research_kagome_N12`. Reference commit: see
   `git log -- scripts/research_kagome_N12_convergence.c`.
2. **Bipartite sublattice probe** — `nqs_compute_kagome_bond_phase`
   (see `include/nqs/nqs_diagnostics.h`) measures ⟨ψ(s_{ij})/ψ(s)⟩
   per bond class {A-B, A-C, B-C} from a freshly sampled batch.
   Complex-valued: magnitude captures Marshall-style coherence, arg
   reveals Dirac- vs Z₂-compatible phase structure. Test:
   `tests/test_nqs_chi_F.c::test_kagome_bond_phase_basic`.
3. **Point-group projection (C_3v at K, C_6v at Γ, C_2v at M)** —
   wire `NQS_SYM_POINT_GROUP` to libirrep's `irrep_pg_project` when
   v1.3.0-alpha.1 is vendored. Runs per-irrep-sector.
4. **χ_F from samples** — `nqs_compute_chi_F` in
   `src/nqs/nqs_diagnostics.c` returns Tr(S) = Σ_k (⟨|O_k|²⟩ −
   |⟨O_k⟩|²) using the same complex-gradient path as holomorphic
   SR. Real and complex ansätze both supported. Test:
   `tests/test_nqs_chi_F.c`. Feeds into irrep's diagnostic
   pipeline via the sample-producer contract.
5. **Excited-state SR** — `nqs_sr_{step,run}_excited` in
   `src/nqs/nqs_optimizer.c` implements the Choo-Neupert-Carleo
   orthogonal-ansatz penalty (arXiv:1810.10196 §II). Augments the
   holomorphic-SR local energy by `μ · r(s) · conj(⟨r⟩)` where
   `r(s) = ψ_ref(s)/ψ(s)`. Validated on 2-site Heisenberg (exact
   E₀ = -0.75, E₁ = +0.25): reference cRBM converges to
   E_ref = -0.7528, excited run with μ = 5 reaches E = +0.2498
   (4-decimal agreement with the exact triplet). Test:
   `tests/test_nqs_excited.c`. This unblocks diagnostic (1) — the
   spin-gap Δ_S(N→∞) estimator now has a principled path on any
   Hamiltonian supported by the existing local-energy kernels.
6. **Exact reference via Lanczos** —
   `nqs_lanczos_k_lowest_kagome_heisenberg` builds the full 2^N-dim
   Heisenberg matvec (dim=4096 at N=12) and returns the k smallest
   Ritz values to machine precision. On our 2×2 PBC cluster:
     - E₀ = −5.44487522 J
     - E₁ = −5.32839240 J
     - E₂ = −5.29823654 J
     - **spin gap Δ = E₁ − E₀ = 0.116483 J**
   Feasible for N ≤ 24 (dim 2^24 = 16M, about 128 MB of working
   memory at one double per basis state — fits comfortably on a
   consumer Mac). The N-schedule diagnostic (Δ_S(N→∞) extrapolation)
   now has exact anchors at N = 12, 18, 24 for cross-checking any
   VMC estimate.

## Publication target

Still open. If the 3-of-5 bar resolves cleanly, a Letter-length piece
is the natural form. If it doesn't, a longer methods paper describing
the diagnostic stack itself (independent of which phase wins).

## Revision history

- 2026-04-23: initial plan; capability kernel landed; coordination
  protocol set with libirrep 1.3.0-alpha.1 cycle.
