# libirrep 1.2 clarifications — response cycle

Follow-up on the five technical points raised by the libirrep maintainer
after the first-pass 1.2 coordination doc landed. Answers below are
committed here as the formal record; the main coordination doc
(`libirrep_1_2_coordination.md`) links to this one for the details.

## 1. `_apply_forces` recomputes forward state — defer the split

**Maintainer's concern:** `_apply_forces` re-runs SH + RBF + TP forward
per edge internally. In a `apply → backward → forces` training loop,
that's 3× SH evaluations per edge per iteration. Currently 2.3× apply
cost at small shapes.

**Response:** accept current design. At our current benchmark scales
(edge counts ≤ 1024) the 3× SH cost is dominated by the SR
conjugate-gradient preconditioner on top. The `_apply_with_state` /
`_forces_from_state` split is deferred until edge counts reach 16k+
on real LLG device simulations, at which point we profile and ping.

No action required on libirrep 1.2.

## 2. `tp_weight_l2_per_path_uvw` output layout — per-path is right

**Maintainer's offer:** add `irrep_tp_weight_l2_per_lc_group_uvw` for
aggregation-by-`l_c` if we wanted pooled groupings.

**Response:** per-path form is correct for our SR optimizer. We apply
regularisation element-wise inside the CG preconditioner on
`(G + λI)⁻¹ F`; no libirrep-side grouping needed. Any future
symmetry-projected training that wanted per-`l_c` pooling can do the
aggregation on top of per-path values cheaply.

No additional libirrep API required.

## 3. Bench baseline edge-count regime

**Maintainer's concern:** 16-node / 32-edge baseline in
`benchmarks/results/baseline/darwin-arm64.json` won't catch regressions
that only appear at our operational scale.

**Response — our edge-count envelope:**

| Workload | Lattice | Edges (periodic) |
|---|---|---|
| Unit / regression tests | 4×4 | 64 |
| Paper 1 THQCP simulation | 8×8 → 16×16 | 256 → 1024 |
| Paper 2 benchmarks | 32×32 | 4096 |
| Device simulation (Paper 4+) | 64×64 | 16384 |
| µMAG micromagnetic follow-up | up to 256×256 | 262144 |

**Suggestion for libirrep 1.2 benchmarks:** log-spaced edge counts
`{64, 256, 1024, 4096, 16384}`. The 262k-edge µMAG case can wait for
1.3 — it's a specific follow-up application, not a core workload.

**Offer:** we can vendor actual connectivity arrays for 4×4 / 8×8 /
16×16 / 32×32 / 64×64 periodic graphs as JSON files under
`tests/test_downstream_compat/graphs/`, which libirrep's bench can
load directly. No need for both sides to re-implement the grid
construction.

## 4. Forces FD check — golden vectors are the real pin

**Maintainer's ask:** once `_apply_forces` ships, replace the one-edge
FD smoke test with real downstream-coupled golden vectors.

**Response — committed:**

When libirrep 1.2 tags with `_apply_forces`, we extend
`tests/test_downstream_compat/torque_net_tp_paths/` with:

- `expected_forces.json` — `∂h_out/∂edge_vec` for each of the 5
  existing configurations, pinned with the same bit-exact contract as
  `expected_outputs.json`.
- `generate_golden_forces.c` — regenerator for the forces JSON, so
  convention evolution can be re-captured as a single rerun.

libirrep's `tests/test_downstream_compat/` vendors both files. FD
smoke test stays as a belt-and-suspenders assertion; golden-vector
match is the enforcing gate.

## 5. Path-weight initialization scheme

**Maintainer's ask:** what init does our training use? `0.013·(i+1)`
placeholder risks pathological short-circuits on `wt == 0.0` in the
TP inner loop.

**Response — NequIP's equivariance-preserving init:**

For each path `p` with input irrep `l_in`, spherical harmonic `l_sh`,
and output irrep `l_out`:

    w_p  ~  TruncNormal(0, σ_p²),   σ_p = 1 / √(n_paths(l_out))

where `n_paths(l_out)` is the count of TP paths feeding irrep `l_out`
in the output multiset. This matches e3nn's uvw-TP convention and
NequIP's Table-2 init. Keeps output-feature variance ≈ O(1) per
output-irrep channel at init.

**For benchmark hygiene:** the current `0.013 · (i+1)` placeholder is
dense non-zero so the short-circuit is fine. When the "realistic init"
benchmark lands, use the formula above; weights sit around 0.1–0.5,
no zeros.

**Offer:** we can commit `scripts/compute_nequip_init.c` (pure C
helper) that takes a spec string and emits the per-path σ_p array.
Both trees use the same helper; no divergence on init convention.

## Meta — on the coordination pattern

Agreed on the shape:
- Mirrored benchmarks in both trees.
- Shared golden vectors gating CI on both sides.
- Single coordination doc per milestone cycle.
- Pinned compatibility declared in both trees.

For libirrep 1.3, when we start layering P1.3 learned-decoder work
that wants additional primitives (CG-scalar broadcast, symmetry-
projected feature contraction, etc.), open a
`docs/coordination/libirrep_1_3_*.md` pair early so scope-lock
happens before either side over-commits. Template unchanged.

## Revised scope table for libirrep 1.2

| Item | Owner | Status after clarifications |
|---|---|---|
| `irrep_nequip_layer_from_spec` | libirrep | PR #5 draft ready to land |
| `irrep/point_group.h` (C₄ᵥ, D₆) | libirrep | PR #6 draft ready to land |
| `irrep_nequip_layer_apply_forces` | libirrep | FD smoke test sufficient; golden-vector gate follows from our side post-tag |
| `irrep_tp_weight_l2_per_path_uvw` | libirrep | per-path form confirmed, no grouping variant needed |
| `irrep_rbf_bessel_d` + batched | libirrep | unchanged |
| NEON SH l=1,2 | libirrep | may defer to 1.3 (unchanged) |
| Torque-net migration to TP paths | downstream (this repo) | waits on libirrep 1.2 tag |
| `benchmarks/libirrep_vs_handrolled/` | downstream (this repo) | scaffold committed; full impl post-tag |
| `expected_forces.json` golden addendum | downstream (this repo) | commitment, follows 1.2 tag |
| Vendored connectivity graphs | downstream (this repo) | ~64 → 16k edge suite, offer open |
| NequIP-init helper | downstream (this repo) | `compute_nequip_init.c` on request |
| Benchmark edge-count expansion | libirrep | 64 → 16k log-spaced suggested |

No blockers remaining on either side. libirrep 1.2 ships on schedule
from the coordination standpoint.
