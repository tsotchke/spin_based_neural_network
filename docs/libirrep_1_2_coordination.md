# libirrep 1.2 coordination

> **Status (2026-04-23, v0.4.1).** This document is historical — it
> records the 1.2 cycle now closed. libirrep 1.2.0 has shipped; see
> `VERSION_PINS` for the current `LIBIRREP_MIN` (now `1.3.0-alpha`,
> which ships kagome geometry, p6mm wallpaper group,
> `config_project.h`, `rdm.h`, `sym_group.h`, `spin_project.h`, and a
> half-integer path in `tensor_product.h`). Live coordination notes
> between the two projects go through
> `/Users/tyr/Desktop/agent-notes/inbox-spin/` and
> `/Users/tyr/Desktop/agent-notes/inbox-irrep/`.

Companion document to [`docs/cross_project_integration.md`](cross_project_integration.md).
Tracks the technical commitments between this repo (`spin_based_neural_network`)
and [`libirrep`](https://github.com/tsotchke/libirrep) for the libirrep 1.2
release cycle.

libirrep 1.1.0 is final at `release/1.1.0/`, ABI hash `48be5f00…2a3`.
All items below ship in libirrep 1.2.

## Deliverables owed by libirrep 1.2

| Item | Purpose here | API form |
|---|---|---|
| `irrep_nequip_layer_from_spec` | Spec-string constructor — replaces multiset handle-lifecycle boilerplate in the bridge. | `irrep_nequip_layer_t *layer = irrep_nequip_layer_from_spec("2x0e+1x1o -> 1x1o [sh=2, radial=8, cutoff=polynomial(6), r_cut=1.5]")` |
| `irrep_point_group.h` (C₄ᵥ, D₆) | Enables symmetry-projected NQS ansätze on square + kagome lattices (pillar P1.1.b follow-up). | `irrep_pg_project(table, mu, spec, in, out)` + `irrep_pg_reduce(table, spec, out_mult)` |
| `irrep_nequip_layer_apply_forces` | Edge-geometry gradients. Required for training the equivariant torque net against measured device dynamics. | `void irrep_nequip_layer_apply_forces(layer, tp_weights, n_nodes, n_edges, edge_src, edge_dst, edge_vec, h_in, grad_h_out, grad_edge_vec)` |
| `irrep_tp_weight_l2_per_path_uvw` | Per-path L2 regulariser for training stability. | `irrep_tp_weight_l2_per_path_uvw(desc, weights, out)` + backward variant |
| `irrep_rbf_bessel_d` + batched | Completes the RBF chain-rule for `_apply_forces`. | Mirror of existing `irrep_rbf_bessel` with `_d` suffix and batched sibling |
| NEON SH at l=1,2 (may defer to 1.3) | Performance — hot path in the Paper-1 equivariant-torque benchmarks. | No API change; speedup only |

Upstream acceptance: bit-equal vs scalar implementations, finite-difference gradient cross-check < 1e-6, Bradley–Cracknell character-table agreement for C₄ᵥ / D₆.

## Deliverables owed by this repo

### PR #5 — `irrep_nequip_layer_from_spec` parser spec + tests

- BNF grammar for the spec string, strict-in-token / lenient-between-tokens whitespace.
- 10-case test suite in libirrep's tree: 5 valid-parse cases (default options + each field overridden individually + all-fields-overridden), 5 malformed-input cases (missing arrow, trailing operator, invalid irrep, invalid cutoff, unknown option).
- Acceptance gate: every pass-case constructs a layer bit-equal to the verbose `irrep_nequip_layer_build(...)` API, verified by round-trip through `irrep_nequip_layer_apply` on a fixed `(h_in, edge_vec)` input.

### PR #6 — `irrep_point_group.h` API proposal

- Header skeleton per the coordinator's spec:
  ```c
  typedef enum { IRREP_PG_C4V, IRREP_PG_D6, IRREP_PG_C3V, IRREP_PG_D3 /* more */ } irrep_point_group_t;
  typedef struct irrep_pg_table irrep_pg_table_t;
  irrep_pg_table_t *irrep_pg_table_build(irrep_point_group_t g);
  int  irrep_pg_num_irreps(const irrep_pg_table_t *);
  int  irrep_pg_order     (const irrep_pg_table_t *);
  void irrep_pg_project   (const irrep_pg_table_t *, int mu, const irrep_multiset_t *, const double *in, double *out);
  void irrep_pg_reduce    (const irrep_pg_table_t *, const irrep_multiset_t *, int *out_mult);
  ```
- Coordinator takes point on build / apply internals.

### Golden vectors for torque-net migration

- `tests/test_downstream_compat/torque_net_tp_paths/` holds five fixed `(h_in, edge_vec)` configurations with their expected tensor-product outputs, vendored into libirrep's tree as `tests/test_downstream_compat/`.
- Pinned prefactors `{−1/√2, 1/√3}` documented inline so any convention drift fails in both test trees simultaneously.
- Running `make test_downstream_compat` in libirrep's tree regenerates the expected values from the libirrep installation and asserts bit-equality with the pinned vectors.

### `benchmarks/libirrep_vs_handrolled/`

- Measures our current hand-rolled `src/equivariant_gnn/torque_net.c` against a libirrep-backed implementation (via `irrep_nequip_layer_apply`) on three NequIP shapes — identical to libirrep's downstream shapes (`benchmarks/bench_downstream_shapes.c`):
  - small:  `2x0e + 1x1o → 1x1o`, sh=2
  - medium: `4x0e + 2x1o + 1x2e → 2x0e + 1x1o`, sh=3
  - large:  `8x0e + 4x1o + 2x2e + 1x3o → 4x0e + 2x1o + 1x2e`, sh=4
- JSON output under `benchmarks/results/libirrep_vs_handrolled/`, mirrors libirrep's `benchmarks/results/baseline/`, gated by `scripts/perf_compare.sh` in CI.

### Torque-net migration

- Rewrite `src/equivariant_gnn/torque_net.c` forward pass to route through `irrep_nequip_layer_apply` once libirrep 1.2 is tagged. Hand-rolled primitives stay as a reference implementation for the regression harness.
- Backward pass uses `irrep_nequip_layer_apply_backward` + `irrep_nequip_layer_apply_forces`.
- Closed-form fitter (`torque_net_fit_weights`) extended with `irrep_tp_weight_l2_per_path_uvw` as an optional regulariser term.

## Pinned compatibility

- This repo currently builds against libirrep ≥ 1.0.0 (primitives) and ≥ 1.1.0 (NequIP).
- libirrep 1.2 upgrade: one-line change to `VERSION_PINS`, plus activate the migration items above.
- No breaking changes on either side.

## Schedule

| Milestone | Owner | Target |
|---|---|---|
| PR #5 draft (spec + 10-case test) | this repo | this sprint |
| PR #6 API spec for `point_group.h` | this repo | this sprint |
| libirrep 1.2 implementation of PR #5/#6 | libirrep | next sprint |
| `_apply_forces` FD cross-check | libirrep | next sprint |
| `_l2_per_path_uvw` + backward | libirrep | next sprint |
| `irrep_rbf_bessel_d` + batched | libirrep | next sprint |
| Golden vectors in both trees | this repo + libirrep | alongside libirrep 1.2 tag |
| Torque-net migration + benchmark | this repo | immediately after libirrep 1.2 tag |

## Attribution

`tsotchke` owns both sides; the "owner" column distinguishes which repo carries the commit, not authorship.
