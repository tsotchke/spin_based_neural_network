# libirrep 1.2 — Round-2 handoff artifacts

Second round of the 1.2 coordination. Maintainer accepted all
clarifications from
[`libirrep_1_2_clarifications.md`](libirrep_1_2_clarifications.md)
and requested three concrete deliverables to land on our side before
their implementation cycle. This doc records what we shipped in this
turn.

## Delivered in this commit

### Lattice connectivity JSON — `tests/test_downstream_compat/lattice_connectivity.json`

Vendorable bit-identical periodic 2D lattice graphs covering the
envelope:

| Shape | Nodes | Edges |
|---|---:|---:|
| `4x4_periodic_64_edges` | 16 | 64 |
| `8x8_periodic_256_edges` | 64 | 256 |
| `16x16_periodic_1024_edges` | 256 | 1024 |
| `32x32_periodic_4096_edges` | 1024 | 4096 |
| `64x64_periodic_16384_edges` | 4096 | 16384 |
| `256x256_periodic_262144_edges` | 65536 | 262144 |

Each shape carries `{Lx, Ly, periodic, num_nodes, num_edges,
edge_src[], edge_dst[], edge_vec[3·E]}`. Edge enumeration matches
`torque_net_build_grid(Lx, Ly, periodic=1)` exactly.

Total file size ~7.4 MB, dominated by the 256×256 case.

Generator: `tests/test_downstream_compat/generate_lattice_connectivity.c`
(also committed). libirrep's CI can optionally regenerate and
bit-compare as a drift detector.

**Vendor path:** drop at `tests/test_downstream_compat/lattice_connectivity.json`
in libirrep's tree.

### NequIP init helper — `scripts/compute_nequip_init.c`

Pure ANSI C, ~230 LOC, no deps. CLI:

```sh
./compute_nequip_init --input "2x0e+1x1o" --output "1x1o" --sh 2
./compute_nequip_init --input "4x0e+2x1o+1x2e" --output "2x0e+1x1o" --sh 3 \
                     --dump-weights --seed 42
```

Outputs per-output-irrep JSON with `{l, parity, n_paths, sigma}`
and optional sampled `weights[]` (TruncNormal cutoff ±2σ, Box-Muller).

Implements exactly the rule agreed in the clarifications:
`σ_p = 1 / √n_paths(l_out)` where `n_paths(l_out)` counts the TP
paths `(l_in, l_sh)` contributing to output irrep
`(l_out, parity_out)` under the angular-momentum triangle
inequality + parity product.

**Validated:**
- `2x0e+1x1o → 1x1o` @ sh=2 → n_paths(l=1,o) = 3, σ = 1/√3 ≈ 0.577.
- `4x0e+2x1o+1x2e → 2x0e+1x1o` @ sh=3 →
  n_paths(l=0,e) = 3, σ(l=0,e) = 1/√3;
  n_paths(l=1,o) = 5, σ(l=1,o) = 1/√5 ≈ 0.447.

**Maintainer's choice:** link against this file, or copy the
`σ_p = 1/√n_paths(l_out)` formula into a 30-line bench helper.
Either keeps the surface clean.

## Queued, waiting on libirrep 1.2 tag

### `generate_golden_forces.c` + `expected_forces.json`

Triggered by libirrep 1.2's `irrep_nequip_layer_apply_forces`
landing. On 1.2 tag I will extend
`tests/test_downstream_compat/torque_net_tp_paths/` with:

- Generator (parallels existing `generate_golden.c`): runs
  `irrep_nequip_layer_apply_forces` on the five fixed configurations,
  emits pinned `∂h_out/∂edge_vec` arrays.
- `expected_forces.json`: pinned forces residuals.

Vendor path is identical to the existing golden-outputs artefacts
(libirrep's `tests/test_downstream_compat/torque_net_tp_paths/`).

### PR #5 and PR #6

Drafts ready:

- [`libirrep_pr5_nequip_layer_from_spec.md`](libirrep_pr5_nequip_layer_from_spec.md)
- [`libirrep_pr6_point_group.md`](libirrep_pr6_point_group.md)

I'll file them against libirrep's tree when ready to go; content
above is what lands in the respective PR descriptions.

## Revised scope table

| Item | Owner | Status |
|---|---|---|
| `irrep_nequip_layer_from_spec` | libirrep | waits on our PR #5 |
| `irrep/point_group.h` (C₄ᵥ, D₆) | libirrep | waits on our PR #6 |
| `irrep_nequip_layer_apply_forces` | libirrep | unblocked |
| `irrep_tp_weight_l2_per_path_uvw` | libirrep | unblocked, per-path final |
| `irrep_rbf_bessel_d` + batched | libirrep | unblocked |
| NEON SH l=1,2 | libirrep | may defer to 1.3 |
| Bench edge-count expansion to {64,…,16384} | libirrep | unblocked; uses our lattice_connectivity.json |
| Realistic-init bench entry | libirrep | unblocked; uses our compute_nequip_init.c |
| Torque-net migration to TP paths | downstream | waits on libirrep 1.2 tag |
| `expected_forces.json` + generator | downstream | waits on libirrep 1.2 tag |
| `benchmarks/libirrep_vs_handrolled/` impl | downstream | waits on libirrep 1.2 tag |

No blockers on libirrep's side from this repo's commitments.

## 1.3 preview per maintainer's ask

When P1.3 learned-decoder work starts, open
`docs/coordination/libirrep_1_3_cg_scalar_broadcast.md` (or named
per whichever primitive is first needed) with API sketch + use case
+ acceptance criterion *before* implementation begins. CG-scalar
broadcast is the first candidate I've flagged; exact API to be
proposed when the use case crystallises.
