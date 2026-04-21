# `libirrep_vs_handrolled` benchmark

Perf-regression harness comparing our current hand-rolled
`src/equivariant_gnn/torque_net.c` implementation with a
libirrep-backed implementation routed through
`irrep_nequip_layer_apply`. Required for the libirrep 1.2
coordination agreement — see
[`docs/libirrep_1_2_coordination.md`](../../docs/libirrep_1_2_coordination.md).

Mirrors libirrep's own `benchmarks/bench_downstream_shapes.c` so that
regressions in either tree are caught before they bite the other.

## Three descriptor shapes

All three use a 4×4 periodic grid, 64 directed edges, 100 node-feature
configurations (deterministic splitmix-seeded). Each shape specifies a
NequIP input multiset, output multiset, and SH order `l_sh_max`:

| Shape | Input multiset | Output multiset | `l_sh_max` |
|---|---|---|---|
| small  | `2x0e + 1x1o` | `1x1o` | 2 |
| medium | `4x0e + 2x1o + 1x2e` | `2x0e + 1x1o` | 3 |
| large  | `8x0e + 4x1o + 2x2e + 1x3o` | `4x0e + 2x1o + 1x2e` | 4 |

## What's measured

For each shape:
- Forward-pass throughput (configurations / second).
- Backward-pass throughput (when libirrep 1.2 `_apply_backward` is
  available).
- Forces pass throughput (when libirrep 1.2 `_apply_forces` is
  available).
- Memory footprint (RSS peak during the benchmark).
- Numerical agreement (L∞ residual of hand-rolled vs libirrep-backed
  outputs on the same input, expected < 1e-14 in normalised units).

## JSON output format

Each run writes `benchmarks/results/libirrep_vs_handrolled/<shape>.json`
with:

```json
{
  "shape": "small",
  "n_configs": 100,
  "hand_rolled": {
    "forward_throughput_per_sec": 1234.5,
    "mem_peak_bytes": 1048576
  },
  "libirrep": {
    "forward_throughput_per_sec": 987.6,
    "backward_throughput_per_sec": 654.3,
    "forces_throughput_per_sec": 321.0,
    "mem_peak_bytes": 2097152
  },
  "numerical_agreement_l_inf": 1.4e-15,
  "git_sha": "…",
  "libirrep_version": "1.2.0",
  "cpu": "Apple M2 / Intel Xeon …",
  "timestamp_utc": "2026-…"
}
```

## Regression gating

`scripts/perf_compare.sh` runs this benchmark on every CI cycle and
compares to the baseline under `benchmarks/results/baseline/`. A
regression is any metric degrading by more than 10% on either the
hand-rolled or libirrep side. Numerical agreement is a hard gate:
any residual > 1e-12 fails CI.

## Status

- [x] Directory + README committed.
- [ ] `bench_libirrep_vs_handrolled.c` — waiting on libirrep 1.2 tag
      (depends on `irrep_nequip_layer_from_spec` to construct layers
      from the above multiset strings).
- [ ] `scripts/perf_compare.sh` — to be added alongside the benchmark.
- [ ] Baseline-JSON seeding — populated after first CI run with
      libirrep 1.2.

## Running once implemented

```sh
# One-shot with JSON outputs:
make bench_libirrep_vs_handrolled \
     IRREP_ENABLE=1 \
     IRREP_ROOT=/path/to/libirrep/release/1.2.0 \
     IRREP_LIBDIR=/path/to/libirrep/release/1.2.0/lib/macos-arm64

build/bench_libirrep_vs_handrolled --shapes small,medium,large \
    --n-configs 100 --output-dir benchmarks/results/libirrep_vs_handrolled
```
