# Benchmarks

v0.4 foundation benchmark suites. See [`docs/benchmarks.md`](../docs/benchmarks.md)
for the full description, JSON schema, and reference numbers.

## TL;DR

```sh
./scripts/run_benchmarks.sh
ls benchmarks/results/
```

Emits one JSON record per suite × configuration under
`benchmarks/results/<suite>/<name>.json`.

## Provenance note

The JSON files shipped in this directory are the v0.4 reference
baseline, measured on an Apple-Silicon M-series Mac. Each record
includes the `hostname`, `utc_epoch`, `os`, and `arch` fields of the
machine that produced it; treat absolute metric values as a ballpark
on that hardware and reproduce locally with the command above to get
your own numbers.

## Suites

- `bench_ising`         — Metropolis sweeps/s at L = 8, 16, 32
- `bench_kitaev`        — Metropolis sweeps/s at L = 8, 16, 32
- `bench_majorana_braid` — Hilbert-space braids/s at N = 8, 12, 16, 20
- `bench_toric_decoder` — greedy-decoder logical-error-rate curves at
                          d = 3, 5, 7 × p = 1%, 3%, 5%

The `bench_toric_decoder` output is the baseline that the learned
surface-code decoder in v0.5 pillar P1.3 (see
[`docs/architecture_v0.4.md`](../docs/architecture_v0.4.md) §P1.3) has
to beat.
