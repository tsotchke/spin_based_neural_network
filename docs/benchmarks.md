# Benchmarks

v0.4 ships 4 benchmark suites under `benchmarks/`, emitting JSON records
under `benchmarks/results/<suite>/<name>.json`. Each result is a
self-describing record (hostname, timestamp, OS, arch, plus the measured
metrics).

## Running

```sh
./scripts/run_benchmarks.sh
```

Builds all 4 suites (`make bench`) and runs them in sequence. Each run
overwrites the previous result file in place.

Individual suites:

```sh
make bench_ising_bin && ./build/bench_ising
```

## Suites

### `bench_ising` / `bench_kitaev`

Metropolis sweeps-per-second on the 3D Ising / Kitaev models at lattice
sizes L = 8, 16, 32. One "sweep" is N = L³ random-site proposals;
reported metric is `sweeps_per_second`.

Results on an M-series Mac (Apple Silicon, NEON enabled, `-O2`):

| L  | Ising sweeps/s | Kitaev sweeps/s |
|---|---|---|
| 8  | 65 850 | 94 406 |
| 16 | 8 087  | 11 716 |
| 32 | 971    | 1 450  |

Scaling is close to the expected 1/L³ asymptote with cache-effect
deviations at small L.

### `bench_majorana_braid`

Braids-per-second for the v0.4 Hilbert-space braiding implementation
(`apply_braid_unitary`). Measured at N = 8, 12, 16, 20 Majorana
operators, corresponding to Fock-space dimensions 16, 64, 256, 1024.

| N (Majoranas) | Hilbert dim | Braids/s |
|---|---|---|
| 8  | 16   | 12 755 102 |
| 12 | 64   | 4 566 210  |
| 16 | 256  | 1 418 440  |
| 20 | 1024 | 338 983    |

Scales roughly linearly with Hilbert dimension, as expected for an
O(dim) Jordan-Wigner kernel plus a vector add-and-scale.

### `bench_toric_decoder`

Greedy-matching decoder throughput + logical-error-rate curves for the
v0.4 baseline decoder. Measured at code distances d = 3, 5, 7 ×
physical error rates p = 1%, 3%, 5%, 500 samples each.

| d | p | logical error rate | decodes/s |
|---|---|---|---|
| 3 | 1% | 0.008 | 1 072 961 |
| 3 | 3% | 0.068 | 978 474 |
| 3 | 5% | 0.154 | 980 392 |
| 5 | 1% | 0.002 | 421 230 |
| 5 | 3% | 0.056 | 404 858 |
| 5 | 5% | 0.140 | 391 850 |
| 7 | 1% | 0.000 | 213 767 |
| 7 | 3% | 0.024 | 204 834 |
| 7 | 5% | 0.092 | 187 266 |

At p = 1% the logical error rate decreases with distance — the greedy
decoder is operating *below* its effective threshold. At p = 5% all
distances show high logical error rates, indicating operation *above*
threshold for this decoder. The learned surface-code decoder in v0.5
pillar P1.3 (transformer / Mamba variants, see [3, 4] below) has to beat
this baseline.

The published threshold for optimal minimum-weight perfect matching
decoding on the toric code is ≈ 10.3 % for independent X / Z noise [5].
The greedy decoder's effective threshold is lower (below ≈ 5 %, based on
these curves).

## JSON schema

Each result file looks like:

```json
{
  "suite": "toric_decoder",
  "name": "d5_p003",
  "utc_epoch": 1776459352,
  "os": "Darwin",
  "arch": "arm64",
  "hostname": "localhost",
  "metrics": {
    "distance": 5,
    "physical_error_rate": 0.03,
    "decodes_run": 500,
    "logical_errors": 28,
    "logical_error_rate": 0.056,
    "decodes_per_second": 404858.303
  }
}
```

The `metrics` object is suite-specific; the outer envelope is uniform.
Parsing is one-liner in any language.

## Provenance caveats

- `hostname` is baked into each result file. Treat the shipped
  `benchmarks/results/*.json` as the v0.4 reference baseline measured
  on an Apple-Silicon M-series Mac; reproduce on your own hardware with
  `./scripts/run_benchmarks.sh`.
- Small-L Ising / Kitaev runs are sensitive to L1 cache behavior; treat
  absolute sweeps/s as "ballpark on this hardware" rather than
  portable ground truth.
- Toric-decoder logical error rates are averages over 500 trials; at
  d = 7, p = 1 % we observed 0 / 500 failures — the reported rate is a
  point estimate with large relative uncertainty at the tail.

## Adding a new benchmark

1. Create `benchmarks/bench_<name>.c`, including `benchmarks/bench_common.h`.
2. Use `bench_now_seconds()` to time, `bench_emit_begin/metric/int/end()`
   to write JSON.
3. Add a `bench_<name>_bin:` target and append to `bench:` in the
   Makefile.
4. Add the run to `scripts/run_benchmarks.sh`.

## References

1. A. Y. Kitaev, "Fault-tolerant quantum computation by anyons," Annals of Physics, vol. 303, pp. 2-30, 2003.
2. J. Edmonds, "Paths, trees, and flowers," Canadian Journal of Mathematics, vol. 17, pp. 449-467, 1965.
3. J. Bausch, A. Senior, F. Heras, T. Edlich, A. Davies, M. Newman, C. Jones, K. Satzinger, M. Y. Niu, S. Blackwell, G. Holland, D. Kafri, J. Atalaya, C. Gidney, D. Hassabis, S. Boixo, H. Neven, and P. Kohli, "Learning high-accuracy error decoding for quantum processors," Nature, vol. 635, pp. 834-840, 2024. DOI: 10.1038/s41586-024-08148-8.
4. V. Ninkovic, O. Kundacina, D. Vukobratovic, and C. Häger, "Scalable Neural Decoders for Practical Real-Time Quantum Error Correction," arXiv:2510.22724, 2025.
5. E. Dennis, A. Kitaev, A. Landahl, and J. Preskill, "Topological quantum memory," J. Math. Phys., vol. 43, pp. 4452-4505, 2002.
