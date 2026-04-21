# Reproducibility

`spin_based_neural_network` takes reproducibility seriously. Every
public result in this repository is meant to be bit-reproducible or
tolerance-reproducible on the stated toolchain.

## Deterministic RNG

Every test uses a fixed RNG seed. Grep: `srand(` appears in the test
suite only with integer-literal arguments (42, 7, 11, 55, 88, 99,
etc.). Zero occurrences of `time(NULL)` or `/dev/urandom`. NQS tests
use `rng_seed` fields in `nqs_config_t` rather than `srand`.

If you write a new test, follow the same discipline: fixed
integer-literal seed; no wall-clock sources.

## Floating-point

The framework uses IEEE 754 double precision (`double`) throughout
the public API. Compile with:

- `-std=c11`
- `-O2` (default in `CFLAGS_COMMON`)
- No `-ffast-math`. Fast-math breaks NaN / Inf semantics and
  makes `Cov` / variance identities drift from the textbook
  derivation.
- No `-funsafe-math-optimizations`.

## Toolchain matrix

The CI gate (`.github/workflows/test.yml`) exercises:

- Ubuntu latest + GCC.
- macOS latest + Clang.

Local known-good configurations:

- macOS 14+ on Apple Silicon, Homebrew Clang, SDL2 2.28+, `make`
  4.3.
- Ubuntu 22.04, GCC 11.4, libsdl2-dev 2.24+.

`make universal` detects NEON at runtime on ARM and falls back to
scalar on x86_64.

## Build reproducibility

`make clean && make test` from a fresh checkout should produce the
same pass/fail outcome on any machine matching the toolchain matrix.
Benchmarks will naturally vary in wall-clock time but should match
their golden JSON schemas in `benchmarks/results/`.

## Numerical tolerances

Each public test `tests/test_*.c` states its tolerance at the top of
the file. As a rule:

- Closed-form algebraic identities assert agreement within IEEE double
  machine epsilon (≈ 1e-15).
- Finite-difference derivative checks assert agreement at the expected
  truncation order for the stencil step size used.
- Monte Carlo / Metropolis routines are checked against exact
  enumeration on small systems when tractable, and against known
  analytical limits (low / high temperature, strong / weak coupling)
  elsewhere.

These tolerances are chosen to exceed the expected arithmetic or
truncation error at the operating point, not to hide imprecision.
Re-running on hardware with identical floating-point behaviour should
match within the same bounds.

## Benchmark reproducibility

Each `benchmarks/bench_*.c` emits a JSON block with:

- `hostname`, `os`, `arch`, `cpu_model`
- `compiler_version`, `cflags`
- `git_commit` of the benchmarked build
- Per-run wall-clock and iteration count

Wall-clock times are not reproducible across different hardware. Use
the JSON metadata to identify whether a reported number was produced
on a comparable machine before treating it as a baseline.

## Dependency versions

v0.4 does not yet ship a `VERSION_PINS` file; this is planned for
v0.5 per `docs/architecture_v0.4.md`. Until then:

- `libirrep` ≥ 1.2 (soft dep, only required for
  `IRREP_ENABLE=1`).
- `libquantumsim` from the `quantum_simulator` sibling project (soft
  dep, only for `MOONLAB_ENABLE=1`).
- No mandatory third-party C library beyond SDL2 for the
  visualization binary.

## Reproducing a published result

When a result in this repository backs a published paper (in
preparation or on arXiv), the reproduction recipe is:

1. Check out the commit or tag referenced in the paper.
2. Install SDL2 via the system package manager (only needed for
   `make visualization`).
3. `make clean && make <target>`.
4. `build/<target>` — output must pass all assertions at the tolerance
   stated in the manuscript.

Any discrepancy is a bug worth investigating.

## Version history of this document

- 2026-04-18: initial write for v0.4.
