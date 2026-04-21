# Development guide

Prerequisites, build mechanics, and debugging workflow for contributors.
Users of the framework only need `README.md`.

## Prerequisites

- A C11 compiler — GCC 11+ or Clang 14+ recommended. Apple Silicon:
  Xcode Clang ships ARM NEON support out of the box. Linux: either
  compiler works.
- GNU `make` 4.0+.
- `pkg-config` (for SDL2 detection).
- `libsdl2-dev` (Linux) or `brew install sdl2` (macOS) — only
  required to build the `visualization` target; not needed for
  tests or benchmarks.
- For coverage/sanitizers: the same compiler with ASAN/UBSAN
  support enabled in its build.

No Python runtime is required for the default build.
`scripts/plot_trends.py` is a single optional plotting utility.

## Quickstart

```
git clone https://github.com/tsotchke/spin_based_neural_network
cd spin_based_neural_network
make test
```

That should produce ~46 passing TAP blocks. If any fail, capture
the full output and open an issue.

## Build targets

Full list: `grep -E '^[a-z_][a-z0-9_]*:' Makefile | head -50`.
The most common:

| Target | What it builds |
|---|---|
| `make all` | `arm` + `non_arm` main binaries |
| `make universal` | Single main binary with runtime NEON auto-detect |
| `make topo_example` | Standalone topological demo |
| `make visualization` | SDL2 interactive viewer |
| `make test` | Build and run the full test aggregate (~46 binaries) |
| `make test_<name>` | Build and run a single test |
| `make bench` | Build all benchmark binaries |
| `make clean` | Delete `build/` |
| `make check_stack` | Advisory probe for optional sibling projects |

## Enabling optional bridges

The seven cross-project bridges are disabled by default. To enable:

```
# libirrep (public sibling project)
make IRREP_ENABLE=1 IRREP_ROOT=/path/to/libirrep test_torque_net_irrep

# moonlab / quantum_simulator
make MOONLAB_ENABLE=1 MOONLAB_ROOT=/path/to/quantum_simulator test_moonlab_bridge

# internal NN engine
make ENGINE_ENABLE=1 ENGINE_ROOT=/path/to/engine ENGINE_VERSION=0.1 all
```

`MOONLAB_ENABLE=1` without `MOONLAB_ROOT` set now errors out — no
more silent developer-path default.

## Running a single test

```
make test_theory_rao_blackwell
build/test_theory_rao_blackwell
```

The `make test_<name>` target builds the binary; running it is a
separate step so you can repeat without recompiling.

## Debugging

### GDB / LLDB

Tests are standalone executables compiled at `-O2`. For debuggable
builds, override the flags:

```
make clean
make CFLAGS_COMMON='-Wall -std=c11 -Iinclude -O0 -g' test_<name>
lldb build/test_<name>
```

### AddressSanitizer / UBSAN

Same override:

```
make clean
make CFLAGS_COMMON='-Wall -std=c11 -Iinclude -O1 -g -fsanitize=address,undefined' \
     LDFLAGS='-lm -fsanitize=address,undefined' \
     test_<name>
build/test_<name>
```

Any heap corruption, use-after-free, or signed-overflow UB will
fire cleanly.

### Valgrind (Linux)

```
valgrind --leak-check=full --track-origins=yes build/test_<name>
```

## Adding a new test

1. Copy `tests/pillars/test_pillar_template.c` to `tests/test_<name>.c`.
2. Add a Makefile target block (grep `test_theory_bps_qgt_dictionary:`
   for the template).
3. Add the target to both the dependency list and the run block of
   `test:` in the Makefile.
4. Use fixed integer seeds — `srand(N)` where `N` is a literal.
   No `time(NULL)`, no `/dev/urandom`.
5. Each test function starts with `static void test_<what>(void)`
   and ends without a return value. Use `ASSERT_*` macros from
   `tests/harness.h`.
6. `main()` invokes `TEST_RUN(fn)` for each test and ends with
   `TEST_SUMMARY()`.

## Adding a new bridge

The seven existing bridges (`src/*_bridge.c`) follow a uniform
pattern:

1. Header `include/<bridge>_bridge.h` declares opaque types,
   `<BRIDGE>_OK` / `<BRIDGE>_EDISABLED` / `<BRIDGE>_EARG` error
   codes, lifecycle hooks (`_init`, `_shutdown`), feature-probe
   (`_is_available`, `_version`), and call-through APIs.
2. Source `src/<bridge>_bridge.c` guards the live implementation
   behind `#ifdef SPIN_NN_HAS_<BRIDGE>`. Disabled-path returns
   `<BRIDGE>_EDISABLED`.
3. Test `tests/test_<bridge>_bridge.c` has two `main()` blocks —
   one for live path, one for disabled path. The disabled path
   asserts the EDISABLED return codes and null-arg safety.

Do not invent a new pattern. Follow the existing seven.

## Coding standards

- C11, `-Wall -Wextra` clean (no warnings).
- 4-space indent, no tabs.
- Snake_case for functions and variables. `UPPER_CASE` for macros
  and enum members.
- One logical change per commit. Commit messages in present tense
  (`add`, `fix`, `tighten`, not past tense).
- No trailing whitespace. Unix line endings.

## Performance profiling

- Apple Silicon: use Instruments (Xcode). Target the binary built
  with `-O2` (default) or `-O3 -g` for richer symbol info.
- Linux: `perf record -g build/bench_<name>` then `perf report`.

Benchmark suite schema is JSON lines written to
`benchmarks/results/<bench_name>/<run_id>.json`. Each record
carries `hostname`, `compiler_version`, `cflags`, `git_commit` so
past runs on different hardware remain comparable.

## Release process

1. All tests pass on the CI matrix (Ubuntu + macOS).
2. `CHANGELOG.md` moved from `[Unreleased]` to a new `[v0.X.Y]`
   heading with today's date.
3. `RELEASE_NOTES.md` written (focus on user-visible changes).
4. Bump version strings in `README.md` if they reference the tag.
5. `git tag vX.Y.Z -a -m "Release vX.Y.Z"`.
6. No push until maintainer explicitly runs `git push origin vX.Y.Z`.

## Contact

See `CONTRIBUTING.md` for how to open issues and PRs.

## Version history of this document

- 2026-04-18: initial write for v0.4.
