# Pillar test suite template

Every v0.5 pillar lands with at least one test suite in this format.
Copy the `.c` template, rename, wire into the Makefile as described
below, and add cases for each public API function.

## Files to create

1. `include/<pillar>/*.h` — public API.
2. `src/<pillar>/*.c` — implementation.
3. `tests/pillars/test_<pillar>.c` — test suite (start from
   `test_pillar_template.c` in this directory).
4. `benchmarks/pillars/bench_<pillar>.c` — throughput suite (start from
   `bench_pillar_template.c` in `benchmarks/pillars/`).

## Makefile rules

Add in `Makefile` next to the existing `test_*` rules:

```make
<PILLAR>_SRCS = src/<pillar>/file1.c src/<pillar>/file2.c
test_<pillar>: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_<pillar> \
	    tests/pillars/test_<pillar>.c $(<PILLAR>_SRCS) $(LDFLAGS)
bench_<pillar>_bin: $(BIN_DIR)
	$(CC) $(BENCH_CFLAGS) -o $(BIN_DIR)/bench_<pillar> \
	    benchmarks/pillars/bench_<pillar>.c $(<PILLAR>_SRCS) $(LDFLAGS)
```

Append `test_<pillar>` to the `test:` target dependency list and
recipe, and `bench_<pillar>_bin` to the `bench:` target.

## Required tests per pillar

Minimum acceptance bar:

1. **Lifecycle** — create, destroy, free NULL-safe.
2. **Happy path** — one test per public function.
3. **Argument validation** — negative numerics return
   `<PILLAR>_EARG` (or equivalent).
4. **Disabled-mode** — if the pillar has a
   `#ifdef SPIN_NN_HAS_<PILLAR>` guard, assert the disabled-path
   behavior.
5. **Known-answer physics** — at least one test computes a published
   analytical result to 1e-6 relative tolerance.

See `tests/test_nqs.c` for a worked example (14 tests across lifecycle,
sampler, local-energy, and SR step).
