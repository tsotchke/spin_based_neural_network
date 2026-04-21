/*
 * Minimal TAP-style assert harness. No framework dependency.
 * Each test file defines main() that calls TEST_RUN(...) for each test
 * function; the last line prints TAP summary.
 */
#ifndef SPIN_NN_TEST_HARNESS_H
#define SPIN_NN_TEST_HARNESS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

static int _tests_run = 0;
static int _tests_passed = 0;
static int _tests_failed = 0;
static int _current_test_ok = 1;
static const char *_current_test_name = "";
static int _tests_skipped = 0;

#define TEST_RUN(fn) do {                                                 \
    _current_test_ok = 1;                                                 \
    _current_test_name = #fn;                                             \
    _tests_run++;                                                         \
    (fn)();                                                               \
    if (_current_test_ok) {                                               \
        _tests_passed++;                                                  \
        printf("ok %d - %s\n", _tests_run, _current_test_name);           \
    } else {                                                              \
        _tests_failed++;                                                  \
        printf("not ok %d - %s\n", _tests_run, _current_test_name);       \
    }                                                                     \
} while (0)

/* Mark the test suite as skipped and emit TAP "ok N # SKIP" directives.
 * Use this in main() for tests that require an unavailable dependency
 * (e.g. disabled bridge, missing external library) INSTEAD of silently
 * returning 0 without running anything. Makes it visible in `make test`
 * output that the suite was skipped, not that it trivially passed. */
#define SKIP_SUITE(reason) do {                                           \
    printf("# SKIP: %s\n", reason);                                       \
    printf("1..0 # SKIP %s\n", reason);                                   \
    return 0;                                                             \
} while (0)

#define TEST_FAIL(fmt, ...) do {                                          \
    _current_test_ok = 0;                                                 \
    fprintf(stderr, "    # %s: " fmt " (%s:%d)\n",                        \
            _current_test_name, ##__VA_ARGS__, __FILE__, __LINE__);       \
} while (0)

#define ASSERT_TRUE(cond) do {                                            \
    if (!(cond)) { TEST_FAIL("expected " #cond); return; }                \
} while (0)

#define ASSERT_EQ_INT(a, b) do {                                          \
    long _a = (long)(a), _b = (long)(b);                                  \
    if (_a != _b) { TEST_FAIL(#a " (%ld) != " #b " (%ld)", _a, _b); return; }\
} while (0)

#define ASSERT_NEAR(a, b, eps) do {                                       \
    double _a = (double)(a), _b = (double)(b);                            \
    if (fabs(_a - _b) > (eps)) {                                          \
        TEST_FAIL(#a "=%.12g, " #b "=%.12g, eps=%g",                      \
                  _a, _b, (double)(eps)); return;                         \
    }                                                                     \
} while (0)

#define ASSERT_NEAR_COMPLEX(a, b, eps) do {                               \
    double _Complex _a = (a), _b = (b);                                   \
    if (cabs(_a - _b) > (eps)) {                                          \
        TEST_FAIL(#a "=%.12g+%.12gi, " #b "=%.12g+%.12gi, eps=%g",        \
                  creal(_a), cimag(_a), creal(_b), cimag(_b),             \
                  (double)(eps)); return;                                 \
    }                                                                     \
} while (0)

#define TEST_SUMMARY() do {                                               \
    printf("1..%d\n", _tests_run);                                        \
    printf("# passed: %d / %d\n", _tests_passed, _tests_run);             \
    if (_tests_failed) return 1;                                          \
    return 0;                                                             \
} while (0)

#endif /* SPIN_NN_TEST_HARNESS_H */
