/*
 * benchmarks/pillars/bench_pillar_template.c
 *
 * Copy this file to benchmarks/pillars/bench_<pillar>.c when adding a
 * v0.5 pillar benchmark. The resulting binary emits one JSON record
 * per measurement under benchmarks/results/<pillar>/<name>.json.
 *
 * Minimum content: one throughput measurement that represents the
 * pillar's critical-path cost. More detailed benchmarks (per-operation
 * breakdowns, scaling studies) can be added incrementally.
 *
 * See benchmarks/bench_nqs.c for a fully worked example.
 */
#include <stdio.h>
#include "bench_common.h"

int main(void) {
    /* TODO: pillar-specific setup. */

    double t0 = bench_now_seconds();
    /* TODO: call into the pillar for a representative workload. */
    double dt = bench_now_seconds() - t0;

    double ops_per_second = 1.0 / (dt > 0 ? dt : 1.0);

    bench_emitter_t em;
    bench_emit_begin(&em, "pillar_template", "default");
    bench_emit_metric(&em, "seconds", dt);
    bench_emit_metric(&em, "ops_per_second", ops_per_second);
    bench_emit_end(&em);

    printf("pillar_template: %.3f s (%.1f ops/sec)\n", dt, ops_per_second);
    return 0;
}
