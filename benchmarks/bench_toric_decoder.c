/*
 * benchmarks/bench_toric_decoder.c
 *
 * Measures greedy-matching decoder throughput (decodes/sec) at several
 * lattice distances and physical error rates. Serves as the baseline for
 * the learned decoder that lands in v0.5 (pillar P1.3).
 */
#include "bench_common.h"
#include "toric_code.h"
#include <stdlib.h>

static double bench_one(int L, double p, int n_decodes, int *out_logical_err_rate) {
    int logical_err_count = 0;
    double t0 = bench_now_seconds();
    for (int i = 0; i < n_decodes; i++) {
        ToricCode *c = initialize_toric_code(L, L);
        apply_random_errors(c, p);
        toric_code_decode_greedy(c);
        if (toric_code_has_logical_error(c)) logical_err_count++;
        free_toric_code(c);
    }
    double dt = bench_now_seconds() - t0;
    if (out_logical_err_rate) *out_logical_err_rate = logical_err_count;
    return (double)n_decodes / dt;
}

int main(void) {
    int dists[] = {3, 5, 7};
    double rates[] = {0.01, 0.03, 0.05};
    int n_decodes = 500;

    srand(2026);
    for (int di = 0; di < 3; di++) {
        for (int ri = 0; ri < 3; ri++) {
            int L = dists[di];
            double p = rates[ri];
            int le = 0;
            double dps = bench_one(L, p, n_decodes, &le);
            char name[32];
            snprintf(name, sizeof(name), "d%d_p%03d", L, (int)(p * 100));
            bench_emitter_t em;
            bench_emit_begin(&em, "toric_decoder", name);
            bench_emit_int(&em, "distance", L);
            bench_emit_metric(&em, "physical_error_rate", p);
            bench_emit_int(&em, "decodes_run", n_decodes);
            bench_emit_int(&em, "logical_errors", le);
            bench_emit_metric(&em, "logical_error_rate", (double)le / n_decodes);
            bench_emit_metric(&em, "decodes_per_second", dps);
            bench_emit_end(&em);
            printf("toric d=%d p=%.2f: %.1f decodes/sec, logical-err-rate=%.3f\n",
                   L, p, dps, (double)le / n_decodes);
        }
    }
    return 0;
}
