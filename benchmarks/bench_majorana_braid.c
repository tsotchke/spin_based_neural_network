/*
 * benchmarks/bench_majorana_braid.c
 *
 * Measures Majorana-braiding throughput (braids-per-second) on the
 * Hilbert-space implementation added in v0.4 (P0.2).
 */
#include "bench_common.h"
#include "majorana_modes.h"

static double bench_braids(int num_majoranas, int num_braids) {
    MajoranaHilbertState *psi = initialize_majorana_hilbert_state(num_majoranas);
    if (!psi) return 0.0;
    double t0 = bench_now_seconds();
    for (int i = 0; i < num_braids; i++) {
        apply_braid_unitary(psi, i & 1, (i & 1) ^ 2);
    }
    double dt = bench_now_seconds() - t0;
    free_majorana_hilbert_state(psi);
    return (double)num_braids / dt;
}

int main(void) {
    int num_majoranas_list[] = {8, 12, 16, 20};
    int num_braids_list[]    = {20000, 2000, 200, 20};
    for (int i = 0; i < 4; i++) {
        double bps = bench_braids(num_majoranas_list[i], num_braids_list[i]);
        char name[32];
        snprintf(name, sizeof(name), "N%d", num_majoranas_list[i]);
        bench_emitter_t em;
        bench_emit_begin(&em, "majorana", name);
        bench_emit_int(&em, "num_majoranas", num_majoranas_list[i]);
        bench_emit_int(&em, "num_braids", num_braids_list[i]);
        bench_emit_int(&em, "hilbert_dim", 1 << (num_majoranas_list[i] / 2));
        bench_emit_metric(&em, "braids_per_second", bps);
        bench_emit_end(&em);
        printf("majorana N=%d: %.2f braids/sec (Hilbert dim = %d)\n",
               num_majoranas_list[i], bps, 1 << (num_majoranas_list[i] / 2));
    }
    return 0;
}
