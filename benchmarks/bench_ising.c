/*
 * benchmarks/bench_ising.c
 *
 * Measures Metropolis sweeps-per-second on the 3D Ising lattice at a
 * handful of lattice sizes. Emits JSON to benchmarks/results/ising/.
 */
#include "bench_common.h"
#include "ising_model.h"

static double bench_sweeps(int L, int sweeps) {
    IsingLattice *l = initialize_ising_lattice(L, L, L, "random");
    int N = L * L * L;
    double t0 = bench_now_seconds();
    for (int s = 0; s < sweeps; s++) {
        for (int i = 0; i < N; i++) flip_random_spin_ising(l);
    }
    double dt = bench_now_seconds() - t0;
    free_ising_lattice(l);
    return (double)sweeps / dt;
}

int main(void) {
    int sizes[] = {8, 16, 32};
    int sweeps[] = {2000, 500, 100};
    for (int i = 0; i < 3; i++) {
        int L = sizes[i];
        double sps = bench_sweeps(L, sweeps[i]);
        char name[32];
        snprintf(name, sizeof(name), "L%d", L);
        bench_emitter_t em;
        bench_emit_begin(&em, "ising", name);
        bench_emit_int(&em, "lattice_L", L);
        bench_emit_int(&em, "sweeps_run", sweeps[i]);
        bench_emit_metric(&em, "sweeps_per_second", sps);
        bench_emit_end(&em);
        printf("ising L=%d: %.2f sweeps/sec\n", L, sps);
    }
    return 0;
}
