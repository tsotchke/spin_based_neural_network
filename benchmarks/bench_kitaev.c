/*
 * benchmarks/bench_kitaev.c
 *
 * Metropolis sweeps-per-second for the 3D Kitaev lattice.
 */
#include "bench_common.h"
#include "kitaev_model.h"

static double bench_sweeps(int L, int sweeps) {
    KitaevLattice *l = initialize_kitaev_lattice(L, L, L, 1.0, 1.0, -1.0, "random");
    int N = L * L * L;
    double t0 = bench_now_seconds();
    for (int s = 0; s < sweeps; s++) {
        for (int i = 0; i < N; i++) flip_random_spin_kitaev(l);
    }
    double dt = bench_now_seconds() - t0;
    free_kitaev_lattice(l);
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
        bench_emit_begin(&em, "kitaev", name);
        bench_emit_int(&em, "lattice_L", L);
        bench_emit_int(&em, "sweeps_run", sweeps[i]);
        bench_emit_metric(&em, "sweeps_per_second", sps);
        bench_emit_end(&em);
        printf("kitaev L=%d: %.2f sweeps/sec\n", L, sps);
    }
    return 0;
}
