/*
 * tests/test_downstream_compat/generate_lattice_connectivity.c
 *
 * Emits `lattice_connectivity.json` — a vendorable artifact shared
 * with libirrep's `tests/test_downstream_compat/` so both repos
 * generate bit-identical periodic-lattice graphs from the same source
 * of truth.
 *
 * Contents: {4×4, 8×8, 16×16, 32×32, 64×64, 256×256} periodic grids,
 * each emitted as (num_nodes, num_edges, edge_src[], edge_dst[],
 * edge_vec[3·num_edges]) arrays. Uses torque_net_build_grid for
 * the enumeration so any future change to edge ordering drops into
 * both trees simultaneously.
 *
 * Build / run:
 *   gcc -Wall -std=c11 -Iinclude -O2 \
 *       tests/test_downstream_compat/generate_lattice_connectivity.c \
 *       src/equivariant_gnn/torque_net.c -lm -o /tmp/gen_lattice
 *   /tmp/gen_lattice tests/test_downstream_compat/lattice_connectivity.json
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "equivariant_gnn/torque_net.h"
typedef struct { int Lx; int Ly; const char *name; } shape_t;
static void emit_grid(FILE *f, const shape_t *sh, int last) {
    int *src, *dst; double *vec; int E;
    int rc = torque_net_build_grid(sh->Lx, sh->Ly, 1, &src, &dst, &vec, &E);
    if (rc != 0) { fprintf(stderr, "build_grid failed for %s\n", sh->name); exit(1); }
    int N = sh->Lx * sh->Ly;
    fprintf(f, "    \"%s\": {\n", sh->name);
    fprintf(f, "      \"Lx\": %d,\n", sh->Lx);
    fprintf(f, "      \"Ly\": %d,\n", sh->Ly);
    fprintf(f, "      \"periodic\": true,\n");
    fprintf(f, "      \"num_nodes\": %d,\n", N);
    fprintf(f, "      \"num_edges\": %d,\n", E);
    fprintf(f, "      \"edge_src\": [");
    for (int e = 0; e < E; e++) {
        if (e) fprintf(f, (e % 32 == 0) ? ",\n        " : ", ");
        fprintf(f, "%d", src[e]);
    }
    fprintf(f, "],\n");
    fprintf(f, "      \"edge_dst\": [");
    for (int e = 0; e < E; e++) {
        if (e) fprintf(f, (e % 32 == 0) ? ",\n        " : ", ");
        fprintf(f, "%d", dst[e]);
    }
    fprintf(f, "],\n");
    fprintf(f, "      \"edge_vec\": [");
    for (int e = 0; e < E; e++) {
        if (e) fprintf(f, (e % 16 == 0) ? ",\n        " : ", ");
        fprintf(f, "[%.17g, %.17g, %.17g]",
                vec[3*e], vec[3*e+1], vec[3*e+2]);
    }
    fprintf(f, "]\n");
    fprintf(f, "    }%s\n", last ? "" : ",");
    free(src); free(dst); free(vec);
}
int main(int argc, char **argv) {
    const char *outpath = argc > 1
        ? argv[1]
        : "tests/test_downstream_compat/lattice_connectivity.json";
    /* Log-spaced edge counts per libirrep 1.2 coordination.
     * Periodic 2D grid → 4N edges per lattice.
     *   4×4    N=16    E=64
     *   8×8    N=64    E=256
     *   16×16  N=256   E=1024
     *   32×32  N=1024  E=4096
     *   64×64  N=4096  E=16384
     *   256×256 N=65536 E=262144  (µMAG follow-up regime)
     */
    shape_t shapes[] = {
        {   4,   4, "4x4_periodic_64_edges"        },
        {   8,   8, "8x8_periodic_256_edges"       },
        {  16,  16, "16x16_periodic_1024_edges"    },
        {  32,  32, "32x32_periodic_4096_edges"    },
        {  64,  64, "64x64_periodic_16384_edges"   },
        { 256, 256, "256x256_periodic_262144_edges" }
    };
    int n_shapes = (int)(sizeof(shapes) / sizeof(shapes[0]));
    FILE *f = fopen(outpath, "w");
    if (!f) { fprintf(stderr, "cannot open %s for writing\n", outpath); return 1; }
    fprintf(f, "{\n");
    fprintf(f, "  \"_version\": \"1\",\n");
    fprintf(f, "  \"_note\": \"Periodic 2D square-lattice connectivity for libirrep 1.2 coordination. Shared between spin_based_neural_network and libirrep tests/test_downstream_compat/. Enumeration matches torque_net_build_grid(Lx, Ly, periodic=1).\",\n");
    fprintf(f, "  \"generator\": \"tests/test_downstream_compat/generate_lattice_connectivity.c\",\n");
    fprintf(f, "  \"shapes\": {\n");
    for (int i = 0; i < n_shapes; i++) {
        emit_grid(f, &shapes[i], i == n_shapes - 1);
    }
    fprintf(f, "  }\n");
    fprintf(f, "}\n");
    fclose(f);
    fprintf(stderr, "wrote %s with %d shapes\n", outpath, n_shapes);
    return 0;
}