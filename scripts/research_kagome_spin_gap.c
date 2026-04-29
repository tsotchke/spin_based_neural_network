/*
 * scripts/research_kagome_spin_gap.c
 *
 * TRUE spin gap of the kagome AFM Heisenberg ground state via Sz-
 * projection.  See the .h docstring for the methodology and the
 * companion synthesis tool.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "nqs/nqs_lanczos.h"
#include "mps/lanczos.h"

static double wall_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

typedef struct {
    int target_popcount;
} sz_proj_ctx_t;

static void sz_project(double *vec, long dim, void *user) {
    sz_proj_ctx_t *pc = (sz_proj_ctx_t *)user;
    int target = pc->target_popcount;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (long s = 0; s < dim; s++) {
        if (__builtin_popcountll((unsigned long long)s) != target) {
            vec[s] = 0.0;
        }
    }
}

typedef struct {
    int Lx_cells, Ly_cells, N;
    double J;
    int pbc;
} heis_ctx_t;

/* Inline kagome H matvec — moved out of main() because Apple Clang
 * doesn't support nested function definitions.  Identical structure
 * to kagome_heis_matvec in src/nqs/nqs_lanczos.c. */
static void kagome_matvec_inline(const double *in, double *out, long dim, void *ud) {
    heis_ctx_t *c = (heis_ctx_t *)ud;
    int Lx = c->Lx_cells, Ly = c->Ly_cells;
    double J = c->J;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (long s = 0; s < dim; s++) {
        double e_diag = 0.0;
        for (int cx = 0; cx < Lx; cx++) for (int cy = 0; cy < Ly; cy++) {
            int A = 3*(cx*Ly+cy)+0, B = 3*(cx*Ly+cy)+1, C = 3*(cx*Ly+cy)+2;
            int sA = ((s >> A) & 1) ? -1 : 1;
            int sB = ((s >> B) & 1) ? -1 : 1;
            int sC = ((s >> C) & 1) ? -1 : 1;
            e_diag += 0.25 * J * (sA*sB + sA*sC + sB*sC);
            int cxm = (cx-1+Lx)%Lx, cym = (cy-1+Ly)%Ly;
            int Bm = 3*(cxm*Ly+cy)+1, Cm = 3*(cx*Ly+cym)+2;
            int sBm = ((s >> Bm) & 1) ? -1 : 1;
            int sCm = ((s >> Cm) & 1) ? -1 : 1;
            e_diag += 0.25 * J * (sA*sBm + sA*sCm + sBm*sCm);
        }
        double y = e_diag * in[s];
        for (int cx = 0; cx < Lx; cx++) for (int cy = 0; cy < Ly; cy++) {
            int A = 3*(cx*Ly+cy)+0, B = 3*(cx*Ly+cy)+1, C = 3*(cx*Ly+cy)+2;
            int up_bonds[3][2] = {{A,B},{A,C},{B,C}};
            for (int b = 0; b < 3; b++) {
                int u = up_bonds[b][0], v = up_bonds[b][1];
                int su = ((s >> u) & 1) ? -1 : 1;
                int sv = ((s >> v) & 1) ? -1 : 1;
                if (su != sv) y += 0.5 * J * in[s ^ (1L<<u) ^ (1L<<v)];
            }
            int cxm = (cx-1+Lx)%Lx, cym = (cy-1+Ly)%Ly;
            int Bm = 3*(cxm*Ly+cy)+1, Cm = 3*(cx*Ly+cym)+2;
            int dn_bonds[3][2] = {{A,Bm},{A,Cm},{Bm,Cm}};
            for (int b = 0; b < 3; b++) {
                int u = dn_bonds[b][0], v = dn_bonds[b][1];
                int su = ((s >> u) & 1) ? -1 : 1;
                int sv = ((s >> v) & 1) ? -1 : 1;
                if (su != sv) y += 0.5 * J * in[s ^ (1L<<u) ^ (1L<<v)];
            }
        }
        out[s] = y;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s L Sz_target_2x [iters]\n", argv[0]);
        return 1;
    }
    int L = atoi(argv[1]);
    int Sz_2x = atoi(argv[2]);
    int max_iters = (argc > 3) ? atoi(argv[3]) : 200;

    int N = 3 * L * L;
    long dim = 1L << N;

    if ((Sz_2x % 2) != (N % 2)) {
        fprintf(stderr, "Sz_2x=%d incompatible with N=%d (parity mismatch)\n",
                Sz_2x, N);
        return 1;
    }
    int popcount_target = (N - Sz_2x) / 2;

    long sector_dim = 1;
    for (int k = 0; k < popcount_target; k++) {
        sector_dim = sector_dim * (N - k) / (k + 1);
    }
    fprintf(stderr, "# kagome %d×%d PBC, Sz=%d/2 sector dim = %ld (%.2f%% of total)\n",
            L, L, Sz_2x, sector_dim, 100.0 * sector_dim / dim);

    sz_proj_ctx_t pc = { .target_popcount = popcount_target };
    heis_ctx_t ctx = { .Lx_cells = L, .Ly_cells = L, .N = N, .J = 1.0, .pbc = 1 };

    double *seed = malloc((size_t)dim * sizeof(double));
    if (!seed) return 1;
    unsigned long long rng = 0xA5A5A5A5A5A5A5A5ULL ^ (unsigned long long)dim;
    for (long i = 0; i < dim; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        double u = (double)(rng >> 11) / 9007199254740992.0;
        seed[i] = u - 0.5;
    }
    sz_project(seed, dim, &pc);
    double n2 = 0.0;
    for (long i = 0; i < dim; i++) n2 += seed[i] * seed[i];
    if (n2 > 0) { double inv = 1.0 / sqrt(n2);
                   for (long i = 0; i < dim; i++) seed[i] *= inv; }

    double t0 = wall_seconds();
    lanczos_result_t lr = (lanczos_result_t){0};
    double e0 = 0.0;
    int rc = lanczos_smallest_projected_lean(
        kagome_matvec_inline, &ctx, dim,
        max_iters, 1e-10, seed,
        sz_project, &pc,
        &e0, &lr);
    double dt = wall_seconds() - t0;
    if (rc != 0) { fprintf(stderr, "Lanczos failed\n"); return 1; }

    fprintf(stderr, "# E_0(Sz=%d/2) = %.10f, iters=%d, conv=%d, wall=%.1f s\n",
            Sz_2x, e0, lr.iterations, lr.converged, dt);
    printf("{\n");
    printf("  \"system\": {\"L\": %d, \"N\": %d, \"Sz_2x\": %d, \"popcount\": %d, \"sector_dim\": %ld},\n",
           L, N, Sz_2x, popcount_target, sector_dim);
    printf("  \"lanczos\": {\"E_0\": %.12f, \"iters\": %d, \"converged\": %d, \"wall_s\": %.1f}\n",
           e0, lr.iterations, lr.converged, dt);
    printf("}\n");
    free(seed);
    return 0;
}
