/*
 * scripts/research_kagome_sz_spatial.c
 *
 * Joint Sz + spatial-irrep projected Lanczos for the kagome AFM
 * Heisenberg model on PBC L×L torus.
 *
 * Motivation: at L=3 PBC, the lowest state in the (Γ, B_2) C_6v
 * sector is the SPIN-3/2 multiplet at E_0 = -11.434 J — NOT a
 * singlet.  But Z_2 Toric Code topological order predicts 4 SINGLET
 * ground states on the torus.  To resolve whether the missing singlet
 * lives somewhere in B_2 (above the spin-3/2 state), we run Lanczos
 * with the COMBINED projector:
 *
 *     P = P_{Sz=0}  ∘  P_{Γ, irrep}
 *
 * Both projectors commute (Sz commutes with the spatial p6m action)
 * and are idempotent so applying them sequentially is correct.
 *
 * Build:  make OPENMP=1 research_kagome_sz_spatial
 * Run:    env OMP_NUM_THREADS=14 ./build/research_kagome_sz_spatial L irrep [iters [eigvec_path]]
 *
 *   irrep = 0..3 → A_1, A_2, B_1, B_2
 *
 * Output: JSON with E_0, S_total^2 (computed from psi), wall time.
 *
 * Companion: research_kagome_full_analysis.c does the same WITHOUT
 * Sz projection — use this tool to find the lowest singlet within
 * each spatial irrep.
 */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "nqs/nqs_symproj.h"
#include "mps/lanczos.h"

static double wall_seconds(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}

/* ---- Hamiltonian context (kagome PBC Heisenberg) ----------------- */
typedef struct {
    int Lx, Ly, N;
    double J;
} kheis_ctx_t;

static void kheis_matvec(const double *in, double *out, long dim, void *ud) {
    kheis_ctx_t *c = (kheis_ctx_t *)ud;
    int Lx = c->Lx, Ly = c->Ly;
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
            int up[3][2] = {{A,B},{A,C},{B,C}};
            for (int b = 0; b < 3; b++) {
                int u = up[b][0], v = up[b][1];
                int su = ((s >> u) & 1) ? -1 : 1;
                int sv = ((s >> v) & 1) ? -1 : 1;
                if (su != sv) y += 0.5 * J * in[s ^ (1L<<u) ^ (1L<<v)];
            }
            int cxm = (cx-1+Lx)%Lx, cym = (cy-1+Ly)%Ly;
            int Bm = 3*(cxm*Ly+cy)+1, Cm = 3*(cx*Ly+cym)+2;
            int dn[3][2] = {{A,Bm},{A,Cm},{Bm,Cm}};
            for (int b = 0; b < 3; b++) {
                int u = dn[b][0], v = dn[b][1];
                int su = ((s >> u) & 1) ? -1 : 1;
                int sv = ((s >> v) & 1) ? -1 : 1;
                if (su != sv) y += 0.5 * J * in[s ^ (1L<<u) ^ (1L<<v)];
            }
        }
        out[s] = y;
    }
}

/* ---- Combined Sz=target + spatial-irrep projector --------------- */
typedef struct {
    int popcount_target;
    int N;
    int G;
    const int *perm;
    const double *chars;
} sz_spatial_ctx_t;

static void sz_spatial_project(double *vec, long dim, void *user) {
    sz_spatial_ctx_t *pc = (sz_spatial_ctx_t *)user;

    /* Step 1: zero out non-target-Sz amplitudes. */
    int target = pc->popcount_target;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (long s = 0; s < dim; s++) {
        if (__builtin_popcountll((unsigned long long)s) != target) {
            vec[s] = 0.0;
        }
    }

    /* Step 2: in-place spatial-irrep projection. */
    nqs_kagome_p6m_project_inplace(vec, pc->N, pc->G, pc->perm, pc->chars);
}

/* Compute total spin sum rule  ⟨S²⟩ = N·(3/4) + 2·Σ_{i<j} ⟨S_i·S_j⟩
 * via diagonal + off-diagonal contributions on a real-amplitude state. */
static double total_S2_from_psi(const double *psi, long dim, int N) {
    /* diagonal: Σ_{i<j} ⟨S^z_i S^z_j⟩ */
    double zz_off = 0.0;
    /* Σ_{i<j} (S^+_i S^-_j + S^-_i S^+_j) — bit flips */
    double xy_off = 0.0;
    /* Use compute pairs from i, j in [0, N) */
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            double zz = 0.0;
            double xy = 0.0;
            long mask_ij = (1L<<i) | (1L<<j);
            #ifdef _OPENMP
            #pragma omp parallel for reduction(+:zz,xy) schedule(static)
            #endif
            for (long s = 0; s < dim; s++) {
                int si = ((s >> i) & 1) ? -1 : 1;
                int sj = ((s >> j) & 1) ? -1 : 1;
                zz += 0.25 * (double)(si*sj) * psi[s] * psi[s];
                if (si != sj) {
                    long s2 = s ^ mask_ij;
                    if (s < s2) xy += psi[s] * psi[s2];
                }
            }
            zz_off += zz;
            xy_off += xy;
        }
    }
    return N * 0.75 + 2.0 * (zz_off + xy_off);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s L irrep Sz_2x [iters [eigvec_path]]\n", argv[0]);
        fprintf(stderr, "  irrep   = 0..5 (A_1, A_2, B_1, B_2, E_1, E_2)\n");
        fprintf(stderr, "  Sz_2x   = 2 × target Sz (must match parity of N).\n");
        fprintf(stderr, "             For N even use 0; for N odd the lowest-|Sz|\n");
        fprintf(stderr, "             allowed is ±1.\n");
        return 1;
    }
    int L = atoi(argv[1]);
    int ir = atoi(argv[2]);
    int Sz_2x = atoi(argv[3]);
    int max_iters = (argc > 4) ? atoi(argv[4]) : 200;
    const char *eigvec_path = (argc > 5) ? argv[5] : NULL;
    if (L <= 0 || ir < 0 || ir > 5) return 1;

    int N = 3 * L * L;
    long dim = 1L << N;
    if ((Sz_2x % 2) != (N % 2)) {
        fprintf(stderr, "Sz_2x=%d incompatible with N=%d (parity mismatch)\n",
                Sz_2x, N);
        return 1;
    }
    int popcount_target = (N - Sz_2x) / 2;
    if (popcount_target < 0 || popcount_target > N) {
        fprintf(stderr, "Sz_2x=%d out of range for N=%d\n", Sz_2x, N);
        return 1;
    }

    nqs_symproj_kagome_irrep_t irrep_codes[6] = {
        NQS_SYMPROJ_KAGOME_GAMMA_A1, NQS_SYMPROJ_KAGOME_GAMMA_A2,
        NQS_SYMPROJ_KAGOME_GAMMA_B1, NQS_SYMPROJ_KAGOME_GAMMA_B2,
        NQS_SYMPROJ_KAGOME_GAMMA_E1, NQS_SYMPROJ_KAGOME_GAMMA_E2
    };
    const char *irrep_names[6] = {"A_1", "A_2", "B_1", "B_2", "E_1", "E_2"};

    fprintf(stderr, "# kagome %d×%d PBC AFM, sector (Sz=%d/2, Γ, %s)  N=%d  dim=%ld  popcount=%d\n",
            L, L, Sz_2x, irrep_names[ir], N, dim, popcount_target);

    int *perm = NULL; double *chars = NULL; int G = 0;
    if (nqs_kagome_p6m_perm_irrep(L, irrep_codes[ir], &perm, &chars, &G) != 0) {
        fprintf(stderr, "FAIL: perm_irrep build failed\n");
        return 1;
    }

    sz_spatial_ctx_t pc = {
        .popcount_target = popcount_target,
        .N = N, .G = G, .perm = perm, .chars = chars
    };
    kheis_ctx_t hctx = { .Lx = L, .Ly = L, .N = N, .J = 1.0 };

    /* Build a Sz=0 + spatial-projected seed. */
    double *seed = malloc((size_t)dim * sizeof(double));
    if (!seed) return 1;
    unsigned long long rng = 0xC3C3C3C3C3C3C3C3ULL ^ (unsigned long long)dim
                            ^ ((unsigned long long)ir * 0x9E3779B97F4A7C15ULL);
    for (long i = 0; i < dim; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        double u = (double)(rng >> 11) / 9007199254740992.0;
        seed[i] = u - 0.5;
    }
    sz_spatial_project(seed, dim, &pc);
    double n2 = 0.0;
    for (long i = 0; i < dim; i++) n2 += seed[i] * seed[i];
    if (n2 <= 0) {
        fprintf(stderr, "FAIL: projected seed is zero — sector likely empty\n");
        return 1;
    }
    double inv = 1.0 / sqrt(n2);
    for (long i = 0; i < dim; i++) seed[i] *= inv;
    fprintf(stderr, "# projected-seed norm² before normalisation = %.6e\n", n2);

    /* Lanczos with eigenvector. */
    double *psi = malloc((size_t)dim * sizeof(double));
    if (!psi) return 1;
    lanczos_result_t lr = (lanczos_result_t){0};
    double e0 = 0.0;
    double t0 = wall_seconds();
    int rc = lanczos_smallest_projected_lean_eigvec(
        kheis_matvec, &hctx, dim,
        max_iters, 1e-10, seed,
        sz_spatial_project, &pc,
        &e0, psi, &lr);
    double dt = wall_seconds() - t0;
    if (rc != 0) {
        fprintf(stderr, "Lanczos failed (rc=%d)\n", rc);
        return 1;
    }
    fprintf(stderr, "# E_0 = %.10f  iters=%d  conv=%d  wall=%.1f s\n",
            e0, lr.iterations, lr.converged, dt);

    /* Compute S_total via the sum rule.  Slow but exact. */
    double S2 = total_S2_from_psi(psi, dim, N);
    double S_total = -0.5 + sqrt(0.25 + S2);
    fprintf(stderr, "# total S² = %.6f → S_total = %.6f\n", S2, S_total);

    /* Save eigvec if requested.  Check fwrite return values — we burned
     * a 6500-second Lanczos run earlier when /tmp filled mid-write and
     * fwrite silently returned a short count, leaving a 4 KB stub on
     * disk. */
    if (eigvec_path) {
        FILE *f = fopen(eigvec_path, "wb");
        if (!f) {
            fprintf(stderr, "# ERROR: cannot open %s for write: %s\n",
                    eigvec_path, strerror(errno));
        } else {
            int header[4] = {N, L, ir, (int)lr.iterations};
            double meta[2] = {e0, (double)dim};
            int ok = 1;
            if (fwrite(header, sizeof(int), 4, f) != 4) {
                fprintf(stderr, "# ERROR: header fwrite failed: %s\n",
                        strerror(errno));
                ok = 0;
            }
            if (ok && fwrite(meta, sizeof(double), 2, f) != 2) {
                fprintf(stderr, "# ERROR: meta fwrite failed: %s\n",
                        strerror(errno));
                ok = 0;
            }
            if (ok) {
                size_t want = (size_t)dim;
                size_t got = fwrite(psi, sizeof(double), want, f);
                if (got != want) {
                    fprintf(stderr,
                            "# ERROR: eigvec fwrite truncated: wrote %zu/%zu doubles (%s)\n",
                            got, want, strerror(errno));
                    ok = 0;
                }
            }
            if (fflush(f) != 0) {
                fprintf(stderr, "# ERROR: fflush failed: %s\n", strerror(errno));
                ok = 0;
            }
            if (fclose(f) != 0) {
                fprintf(stderr, "# ERROR: fclose failed: %s\n", strerror(errno));
                ok = 0;
            }
            if (ok) {
                fprintf(stderr, "# eigvec saved to %s (%.2f GB)\n",
                        eigvec_path, (double)dim * 8.0 / 1e9);
            } else {
                fprintf(stderr, "# WARNING: eigvec save failed; truncated file at %s should be deleted\n",
                        eigvec_path);
            }
        }
    }

    printf("{\n");
    printf("  \"system\": {\"L\": %d, \"N\": %d, \"sector\": \"(Sz=%d/2, Γ, %s)\", \"Sz_2x\": %d, \"dim\": %ld, \"popcount\": %d},\n",
           L, N, Sz_2x, irrep_names[ir], Sz_2x, dim, popcount_target);
    printf("  \"lanczos\": {\"E_0\": %.12f, \"iters\": %d, \"converged\": %d, \"wall_s\": %.1f},\n",
           e0, lr.iterations, lr.converged, dt);
    printf("  \"total_spin\": {\"S_squared\": %.10f, \"S_total\": %.10f},\n",
           S2, S_total);
    printf("  \"interpretation\": \"Lowest singlet (or lowest state) of the (Γ, %s) sector restricted to Sz=0 manifold.  Compares to the unprojected sector ground state from research_kagome_full_analysis.c.\"\n",
           irrep_names[ir]);
    printf("}\n");

    free(perm); free(chars); free(seed); free(psi);
    return 0;
}
