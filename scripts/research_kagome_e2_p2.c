/*
 * scripts/research_kagome_e2_p2.c
 *
 * Find the SECOND partner of the (Γ, E_2, Sz=1/2) doublet on the L=3
 * PBC kagome AFM Heisenberg lattice via orthogonal-projection penalty
 * Lanczos.
 *
 * Method: load the first partner ψ_p1 from disk, then run projecting
 * Lanczos on the modified Hamiltonian
 *
 *     H' = H + λ · |ψ_p1⟩⟨ψ_p1|     (λ = 10 J ≫ band width)
 *
 * H' shifts ψ_p1's eigenvalue up by λ but leaves all states orthogonal
 * to ψ_p1 unchanged.  Restricted to the (Sz=1/2, Γ, E_2) projected
 * subspace, the lowest H' eigenstate is therefore the second doublet
 * partner ψ_p2 with E(p2) = E(p1) (degenerate doublet).
 *
 * Validation at output time:
 *   - |⟨p1 | p2⟩| < 1e-8   (orthogonality)
 *   - |E(p2) - E(p1)| < 1e-6 J   (doublet degeneracy)
 *
 * Build:  make IRREP_ENABLE=1 OPENMP=1 research_kagome_e2_p2
 * Run:    env OMP_NUM_THREADS=14 ./build/research_kagome_e2_p2 \
 *           research_data/eigvecs/kagome_3x3_E2_sz1_eigvec.bin \
 *           research_data/eigvecs/kagome_3x3_E2_p2_sz1_eigvec.bin
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

/* ---- Hamiltonian context with orthogonality penalty -------------- */
typedef struct {
    int Lx, Ly, N;
    double J;
    const double *psi_p1;     /* the first partner — penalised against */
    double lambda;            /* penalty strength (J units) */
} kheis_penal_ctx_t;

static void kheis_matvec_penal(const double *in, double *out, long dim, void *ud) {
    kheis_penal_ctx_t *c = (kheis_penal_ctx_t *)ud;
    int Lx = c->Lx, Ly = c->Ly;
    double J = c->J;
    /* Step 1: H · in (same kernel as research_kagome_sz_spatial.c) */
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
    /* Step 2: penalty term  out += λ · ⟨p1|in⟩ · p1 */
    double overlap = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:overlap) schedule(static)
    #endif
    for (long s = 0; s < dim; s++) overlap += c->psi_p1[s] * in[s];
    double w = c->lambda * overlap;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (long s = 0; s < dim; s++) out[s] += w * c->psi_p1[s];
}

/* ---- Combined Sz=target + spatial-irrep + orthogonal projector --- */
typedef struct {
    int popcount_target;
    int N;
    int G;
    const int *perm;
    const double *chars;
    const double *psi_p1;     /* project orthogonal to this each step */
    long dim;
} sz_spatial_orth_ctx_t;

static void sz_spatial_orth_project(double *vec, long dim, void *user) {
    sz_spatial_orth_ctx_t *pc = (sz_spatial_orth_ctx_t *)user;
    int target = pc->popcount_target;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (long s = 0; s < dim; s++) {
        if (__builtin_popcountll((unsigned long long)s) != target) {
            vec[s] = 0.0;
        }
    }
    nqs_kagome_p6m_project_inplace(vec, pc->N, pc->G, pc->perm, pc->chars);
    /* Project orthogonal to p1: vec ← vec − ⟨p1|vec⟩ p1.  This is
     * belt-and-braces alongside the matvec penalty — Lanczos full
     * reorthogonalisation against p1 inside the projector keeps the
     * Krylov subspace clean even as round-off accumulates. */
    double overlap = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:overlap) schedule(static)
    #endif
    for (long s = 0; s < dim; s++) overlap += pc->psi_p1[s] * vec[s];
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (long s = 0; s < dim; s++) vec[s] -= overlap * pc->psi_p1[s];
}

/* Compute total spin sum rule  ⟨S²⟩ via diagonal + off-diagonal. */
static double total_S2_from_psi(const double *psi, long dim, int N) {
    double zz_off = 0.0;
    double xy_off = 0.0;
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

static int load_eigvec_real(const char *path, int *out_N, int *out_ir,
                              double *out_E0, double **out_psi, long *out_dim) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int header[4]; double meta[2];
    if (fread(header, sizeof(int), 4, f) != 4) { fclose(f); return -1; }
    if (fread(meta, sizeof(double), 2, f) != 2) { fclose(f); return -1; }
    int N = header[0], ir = header[2];
    long dim = (long)meta[1];
    if (dim != (1L << N)) { fclose(f); return -1; }
    double *psi = malloc((size_t)dim * sizeof(double));
    if (!psi) { fclose(f); return -1; }
    if ((long)fread(psi, sizeof(double), (size_t)dim, f) != dim) {
        free(psi); fclose(f); return -1;
    }
    fclose(f);
    *out_N = N; *out_ir = ir; *out_E0 = meta[0];
    *out_psi = psi; *out_dim = dim;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <p1_eigvec_path> <p2_eigvec_save_path> "
                "[L=3] [iters=200] [lambda=10.0]\n", argv[0]);
        return 1;
    }
    const char *p1_path = argv[1];
    const char *p2_path = argv[2];
    int L = (argc > 3) ? atoi(argv[3]) : 3;
    int max_iters = (argc > 4) ? atoi(argv[4]) : 200;
    double lambda = (argc > 5) ? atof(argv[5]) : 10.0;
    int N = 3 * L * L;
    long dim = 1L << N;
    int Sz_2x = 1;     /* hard-coded: this driver targets the E_2 Sz=1/2 sector */
    int popcount_target = (N - Sz_2x) / 2;

    fprintf(stderr,
            "# E_2 partner-2 finder, L=%d, N=%d, dim=%ld, λ=%.2f J\n",
            L, N, dim, lambda);

    /* Load p1. */
    int Np1, irp1; double E0_p1; double *psi_p1 = NULL; long dim_p1;
    if (load_eigvec_real(p1_path, &Np1, &irp1, &E0_p1, &psi_p1, &dim_p1) != 0) {
        fprintf(stderr, "FAIL: cannot load p1 from %s\n", p1_path);
        return 1;
    }
    if (Np1 != N || dim_p1 != dim) {
        fprintf(stderr, "FAIL: p1 N/dim mismatch (got N=%d dim=%ld)\n",
                Np1, dim_p1);
        return 1;
    }
    fprintf(stderr, "# loaded p1 from %s  (irrep_code=%d, E_0=%.10f J)\n",
            p1_path, irp1, E0_p1);

    /* Sanity check ⟨p1|p1⟩ ≈ 1. */
    double n2_p1 = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:n2_p1) schedule(static)
    #endif
    for (long s = 0; s < dim; s++) n2_p1 += psi_p1[s] * psi_p1[s];
    fprintf(stderr, "# ⟨p1|p1⟩ = %.10f (should be 1)\n", n2_p1);
    if (fabs(n2_p1 - 1.0) > 1e-8) {
        double inv = 1.0 / sqrt(n2_p1);
        for (long s = 0; s < dim; s++) psi_p1[s] *= inv;
        fprintf(stderr, "# re-normalised p1 in place\n");
    }

    /* Build the (Γ, E_2) symmetry projector. */
    int *perm = NULL; double *chars = NULL; int G = 0;
    if (nqs_kagome_p6m_perm_irrep(L, NQS_SYMPROJ_KAGOME_GAMMA_E2,
                                    &perm, &chars, &G) != 0) {
        fprintf(stderr, "FAIL: perm_irrep build failed\n");
        return 1;
    }

    sz_spatial_orth_ctx_t pc = {
        .popcount_target = popcount_target,
        .N = N, .G = G, .perm = perm, .chars = chars,
        .psi_p1 = psi_p1, .dim = dim
    };
    kheis_penal_ctx_t hctx = {
        .Lx = L, .Ly = L, .N = N, .J = 1.0,
        .psi_p1 = psi_p1, .lambda = lambda
    };

    /* Build a projected seed orthogonal to p1. */
    double *seed = malloc((size_t)dim * sizeof(double));
    if (!seed) return 1;
    unsigned long long rng = 0xA5A5A5A5A5A5A5A5ULL ^ (unsigned long long)dim;
    for (long i = 0; i < dim; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        double u = (double)(rng >> 11) / 9007199254740992.0;
        seed[i] = u - 0.5;
    }
    sz_spatial_orth_project(seed, dim, &pc);
    double n2 = 0.0;
    for (long i = 0; i < dim; i++) n2 += seed[i] * seed[i];
    if (n2 <= 0) {
        fprintf(stderr,
                "FAIL: orth-projected seed is zero — p1 may already span (Γ,E_2,Sz=1/2)?\n");
        return 1;
    }
    double inv = 1.0 / sqrt(n2);
    for (long i = 0; i < dim; i++) seed[i] *= inv;
    fprintf(stderr, "# orth-projected-seed norm² before normalisation = %.6e\n", n2);

    /* Lanczos with eigenvector. */
    double *psi_p2 = malloc((size_t)dim * sizeof(double));
    if (!psi_p2) return 1;
    lanczos_result_t lr = (lanczos_result_t){0};
    double e0_penal = 0.0;
    double t0 = wall_seconds();
    int rc = lanczos_smallest_projected_lean_eigvec(
        kheis_matvec_penal, &hctx, dim,
        max_iters, 1e-10, seed,
        sz_spatial_orth_project, &pc,
        &e0_penal, psi_p2, &lr);
    double dt = wall_seconds() - t0;
    if (rc != 0) {
        fprintf(stderr, "Lanczos failed (rc=%d)\n", rc);
        return 1;
    }

    /* The penalised Lanczos energy includes λ⟨ψ_p2|p1⟩²; if p2 is
     * truly orthogonal to p1, the penalty contribution is zero and
     * e0_penal == E(p2).  Compute the bare energy ⟨p2|H|p2⟩ as the
     * canonical reported value. */
    kheis_penal_ctx_t bare = hctx; bare.lambda = 0.0;
    double *Hp2 = malloc((size_t)dim * sizeof(double));
    kheis_matvec_penal(psi_p2, Hp2, dim, &bare);
    double e0_bare = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:e0_bare) schedule(static)
    #endif
    for (long s = 0; s < dim; s++) e0_bare += psi_p2[s] * Hp2[s];
    free(Hp2);

    /* Orthogonality: ⟨p1|p2⟩. */
    double overlap = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:overlap) schedule(static)
    #endif
    for (long s = 0; s < dim; s++) overlap += psi_p1[s] * psi_p2[s];

    fprintf(stderr,
            "# Lanczos done: e0_penal=%.10f, e0_bare=%.10f, iters=%d, conv=%d, wall=%.1f s\n",
            e0_penal, e0_bare, lr.iterations, lr.converged, dt);
    fprintf(stderr, "# orthogonality ⟨p1|p2⟩ = %.3e (should be < 1e-8)\n", overlap);
    fprintf(stderr, "# doublet check |E(p2) - E(p1)| = %.3e J (should be < 1e-6)\n",
            fabs(e0_bare - E0_p1));

    /* Total spin sum rule. */
    double S2 = total_S2_from_psi(psi_p2, dim, N);
    double S_total = -0.5 + sqrt(0.25 + S2);
    fprintf(stderr, "# total S² = %.6f → S_total = %.6f (expect 0.5)\n", S2, S_total);

    /* Save eigvec. */
    {
        FILE *f = fopen(p2_path, "wb");
        if (!f) {
            fprintf(stderr, "# ERROR: cannot open %s for write: %s\n",
                    p2_path, strerror(errno));
        } else {
            int header[4] = {N, L, irp1, (int)lr.iterations};
            double meta[2] = {e0_bare, (double)dim};
            int ok = 1;
            if (fwrite(header, sizeof(int), 4, f) != 4) { ok = 0; }
            if (ok && fwrite(meta, sizeof(double), 2, f) != 2) { ok = 0; }
            if (ok) {
                size_t want = (size_t)dim;
                size_t got = fwrite(psi_p2, sizeof(double), want, f);
                if (got != want) {
                    fprintf(stderr,
                            "# ERROR: eigvec fwrite truncated: %zu/%zu doubles (%s)\n",
                            got, want, strerror(errno));
                    ok = 0;
                }
            }
            if (fflush(f) != 0) { ok = 0; }
            if (fclose(f) != 0) { ok = 0; }
            if (ok) {
                fprintf(stderr, "# p2 eigvec saved to %s (%.2f GB)\n",
                        p2_path, (double)dim * 8.0 / 1e9);
            }
        }
    }

    printf("{\n");
    printf("  \"system\": {\"L\": %d, \"N\": %d, \"sector\": \"(Sz=1/2, Γ, E_2 partner 2)\", \"dim\": %ld, \"popcount\": %d},\n",
           L, N, dim, popcount_target);
    printf("  \"E0_p1_loaded\": %.12f,\n", E0_p1);
    printf("  \"E0_p2_bare\":   %.12f,\n", e0_bare);
    printf("  \"E0_p2_penal\":  %.12f,\n", e0_penal);
    printf("  \"penalty_lambda\": %.4f,\n", lambda);
    printf("  \"orthogonality_p1_p2\": %.6e,\n", overlap);
    printf("  \"doublet_split\": %.6e,\n", fabs(e0_bare - E0_p1));
    printf("  \"lanczos\": {\"iters\": %d, \"converged\": %d, \"wall_s\": %.1f},\n",
           lr.iterations, lr.converged, dt);
    printf("  \"total_spin\": {\"S_squared\": %.10f, \"S_total\": %.10f}\n",
           S2, S_total);
    printf("}\n");

    free(perm); free(chars); free(seed); free(psi_p1); free(psi_p2);
    return 0;
}
