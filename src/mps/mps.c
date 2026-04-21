/*
 * src/mps/mps.c
 *
 * Dense-Hilbert-space ground-state solver that builds H·v matrix-free
 * and feeds it to the Lanczos eigensolver. Correct for spin-1/2 chains
 * up to num_sites ≈ 14 (2^14 = 16384 Hilbert dim).
 *
 * This is the reference solver that the full tensor-network DMRG in
 * v0.6+ will replace behind the same mps_ground_state_dense API.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "mps/mps.h"

typedef struct {
    int N;
    mps_hamiltonian_kind_t ham;
    double J, Gamma, Jz;
} mps_matvec_ctx_t;

static inline int bit(int state, int i) { return (state >> i) & 1; }
/* Spin convention: bit 0 → +1, bit 1 → -1.  s_i = 1 - 2 * bit(state, i). */

static void mps_tfim_matvec(const double *in, double *out, long dim, void *u) {
    mps_matvec_ctx_t *ctx = (mps_matvec_ctx_t *)u;
    int N = ctx->N;
    double J = ctx->J;
    double G = ctx->Gamma;
    for (long s = 0; s < dim; s++) {
        /* Diagonal: -J Σ σ^z σ^z on neighbouring bonds (open BC). */
        double diag = 0.0;
        for (int i = 0; i < N - 1; i++) {
            int si = 1 - 2 * bit((int)s, i);
            int sj = 1 - 2 * bit((int)s, i + 1);
            diag += -J * (double)(si * sj);
        }
        double acc = diag * in[s];
        /* Off-diagonal: -Γ σ^x per site — single-bit flips. */
        for (int i = 0; i < N; i++) {
            long s2 = s ^ (1L << i);
            acc += -G * in[s2];
        }
        out[s] = acc;
    }
}

/* Heisenberg:  H = J Σ_⟨ij⟩ (S^x_i S^x_j + S^y_i S^y_j + S^z_i S^z_j)
 *            = J Σ_⟨ij⟩ [ (1/2)(S^+_i S^-_j + S^-_i S^+_j) + S^z_i S^z_j ]
 *  with S^± flipping a bit and S^z = (1/2) σ^z (taking s_i = ±1 means
 *  S^z_i = s_i / 2). The XY term contributes J/2 for each antiparallel
 *  neighbour pair. */
static void mps_heisenberg_matvec(const double *in, double *out, long dim, void *u) {
    mps_matvec_ctx_t *ctx = (mps_matvec_ctx_t *)u;
    int N = ctx->N;
    double J = ctx->J;
    for (long s = 0; s < dim; s++) {
        double diag = 0.0;
        double offd = 0.0;
        for (int i = 0; i < N - 1; i++) {
            int si = 1 - 2 * bit((int)s, i);
            int sj = 1 - 2 * bit((int)s, i + 1);
            diag += 0.25 * J * (double)(si * sj);
            if (si != sj) {
                long s2 = s ^ (1L << i) ^ (1L << (i + 1));
                offd += 0.5 * J * in[s2];
            }
        }
        out[s] = diag * in[s] + offd;
    }
}

static void mps_xxz_matvec(const double *in, double *out, long dim, void *u) {
    mps_matvec_ctx_t *ctx = (mps_matvec_ctx_t *)u;
    int N = ctx->N;
    double J = ctx->J;
    double Jz = ctx->Jz;
    for (long s = 0; s < dim; s++) {
        double diag = 0.0;
        double offd = 0.0;
        for (int i = 0; i < N - 1; i++) {
            int si = 1 - 2 * bit((int)s, i);
            int sj = 1 - 2 * bit((int)s, i + 1);
            diag += 0.25 * Jz * (double)(si * sj);
            if (si != sj) {
                long s2 = s ^ (1L << i) ^ (1L << (i + 1));
                offd += 0.5 * J * in[s2];
            }
        }
        out[s] = diag * in[s] + offd;
    }
}

int mps_ground_state_dense(const mps_config_t *cfg,
                           double *out_energy,
                           double *out_state,
                           lanczos_result_t *out_info) {
    if (!cfg || !out_energy) return -1;
    if (cfg->num_sites <= 0 || cfg->num_sites > 14) return -1;

    long dim = 1L << cfg->num_sites;
    mps_matvec_ctx_t ctx = {
        .N = cfg->num_sites,
        .ham = cfg->ham,
        .J = cfg->J,
        .Gamma = cfg->Gamma,
        .Jz = cfg->Jz
    };
    lanczos_matvec_fn_t fn = mps_tfim_matvec;
    if (cfg->ham == MPS_HAM_HEISENBERG) fn = mps_heisenberg_matvec;
    else if (cfg->ham == MPS_HAM_XXZ)   fn = mps_xxz_matvec;

    lanczos_result_t res;
    int rc = lanczos_smallest(fn, &ctx, dim,
                               cfg->lanczos_max_iters, cfg->lanczos_tol,
                               out_state, &res);
    if (rc != 0) return rc;

    *out_energy = res.eigenvalue;
    if (out_info) *out_info = res;
    return 0;
}
