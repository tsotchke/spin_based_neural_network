/*
 * src/nqs/nqs_lanczos.c
 *
 * Dense exact materialisation + Lanczos post-processing of a trained
 * NQS ansatz (pillar P2.6). For small N only (dim = 2^N fits in RAM).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nqs/nqs_lanczos.h"

/* Map a bit state (0..2^N-1) to an int spin vector of ±1.
 *   bit 0 in state → site index 0; bit = 1 → spin = -1. */
static void state_to_spins(long state, int N, int *out) {
    for (int i = 0; i < N; i++) {
        out[i] = ((state >> i) & 1) ? -1 : +1;
    }
}

int nqs_materialise_state_with_cb(nqs_log_amp_fn_t log_amp, void *user,
                                   int Lx, int Ly,
                                   double **out_psi, long *out_dim) {
    if (!log_amp || !out_psi || !out_dim || Lx <= 0 || Ly <= 0) return -1;
    int N = Lx * Ly;
    if (N > 24) return -1;
    long dim = 1L << N;
    double *psi = malloc((size_t)dim * sizeof(double));
    if (!psi) return -1;
    int *spins = malloc((size_t)N * sizeof(int));
    if (!spins) { free(psi); return -1; }
    double lp_max = -INFINITY;
    double *lp = malloc((size_t)dim * sizeof(double));
    double *arg_arr = malloc((size_t)dim * sizeof(double));
    if (!lp || !arg_arr) { free(psi); free(spins); free(lp); free(arg_arr); return -1; }
    for (long s = 0; s < dim; s++) {
        state_to_spins(s, N, spins);
        double lp_s, arg_s;
        log_amp(spins, N, user, &lp_s, &arg_s);
        lp[s]      = lp_s;
        arg_arr[s] = arg_s;
        if (lp_s > lp_max) lp_max = lp_s;
    }
    double norm2 = 0.0;
    for (long s = 0; s < dim; s++) {
        /* For real wavefunctions arg ∈ {0, π}; cos(arg) = ±1. */
        psi[s] = cos(arg_arr[s]) * exp(lp[s] - lp_max);
        norm2 += psi[s] * psi[s];
    }
    double inv = norm2 > 0 ? 1.0 / sqrt(norm2) : 1.0;
    for (long s = 0; s < dim; s++) psi[s] *= inv;
    free(lp); free(arg_arr); free(spins);
    *out_psi = psi;
    *out_dim = dim;
    return 0;
}

int nqs_materialise_state(nqs_ansatz_t *a, int Lx, int Ly,
                          double **out_psi, long *out_dim) {
    if (!a) return -1;
    return nqs_materialise_state_with_cb(nqs_ansatz_log_amp, a, Lx, Ly,
                                          out_psi, out_dim);
}

/* Build the dense TFIM Hamiltonian matrix in the computational basis.
 *   H = -J Σ_<ij> σ^z_i σ^z_j - Γ Σ_i σ^x_i
 * with open boundary conditions on an (Lx × Ly) lattice. */
typedef struct {
    int Lx, Ly, N;
    double J, Gamma;
} tfim_ctx_t;

static double tfim_diag_energy(long state, const tfim_ctx_t *ctx) {
    double e = 0.0;
    for (int x = 0; x < ctx->Lx; x++) {
        for (int y = 0; y < ctx->Ly; y++) {
            int idx = x * ctx->Ly + y;
            int sxy = ((state >> idx) & 1) ? -1 : +1;
            if (x + 1 < ctx->Lx) {
                int j = (x + 1) * ctx->Ly + y;
                int sj = ((state >> j) & 1) ? -1 : +1;
                e += -ctx->J * (double)(sxy * sj);
            }
            if (y + 1 < ctx->Ly) {
                int j = x * ctx->Ly + (y + 1);
                int sj = ((state >> j) & 1) ? -1 : +1;
                e += -ctx->J * (double)(sxy * sj);
            }
        }
    }
    return e;
}

static void tfim_matvec(const double *in, double *out, long dim, void *ud) {
    tfim_ctx_t *ctx = (tfim_ctx_t *)ud;
    int N = ctx->N;
    for (long s = 0; s < dim; s++) {
        double y = tfim_diag_energy(s, ctx) * in[s];
        for (int i = 0; i < N; i++) {
            long s2 = s ^ (1L << i);
            y += -ctx->Gamma * in[s2];
        }
        out[s] = y;
    }
}

int nqs_exact_energy_tfim(nqs_ansatz_t *a, int Lx, int Ly,
                           double J, double Gamma, double *out_energy) {
    if (!a || !out_energy) return -1;
    double *psi;
    long dim;
    if (nqs_materialise_state(a, Lx, Ly, &psi, &dim) != 0) return -1;
    tfim_ctx_t ctx = { .Lx = Lx, .Ly = Ly, .N = Lx * Ly, .J = J, .Gamma = Gamma };
    double *Hpsi = malloc((size_t)dim * sizeof(double));
    if (!Hpsi) { free(psi); return -1; }
    tfim_matvec(psi, Hpsi, dim, &ctx);
    double num = 0.0, den = 0.0;
    for (long s = 0; s < dim; s++) { num += psi[s] * Hpsi[s]; den += psi[s] * psi[s]; }
    *out_energy = num / den;
    free(psi); free(Hpsi);
    return 0;
}

int nqs_lanczos_refine_tfim(nqs_ansatz_t *a, int Lx, int Ly,
                             double J, double Gamma,
                             int max_iters, double tol,
                             double *out_eigenvalue,
                             double *out_eigenvector,
                             lanczos_result_t *out_result) {
    if (!a || !out_eigenvalue) return -1;
    tfim_ctx_t ctx = { .Lx = Lx, .Ly = Ly, .N = Lx * Ly, .J = J, .Gamma = Gamma };
    long dim = 1L << ctx.N;
    int rc = lanczos_smallest(tfim_matvec, &ctx, dim,
                               max_iters, tol,
                               out_eigenvector, out_result);
    if (rc == 0 && out_result) *out_eigenvalue = out_result->eigenvalue;
    return rc;
}

/* Heisenberg XXZ:
 *   H = J Σ_<ij> ½(S^+_i S^-_j + S^-_i S^+_j) + Jz Σ_<ij> S^z_i S^z_j
 * with S = σ/2. Open boundary conditions on an Lx × Ly lattice. */
typedef struct {
    int Lx, Ly, N;
    double J, Jz;
} heis_ctx_t;

static double heis_diag_energy(long state, const heis_ctx_t *ctx) {
    double e = 0.0;
    for (int x = 0; x < ctx->Lx; x++) {
        for (int y = 0; y < ctx->Ly; y++) {
            int idx = x * ctx->Ly + y;
            int sxy = ((state >> idx) & 1) ? -1 : +1;
            if (x + 1 < ctx->Lx) {
                int j = (x + 1) * ctx->Ly + y;
                int sj = ((state >> j) & 1) ? -1 : +1;
                e += 0.25 * ctx->Jz * (double)(sxy * sj);
            }
            if (y + 1 < ctx->Ly) {
                int j = x * ctx->Ly + (y + 1);
                int sj = ((state >> j) & 1) ? -1 : +1;
                e += 0.25 * ctx->Jz * (double)(sxy * sj);
            }
        }
    }
    return e;
}

static void heis_matvec(const double *in, double *out, long dim, void *ud) {
    heis_ctx_t *ctx = (heis_ctx_t *)ud;
    int Lx = ctx->Lx, Ly = ctx->Ly;
    for (long s = 0; s < dim; s++) {
        double y = heis_diag_energy(s, ctx) * in[s];
        /* Off-diagonal S+S-/S-S+ hopping over each bond. */
        for (int x = 0; x < Lx; x++) {
            for (int yy = 0; yy < Ly; yy++) {
                int a = x * Ly + yy;
                int neighbors[2];
                int nb = 0;
                if (x + 1 < Lx) neighbors[nb++] = (x + 1) * Ly + yy;
                if (yy + 1 < Ly) neighbors[nb++] = x * Ly + (yy + 1);
                int s_a = ((s >> a) & 1) ? -1 : +1;
                for (int k = 0; k < nb; k++) {
                    int b = neighbors[k];
                    int s_b = ((s >> b) & 1) ? -1 : +1;
                    if (s_a == -s_b) {
                        long s2 = s ^ (1L << a) ^ (1L << b);
                        y += 0.5 * ctx->J * in[s2];
                    }
                }
            }
        }
        out[s] = y;
    }
}

int nqs_exact_energy_heisenberg(nqs_ansatz_t *a, int Lx, int Ly,
                                 double J, double Jz, double *out_energy) {
    if (!a || !out_energy) return -1;
    double *psi;
    long dim;
    if (nqs_materialise_state(a, Lx, Ly, &psi, &dim) != 0) return -1;
    heis_ctx_t ctx = { .Lx = Lx, .Ly = Ly, .N = Lx * Ly, .J = J, .Jz = Jz };
    double *Hpsi = malloc((size_t)dim * sizeof(double));
    if (!Hpsi) { free(psi); return -1; }
    heis_matvec(psi, Hpsi, dim, &ctx);
    double num = 0.0, den = 0.0;
    for (long s = 0; s < dim; s++) { num += psi[s] * Hpsi[s]; den += psi[s] * psi[s]; }
    *out_energy = num / den;
    free(psi); free(Hpsi);
    return 0;
}

int nqs_lanczos_refine_heisenberg(nqs_ansatz_t *a, int Lx, int Ly,
                                   double J, double Jz,
                                   int max_iters, double tol,
                                   double *out_eigenvalue,
                                   double *out_eigenvector,
                                   lanczos_result_t *out_result) {
    if (!a || !out_eigenvalue) return -1;
    heis_ctx_t ctx = { .Lx = Lx, .Ly = Ly, .N = Lx * Ly, .J = J, .Jz = Jz };
    long dim = 1L << ctx.N;
    int rc = lanczos_smallest(heis_matvec, &ctx, dim,
                               max_iters, tol,
                               out_eigenvector, out_result);
    if (rc == 0 && out_result) *out_eigenvalue = out_result->eigenvalue;
    return rc;
}
