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
#include "nqs/nqs_symproj.h"

/* Map a bit state (0..2^N-1) to an int spin vector of ±1.
 *   bit 0 in state → site index 0; bit = 1 → spin = -1. */
static void state_to_spins(long state, int N, int *out) {
    for (int i = 0; i < N; i++) {
        out[i] = ((state >> i) & 1) ? -1 : +1;
    }
}

/*
 * Check whether dropping Im(ψ) is safe for the materialisation path.
 *
 * The Lanczos seed is built from Re(ψ) = |ψ|·cos(arg ψ).  This is exact
 * for ground states that are real up to a global phase (stoquastic
 * Hamiltonians on bipartite lattices under Marshall).  For non-stoquastic
 * problems — kagome Heisenberg, kagome KH, frustrated J1-J2 — the ground
 * state has a non-trivial phase structure and the projection silently
 * discards physical information, producing wrong Lanczos eigenvalues.
 *
 * We compute ||Im(ψ)||² / ||ψ||² and warn when it exceeds a tolerance.
 * The routine still proceeds — callers see the warning on stderr and can
 * interpret their Lanczos result accordingly (or decide to fall back to
 * a complex Krylov implementation once one is available).
 *
 * Returns the imaginary fraction f = ||Im(ψ)||² / (||Re(ψ)||² + ||Im(ψ)||²).
 */
static double nqs_lanczos_check_stoquastic(const double *lp, const double *arg_arr,
                                           long dim, double lp_max)
{
    double norm_re2 = 0.0, norm_im2 = 0.0;
    for (long s = 0; s < dim; s++) {
        double a   = exp(lp[s] - lp_max);
        double rc  = cos(arg_arr[s]) * a;
        double ic  = sin(arg_arr[s]) * a;
        norm_re2  += rc * rc;
        norm_im2  += ic * ic;
    }
    double total = norm_re2 + norm_im2;
    double frac  = (total > 0.0) ? norm_im2 / total : 0.0;
    if (frac > 1e-6) {
        fprintf(stderr,
                "nqs_lanczos: WARNING — Im(ψ) accounts for %.3e of the "
                "wavefunction norm.  The materialisation path projects onto "
                "Re(ψ) only and is correct only for stoquastic ground states "
                "(bipartite Heisenberg + Marshall, TFIM).  Kagome / J1-J2 / "
                "frustrated KH ground states are non-stoquastic; Lanczos "
                "eigenvalues returned below will be contaminated by phase "
                "projection.\n", frac);
    }
    return frac;
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
    /* Stoquasticity guard: warn if Im(ψ) carries significant weight. */
    (void)nqs_lanczos_check_stoquastic(lp, arg_arr, dim, lp_max);
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

/* Same as nqs_materialise_state_with_cb but takes an explicit site
 * count. Lets kagome (N = 3·Lx·Ly, i.e. not size_x·size_y) reuse the
 * materialisation path. Mirrors the cb variant above verbatim with
 * only the N = Lx·Ly line replaced. */
int nqs_materialise_state_with_cb_N(nqs_log_amp_fn_t log_amp, void *user,
                                     int N,
                                     double **out_psi, long *out_dim) {
    if (!log_amp || !out_psi || !out_dim || N <= 0 || N > 24) return -1;
    long dim = 1L << N;
    double *psi = malloc((size_t)dim * sizeof(double));
    if (!psi) return -1;
    int *spins = malloc((size_t)N * sizeof(int));
    if (!spins) { free(psi); return -1; }
    double lp_max = -INFINITY;
    double *lp      = malloc((size_t)dim * sizeof(double));
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
    (void)nqs_lanczos_check_stoquastic(lp, arg_arr, dim, lp_max);
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
    double *psi_seed = NULL; long pdim = 0;
    int rc_seed = nqs_materialise_state(a, Lx, Ly, &psi_seed, &pdim);
    int rc = (rc_seed == 0 && pdim == dim)
        ? lanczos_smallest_with_init(heis_matvec, &ctx, dim,
                                       max_iters, tol,
                                       psi_seed, out_eigenvector, out_result)
        : lanczos_smallest(heis_matvec, &ctx, dim,
                             max_iters, tol,
                             out_eigenvector, out_result);
    free(psi_seed);
    if (rc == 0 && out_result) *out_eigenvalue = out_result->eigenvalue;
    return rc;
}

/* ===== Kagome Heisenberg =================================================
 *
 *   H = J Σ_<ij> S_i · S_j
 *
 * Bonds follow the same up-triangle / down-triangle enumeration as
 * `local_energy_kagome_heisenberg` in `src/nqs/nqs_gradient.c`, so the
 * Lanczos Hamiltonian matches the VMC local-energy kernel by
 * construction.
 * =======================================================================*/

typedef struct {
    int Lx_cells, Ly_cells, N;
    double J;
    int pbc;
} kagome_heis_ctx_t;

static inline int kg_site(int cx, int cy, int sub, int Ly_cells) {
    return 3 * (cx * Ly_cells + cy) + sub;
}

static double kagome_heis_diag_energy(long state, const kagome_heis_ctx_t *ctx) {
    double e = 0.0;
    int Lx = ctx->Lx_cells, Ly = ctx->Ly_cells;
    double J = ctx->J;
    for (int cx = 0; cx < Lx; cx++) {
        for (int cy = 0; cy < Ly; cy++) {
            int A = kg_site(cx, cy, 0, Ly);
            int B = kg_site(cx, cy, 1, Ly);
            int C = kg_site(cx, cy, 2, Ly);
            int sA = ((state >> A) & 1) ? -1 : +1;
            int sB = ((state >> B) & 1) ? -1 : +1;
            int sC = ((state >> C) & 1) ? -1 : +1;
            e += 0.25 * J * (double)(sA * sB + sA * sC + sB * sC);

            int cxm, cym;
            if (ctx->pbc) {
                cxm = (cx - 1 + Lx) % Lx;
                cym = (cy - 1 + Ly) % Ly;
            } else if (cx == 0 || cy == 0) {
                continue;
            } else {
                cxm = cx - 1; cym = cy - 1;
            }
            int Bm = kg_site(cxm, cy, 1, Ly);
            int Cm = kg_site(cx, cym, 2, Ly);
            int sBm = ((state >> Bm) & 1) ? -1 : +1;
            int sCm = ((state >> Cm) & 1) ? -1 : +1;
            e += 0.25 * J * (double)(sA * sBm + sA * sCm + sBm * sCm);
        }
    }
    return e;
}

/* For each bond (u, v) on an opposite-spin configuration, S^+S^- + S^-S^+
 * flips both spins and contributes coefficient (1/2)·J to the off-
 * diagonal matrix element. Identical structure to the square-lattice
 * heis_matvec, just with kagome's bond list. */
static void kagome_heis_matvec(const double *in, double *out,
                                long dim, void *ud) {
    kagome_heis_ctx_t *ctx = (kagome_heis_ctx_t *)ud;
    int Lx = ctx->Lx_cells, Ly = ctx->Ly_cells;
    double J = ctx->J;
    int pbc = ctx->pbc;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (long s = 0; s < dim; s++) {
        double y = kagome_heis_diag_energy(s, ctx) * in[s];
        for (int cx = 0; cx < Lx; cx++) {
            for (int cy = 0; cy < Ly; cy++) {
                int A = kg_site(cx, cy, 0, Ly);
                int B = kg_site(cx, cy, 1, Ly);
                int C = kg_site(cx, cy, 2, Ly);
                int bonds_up[3][2] = { {A, B}, {A, C}, {B, C} };
                for (int b = 0; b < 3; b++) {
                    int u = bonds_up[b][0], v = bonds_up[b][1];
                    int su = ((s >> u) & 1) ? -1 : +1;
                    int sv = ((s >> v) & 1) ? -1 : +1;
                    if (su != sv) {
                        long s2 = s ^ (1L << u) ^ (1L << v);
                        y += 0.5 * J * in[s2];
                    }
                }

                int cxm, cym;
                if (pbc) {
                    cxm = (cx - 1 + Lx) % Lx;
                    cym = (cy - 1 + Ly) % Ly;
                } else if (cx == 0 || cy == 0) {
                    continue;
                } else {
                    cxm = cx - 1; cym = cy - 1;
                }
                int Bm = kg_site(cxm, cy, 1, Ly);
                int Cm = kg_site(cx, cym, 2, Ly);
                int bonds_dn[3][2] = { {A, Bm}, {A, Cm}, {Bm, Cm} };
                for (int b = 0; b < 3; b++) {
                    int u = bonds_dn[b][0], v = bonds_dn[b][1];
                    int su = ((s >> u) & 1) ? -1 : +1;
                    int sv = ((s >> v) & 1) ? -1 : +1;
                    if (su != sv) {
                        long s2 = s ^ (1L << u) ^ (1L << v);
                        y += 0.5 * J * in[s2];
                    }
                }
            }
        }
        out[s] = y;
    }
}

int nqs_exact_energy_kagome_heisenberg(nqs_ansatz_t *a,
                                        int Lx_cells, int Ly_cells,
                                        double J, int pbc,
                                        double *out_energy) {
    if (!a || !out_energy) return -1;
    int N = 3 * Lx_cells * Ly_cells;
    double *psi = NULL; long dim = 0;
    if (nqs_materialise_state_with_cb_N(nqs_ansatz_log_amp, a, N,
                                          &psi, &dim) != 0) return -1;
    kagome_heis_ctx_t ctx = { .Lx_cells = Lx_cells, .Ly_cells = Ly_cells,
                               .N = N, .J = J, .pbc = pbc };
    double *Hpsi = malloc((size_t)dim * sizeof(double));
    if (!Hpsi) { free(psi); return -1; }
    kagome_heis_matvec(psi, Hpsi, dim, &ctx);
    double num = 0.0, den = 0.0;
    for (long s = 0; s < dim; s++) { num += psi[s] * Hpsi[s]; den += psi[s] * psi[s]; }
    *out_energy = num / den;
    free(psi); free(Hpsi);
    return 0;
}

int nqs_exact_energy_kagome_heisenberg_with_cb(nqs_log_amp_fn_t log_amp,
                                                 void *user,
                                                 int Lx_cells, int Ly_cells,
                                                 double J, int pbc,
                                                 double *out_energy) {
    if (!log_amp || !out_energy) return -1;
    int N = 3 * Lx_cells * Ly_cells;
    double *psi = NULL; long dim = 0;
    if (nqs_materialise_state_with_cb_N(log_amp, user, N,
                                          &psi, &dim) != 0) return -1;
    kagome_heis_ctx_t ctx = { .Lx_cells = Lx_cells, .Ly_cells = Ly_cells,
                               .N = N, .J = J, .pbc = pbc };
    double *Hpsi = malloc((size_t)dim * sizeof(double));
    if (!Hpsi) { free(psi); return -1; }
    kagome_heis_matvec(psi, Hpsi, dim, &ctx);
    double num = 0.0, den = 0.0;
    for (long s = 0; s < dim; s++) { num += psi[s] * Hpsi[s]; den += psi[s] * psi[s]; }
    *out_energy = num / den;
    free(psi); free(Hpsi);
    return 0;
}

int nqs_lanczos_refine_kagome_heisenberg(nqs_ansatz_t *a,
                                          int Lx_cells, int Ly_cells,
                                          double J, int pbc,
                                          int max_iters, double tol,
                                          double *out_eigenvalue,
                                          double *out_eigenvector,
                                          lanczos_result_t *out_result) {
    if (!a || !out_eigenvalue) return -1;
    int N = 3 * Lx_cells * Ly_cells;
    kagome_heis_ctx_t ctx = { .Lx_cells = Lx_cells, .Ly_cells = Ly_cells,
                               .N = N, .J = J, .pbc = pbc };
    long dim = 1L << N;
    double *psi_seed = NULL; long pdim = 0;
    int rc_seed = nqs_materialise_state_with_cb_N(nqs_ansatz_log_amp, a, N,
                                                    &psi_seed, &pdim);
    int rc = (rc_seed == 0 && pdim == dim)
        ? lanczos_smallest_with_init(kagome_heis_matvec, &ctx, dim,
                                       max_iters, tol,
                                       psi_seed, out_eigenvector, out_result)
        : lanczos_smallest(kagome_heis_matvec, &ctx, dim,
                             max_iters, tol,
                             out_eigenvector, out_result);
    free(psi_seed);
    if (rc == 0 && out_result) *out_eigenvalue = out_result->eigenvalue;
    return rc;
}

int nqs_lanczos_refine_kagome_heisenberg_with_cb(nqs_log_amp_fn_t log_amp,
                                                  void *user,
                                                  int Lx_cells, int Ly_cells,
                                                  double J, int pbc,
                                                  int max_iters, double tol,
                                                  double *out_eigenvalue,
                                                  double *out_eigenvector,
                                                  lanczos_result_t *out_result) {
    if (!log_amp || !out_eigenvalue) return -1;
    int N = 3 * Lx_cells * Ly_cells;
    kagome_heis_ctx_t ctx = { .Lx_cells = Lx_cells, .Ly_cells = Ly_cells,
                               .N = N, .J = J, .pbc = pbc };
    long dim = 1L << N;
    double *psi_seed = NULL; long pdim = 0;
    int rc_seed = nqs_materialise_state_with_cb_N(log_amp, user, N,
                                                    &psi_seed, &pdim);
    int rc = (rc_seed == 0 && pdim == dim)
        ? lanczos_smallest_with_init(kagome_heis_matvec, &ctx, dim,
                                       max_iters, tol,
                                       psi_seed, out_eigenvector, out_result)
        : lanczos_smallest(kagome_heis_matvec, &ctx, dim,
                             max_iters, tol,
                             out_eigenvector, out_result);
    free(psi_seed);
    if (rc == 0 && out_result) *out_eigenvalue = out_result->eigenvalue;
    return rc;
}

int nqs_lanczos_k_lowest_kagome_heisenberg_with_cb(nqs_log_amp_fn_t log_amp,
                                                    void *user,
                                                    int Lx_cells, int Ly_cells,
                                                    double J, int pbc,
                                                    int max_iters, int k,
                                                    double *out_eigenvalues,
                                                    lanczos_result_t *out_result) {
    if (!log_amp || !out_eigenvalues || k <= 0) return -1;
    int N = 3 * Lx_cells * Ly_cells;
    kagome_heis_ctx_t ctx = { .Lx_cells = Lx_cells, .Ly_cells = Ly_cells,
                               .N = N, .J = J, .pbc = pbc };
    long dim = 1L << N;
    double *psi_seed = NULL; long pdim = 0;
    int rc_seed = nqs_materialise_state_with_cb_N(log_amp, user, N,
                                                    &psi_seed, &pdim);
    const double *seed = (rc_seed == 0 && pdim == dim) ? psi_seed : NULL;
    int rc = lanczos_k_smallest_with_init(kagome_heis_matvec, &ctx, dim,
                                            max_iters, seed, k,
                                            out_eigenvalues, out_result);
    free(psi_seed);
    return rc;
}

int nqs_lanczos_k_lowest_kagome_heisenberg(nqs_ansatz_t *a,
                                            int Lx_cells, int Ly_cells,
                                            double J, int pbc,
                                            int max_iters, int k,
                                            double *out_eigenvalues,
                                            lanczos_result_t *out_result) {
    if (!a || !out_eigenvalues || k <= 0) return -1;
    int N = 3 * Lx_cells * Ly_cells;
    kagome_heis_ctx_t ctx = { .Lx_cells = Lx_cells, .Ly_cells = Ly_cells,
                               .N = N, .J = J, .pbc = pbc };
    long dim = 1L << N;
    double *psi_seed = NULL; long pdim = 0;
    int rc_seed = nqs_materialise_state_with_cb_N(nqs_ansatz_log_amp, a, N,
                                                    &psi_seed, &pdim);
    const double *seed = (rc_seed == 0 && pdim == dim) ? psi_seed : NULL;
    int rc = lanczos_k_smallest_with_init(kagome_heis_matvec, &ctx, dim,
                                            max_iters, seed, k,
                                            out_eigenvalues, out_result);
    free(psi_seed);
    return rc;
}

/* Sector-projected projector callback for the projecting Lanczos.  The
 * projector is closed over a context holding (perm, characters, N, G). */
typedef struct {
    int N, G;
    const int *perm;
    const double *characters;
} kagome_p6m_proj_ctx_t;

static void kagome_p6m_project_step(double *vec, long dim, void *user) {
    kagome_p6m_proj_ctx_t *pc = (kagome_p6m_proj_ctx_t *)user;
    if (!pc) return;
    long expected_dim = 1L << pc->N;
    if (dim != expected_dim) return;
    nqs_kagome_p6m_project_inplace(vec, pc->N, pc->G, pc->perm, pc->characters);
}

int nqs_lanczos_k_lowest_kagome_heisenberg_projected(
    nqs_log_amp_fn_t log_amp, void *user,
    int Lx_cells, int Ly_cells, double J, int pbc,
    const int *perm, const double *characters, int G,
    int max_iters, int k,
    double *out_eigenvalues,
    lanczos_result_t *out_result) {
    if (!log_amp || !out_eigenvalues || !perm || !characters || k <= 0)
        return -1;
    int N = 3 * Lx_cells * Ly_cells;
    kagome_heis_ctx_t ctx = { .Lx_cells = Lx_cells, .Ly_cells = Ly_cells,
                               .N = N, .J = J, .pbc = pbc };
    kagome_p6m_proj_ctx_t pc = { .N = N, .G = G,
                                  .perm = perm, .characters = characters };
    long dim = 1L << N;
    double *psi_seed = NULL; long pdim = 0;
    int rc_seed = nqs_materialise_state_with_cb_N(log_amp, user, N,
                                                    &psi_seed, &pdim);
    const double *seed = (rc_seed == 0 && pdim == dim) ? psi_seed : NULL;
    int rc = lanczos_k_smallest_projected(kagome_heis_matvec, &ctx, dim,
                                           max_iters, seed, k,
                                           kagome_p6m_project_step, &pc,
                                           out_eigenvalues, out_result);
    free(psi_seed);
    return rc;
}

/* Memory-lean projecting Lanczos for E_0 only.  Bypasses the materialise
 * step entirely — starts from a deterministic random vector, projects
 * it once into the sector, then runs 3-term-recurrence Lanczos with
 * in-loop sector projection.  Works at large N where full reorth would
 * blow memory.
 *
 * No NQS callback is used — the seed is just a pseudo-random vector,
 * sufficient to span the projected sector.  Returns the (k=Γ, irrep
 * α)-sector ground-state energy. */
int nqs_lanczos_e0_kagome_heisenberg_projected_lean(
    int Lx_cells, int Ly_cells, double J, int pbc,
    const int *perm, const double *characters, int G,
    int max_iters, double tol,
    double *out_eigenvalue,
    lanczos_result_t *out_result) {
    if (!out_eigenvalue || !perm || !characters || G <= 0) return -1;
    int N = 3 * Lx_cells * Ly_cells;
    if (N <= 0 || N > 30) return -1;
    long dim = 1L << N;
    kagome_heis_ctx_t ctx = { .Lx_cells = Lx_cells, .Ly_cells = Ly_cells,
                               .N = N, .J = J, .pbc = pbc };
    kagome_p6m_proj_ctx_t pc = { .N = N, .G = G,
                                  .perm = perm, .characters = characters };
    return lanczos_smallest_projected_lean(kagome_heis_matvec, &ctx, dim,
                                            max_iters, tol, NULL,
                                            kagome_p6m_project_step, &pc,
                                            out_eigenvalue, out_result);
}

int nqs_lanczos_refine_kagome_heisenberg_projected(
    nqs_log_amp_fn_t log_amp, void *user,
    int Lx_cells, int Ly_cells, double J, int pbc,
    const int *perm, const double *characters, int G,
    int max_iters, double tol,
    double *out_eigenvalue,
    double *out_eigenvector,
    lanczos_result_t *out_result) {
    if (!log_amp || !out_eigenvalue || !perm || !characters) return -1;
    int N = 3 * Lx_cells * Ly_cells;
    kagome_heis_ctx_t ctx = { .Lx_cells = Lx_cells, .Ly_cells = Ly_cells,
                               .N = N, .J = J, .pbc = pbc };
    kagome_p6m_proj_ctx_t pc = { .N = N, .G = G,
                                  .perm = perm, .characters = characters };
    long dim = 1L << N;
    double *psi_seed = NULL; long pdim = 0;
    int rc_seed = nqs_materialise_state_with_cb_N(log_amp, user, N,
                                                    &psi_seed, &pdim);
    int rc = (rc_seed == 0 && pdim == dim)
        ? lanczos_smallest_projected(kagome_heis_matvec, &ctx, dim,
                                       max_iters, tol,
                                       psi_seed,
                                       kagome_p6m_project_step, &pc,
                                       out_eigenvector, out_result)
        : lanczos_smallest_projected(kagome_heis_matvec, &ctx, dim,
                                       max_iters, tol, NULL,
                                       kagome_p6m_project_step, &pc,
                                       out_eigenvector, out_result);
    free(psi_seed);
    if (rc == 0 && out_result) *out_eigenvalue = out_result->eigenvalue;
    return rc;
}
