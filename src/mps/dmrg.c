/*
 * src/mps/dmrg.c
 *
 * Two-site DMRG for the 1D XXZ chain with open boundary conditions.
 * Everything is self-contained: the MPO is hard-coded as the standard
 * bond-dim-5 XXZ form, tensor contractions are written out explicitly,
 * the two-site eigenproblem is solved matrix-free by the shared
 * Lanczos driver in mps/lanczos.c, and the Schmidt decomposition uses
 * the Jacobi SVD in mps/svd.c. No BLAS/LAPACK.
 *
 * Bond dimensions grow during the first sweep from 1 to at most D_max
 * and are truncated on each two-site split by keeping the D_max
 * largest singular values of the merged tensor.
 *
 * Physical index convention: 0 = ↑, 1 = ↓. Spin operators:
 *   S^z = diag(+½, -½)
 *   S^+ = [[0, 1], [0, 0]]   (|↓⟩ → |↑⟩)
 *   S^- = [[0, 0], [1, 0]]   (|↑⟩ → |↓⟩)
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mps/dmrg.h"
#include "mps/svd.h"
#include "mps/lanczos.h"

#define MPS_D 2     /* physical dim (spin-1/2) */
#define MPO_BW 5    /* MPO bond dim for XXZ */

typedef struct {
    int D_left, D_right;   /* MPS bond dimensions at left/right of this site */
    double *A;             /* shape (D_left, MPS_D, D_right), row-major      */
} mps_site_t;

/* Bulk XXZ MPO tensor. W[wl, s, s', wr], shape (MPO_BW, MPS_D, MPS_D, MPO_BW).
 * Packed row-major with strides (d*d*bw, d*bw, bw, 1). */
static void build_xxz_bulk_mpo(double W[MPO_BW * MPS_D * MPS_D * MPO_BW],
                                double Jx, double Jz) {
    memset(W, 0, sizeof(double) * MPO_BW * MPS_D * MPS_D * MPO_BW);

    #define W_SET(wl, s, sp, wr, v) \
        W[((wl)*MPS_D*MPS_D*MPO_BW) + ((s)*MPS_D*MPO_BW) + ((sp)*MPO_BW) + (wr)] = (v)

    /* S^z = diag(+½, -½)  S^+ only [0,1] = 1  S^- only [1,0] = 1  I = delta */
    /* Top row (wl=0): coupling terms that *start* an interaction with the
     * next site. The "end" column (wr=4) carries the completed term. */
    /* wl=0, wr=0: I */
    W_SET(0, 0, 0, 0,  1.0);
    W_SET(0, 1, 1, 0,  1.0);
    /* wl=0, wr=1: (Jx/2) S^+       (S^+ on site i) */
    W_SET(0, 0, 1, 1,  0.5 * Jx);
    /* wl=0, wr=2: (Jx/2) S^-       (S^- on site i) */
    W_SET(0, 1, 0, 2,  0.5 * Jx);
    /* wl=0, wr=3: Jz · S^z */
    W_SET(0, 0, 0, 3,  0.5 * Jz);
    W_SET(0, 1, 1, 3, -0.5 * Jz);
    /* wl=0, wr=4: 0  (no single-site term in pure XXZ) */
    /* wl=1 completes a J_x/2 S^+ S^-   (S^- on site i+1) */
    W_SET(1, 1, 0, 4,  1.0);
    /* wl=2 completes a J_x/2 S^- S^+   (S^+ on site i+1) */
    W_SET(2, 0, 1, 4,  1.0);
    /* wl=3 completes Jz S^z S^z        (S^z on site i+1) */
    W_SET(3, 0, 0, 4,  0.5);
    W_SET(3, 1, 1, 4, -0.5);
    /* wl=4, wr=4: I   (propagate completed term) */
    W_SET(4, 0, 0, 4,  1.0);
    W_SET(4, 1, 1, 4,  1.0);
    #undef W_SET
}

/* ------------------------------- utilities ----------------------------- */

static double rand01(unsigned long long *rng) {
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    return (double)(*rng >> 11) / 9007199254740992.0;
}

static void mat_transpose(const double *A, int m, int n, double *out) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            out[(size_t)j * m + i] = A[(size_t)i * n + j];
}


/* ------------------ MPS construction + canonicalisation --------------- */

static void mps_init_random(mps_site_t *sites, int N, int D_max,
                             unsigned long long *rng) {
    /* Start with bond dim = min(2^i, 2^(N-i), D_max). This is the
     * natural bond dim from a product state; it avoids any need for
     * wide-matrix SVDs in the initial canonicalisation step — every
     * reshape is square or tall. The sweep will grow bond dim only
     * where the SVD says it should. */
    int *dims = calloc((size_t)(N + 1), sizeof(int));
    dims[0] = 1;
    dims[N] = 1;
    for (int i = 1; i < N; i++) {
        int left_cap = 1; for (int j = 0; j < i && left_cap < D_max; j++) left_cap *= 2;
        int right_cap = 1; for (int j = 0; j < N - i && right_cap < D_max; j++) right_cap *= 2;
        int d = left_cap < right_cap ? left_cap : right_cap;
        if (d > D_max) d = D_max;
        if (d < 1) d = 1;
        dims[i] = d;
    }
    for (int i = 0; i < N; i++) {
        int Dl = dims[i];
        int Dr = dims[i + 1];
        sites[i].D_left  = Dl;
        sites[i].D_right = Dr;
        sites[i].A = calloc((size_t)Dl * MPS_D * Dr, sizeof(double));
        for (int k = 0; k < Dl * MPS_D * Dr; k++)
            sites[i].A[k] = 0.1 * (rand01(rng) - 0.5);
    }
    free(dims);
}

/* Right-canonicalise the MPS from site N-1 back to site 0. After the
 * sweep, each A[i] satisfies   Σ_{s, r} A[i][l, s, r] · A[i][l', s, r] = δ_{l, l'}
 * (row-orthogonal), and A[0] holds the whole state's norm. */
static int mps_right_canonicalize(mps_site_t *sites, int N) {
    for (int i = N - 1; i > 0; i--) {
        int Dl = sites[i].D_left, Dr = sites[i].D_right;
        int rows = Dl, cols = MPS_D * Dr;
        /* SVD: A[i] reshaped as (Dl, d*Dr). We need m ≥ n for svd_jacobi. */
        if (rows >= cols) {
            double *U = malloc(sizeof(double) * rows * cols);
            double *sv = malloc(sizeof(double) * cols);
            double *Vt = malloc(sizeof(double) * cols * cols);
            svd_jacobi(sites[i].A, rows, cols, U, sv, Vt, 1e-14);
            /* A[i] becomes V^T (first `cols` rows of V^T — all rows). New
             * bond dim = cols (could be cols ≤ rows; pick min). */
            int D_new = cols;
            double *A_new = malloc(sizeof(double) * D_new * cols);
            memcpy(A_new, Vt, sizeof(double) * D_new * cols);
            /* Absorb U · Σ into A[i-1]. */
            double *USig = malloc(sizeof(double) * rows * D_new);
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < D_new; c++)
                    USig[(size_t)r * D_new + c] = U[(size_t)r * cols + c] * sv[c];
            /* A[i-1] shape (D_{i-1}, d, D_left) = (D_{i-1}, d, rows). New right
             * bond is D_new. Contract A[i-1][l, s, r] · USig[r, r'] → new. */
            int Dl_prev = sites[i-1].D_left;
            double *A_prev_new = malloc(sizeof(double) * Dl_prev * MPS_D * D_new);
            for (int l = 0; l < Dl_prev; l++) {
                for (int s = 0; s < MPS_D; s++) {
                    for (int c = 0; c < D_new; c++) {
                        double val = 0.0;
                        for (int r = 0; r < rows; r++) {
                            val += sites[i-1].A[(size_t)((l * MPS_D) + s) * sites[i-1].D_right + r]
                                 * USig[(size_t)r * D_new + c];
                        }
                        A_prev_new[(size_t)((l * MPS_D) + s) * D_new + c] = val;
                    }
                }
            }
            free(sites[i].A);  sites[i].A = A_new;
            sites[i].D_left = D_new;
            free(sites[i-1].A); sites[i-1].A = A_prev_new;
            sites[i-1].D_right = D_new;
            free(U); free(sv); free(Vt); free(USig);
        } else {
            /* rows < cols — use LQ via modified Gram-Schmidt on the
             * rows of A. For A (rows, cols), orthonormalise rows
             * bottom-up to get A = L · Q with L (rows, rows) lower-
             * triangular and Q (rows, cols) row-orthonormal. Set
             * sites[i] = Q; absorb L into sites[i-1]. */
            double *Q = malloc(sizeof(double) * rows * cols);
            double *L = calloc((size_t)rows * rows, sizeof(double));
            memcpy(Q, sites[i].A, sizeof(double) * rows * cols);
            for (int k = 0; k < rows; k++) {
                /* Orthogonalise row k against previously-processed rows. */
                for (int p = 0; p < k; p++) {
                    double dot = 0.0;
                    for (int c = 0; c < cols; c++)
                        dot += Q[(size_t)p * cols + c] * Q[(size_t)k * cols + c];
                    L[(size_t)k * rows + p] = dot;
                    for (int c = 0; c < cols; c++)
                        Q[(size_t)k * cols + c] -= dot * Q[(size_t)p * cols + c];
                }
                double nrm2 = 0.0;
                for (int c = 0; c < cols; c++)
                    nrm2 += Q[(size_t)k * cols + c] * Q[(size_t)k * cols + c];
                double nrm = sqrt(nrm2);
                L[(size_t)k * rows + k] = nrm;
                if (nrm > 1e-15) {
                    for (int c = 0; c < cols; c++)
                        Q[(size_t)k * cols + c] /= nrm;
                }
            }
            /* sites[i] ← Q. */
            free(sites[i].A);
            sites[i].A = Q;
            sites[i].D_left = rows;
            /* Absorb L into sites[i-1]: contract sites[i-1][l, s, r] · L[r, c] → new. */
            int Dl_prev = sites[i-1].D_left;
            double *A_prev_new = malloc(sizeof(double) * Dl_prev * MPS_D * rows);
            for (int l = 0; l < Dl_prev; l++) {
                for (int s = 0; s < MPS_D; s++) {
                    for (int c = 0; c < rows; c++) {
                        double val = 0.0;
                        for (int r = 0; r < rows; r++) {
                            val += sites[i-1].A[(size_t)((l * MPS_D) + s) * sites[i-1].D_right + r]
                                 * L[(size_t)r * rows + c];
                        }
                        A_prev_new[(size_t)((l * MPS_D) + s) * rows + c] = val;
                    }
                }
            }
            free(sites[i-1].A);
            sites[i-1].A = A_prev_new;
            sites[i-1].D_right = rows;
            free(L);
        }
    }
    /* Normalise A[0]. */
    int Dl = sites[0].D_left, Dr = sites[0].D_right;
    double n2 = 0.0;
    for (int k = 0; k < Dl * MPS_D * Dr; k++) n2 += sites[0].A[k] * sites[0].A[k];
    double inv = n2 > 0 ? 1.0 / sqrt(n2) : 1.0;
    for (int k = 0; k < Dl * MPS_D * Dr; k++) sites[0].A[k] *= inv;
    return 0;
}

/* ----------------- environment tensors + H_eff action ----------------- */

/* Each L[i] has shape (D_{i}, MPO_BW, D_{i}). Identity at boundaries.
 * R[i] same. Index: (l * MPO_BW + w) * D + l'. */

/* Left environment update: given L[i] of shape (D_l, bw, D_l) and
 * A[i] of shape (D_l, d, D_r), MPO bulk W, produce L[i+1] of shape
 * (D_r, bw, D_r):
 *   L'[r, wr, r'] = Σ L[l, wl, l'] A[l, s, r] W[wl, s, s', wr] A[l', s', r']
 */
static void env_update_left(const double *L, int D_l,
                            const double *A, int D_r,
                            const double *W,      /* shape (bw, d, d, bw) */
                            int bw_l, int bw_r,   /* 1 or MPO_BW */
                            double *Lp) {
    memset(Lp, 0, sizeof(double) * D_r * MPO_BW * D_r);
    for (int l = 0; l < D_l; l++) {
        for (int lp = 0; lp < D_l; lp++) {
            for (int wl = 0; wl < bw_l; wl++) {
                double Lv = L[(l * MPO_BW + wl) * D_l + lp];
                if (Lv == 0.0) continue;
                for (int s = 0; s < MPS_D; s++) {
                    for (int sp = 0; sp < MPS_D; sp++) {
                        for (int wr = 0; wr < bw_r; wr++) {
                            double Wv = W[((wl * MPS_D + s) * MPS_D + sp) * bw_r + wr];
                            if (Wv == 0.0) continue;
                            for (int r = 0; r < D_r; r++) {
                                for (int rp = 0; rp < D_r; rp++) {
                                    Lp[(r * MPO_BW + wr) * D_r + rp] +=
                                        Lv * Wv
                                        * A[(l * MPS_D + s) * D_r + r]
                                        * A[(lp * MPS_D + sp) * D_r + rp];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static void env_update_right(const double *R, int D_r,
                             const double *A, int D_l,
                             const double *W,
                             int bw_l, int bw_r,
                             double *Rp) {
    memset(Rp, 0, sizeof(double) * D_l * MPO_BW * D_l);
    for (int r = 0; r < D_r; r++) {
        for (int rp = 0; rp < D_r; rp++) {
            for (int wr = 0; wr < bw_r; wr++) {
                double Rv = R[(r * MPO_BW + wr) * D_r + rp];
                if (Rv == 0.0) continue;
                for (int s = 0; s < MPS_D; s++) {
                    for (int sp = 0; sp < MPS_D; sp++) {
                        for (int wl = 0; wl < bw_l; wl++) {
                            double Wv = W[((wl * MPS_D + s) * MPS_D + sp) * bw_r + wr];
                            if (Wv == 0.0) continue;
                            for (int l = 0; l < D_l; l++) {
                                for (int lp = 0; lp < D_l; lp++) {
                                    Rp[(l * MPO_BW + wl) * D_l + lp] +=
                                        Rv * Wv
                                        * A[(l * MPS_D + s) * D_r + r]
                                        * A[(lp * MPS_D + sp) * D_r + rp];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/* Two-site H_eff action on T[l, s1, s2, r], flattened length D_l · d · d · D_r.
 * Uses left env L (D_l, bw, D_l), right env R (D_r, bw, D_r) and two MPO
 * tensors W1 (bw_l, d, d, bw), W2 (bw, d, d, bw_r). */
typedef struct {
    int D_l, D_r;
    int bw_l, bw_r;
    const double *L, *R;
    const double *W1, *W2;
} h_eff_ctx_t;

static void h_eff_matvec(const double *in, double *out, long dim, void *ud) {
    h_eff_ctx_t *c = (h_eff_ctx_t *)ud;
    int D_l = c->D_l, D_r = c->D_r;
    int bw_l = c->bw_l, bw_r = c->bw_r;
    memset(out, 0, sizeof(double) * dim);
    for (int l = 0; l < D_l; l++) {
        for (int s1 = 0; s1 < MPS_D; s1++) {
            for (int s2 = 0; s2 < MPS_D; s2++) {
                for (int r = 0; r < D_r; r++) {
                    double acc = 0.0;
                    for (int lp = 0; lp < D_l; lp++) {
                        for (int wl = 0; wl < bw_l; wl++) {
                            double Lv = c->L[(l * MPO_BW + wl) * D_l + lp];
                            if (Lv == 0.0) continue;
                            for (int s1p = 0; s1p < MPS_D; s1p++) {
                                for (int wi = 0; wi < MPO_BW; wi++) {
                                    double W1v = c->W1[((wl * MPS_D + s1) * MPS_D + s1p) * MPO_BW + wi];
                                    if (W1v == 0.0) continue;
                                    for (int s2p = 0; s2p < MPS_D; s2p++) {
                                        for (int wr = 0; wr < bw_r; wr++) {
                                            double W2v = c->W2[((wi * MPS_D + s2) * MPS_D + s2p) * bw_r + wr];
                                            if (W2v == 0.0) continue;
                                            for (int rp = 0; rp < D_r; rp++) {
                                                double Rv = c->R[(r * MPO_BW + wr) * D_r + rp];
                                                if (Rv == 0.0) continue;
                                                int idx_in = (((lp * MPS_D + s1p) * MPS_D + s2p) * D_r) + rp;
                                                acc += Lv * W1v * W2v * Rv * in[idx_in];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    int idx_out = (((l * MPS_D + s1) * MPS_D + s2) * D_r) + r;
                    out[idx_out] = acc;
                }
            }
        }
    }
}

/* ------------------------- two-site DMRG step ------------------------- */

/* Merge A[i] (D_l, d, D_m) with A[i+1] (D_m, d, D_r) → T (D_l, d, d, D_r). */
static void merge_two_sites(const double *A1, int D_l, int D_m,
                             const double *A2, int D_r,
                             double *T) {
    for (int l = 0; l < D_l; l++)
        for (int s1 = 0; s1 < MPS_D; s1++)
            for (int s2 = 0; s2 < MPS_D; s2++)
                for (int r = 0; r < D_r; r++) {
                    double v = 0.0;
                    for (int m = 0; m < D_m; m++) {
                        v += A1[(l * MPS_D + s1) * D_m + m]
                           * A2[(m * MPS_D + s2) * D_r + r];
                    }
                    T[((l * MPS_D + s1) * MPS_D + s2) * D_r + r] = v;
                }
}

/* Split merged tensor T back into A1, A2 with bond dim D_new ≤ D_max.
 * T reshaped as matrix M (D_l · d, d · D_r), SVD, truncate, distribute. */
static int split_two_sites(const double *T, int D_l, int D_r, int D_max,
                            mps_site_t *out_A1, mps_site_t *out_A2,
                            int left_sweep,
                            double *out_largest_trunc) {
    int rows = D_l * MPS_D;
    int cols = MPS_D * D_r;
    int n = rows < cols ? rows : cols;
    /* svd_jacobi needs m ≥ n. If rows < cols, work with transpose. */
    double *U, *sv, *Vt;
    int Mrows = rows, Mcols = cols;
    const double *M_in = T;
    double *M_tmp = NULL;
    if (rows < cols) {
        /* Transpose: new matrix is cols × rows, then svd_jacobi(cols, rows). */
        M_tmp = malloc(sizeof(double) * cols * rows);
        mat_transpose(T, rows, cols, M_tmp);
        M_in = M_tmp;
        Mrows = cols;
        Mcols = rows;
    }
    U  = malloc(sizeof(double) * Mrows * Mcols);
    sv = malloc(sizeof(double) * Mcols);
    Vt = malloc(sizeof(double) * Mcols * Mcols);
    svd_jacobi(M_in, Mrows, Mcols, U, sv, Vt, 1e-14);

    int D_new = n < D_max ? n : D_max;
    /* Track the largest truncated singular value for diagnostics. */
    if (D_new < n && out_largest_trunc && sv[D_new] > *out_largest_trunc)
        *out_largest_trunc = sv[D_new];

    /* Back in the original orientation:
     *   if rows >= cols: M = U (rows,cols) · diag(s) · Vt (cols,cols), keep D_new cols.
     *   if rows <  cols: M^T = U' (cols,rows) · diag(s') · Vt' (rows,rows).
     *     Then M = Vt'^T · diag(s') · U'^T. */
    double *A1_new = calloc((size_t)D_l * MPS_D * D_new, sizeof(double));
    double *A2_new = calloc((size_t)D_new * MPS_D * D_r, sizeof(double));
    if (rows >= cols) {
        /* Left sweep: keep U absorbed as A1 (left-canonical); Σ·Vt absorbed as A2. */
        if (left_sweep) {
            /* A1[l, s1, m] = U[l*d + s1, m] for m < D_new */
            for (int i = 0; i < rows; i++)
                for (int m = 0; m < D_new; m++)
                    A1_new[(size_t)i * D_new + m] = U[(size_t)i * Mcols + m];
            /* A2[m, s2, r] = s[m] · Vt[m, s2*D_r + r] */
            for (int m = 0; m < D_new; m++)
                for (int j = 0; j < cols; j++)
                    A2_new[(size_t)m * cols + j] = sv[m] * Vt[(size_t)m * Mcols + j];
        } else {
            /* Right sweep: keep Vt absorbed as A2 (right-canonical); U·Σ as A1. */
            for (int i = 0; i < rows; i++)
                for (int m = 0; m < D_new; m++)
                    A1_new[(size_t)i * D_new + m] = U[(size_t)i * Mcols + m] * sv[m];
            for (int m = 0; m < D_new; m++)
                for (int j = 0; j < cols; j++)
                    A2_new[(size_t)m * cols + j] = Vt[(size_t)m * Mcols + j];
        }
    } else {
        /* Original M was rows × cols with rows < cols. SVD was taken
         * on M^T (cols × rows). With Jacobi output U (cols × rows),
         * s (rows), Vt (rows × rows):
         *     M^T = U · diag(s) · Vt
         *     M   = Vt^T · diag(s) · U^T
         * So the "SVD of M" has:
         *     U_orig = Vt^T         shape (rows, rows)  column-orthonormal
         *     Vt_orig = U^T         shape (rows, cols)  row-orthonormal */
        double *VtT = malloc(sizeof(double) * Mcols * Mcols);    /* (rows,rows) */
        mat_transpose(Vt, Mcols, Mcols, VtT);
        double *UT  = malloc(sizeof(double) * Mcols * Mrows);    /* (rows,cols) */
        mat_transpose(U, Mrows, Mcols, UT);
        int stride_UT = Mrows;   /* row stride of UT = Mrows = cols */
        int stride_VtT = Mcols;  /* row stride of VtT = Mcols = rows */
        if (left_sweep) {
            for (int i = 0; i < rows; i++)
                for (int m = 0; m < D_new; m++)
                    A1_new[(size_t)i * D_new + m] = VtT[(size_t)i * stride_VtT + m];
            for (int m = 0; m < D_new; m++)
                for (int j = 0; j < cols; j++)
                    A2_new[(size_t)m * cols + j] = sv[m] * UT[(size_t)m * stride_UT + j];
        } else {
            for (int i = 0; i < rows; i++)
                for (int m = 0; m < D_new; m++)
                    A1_new[(size_t)i * D_new + m] = VtT[(size_t)i * stride_VtT + m] * sv[m];
            for (int m = 0; m < D_new; m++)
                for (int j = 0; j < cols; j++)
                    A2_new[(size_t)m * cols + j] = UT[(size_t)m * stride_UT + j];
        }
        free(VtT); free(UT);
    }

    free(out_A1->A); out_A1->A = A1_new; out_A1->D_left = D_l; out_A1->D_right = D_new;
    free(out_A2->A); out_A2->A = A2_new; out_A2->D_left = D_new; out_A2->D_right = D_r;

    free(U); free(sv); free(Vt); free(M_tmp);
    return 0;
}

/* ----------------------------- main driver ---------------------------- */

/* Contract an MPS into its full state vector. Works for N ≤ 20.
 * Caller frees *out_psi. */
static int mps_to_state_vector(const mps_site_t *sites, int N,
                                double **out_psi, long *out_dim) {
    if (!sites || !out_psi || !out_dim || N <= 0 || N > 20) return -1;
    /* Incremental contraction: start with a (1, D_0) vector (= sites[0]
     * sliced at left_bond=0), extend by contracting against A[i] to
     * produce a (2^{i+1}, D_{i+1}) block. At the end, D_N = 1. */
    /* Outer-block invariant: sites[0].D_left == 1, so only D_right is used. */
    int Dr = sites[0].D_right;
    long block_states = 1;  /* number of basis states captured so far */
    long nblock = block_states * MPS_D * Dr;
    double *block = malloc(sizeof(double) * nblock);
    for (int s = 0; s < MPS_D; s++) {
        for (int r = 0; r < Dr; r++) {
            block[s * Dr + r] = sites[0].A[(0 * MPS_D + s) * Dr + r];
        }
    }
    block_states = MPS_D;
    int D_in = Dr;
    for (int i = 1; i < N; i++) {
        int D_out = sites[i].D_right;
        long new_states = block_states * MPS_D;
        double *new_block = calloc((size_t)new_states * D_out, sizeof(double));
        for (long st = 0; st < block_states; st++) {
            for (int s = 0; s < MPS_D; s++) {
                for (int r = 0; r < D_out; r++) {
                    double v = 0.0;
                    for (int m = 0; m < D_in; m++) {
                        v += block[st * D_in + m]
                           * sites[i].A[(m * MPS_D + s) * D_out + r];
                    }
                    new_block[(st * MPS_D + s) * D_out + r] = v;
                }
            }
        }
        free(block);
        block = new_block;
        block_states = new_states;
        D_in = D_out;
    }
    /* D_in == 1 at the end (sites[N-1].D_right == 1). */
    double *psi = malloc(sizeof(double) * block_states);
    for (long st = 0; st < block_states; st++) psi[st] = block[st];
    free(block);
    double n2 = 0;
    for (long st = 0; st < block_states; st++) n2 += psi[st] * psi[st];
    double inv = n2 > 0 ? 1.0 / sqrt(n2) : 1.0;
    for (long st = 0; st < block_states; st++) psi[st] *= inv;
    *out_psi = psi;
    *out_dim = block_states;
    return 0;
}

static int mps_dmrg_xxz_impl(const mps_config_t *cfg,
                              mps_dmrg_result_t *out,
                              double **out_psi, long *out_dim);

int mps_dmrg_xxz(const mps_config_t *cfg, mps_dmrg_result_t *out) {
    return mps_dmrg_xxz_impl(cfg, out, NULL, NULL);
}

int mps_dmrg_xxz_with_state(const mps_config_t *cfg,
                             mps_dmrg_result_t *out,
                             double **out_psi, long *out_dim) {
    return mps_dmrg_xxz_impl(cfg, out, out_psi, out_dim);
}

static int mps_dmrg_xxz_impl(const mps_config_t *cfg,
                              mps_dmrg_result_t *out,
                              double **out_psi, long *out_dim) {
    if (!cfg || !out) return -1;
    int N = cfg->num_sites;
    int D_max = cfg->max_bond_dim > 0 ? cfg->max_bond_dim : 16;
    int max_sweeps = cfg->num_sweeps > 0 ? cfg->num_sweeps : 10;
    double sweep_tol = cfg->sweep_tol > 0 ? cfg->sweep_tol : 1e-8;

    mps_site_t *sites = calloc((size_t)N, sizeof(mps_site_t));
    if (!sites) return -1;
    unsigned long long rng = 0xABC;
    mps_init_random(sites, N, D_max, &rng);
    mps_right_canonicalize(sites, N);

    /* Build MPO tensors. Left and right boundaries are slices of the bulk. */
    double W_bulk[MPO_BW * MPS_D * MPS_D * MPO_BW];
    double Jx = cfg->J;
    double Jz = (cfg->ham == MPS_HAM_XXZ) ? cfg->Jz : Jx;
    if (cfg->ham == MPS_HAM_TFIM) {
        /* TFIM uses a bond-dim-3 MPO; pad into the same 5x5 cage by
         * reusing state indices 0, 1, 4 (begin, σ^z-emitter, end). */
        memset(W_bulk, 0, sizeof(W_bulk));
        #define W_TFIM(wl, s, sp, wr, v) \
            W_bulk[((wl)*MPS_D*MPS_D*MPO_BW) + ((s)*MPS_D*MPO_BW) + ((sp)*MPO_BW) + (wr)] = (v)
        W_TFIM(0, 0, 0, 0,  1.0);          /* I at begin */
        W_TFIM(0, 1, 1, 0,  1.0);
        W_TFIM(0, 0, 0, 1,  1.0);          /* σ^z start */
        W_TFIM(0, 1, 1, 1, -1.0);
        W_TFIM(0, 0, 1, 4, -cfg->Gamma);    /* -Γ σ^x single-site */
        W_TFIM(0, 1, 0, 4, -cfg->Gamma);
        W_TFIM(1, 0, 0, 4, -cfg->J);        /* -J σ^z to finish */
        W_TFIM(1, 1, 1, 4,  cfg->J);
        W_TFIM(4, 0, 0, 4,  1.0);          /* I at end */
        W_TFIM(4, 1, 1, 4,  1.0);
        #undef W_TFIM
    } else {
        build_xxz_bulk_mpo(W_bulk, Jx, Jz);
    }

    /* Build environments. L[0] trivial; build R from right to left. */
    double **L = calloc(N + 1, sizeof(double *));
    double **R = calloc(N + 1, sizeof(double *));
    for (int i = 0; i <= N; i++) {
        /* Allocate up-front to max possible bond dim; for simplicity use
         * the actual bond dim of sites[]. */
    }
    /* R[N] trivial 1×bw×1 (only entry wl=4 since we are at the "end" state). */
    L[0] = calloc(MPO_BW, sizeof(double));
    L[0][0] = 1.0;  /* L starts in MPO state 0 */
    R[N] = calloc(MPO_BW, sizeof(double));
    R[N][4] = 1.0;  /* R starts in MPO state 4 */

    /* Build R from right to left using env_update_right. We need a per-
     * site W tensor: bulk for interior, boundary-sliced at edges. For
     * env building, we use the full bulk W because L[0] already
     * implicitly selects wl=0 and R[N] implicitly selects wr=4. */
    for (int i = N - 1; i >= 1; i--) {
        int D_l = sites[i].D_left, D_r = sites[i].D_right;
        R[i] = calloc((size_t)D_l * MPO_BW * D_l, sizeof(double));
        env_update_right(R[i+1], D_r, sites[i].A, D_l,
                          W_bulk, MPO_BW, MPO_BW, R[i]);
    }

    /* Sweep. */
    double last_energy = 1e300;
    out->sweeps_performed = 0;
    out->converged = 0;
    out->largest_truncated_sv = 0.0;

    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        double sweep_energy = 0.0;
        /* Left-to-right sweep. */
        for (int i = 0; i < N - 1; i++) {
            int D_l = sites[i].D_left;
            int D_m = sites[i].D_right;      /* bond between i and i+1 */
            int D_r = sites[i+1].D_right;
            long dim = (long)D_l * MPS_D * MPS_D * D_r;
            double *T = malloc(sizeof(double) * dim);
            merge_two_sites(sites[i].A, D_l, D_m, sites[i+1].A, D_r, T);

            h_eff_ctx_t ctx;
            ctx.D_l = D_l; ctx.D_r = D_r;
            ctx.bw_l = MPO_BW; ctx.bw_r = MPO_BW;
            ctx.L = L[i]; ctx.R = R[i+2]; ctx.W1 = W_bulk; ctx.W2 = W_bulk;
            double *eigvec = malloc(sizeof(double) * dim);
            lanczos_result_t lres;
            int max_iters = cfg->lanczos_max_iters > 0 ? cfg->lanczos_max_iters : 60;
            double tol = cfg->lanczos_tol > 0 ? cfg->lanczos_tol : 1e-9;
            int rc = lanczos_smallest_with_init(h_eff_matvec, &ctx, dim,
                                                 max_iters, tol, T,
                                                 eigvec, &lres);
            (void)rc;
            sweep_energy = lres.eigenvalue;
            split_two_sites(eigvec, D_l, D_r, D_max,
                             &sites[i], &sites[i+1], 1,
                             &out->largest_truncated_sv);
            /* Update L[i+1] from L[i], A[i], W. */
            free(L[i+1]);
            int D_new = sites[i].D_right;
            L[i+1] = calloc((size_t)D_new * MPO_BW * D_new, sizeof(double));
            env_update_left(L[i], D_l, sites[i].A, D_new,
                             W_bulk, MPO_BW, MPO_BW, L[i+1]);
            free(T); free(eigvec);
        }
        /* Right-to-left sweep. */
        for (int i = N - 2; i >= 0; i--) {
            int D_l = sites[i].D_left;
            int D_m = sites[i].D_right;
            int D_r = sites[i+1].D_right;
            long dim = (long)D_l * MPS_D * MPS_D * D_r;
            double *T = malloc(sizeof(double) * dim);
            merge_two_sites(sites[i].A, D_l, D_m, sites[i+1].A, D_r, T);
            h_eff_ctx_t ctx;
            ctx.D_l = D_l; ctx.D_r = D_r;
            ctx.bw_l = MPO_BW; ctx.bw_r = MPO_BW;
            ctx.L = L[i]; ctx.R = R[i+2]; ctx.W1 = W_bulk; ctx.W2 = W_bulk;
            double *eigvec = malloc(sizeof(double) * dim);
            lanczos_result_t lres;
            int max_iters = cfg->lanczos_max_iters > 0 ? cfg->lanczos_max_iters : 60;
            double tol = cfg->lanczos_tol > 0 ? cfg->lanczos_tol : 1e-9;
            lanczos_smallest(h_eff_matvec, &ctx, dim, max_iters, tol, eigvec, &lres);
            sweep_energy = lres.eigenvalue;
            split_two_sites(eigvec, D_l, D_r, D_max,
                             &sites[i], &sites[i+1], 0,
                             &out->largest_truncated_sv);
            /* Update R[i+1] from R[i+2], A[i+1], W. */
            free(R[i+1]);
            int D_new = sites[i].D_right;
            R[i+1] = calloc((size_t)D_new * MPO_BW * D_new, sizeof(double));
            env_update_right(R[i+2], D_r, sites[i+1].A, D_new,
                              W_bulk, MPO_BW, MPO_BW, R[i+1]);
            free(T); free(eigvec);
        }
        out->sweeps_performed = sweep + 1;
        double de = fabs(sweep_energy - last_energy);
        last_energy = sweep_energy;
        if (de < sweep_tol) { out->converged = 1; break; }
    }
    out->final_energy = last_energy;

    if (out_psi && out_dim) {
        *out_psi = NULL;
        *out_dim = 0;
        mps_to_state_vector(sites, N, out_psi, out_dim);
    }

    for (int i = 0; i < N; i++) free(sites[i].A);
    free(sites);
    for (int i = 0; i <= N; i++) { if (L[i]) free(L[i]); if (R[i]) free(R[i]); }
    free(L); free(R);
    return 0;
}
