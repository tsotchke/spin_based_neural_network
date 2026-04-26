/*
 * src/nqs/nqs_symproj.c
 *
 * Generic finite-group symmetry-projection wrapper + standard
 * permutation builders for the kagome lattice.
 *
 * The math mirrors nqs_translation: log-sum-exp the contributions of
 * each orbit member, multiply by the chosen 1-D character.  The two
 * differences are (a) groups beyond pure translation (point ops,
 * inversion, ...) require explicit per-element permutation tables,
 * and (b) characters are stored explicitly so non-trivial irreps
 * (e.g. A₂ sign irrep) plug in without code changes.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nqs/nqs_symproj.h"
#include "nqs/nqs_ansatz.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* (g·s)_i = s_{π_g(i)}.  Caller provides scratch of size N. */
static void apply_perm(const int *perm_g, int N,
                       const int *src, int *dst) {
    for (int i = 0; i < N; i++) dst[i] = src[perm_g[i]];
}

void nqs_symproj_log_amp(const int *spins, int num_sites,
                         void *user,
                         double *out_log_abs, double *out_arg) {
    nqs_symproj_wrapper_t *w = (nqs_symproj_wrapper_t *)user;
    if (!w || !w->base_log_amp || !w->perm || !w->characters) {
        if (out_log_abs) *out_log_abs = 0;
        if (out_arg)     *out_arg     = 0;
        return;
    }
    int N = w->num_sites;
    int G = w->num_group_elements;
    if (num_sites != N || N <= 0 || G <= 0) {
        if (out_log_abs) *out_log_abs = 0;
        if (out_arg)     *out_arg     = 0;
        return;
    }

    int *shifted = (int *)malloc((size_t)N * sizeof(int));
    double *lp_arr  = (double *)malloc((size_t)G * sizeof(double));
    double *cos_arr = (double *)malloc((size_t)G * sizeof(double));
    if (!shifted || !lp_arr || !cos_arr) {
        free(shifted); free(lp_arr); free(cos_arr);
        if (out_log_abs) *out_log_abs = 0;
        if (out_arg)     *out_arg     = 0;
        return;
    }

    double max_lp = -1e300;
    for (int g = 0; g < G; g++) {
        apply_perm(&w->perm[(size_t)g * N], N, spins, shifted);
        double lp, arg;
        w->base_log_amp(shifted, N, w->base_user, &lp, &arg);
        lp_arr[g]  = lp;
        cos_arr[g] = w->characters[g] * cos(arg);
        if (lp > max_lp) max_lp = lp;
    }

    double sum = 0.0;
    for (int g = 0; g < G; g++) sum += cos_arr[g] * exp(lp_arr[g] - max_lp);

    double abs_sum = fabs(sum);
    if (abs_sum > 0.0) {
        if (out_log_abs) *out_log_abs = log(abs_sum) + max_lp - 0.5 * log((double)G);
        if (out_arg)     *out_arg     = (sum < 0.0) ? M_PI : 0.0;
    } else {
        if (out_log_abs) *out_log_abs = -1e300;
        if (out_arg)     *out_arg     = 0.0;
    }
    free(shifted); free(lp_arr); free(cos_arr);
}

int nqs_symproj_gradient(void *grad_user,
                         nqs_ansatz_t *ansatz,
                         const int *spins, int num_sites,
                         double *out_grad) {
    if (!grad_user || !ansatz || !spins || !out_grad) return -1;
    nqs_symproj_wrapper_t *w = (nqs_symproj_wrapper_t *)grad_user;
    int N = w->num_sites;
    int G = w->num_group_elements;
    if (num_sites != N || N <= 0 || G <= 0) return -1;
    long P = nqs_ansatz_num_params(ansatz);

    /* Step 1: ψ_base(g·s), with characters χ(g), accumulate weighted
     * denominator using log-sum-exp for stability. */
    int *shifted_buf  = (int *)malloc((size_t)N * sizeof(int));
    int **shifted_store = (int **)malloc((size_t)G * sizeof(int *));
    double *lp_arr  = (double *)malloc((size_t)G * sizeof(double));
    double *cos_arr = (double *)malloc((size_t)G * sizeof(double));
    if (!shifted_buf || !shifted_store || !lp_arr || !cos_arr) {
        free(shifted_buf); free(shifted_store); free(lp_arr); free(cos_arr);
        return -1;
    }
    double max_lp = -1e300;
    for (int g = 0; g < G; g++) {
        shifted_store[g] = (int *)malloc((size_t)N * sizeof(int));
        if (!shifted_store[g]) {
            for (int gg = 0; gg < g; gg++) free(shifted_store[gg]);
            free(shifted_buf); free(shifted_store); free(lp_arr); free(cos_arr);
            return -1;
        }
        apply_perm(&w->perm[(size_t)g * N], N, spins, shifted_store[g]);
        double lp, arg;
        w->base_log_amp(shifted_store[g], N, w->base_user, &lp, &arg);
        lp_arr[g]  = lp;
        cos_arr[g] = w->characters[g] * cos(arg);
        if (lp > max_lp) max_lp = lp;
    }

    double denom = 0.0;
    double *contrib = (double *)malloc((size_t)G * sizeof(double));
    if (!contrib) {
        for (int g = 0; g < G; g++) free(shifted_store[g]);
        free(shifted_buf); free(shifted_store); free(lp_arr); free(cos_arr);
        return -1;
    }
    for (int g = 0; g < G; g++) {
        contrib[g] = cos_arr[g] * exp(lp_arr[g] - max_lp);
        denom += contrib[g];
    }
    if (denom == 0.0) {
        memset(out_grad, 0, (size_t)P * sizeof(double));
        for (int g = 0; g < G; g++) free(shifted_store[g]);
        free(shifted_buf); free(shifted_store); free(lp_arr); free(cos_arr); free(contrib);
        return 0;
    }

    /* Step 2: weighted sum of base gradients at each transformed config. */
    double *tmp_grad = (double *)malloc((size_t)P * sizeof(double));
    if (!tmp_grad) {
        for (int g = 0; g < G; g++) free(shifted_store[g]);
        free(shifted_buf); free(shifted_store); free(lp_arr); free(cos_arr); free(contrib);
        return -1;
    }
    memset(out_grad, 0, (size_t)P * sizeof(double));
    for (int g = 0; g < G; g++) {
        double w_g = contrib[g] / denom;
        if (w_g == 0.0) continue;
        nqs_ansatz_logpsi_gradient(ansatz, shifted_store[g], N, tmp_grad);
        for (long k = 0; k < P; k++) out_grad[k] += w_g * tmp_grad[k];
    }
    free(tmp_grad);
    for (int g = 0; g < G; g++) free(shifted_store[g]);
    free(shifted_buf); free(shifted_store); free(lp_arr); free(cos_arr); free(contrib);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Permutation builders for kagome.                                   */
/* ------------------------------------------------------------------ */

#define KG_SITE(cx, cy, sub, Ly) (3 * ((cx) * (Ly) + (cy)) + (sub))

int nqs_kagome_translation_perm(int Lx, int Ly,
                                int **out_perm,
                                int *out_num_elements) {
    if (Lx <= 0 || Ly <= 0 || !out_perm || !out_num_elements) return -1;
    int N = 3 * Lx * Ly;
    int G = Lx * Ly;
    int *perm = (int *)malloc((size_t)G * (size_t)N * sizeof(int));
    if (!perm) return -1;

    int g = 0;
    for (int tx = 0; tx < Lx; tx++) {
        for (int ty = 0; ty < Ly; ty++) {
            /* Translation T_{tx,ty}: site at (cx, cy, sub) gets the
             * spin from site at (cx + tx, cy + ty, sub) — i.e.
             * π(i) = source-site of i under the inverse shift.  We
             * follow nqs_translation's convention: shifted[i] picks
             * spins[(cx + tx) % Lx, (cy + ty) % Ly, sub]. */
            int *row = &perm[(size_t)g * N];
            for (int cx = 0; cx < Lx; cx++) {
                int xs = (cx + tx) % Lx;
                for (int cy = 0; cy < Ly; cy++) {
                    int ys = (cy + ty) % Ly;
                    for (int sub = 0; sub < 3; sub++) {
                        row[KG_SITE(cx, cy, sub, Ly)] =
                            KG_SITE(xs, ys, sub, Ly);
                    }
                }
            }
            g++;
        }
    }
    *out_perm = perm;
    *out_num_elements = G;
    return 0;
}

int nqs_kagome_p2_perm(int Lx, int Ly,
                        int **out_perm,
                        double **out_characters,
                        int *out_num_elements) {
    if (Lx <= 0 || Ly <= 0 || !out_perm || !out_characters || !out_num_elements)
        return -1;
    int N = 3 * Lx * Ly;
    int G = 2 * Lx * Ly;

    int *perm = (int *)malloc((size_t)G * (size_t)N * sizeof(int));
    double *chars = (double *)malloc((size_t)G * sizeof(double));
    if (!perm || !chars) { free(perm); free(chars); return -1; }

    /* Half 1: pure translations (12 generators × 1 = G/2). */
    int g = 0;
    for (int tx = 0; tx < Lx; tx++) {
        for (int ty = 0; ty < Ly; ty++) {
            int *row = &perm[(size_t)g * N];
            for (int cx = 0; cx < Lx; cx++) {
                int xs = (cx + tx) % Lx;
                for (int cy = 0; cy < Ly; cy++) {
                    int ys = (cy + ty) % Ly;
                    for (int sub = 0; sub < 3; sub++) {
                        row[KG_SITE(cx, cy, sub, Ly)] =
                            KG_SITE(xs, ys, sub, Ly);
                    }
                }
            }
            chars[g] = 1.0;            /* trivial irrep */
            g++;
        }
    }

    /* Half 2: inversion C₂ composed with translations.  C₂ inversion
     * through the A-site origin sends site (cx, cy, sub) to:
     *
     *   sub = A: (-cx,    -cy)
     *   sub = B: (-cx-1,  -cy)         (since -a₁/2 = -a₁ + a₁/2)
     *   sub = C: (-cx,    -cy-1)        (since -a₂/2 = -a₂ + a₂/2)
     *
     * Sublattice index is preserved.  We then compose with translation
     * (tx, ty), i.e. the "shifted" array picks from the C₂-transformed
     * site shifted further by (tx, ty). */
    for (int tx = 0; tx < Lx; tx++) {
        for (int ty = 0; ty < Ly; ty++) {
            int *row = &perm[(size_t)g * N];
            for (int cx = 0; cx < Lx; cx++) {
                for (int cy = 0; cy < Ly; cy++) {
                    for (int sub = 0; sub < 3; sub++) {
                        int dx = (sub == 1) ? -1 : 0;  /* B → -a₁ */
                        int dy = (sub == 2) ? -1 : 0;  /* C → -a₂ */
                        int xs = ((-cx + dx + tx) % Lx + Lx) % Lx;
                        int ys = ((-cy + dy + ty) % Ly + Ly) % Ly;
                        row[KG_SITE(cx, cy, sub, Ly)] =
                            KG_SITE(xs, ys, sub, Ly);
                    }
                }
            }
            chars[g] = 1.0;            /* A₁ trivial irrep */
            g++;
        }
    }

    *out_perm = perm;
    *out_characters = chars;
    *out_num_elements = G;
    return 0;
}

/* ------------------------------------------------------------------ */
/* p3 (translations × C₃) on an L × L kagome torus.                   */
/*                                                                    */
/* The C₃ centre is the up-triangle (0, 0) centroid at                */
/*   τ₀ = (¼, √3/12)                                                  */
/* in Cartesian coordinates.  C₃ around τ₀ cycles sublattices         */
/* A → B → C → A and shifts cells in a way that depends on the site,  */
/* so we compute the permutation numerically: rotate each Cartesian   */
/* position, then invert the lattice basis to recover (cell, sub).    */
/* ------------------------------------------------------------------ */

/* a₁ = (1, 0), a₂ = (½, √3/2). */
#define KG_AX 1.0
#define KG_AY 0.0
#define KG_BX 0.5
#define KG_BY 0.86602540378443864676      /* √3/2 */
#define KG_RB_X 0.5                        /* B at a₁/2 */
#define KG_RB_Y 0.0
#define KG_RC_X 0.25                       /* C at a₂/2 */
#define KG_RC_Y 0.43301270189221932338     /* √3/4 */

static void kagome_site_position(int cx, int cy, int sub,
                                 double *x, double *y) {
    double rsx = (sub == 1) ? KG_RB_X : (sub == 2) ? KG_RC_X : 0.0;
    double rsy = (sub == 1) ? KG_RB_Y : (sub == 2) ? KG_RC_Y : 0.0;
    *x = cx * KG_AX + cy * KG_BX + rsx;
    *y = cx * KG_AY + cy * KG_BY + rsy;
}

/* Invert a Cartesian position to (cx, cy, sub) on an L × L kagome torus
 * (PBC).  Returns site index in [0, 3L²) on success, -1 on failure.
 *
 * Strategy: try each of the 3 sublattices.  For sublattice s with
 * offset r_s, solve cx · a₁ + cy · a₂ = (x − r_s.x, y − r_s.y) for real
 * (cx, cy).  Round to nearest integer mod L; accept if the round-off
 * residual is within tol. */
static int kagome_position_to_site(double x, double y, int L, double tol) {
    /* Inverse of [a₁  a₂] = [[1, 1/2], [0, √3/2]]:
     *   (a₁ a₂)^{-1} = [[1, -1/√3], [0, 2/√3]]
     * cx_real = x' - y'/√3
     * cy_real = (2/√3) y' */
    const double inv_sqrt3 = 0.57735026918962576451;
    const double two_over_sqrt3 = 1.15470053837925152902;
    for (int sub = 0; sub < 3; sub++) {
        double rsx = (sub == 1) ? KG_RB_X : (sub == 2) ? KG_RC_X : 0.0;
        double rsy = (sub == 1) ? KG_RB_Y : (sub == 2) ? KG_RC_Y : 0.0;
        double xs = x - rsx, ys = y - rsy;
        double cx_real = xs - ys * inv_sqrt3;
        double cy_real = ys * two_over_sqrt3;
        /* Round to nearest integer. */
        long cx_int = (long)floor(cx_real + 0.5);
        long cy_int = (long)floor(cy_real + 0.5);
        if (fabs(cx_real - (double)cx_int) < tol &&
            fabs(cy_real - (double)cy_int) < tol) {
            int cx = (int)(((cx_int % L) + L) % L);
            int cy = (int)(((cy_int % L) + L) % L);
            return 3 * (cx * L + cy) + sub;
        }
    }
    return -1;
}

/* Compose one C₃ point op (k = 0 → identity, k = 1 → 120°, k = 2 →
 * 240°) with a translation (tx, ty) and write the resulting permutation
 * row.  Returns 0 on success, -1 if any site failed to invert. */
static int build_p3_perm_row(int L, int k, int tx, int ty, int *row) {
    int N = 3 * L * L;
    /* C₃ centre: up-triangle (0, 0) centroid. */
    const double tau_x = 0.25;
    const double tau_y = 0.14433756729740643112;   /* √3/12 */
    /* Rotation matrix R(120° · k). */
    double theta = (2.0 * 3.14159265358979323846 * (double)k) / 3.0;
    double cs = cos(theta), sn = sin(theta);

    for (int cx = 0; cx < L; cx++) {
        for (int cy = 0; cy < L; cy++) {
            for (int sub = 0; sub < 3; sub++) {
                /* Source-side site.  We want π_g(i) = the site that
                 * spins[π_g(i)] should be picked up at output index i.
                 * Convention: g · s evaluated at i = s_{π_g(i)}.  So
                 * π_g is the *inverse* permutation of the geometric
                 * transformation.  For a rotation R, the geometric
                 * transformation maps site at position p to position
                 * R p.  So spins at the new site = spins at original
                 * site at position R^{-1} p_new.
                 *
                 * Concretely: row index i corresponds to output site
                 * (cx, cy, sub) at position p_i.  We compute p_src =
                 * R^{-1} (p_i − τ₀) + τ₀, shifted by translation (in
                 * lattice basis), then look up which (cx', cy', sub')
                 * has that Cartesian position. */
                double px, py;
                kagome_site_position(cx, cy, sub, &px, &py);
                /* Apply inverse rotation R^{-1} = R(−θ). */
                double dx = px - tau_x, dy = py - tau_y;
                double sx = cs * dx + sn * dy + tau_x;
                double sy = -sn * dx + cs * dy + tau_y;
                /* Apply inverse translation: p_src ← p_src − tx · a₁ − ty · a₂. */
                sx -= tx * KG_AX + ty * KG_BX;
                sy -= tx * KG_AY + ty * KG_BY;
                int src = kagome_position_to_site(sx, sy, L, 1e-6);
                if (src < 0) return -1;
                row[3 * (cx * L + cy) + sub] = src;
            }
        }
    }
    (void)N;
    return 0;
}

int nqs_kagome_p3_perm(int L,
                        int **out_perm,
                        double **out_characters,
                        int *out_num_elements) {
    if (L <= 0 || !out_perm || !out_characters || !out_num_elements) return -1;
    int N = 3 * L * L;
    int G = 3 * L * L;

    int *perm = (int *)malloc((size_t)G * (size_t)N * sizeof(int));
    double *chars = (double *)malloc((size_t)G * sizeof(double));
    if (!perm || !chars) { free(perm); free(chars); return -1; }

    int g = 0;
    for (int k = 0; k < 3; k++) {
        for (int tx = 0; tx < L; tx++) {
            for (int ty = 0; ty < L; ty++) {
                int *row = &perm[(size_t)g * N];
                if (build_p3_perm_row(L, k, tx, ty, row) != 0) {
                    fprintf(stderr,
                            "nqs_kagome_p3_perm: failed to build row "
                            "g=%d (k=%d, tx=%d, ty=%d) on L=%d torus.\n",
                            g, k, tx, ty, L);
                    free(perm); free(chars);
                    return -1;
                }
                chars[g] = 1.0;            /* A₁ trivial irrep */
                g++;
            }
        }
    }

    *out_perm = perm;
    *out_characters = chars;
    *out_num_elements = G;
    return 0;
}

/* ------------------------------------------------------------------ */
/* p6 (translations × C₆) on an L × L kagome torus.                   */
/*                                                                    */
/* C₆ centre: hexagon centroid at (a₁ + a₂)/2 = (3/4, √3/4).          */
/* This is the unique 6-fold rotation centre per unit cell (verified  */
/* by tools/find_kagome_p6_centre).  Same numerical perm-building     */
/* strategy as p3, just with 6 rotations instead of 3.                */
/* ------------------------------------------------------------------ */

static int build_p6_perm_row(int L, int k, int tx, int ty, int *row) {
    /* C₆ centre in Cartesian. */
    const double hex_x = 0.75;
    const double hex_y = 0.43301270189221932338;   /* √3/4 */
    /* Rotation matrix R(60° · k). */
    double theta = (2.0 * 3.14159265358979323846 * (double)k) / 6.0;
    double cs = cos(theta), sn = sin(theta);

    for (int cx = 0; cx < L; cx++) {
        for (int cy = 0; cy < L; cy++) {
            for (int sub = 0; sub < 3; sub++) {
                /* See nqs_kagome_p3_perm comment for the source-side
                 * convention: π_g(i) is the index that g·s should pick
                 * up, which is the inverse of the geometric transform. */
                double px, py;
                kagome_site_position(cx, cy, sub, &px, &py);
                /* Inverse rotation R(−θ). */
                double dx = px - hex_x, dy = py - hex_y;
                double sx =  cs * dx + sn * dy + hex_x;
                double sy = -sn * dx + cs * dy + hex_y;
                /* Inverse translation. */
                sx -= tx * 1.0   + ty * 0.5;
                sy -= tx * 0.0   + ty * 0.86602540378443864676;
                int src = kagome_position_to_site(sx, sy, L, 1e-6);
                if (src < 0) return -1;
                row[3 * (cx * L + cy) + sub] = src;
            }
        }
    }
    return 0;
}

int nqs_kagome_p6_perm(int L,
                        int **out_perm,
                        double **out_characters,
                        int *out_num_elements) {
    if (L <= 0 || !out_perm || !out_characters || !out_num_elements) return -1;
    int N = 3 * L * L;
    int G = 6 * L * L;

    int *perm = (int *)malloc((size_t)G * (size_t)N * sizeof(int));
    double *chars = (double *)malloc((size_t)G * sizeof(double));
    if (!perm || !chars) { free(perm); free(chars); return -1; }

    int g = 0;
    for (int k = 0; k < 6; k++) {
        for (int tx = 0; tx < L; tx++) {
            for (int ty = 0; ty < L; ty++) {
                int *row = &perm[(size_t)g * N];
                if (build_p6_perm_row(L, k, tx, ty, row) != 0) {
                    fprintf(stderr,
                            "nqs_kagome_p6_perm: failed to build row "
                            "g=%d (k=%d, tx=%d, ty=%d) on L=%d torus.\n",
                            g, k, tx, ty, L);
                    free(perm); free(chars);
                    return -1;
                }
                chars[g] = 1.0;            /* A₁ trivial irrep */
                g++;
            }
        }
    }

    *out_perm = perm;
    *out_characters = chars;
    *out_num_elements = G;
    return 0;
}

#undef KG_SITE
#undef KG_AX
#undef KG_AY
#undef KG_BX
#undef KG_BY
#undef KG_RB_X
#undef KG_RB_Y
#undef KG_RC_X
#undef KG_RC_Y
