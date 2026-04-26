/*
 * tools/find_kagome_p6_center.c
 *
 * Standalone search for the 6-fold rotation centre of the kagome
 * lattice in the convention used by src/nqs/nqs_gradient.c:
 *
 *   sub A at (0, 0)
 *   sub B at a₁/2 = (1/2, 0)
 *   sub C at a₂/2 = (1/4, √3/4)
 *   primitive vectors a₁ = (1, 0), a₂ = (1/2, √3/2)
 *   site index = 3 * (cx * Ly + cy) + sub
 *
 * Iterates a fine grid of candidate centres in the unit cell and
 * reports those for which R(60°) maps every kagome site of an
 * L × L torus to another kagome site (mod PBC, within tolerance).
 *
 * Build:  gcc -O2 -std=c11 -o tools/find_kagome_p6_center tools/find_kagome_p6_center.c -lm
 * Run:    ./tools/find_kagome_p6_center
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define A1X 1.0
#define A1Y 0.0
#define A2X 0.5
#define A2Y 0.86602540378443864676   /* √3/2 */

static void site_pos(int cx, int cy, int sub, double *x, double *y) {
    double rsx = (sub == 1) ? 0.5  : (sub == 2) ? 0.25 : 0.0;
    double rsy = (sub == 1) ? 0.0  : (sub == 2) ? 0.43301270189221932338 : 0.0;
    *x = cx * A1X + cy * A2X + rsx;
    *y = cx * A1Y + cy * A2Y + rsy;
}

/* Invert Cartesian (x, y) to (cx, cy, sub) on L × L torus.
 * Returns site index or -1. */
static int pos_to_site(double x, double y, int L, double tol) {
    const double inv_sqrt3      = 0.57735026918962576451;
    const double two_over_sqrt3 = 1.15470053837925152902;
    for (int sub = 0; sub < 3; sub++) {
        double rsx = (sub == 1) ? 0.5  : (sub == 2) ? 0.25 : 0.0;
        double rsy = (sub == 1) ? 0.0  : (sub == 2) ? 0.43301270189221932338 : 0.0;
        double xs = x - rsx, ys = y - rsy;
        double cx_real = xs - ys * inv_sqrt3;
        double cy_real = ys * two_over_sqrt3;
        long cxi = (long)floor(cx_real + 0.5);
        long cyi = (long)floor(cy_real + 0.5);
        if (fabs(cx_real - cxi) < tol && fabs(cy_real - cyi) < tol) {
            int cx = (int)(((cxi % L) + L) % L);
            int cy = (int)(((cyi % L) + L) % L);
            return 3 * (cx * L + cy) + sub;
        }
    }
    return -1;
}

/* For a candidate centre (cx0, cy0), apply R(60°) to every site and
 * check it lands on a lattice site.  Returns 1 if all OK, 0 otherwise.
 * If verbose, prints the first failing case. */
static int candidate_works(double cx0, double cy0, int L, double tol, int verbose) {
    const double cs60 = 0.5;
    const double sn60 = 0.86602540378443864676;
    int N = 3 * L * L;
    for (int s = 0; s < N; s++) {
        int sub = s % 3;
        int cell = s / 3;
        int cy = cell % L;
        int cx = cell / L;
        double px, py; site_pos(cx, cy, sub, &px, &py);
        double dx = px - cx0, dy = py - cy0;
        double rx = cs60 * dx - sn60 * dy + cx0;
        double ry = sn60 * dx + cs60 * dy + cy0;
        if (pos_to_site(rx, ry, L, tol) < 0) {
            if (verbose)
                fprintf(stderr,
                        "  candidate (%.4f, %.4f) FAIL: site (%d,%d,%d) at "
                        "(%.4f, %.4f) rotates to (%.4f, %.4f), no lattice match\n",
                        cx0, cy0, cx, cy, sub, px, py, rx, ry);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    /* Search a fine grid in the unit cell. */
    const int L = 6;            /* small but big enough for non-trivial PBC */
    const int N_GRID = 200;     /* 200 × 200 = 40 000 candidates */
    const double tol = 1e-4;    /* somewhat loose so a near-miss can be diagnosed */

    int hits = 0;
    for (int i = 0; i < N_GRID; i++) {
        for (int j = 0; j < N_GRID; j++) {
            /* Express candidate in (a₁, a₂) fractional coords: (s, t) ∈ [0, 1)². */
            double s = (double)i / N_GRID;
            double t = (double)j / N_GRID;
            double cx0 = s * A1X + t * A2X;
            double cy0 = s * A1Y + t * A2Y;
            if (candidate_works(cx0, cy0, L, tol, 0)) {
                printf("HIT: fractional (%.4f, %.4f) → cartesian (%.6f, %.6f)\n",
                       s, t, cx0, cy0);
                hits++;
                if (hits >= 6) break;
            }
        }
        if (hits >= 6) break;
    }
    if (hits == 0) {
        fprintf(stderr, "No 6-fold centre found at tol = %.0e on L = %d.\n", tol, L);
        fprintf(stderr, "Trying common candidates with verbose diagnostics:\n");
        double cands[6][2] = {
            { 0.0, 0.0 },
            { 0.5, 0.0 },
            { 0.25, 0.43301270189221932338 },
            { 0.5, 0.28867513459481288225 },                     /* (a₁+a₂)/3 */
            { 0.5 + 0.5*0.5, 0.5 * 0.86602540378443864676 },     /* (a₁+a₂)/2 */
            { 1.0/3.0 + 2.0/3.0*0.5, 2.0/3.0 * 0.86602540378443864676 } /* (a₁+2a₂)/3 */
        };
        for (int k = 0; k < 6; k++) {
            int ok = candidate_works(cands[k][0], cands[k][1], L, tol, 1);
            fprintf(stderr, "  cand %d (%.4f, %.4f): %s\n",
                    k, cands[k][0], cands[k][1], ok ? "OK" : "fail");
        }
    }
    return 0;
}
