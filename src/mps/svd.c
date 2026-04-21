/*
 * src/mps/svd.c
 *
 * One-sided Jacobi SVD. The key invariant: maintain U such that
 * U = A · V. When A has columns a_0 ... a_{n-1} and we apply a Givens
 * rotation on columns (i, j), U's columns (i, j) rotate in tandem.
 * Convergence when all column pairs are mutually orthogonal.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "mps/svd.h"

/* Compute the angle that orthogonalises columns (i, j) of B.
 * Returns cos(θ), sin(θ) such that rotating (b_i, b_j) → (b_i', b_j')
 * zeroes ⟨b_i', b_j'⟩.
 *
 *   α = ||b_i||²,  β = ||b_j||²,  γ = ⟨b_i, b_j⟩
 *   ζ = (β - α) / (2 γ)
 *   t = sign(ζ) / (|ζ| + sqrt(1 + ζ²))
 *   c = 1 / sqrt(1 + t²)
 *   s = t · c
 */
static void jacobi_angle(double alpha, double beta, double gamma,
                         double *out_c, double *out_s) {
    if (fabs(gamma) < 1e-300) { *out_c = 1.0; *out_s = 0.0; return; }
    double zeta = (beta - alpha) / (2.0 * gamma);
    double sign = (zeta >= 0.0) ? 1.0 : -1.0;
    double t = sign / (fabs(zeta) + sqrt(1.0 + zeta * zeta));
    double c = 1.0 / sqrt(1.0 + t * t);
    double s = t * c;
    *out_c = c;
    *out_s = s;
}

int svd_jacobi(const double *a_in, int m, int n,
               double *U, double *s, double *Vt,
               double tol) {
    if (!a_in || !U || !s || !Vt || m <= 0 || n <= 0 || m < n) return -1;
    /* U starts as a copy of A; V starts as I. We rotate columns of U
     * and rows of V^T so that at all times A = U · V^T. */
    memcpy(U, a_in, (size_t)m * (size_t)n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Vt[(size_t)i * n + j] = (i == j) ? 1.0 : 0.0;

    double frob2 = 0.0;
    for (int i = 0; i < m * n; i++) frob2 += a_in[i] * a_in[i];
    if (frob2 == 0.0) {
        for (int i = 0; i < n; i++) s[i] = 0.0;
        return 0;
    }
    double thresh = tol * frob2;

    int max_sweeps = 50;
    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        double off = 0.0;
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                /* Compute column norms and inner product. */
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int i = 0; i < m; i++) {
                    double up = U[(size_t)i * n + p];
                    double uq = U[(size_t)i * n + q];
                    alpha += up * up;
                    beta  += uq * uq;
                    gamma += up * uq;
                }
                off += 2.0 * gamma * gamma;
                if (fabs(gamma) < 1e-300) continue;
                double c, sv;
                jacobi_angle(alpha, beta, gamma, &c, &sv);
                /* Rotate columns p, q of U. */
                for (int i = 0; i < m; i++) {
                    double up = U[(size_t)i * n + p];
                    double uq = U[(size_t)i * n + q];
                    U[(size_t)i * n + p] = c * up - sv * uq;
                    U[(size_t)i * n + q] = sv * up + c * uq;
                }
                /* Rotate rows p, q of V^T. */
                for (int j = 0; j < n; j++) {
                    double vp = Vt[(size_t)p * n + j];
                    double vq = Vt[(size_t)q * n + j];
                    Vt[(size_t)p * n + j] = c * vp - sv * vq;
                    Vt[(size_t)q * n + j] = sv * vp + c * vq;
                }
            }
        }
        if (off < thresh) break;
    }

    /* Extract singular values as U column norms. Normalize U to unit
     * columns so that U Σ V^T = A. */
    for (int p = 0; p < n; p++) {
        double norm2 = 0.0;
        for (int i = 0; i < m; i++) norm2 += U[(size_t)i * n + p] * U[(size_t)i * n + p];
        double sv = sqrt(norm2);
        s[p] = sv;
        if (sv > 1e-300) {
            for (int i = 0; i < m; i++) U[(size_t)i * n + p] /= sv;
        } else {
            for (int i = 0; i < m; i++) U[(size_t)i * n + p] = 0.0;
        }
    }

    /* Sort descending by singular value; permute U columns and V^T rows. */
    for (int i = 0; i < n - 1; i++) {
        int max_j = i;
        for (int j = i + 1; j < n; j++) if (s[j] > s[max_j]) max_j = j;
        if (max_j != i) {
            double tmp = s[i]; s[i] = s[max_j]; s[max_j] = tmp;
            for (int k = 0; k < m; k++) {
                double t = U[(size_t)k * n + i];
                U[(size_t)k * n + i] = U[(size_t)k * n + max_j];
                U[(size_t)k * n + max_j] = t;
            }
            for (int k = 0; k < n; k++) {
                double t = Vt[(size_t)i * n + k];
                Vt[(size_t)i * n + k] = Vt[(size_t)max_j * n + k];
                Vt[(size_t)max_j * n + k] = t;
            }
        }
    }
    return 0;
}
