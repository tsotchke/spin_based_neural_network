/*
 * src/skyrmion/skyrmion_field.c
 *
 * Belavin-Polyakov skyrmion magnetisation field on a 2D square lattice
 * + lattice topological-charge integration via Berg-Lüscher.
 */

#include "skyrmion/skyrmion_field.h"

#include <math.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void skyrmion_field_compute(const skyrmion_field_params_t *params,
                            int Lx, int Ly,
                            double *m_out) {
    if (!params || !m_out || Lx < 1 || Ly < 1) return;
    const double R = params->R;
    const double cx = params->cx;
    const double cy = params->cy;
    const int Q = params->Q;
    const double eta = params->helicity;
    const double R2 = R * R;
    /* Avoid division by zero at the centre (r = 0): the limit is
     * m_z = 1, m_x = m_y = 0 (north pole).  Numerically we just
     * treat r² < eps as the centre. */
    const double eps = 1e-30;

    for (int ix = 0; ix < Lx; ix++) {
        for (int iy = 0; iy < Ly; iy++) {
            double dx = (double)ix - cx;
            double dy = (double)iy - cy;
            double r2 = dx * dx + dy * dy;
            double mx, my, mz;
            if (r2 < eps) {
                mx = 0.0; my = 0.0; mz = 1.0;
            } else {
                double denom = R2 + r2;
                mz = (R2 - r2) / denom;
                double sin_theta = 2.0 * R * sqrt(r2) / denom;
                double phi = atan2(dy, dx);
                double angle = (double)Q * phi + eta;
                mx = sin_theta * cos(angle);
                my = sin_theta * sin(angle);
            }
            size_t base = (size_t)3 * ((size_t)ix * (size_t)Ly + (size_t)iy);
            m_out[base + 0] = mx;
            m_out[base + 1] = my;
            m_out[base + 2] = mz;
        }
    }
}

/* Spherical area of a triangle with unit-vector vertices a, b, c on S².
 * Uses the closed-form Berg-Lüscher formula:
 *      tan(Ω/2) = a · (b × c) / (1 + a·b + b·c + c·a)
 * Ω is signed; the sign tracks the orientation of the triangle.
 */
static double triangle_area_S2(const double *a, const double *b, const double *c) {
    double ab = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    double bc = b[0]*c[0] + b[1]*c[1] + b[2]*c[2];
    double ca = c[0]*a[0] + c[1]*a[1] + c[2]*a[2];
    /* Triple product a · (b × c) */
    double bxcx = b[1]*c[2] - b[2]*c[1];
    double bxcy = b[2]*c[0] - b[0]*c[2];
    double bxcz = b[0]*c[1] - b[1]*c[0];
    double tp = a[0]*bxcx + a[1]*bxcy + a[2]*bxcz;
    double denom = 1.0 + ab + bc + ca;
    return 2.0 * atan2(tp, denom);
}

double skyrmion_topological_charge(int Lx, int Ly, const double *m) {
    if (!m || Lx < 2 || Ly < 2) return 0.0;
    /* On each plaquette (ix, iy) — (ix+1, iy) — (ix+1, iy+1) — (ix, iy+1)
     * there are two oriented triangles:
     *   T1 = (m_00, m_10, m_11)
     *   T2 = (m_00, m_11, m_01)
     * Sum the spherical areas; total / (4π) is Q.
     * Open boundaries: skip plaquettes that would wrap. */
    double total = 0.0;
    for (int ix = 0; ix < Lx - 1; ix++) {
        for (int iy = 0; iy < Ly - 1; iy++) {
            const double *m00 = &m[3 * ((size_t)ix * Ly + iy)];
            const double *m10 = &m[3 * (((size_t)ix + 1) * Ly + iy)];
            const double *m11 = &m[3 * (((size_t)ix + 1) * Ly + (iy + 1))];
            const double *m01 = &m[3 * ((size_t)ix * Ly + (iy + 1))];
            total += triangle_area_S2(m00, m10, m11);
            total += triangle_area_S2(m00, m11, m01);
        }
    }
    return total / (4.0 * M_PI);
}
