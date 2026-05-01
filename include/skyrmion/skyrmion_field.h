/*
 * include/skyrmion/skyrmion_field.h
 *
 * Magnetisation field for a magnetic skyrmion of integer topological
 * charge Q on a 2D square lattice.  Uses the Belavin–Polyakov 1975
 * conformal-mapping profile, which saturates the BPS bound and gives
 * exact integer Q for all R > 0:
 *
 *      m_z(r) = (R² - r²) / (R² + r²)
 *      m_x(r) = sin θ(r) · cos(Q·φ + η)
 *      m_y(r) = sin θ(r) · sin(Q·φ + η)
 *
 *      sin θ(r) = 2 R r / (R² + r²)
 *
 * where r = ‖x − x₀‖, φ = atan2(y − y₀, x − x₀), R is the skyrmion
 * radius, and η ∈ [0, 2π) is the helicity (η = 0: hedgehog / Néel;
 * η = π/2: chiral / Bloch).  The charge Q is the wrapping number of
 * the m: ℝ² → S² map; Q = +1 is the standard skyrmion, Q = −1 the
 * antiskyrmion, |Q| ≥ 2 the higher-charge species (skyrmionium for
 * Q = 2 and similar).
 *
 * The magnetisation field couples linearly to electron spin via
 * H_xc = J m(r)·σ in the BdG Hamiltonian (see bdg2d.h); it is the
 * input to the |2Q|-Majorana-zero-mode result of the companion
 * skyrmion-Majorana-Clifford paper.
 *
 * No external dependencies; pure C11 + libm.
 */

#ifndef SBNN_SKYRMION_FIELD_H
#define SBNN_SKYRMION_FIELD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int Q;          /* topological charge (any nonzero int) */
    double R;       /* skyrmion radius (lattice units) */
    double cx;      /* centre, x (lattice units) */
    double cy;      /* centre, y (lattice units) */
    double helicity;/* η ∈ [0, 2π); 0 = Néel, π/2 = Bloch */
} skyrmion_field_params_t;

/**
 * Compute the unit-vector magnetisation field of a charge-Q skyrmion
 * on an Lx × Ly square lattice.
 *
 * @param params  skyrmion parameters (Q, R, cx, cy, helicity).
 * @param Lx, Ly  lattice extents.  Must be ≥ 1.
 * @param m_out   output buffer of length 3·Lx·Ly, layout
 *                m_out[3·(ix·Ly + iy) + a]  for axis a ∈ {0=x, 1=y, 2=z}.
 *                Caller-allocated.
 */
void skyrmion_field_compute(const skyrmion_field_params_t *params,
                            int Lx, int Ly,
                            double *m_out);

/**
 * Compute the topological charge of a magnetisation field on an
 * Lx × Ly lattice via the lattice Berg–Lüscher discretisation:
 *
 *      Q = (1/4π) Σ_{plaquette} A(m_1, m_2, m_3)
 *
 * where A is the spherical area of the triangle (m_1, m_2, m_3) on
 * S² and the sum runs over all elementary triangles of each square
 * plaquette (two triangles per plaquette, oriented).  This converges
 * to the continuum integer Q in the smooth limit.
 *
 * @param Lx, Ly  lattice extents.
 * @param m       magnetisation field, layout as in skyrmion_field_compute.
 * @return        topological charge (real-valued; round to int for clean fields).
 */
double skyrmion_topological_charge(int Lx, int Ly, const double *m);

#ifdef __cplusplus
}
#endif

#endif /* SBNN_SKYRMION_FIELD_H */
