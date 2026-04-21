/*
 * include/mps/tebd.h
 *
 * Time-Evolving Block Decimation: real-time evolution of an MPS
 * under a nearest-neighbour Hamiltonian. The 2nd-order Suzuki-Trotter
 * split
 *
 *     U(dt) ≈  Π_even e^{-iH_e dt/2}  ·  Π_odd e^{-iH_o dt}
 *                                    ·  Π_even e^{-iH_e dt/2}
 *
 * (where H = Σ_even H_bond + Σ_odd H_bond) decomposes the evolution
 * into products of commuting two-site gates. Each gate is applied by
 * merging two adjacent MPS tensors, multiplying by the gate, and
 * splitting back via SVD with bond-dim truncation.
 *
 * MPS is stored in the same (D_l, 2, D_r) per-site layout as the
 * DMRG module. Tensors are real doubles for stoquastic / real-Hermitian
 * gates; the complex U(dt) case requires complex tensors and is a
 * v0.7 follow-up. For verification purposes this file ships an
 * imaginary-time step (dt = -i τ) that projects onto the ground
 * state and reproduces DMRG energies.
 */
#ifndef MPS_TEBD_H
#define MPS_TEBD_H

#include <stddef.h>
#include "mps/mps.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Imaginary-time TEBD step: applies one second-order Trotter split
 * of e^{-H τ} to a real MPS. The state is re-normalised after each
 * sweep so the ground-state projection is well-defined.
 *
 *   cfg->ham     selects the 2-site gate (TFIM, Heisenberg, XXZ)
 *   cfg->max_bond_dim truncates after each SVD split
 *   tau          imaginary-time increment (τ > 0)
 *   num_sites    length of the chain
 *
 * Returns the ⟨ψ|H|ψ⟩ after the step (real, computed from the
 * standard 2-site-expectation sum) so the caller can track the
 * energy descent toward the ground state. */
int mps_tebd_imaginary_step(const mps_config_t *cfg,
                             double tau, int num_sites,
                             double *out_energy);

/* Sweep driver: applies `num_sweeps` imaginary-time steps, each of
 * size tau, starting from a random right-canonical MPS. Writes the
 * per-sweep energies into `out_energy_trace` (length num_sweeps). */
int mps_tebd_imaginary_run(const mps_config_t *cfg,
                            double tau, int num_sweeps,
                            double *out_energy_trace,
                            double *out_final_energy);

/* Real-time evolution of a 1D spin-1/2 chain under a Hamiltonian
 * described by cfg. For N ≤ 20 we carry the full complex state
 * vector and evolve with 4th-order Runge-Kutta on the Schrödinger
 * equation dψ/dt = -i H ψ. The initial state is a product of
 * single-site pure states whose Bloch-vector directions are given in
 * `initial_sites_xyz` (length num_sites × 3, unit-length each):
 *   |s_i⟩ = cos(θ/2)|↑⟩ + e^{iφ} sin(θ/2)|↓⟩
 * with (θ, φ) = (acos(z), atan2(y, x)).
 *
 * Outputs (either may be NULL):
 *   out_mz_trace[step*num_sites + i] = ⟨σ^z_i⟩(t = step·dt)
 *   out_loschmidt[step] = |⟨ψ(0)|ψ(t)⟩|²
 *
 * The Loschmidt-echo return is the canonical non-equilibrium probe:
 * after a quench into a gapped paramagnet it develops characteristic
 * dynamical phase transitions / zeros. */
int mps_tebd_real_time_run(const mps_config_t *cfg,
                            const double *initial_sites_xyz,
                            double dt, int num_steps,
                            double *out_mz_trace,
                            double *out_loschmidt);

#ifdef __cplusplus
}
#endif

#endif /* MPS_TEBD_H */
