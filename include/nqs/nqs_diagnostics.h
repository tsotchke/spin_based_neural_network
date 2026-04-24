/*
 * include/nqs/nqs_diagnostics.h
 *
 * Sample-based diagnostics for NQS variational wavefunctions. Pure
 * observers — never mutate the ansatz. Each diagnostic consumes a
 * freshly sampled batch from the sampler and returns a scalar (or
 * small array) observable of interest for the research pipeline.
 *
 * Currently shipped:
 *   - nqs_compute_chi_F: trace of the quantum geometric tensor
 *     (fidelity susceptibility / Fubini–Study metric trace). Useful
 *     as a convergence diagnostic and, on symmetry-projected runs,
 *     as the replacement for the falsified scalar-curvature QPT
 *     detector.
 *
 * Planned (per docs/research/kagome_KH_plan.md open-items list):
 *   - per-bond-class mean phase (bipartite phase probe).
 *   - two-point spin-spin correlator ⟨S_i · S_j⟩(r).
 *   - excited-state variance projector for gap estimation.
 */
#ifndef NQS_DIAGNOSTICS_H
#define NQS_DIAGNOSTICS_H

#include "nqs/nqs_config.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_sampler.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Number of kagome bond classes, indexed by sublattice pair:
 *   0 → A-B bonds,  1 → A-C bonds,  2 → B-C bonds. */
#define NQS_KAGOME_NUM_BOND_CLASSES 3

/*
 * Compute the trace of the quantum geometric tensor S from a freshly
 * sampled batch.
 *
 *     S_{k,l} = ⟨O_k* O_l⟩_{|ψ|²} − ⟨O_k*⟩⟨O_l⟩
 *     Tr(S)   = Σ_k ( ⟨|O_k|²⟩ − |⟨O_k⟩|² )
 *
 * where O_k = ∂ log ψ / ∂θ_k evaluated at sample configurations. The
 * trace is related to the fidelity susceptibility χ_F by a well-known
 * Zanardi–Paunković 2006 identity (up to basis/normalisation
 * conventions, Tr(S) = 2·χ_F in the real-θ convention used here).
 *
 * Consumes cfg->num_samples configurations from the sampler. The
 * caller is responsible for thermalising the sampler first if it has
 * not yet been used.
 *
 * Both real and complex-amplitude ansätze are supported; the complex
 * case uses |O_k|² = Re(O_k)² + Im(O_k)².
 *
 * On success writes:
 *   *out_trace_S   ← Tr(S)  (the total)
 *   *out_per_param ← Tr(S) / num_params  (averaged, optional; may be NULL)
 *
 * Returns 0 on success, non-zero on error (bad arguments or allocation).
 */
int nqs_compute_chi_F(const nqs_config_t *cfg,
                      int size_x, int size_y,
                      nqs_ansatz_t *ansatz,
                      nqs_sampler_t *sampler,
                      double *out_trace_S,
                      double *out_per_param);

/*
 * Bipartite-phase / bond-flip amplitude probe on the kagome lattice.
 *
 * For each opposite-spin bond (i,j) of class α encountered in a
 * Metropolis-sampled configuration s, the helper records the complex
 * amplitude ratio
 *
 *     r_{ij}(s) = ψ(s_{ij}) / ψ(s)       (with s_{ij} = s with spins i,j flipped)
 *
 * and accumulates its real and imaginary parts. Normalising by the
 * number of (bond, sample) pairs in each class gives the per-class
 * circular mean
 *
 *     ⟨exp(i Δφ_α) · |ψ(s_{ij})/ψ(s)|⟩
 *
 * which equals −1 for Marshall-sign-locked bipartite antiferromagnets
 * (Heisenberg AFM on a bipartite lattice) and exhibits non-trivial
 * per-class structure on frustrated kagome. The three bond classes
 * correspond to the three sublattice pairs {A-B, A-C, B-C} on both
 * up- and down-triangles.
 *
 * Why the ratio is averaged (not the phase): arg ψ(s) itself is only
 * defined up to a global gauge, but ratios are gauge-invariant, and
 * the complex ratio r keeps both the magnitude change (|ψ'|/|ψ|) and
 * the phase shift (Δφ).
 *
 * Consumes cfg->num_samples configurations from the sampler. The
 * caller must have thermalised the sampler already (or set
 * cfg->num_thermalize so the sampler does it internally).
 *
 * Writes per-class real and imaginary means into
 * out_mean_re[α], out_mean_im[α] for α = 0..2 (arrays length 3, caller
 * must allocate). If out_counts is non-NULL, writes the number of
 * (bond, sample) pairs in each class (useful when sampling rarely hits
 * configurations with all three classes represented).
 *
 * Currently only implemented for cfg->hamiltonian ==
 * NQS_HAM_KAGOME_HEISENBERG; returns a non-zero error otherwise.
 *
 * Returns 0 on success, non-zero on error.
 */
int nqs_compute_kagome_bond_phase(const nqs_config_t *cfg,
                                   int Lx_cells, int Ly_cells,
                                   nqs_ansatz_t *ansatz,
                                   nqs_sampler_t *sampler,
                                   double *out_mean_re,
                                   double *out_mean_im,
                                   long   *out_counts);

#ifdef __cplusplus
}
#endif

#endif /* NQS_DIAGNOSTICS_H */
