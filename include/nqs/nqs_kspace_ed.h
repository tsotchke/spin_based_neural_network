/*
 * include/nqs/nqs_kspace_ed.h
 *
 * libirrep-backed exact diagonalisation for spin-½ Heisenberg
 * Hamiltonians.  Uses irrep_heisenberg_new + irrep_lanczos_eigvals_reorth
 * directly; faster, more numerically robust (full reorthogonalisation)
 * than the homegrown nqs_lanczos full-Hilbert-space path.
 *
 *   - irrep_heisenberg_new builds the bond list and provides
 *     irrep_heisenberg_apply as the matvec callback.
 *   - irrep_lanczos_eigvals_reorth runs Krylov with full Gram-Schmidt
 *     re-orthogonalisation, returning the k_wanted smallest eigenvalues
 *     to ~1e-10 in 50-100 iterations on well-separated spectra.
 *
 * v1 (this header) does FULL-Hilbert-space ED on N ≤ 24 with no
 * symmetry projection.  k-space-projected sectors via
 * irrep_sg_heisenberg_sector_build_at_k are a follow-up — they need
 * the space-group + rep-table + little-group machinery and reduce
 * sector dim by factors of 10-1000× on translationally-invariant
 * lattices.
 *
 * Gated by SPIN_NN_HAS_IRREP.
 */
#ifndef NQS_KSPACE_ED_H
#define NQS_KSPACE_ED_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef SPIN_NN_HAS_IRREP

/*
 * Full-Hilbert-space Heisenberg ED via libirrep + reorthogonalised Lanczos.
 *
 * Bond list: nn pairs (bi[b], bj[b]) for b ∈ [0, num_bonds).  J = +1
 * is antiferromagnetic; ground state is the singlet for bipartite
 * lattices.
 *
 * Caller supplies eigvals_out of length k_wanted; on success it is
 * filled in ascending order with the k_wanted smallest eigenvalues
 * of H = J · Σ_<ij> S_i · S_j.
 *
 * Returns 0 on success, non-zero on libirrep error or argument issue.
 */
int nqs_kspace_ed_heisenberg(int num_sites, int num_bonds,
                              const int *bi, const int *bj,
                              double J,
                              int k_wanted, int max_iters,
                              double *eigvals_out);

#endif /* SPIN_NN_HAS_IRREP */

#ifdef __cplusplus
}
#endif

#endif /* NQS_KSPACE_ED_H */
