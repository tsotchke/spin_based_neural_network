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

/*
 * (Γ, A₁) sector exact diagonalisation for the kagome Heisenberg
 * antiferromagnet on an L × L torus with periodic BC.  Total sites
 * N = 3 L²; sector dim is the count of orbit representatives at
 * popcount = N/2 (Sz = 0) under the full p6mm wallpaper group with
 * non-zero σ-norm at the trivial irrep.
 *
 * Sector dim is roughly  C(N, N/2) / |G|  with |G| = 12 · L² for
 * p6mm.  This is the size that fits Lanczos through to N = 27
 * (L = 3, sector ≲ 10⁵) and N = 48 (L = 4, sector ≲ 10⁷) — the
 * regime where the kagome AFM ground-state physics actually emerges.
 *
 * On success, eigvals_out is filled with the k_wanted smallest
 * eigenvalues of H = J Σ_<ij> S_i · S_j projected onto (Γ, A₁) ∩
 * Sz = 0.  For bipartite-like AFMs the global ground state lives in
 * (Γ, A₁), so this is the absolute E_0; for kagome AFM specifically
 * (frustrated, gapped vs gapless still open), this is the Z₂-spin-
 * liquid-candidate sector ground state.
 *
 * Returns 0 on success; non-zero codes:
 *   -1 invalid argument (e.g. L ≤ 0)
 *   -2 num_sites > 27 (memory cap; raise once you have the budget)
 *   -3 libirrep build failure (likely OOM on rep_table or sector)
 *   -4 Lanczos error
 */
int nqs_kspace_ed_kagome_at_gamma(int L, double J,
                                   int k_wanted, int max_iters,
                                   double *eigvals_out);

#endif /* SPIN_NN_HAS_IRREP */

#ifdef __cplusplus
}
#endif

#endif /* NQS_KSPACE_ED_H */
