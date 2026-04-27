/*
 * src/nqs/nqs_kspace_ed.c
 *
 * libirrep-backed full-Hilbert-space Heisenberg ED.
 */
#include "nqs/nqs_kspace_ed.h"

#ifdef SPIN_NN_HAS_IRREP

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <irrep/config_project.h>
#include <irrep/hamiltonian.h>
#include <irrep/lattice.h>
#include <irrep/rdm.h>
#include <irrep/space_group.h>
#include <irrep/types.h>

int nqs_kspace_ed_heisenberg(int num_sites, int num_bonds,
                              const int *bi, const int *bj,
                              double J,
                              int k_wanted, int max_iters,
                              double *eigvals_out) {
    if (num_sites <= 0 || num_bonds < 0 || !bi || !bj) return -1;
    if (k_wanted <= 0 || max_iters <= 0 || !eigvals_out) return -1;
    if (num_sites > 24) {
        /* Hilbert-space dim 2^25 = 32 M complex doubles ≈ 0.5 GB; cap
         * here for the full-space path.  k-space-projected sectors
         * (forthcoming) handle larger N. */
        fprintf(stderr,
                "nqs_kspace_ed_heisenberg: num_sites=%d > 24 — "
                "full-Hilbert-space path is too memory-heavy.  Use the "
                "k-space-sector builder once it lands.\n", num_sites);
        return -2;
    }

    irrep_heisenberg_t *H = irrep_heisenberg_new(num_sites, num_bonds, bi, bj, J);
    if (!H) return -3;

    long long dim = irrep_heisenberg_dim(H);
    /* Reorthogonalised Lanczos with deterministic pseudo-random seed
     * (NULL → libirrep generates one internally).  k_wanted smallest
     * eigenvalues, max_iters ≥ 2·k_wanted recommended. */
    irrep_status_t rc = irrep_lanczos_eigvals_reorth(
        irrep_heisenberg_apply, H, dim,
        k_wanted, max_iters,
        /*seed*/ NULL, eigvals_out);

    irrep_heisenberg_free(H);
    return (rc == IRREP_OK) ? 0 : -4;
}

int nqs_kspace_ed_kagome_at_gamma(int L, double J,
                                   int k_wanted, int max_iters,
                                   double *eigvals_out) {
    if (L <= 0 || k_wanted <= 0 || max_iters <= 0 || !eigvals_out) return -1;
    int N = 3 * L * L;
    if (N > 27) {
        fprintf(stderr,
                "nqs_kspace_ed_kagome_at_gamma: N = %d > 27 (L = %d) — "
                "rep_table_build is capped at N ≤ 27 in this build.\n", N, L);
        return -2;
    }
    /* p6mm requires Lx = Ly for the full point group to act on the torus. */
    irrep_lattice_t *lat = irrep_lattice_build(IRREP_LATTICE_KAGOME, L, L);
    if (!lat) return -3;

    irrep_space_group_t *G = irrep_space_group_build(lat, IRREP_WALLPAPER_P6MM);
    if (!G) { irrep_lattice_free(lat); return -3; }

    /* Sz = 0 sector → popcount = N/2 (only meaningful for even N).
     * Kagome N = 3 L² is even iff L is even.  For odd L the
     * minimum-Sz sector is (N-1)/2; we use that. */
    int popcount = N / 2;
    irrep_sg_rep_table_t *T = irrep_sg_rep_table_build(G, popcount);
    if (!T) {
        irrep_space_group_free(G); irrep_lattice_free(lat); return -3;
    }

    /* Γ point: kx = ky = 0. */
    irrep_sg_little_group_t *lg = irrep_sg_little_group_build(G, 0, 0);
    if (!lg) {
        irrep_sg_rep_table_free(T);
        irrep_space_group_free(G); irrep_lattice_free(lat); return -3;
    }
    irrep_sg_little_group_irrep_t *mu_k =
        irrep_sg_little_group_irrep_named(lg, IRREP_LG_IRREP_A1);
    if (!mu_k) {
        irrep_sg_little_group_free(lg);
        irrep_sg_rep_table_free(T);
        irrep_space_group_free(G); irrep_lattice_free(lat); return -3;
    }

    /* Build the Heisenberg Hamiltonian on the kagome bond list. */
    int num_bonds = irrep_lattice_num_bonds_nn(lat);
    int *bi = (int *)malloc((size_t)num_bonds * sizeof(int));
    int *bj = (int *)malloc((size_t)num_bonds * sizeof(int));
    if (!bi || !bj) {
        free(bi); free(bj);
        irrep_sg_little_group_irrep_free(mu_k);
        irrep_sg_little_group_free(lg);
        irrep_sg_rep_table_free(T);
        irrep_space_group_free(G); irrep_lattice_free(lat); return -3;
    }
    irrep_lattice_fill_bonds_nn(lat, bi, bj);
    irrep_heisenberg_t *H = irrep_heisenberg_new(N, num_bonds, bi, bj, J);
    if (!H) {
        free(bi); free(bj);
        irrep_sg_little_group_irrep_free(mu_k);
        irrep_sg_little_group_free(lg);
        irrep_sg_rep_table_free(T);
        irrep_space_group_free(G); irrep_lattice_free(lat); return -3;
    }

    irrep_sg_heisenberg_sector_t *S =
        irrep_sg_heisenberg_sector_build_at_k(H, T, lg, mu_k);
    if (!S) {
        irrep_heisenberg_free(H);
        free(bi); free(bj);
        irrep_sg_little_group_irrep_free(mu_k);
        irrep_sg_little_group_free(lg);
        irrep_sg_rep_table_free(T);
        irrep_space_group_free(G); irrep_lattice_free(lat); return -3;
    }

    long long sec_dim = irrep_sg_heisenberg_sector_dim(S);
    fprintf(stderr,
            "kagome (Γ, A₁) sector: L=%d, N=%d, popcount=%d, sector_dim=%lld\n",
            L, N, popcount, sec_dim);

    /* Sector matvec callback signature matches Lanczos contract. */
    irrep_status_t rc = irrep_lanczos_eigvals_reorth(
        irrep_sg_heisenberg_sector_apply, S,
        sec_dim, k_wanted, max_iters,
        /*seed*/ NULL, eigvals_out);

    irrep_sg_heisenberg_sector_free(S);
    irrep_heisenberg_free(H);
    free(bi); free(bj);
    irrep_sg_little_group_irrep_free(mu_k);
    irrep_sg_little_group_free(lg);
    irrep_sg_rep_table_free(T);
    irrep_space_group_free(G);
    irrep_lattice_free(lat);
    return (rc == IRREP_OK) ? 0 : -4;
}

#endif /* SPIN_NN_HAS_IRREP */
