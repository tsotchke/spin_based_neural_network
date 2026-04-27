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

#include <irrep/hamiltonian.h>
#include <irrep/rdm.h>
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

#endif /* SPIN_NN_HAS_IRREP */
