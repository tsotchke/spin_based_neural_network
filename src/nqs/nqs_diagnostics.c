/*
 * src/nqs/nqs_diagnostics.c
 *
 * Sample-based diagnostics for NQS variational wavefunctions.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_diagnostics.h"
#include "nqs/nqs_sampler.h"

int nqs_compute_chi_F(const nqs_config_t *cfg,
                      int size_x, int size_y,
                      nqs_ansatz_t *ansatz,
                      nqs_sampler_t *sampler,
                      double *out_trace_S,
                      double *out_per_param) {
    if (!cfg || !ansatz || !sampler || !out_trace_S) return -1;

    long num_params = nqs_ansatz_num_params(ansatz);
    if (num_params <= 0) return -1;

    int N = nqs_sampler_num_sites(sampler);
    int batch_size = cfg->num_samples;
    if (batch_size <= 0) return -1;
    (void)size_x; (void)size_y;  /* signature parity with nqs_sr_step */

    /* Pre-sample the batch. */
    int *batch = malloc((size_t)batch_size * (size_t)N * sizeof(int));
    if (!batch) return -1;
    if (nqs_sampler_batch(sampler, batch_size, batch) != 0) {
        free(batch);
        return -1;
    }

    /* Accumulators for mean and mean-of-squares over samples. Both are
     * length num_params, initialised to zero. */
    double *mean_re = calloc((size_t)num_params, sizeof(double));
    double *mean_im = calloc((size_t)num_params, sizeof(double));
    double *msq     = calloc((size_t)num_params, sizeof(double));
    double *grad_re = malloc((size_t)num_params * sizeof(double));
    double *grad_im = malloc((size_t)num_params * sizeof(double));
    if (!mean_re || !mean_im || !msq || !grad_re || !grad_im) {
        free(batch); free(mean_re); free(mean_im); free(msq);
        free(grad_re); free(grad_im);
        return -1;
    }

    /* Per-sample gradient accumulation. We use the complex gradient
     * path so both real and complex ansätze work without branching:
     * real ansätze simply fill grad_im with zeros. */
    for (int s = 0; s < batch_size; s++) {
        const int *spins = &batch[(size_t)s * (size_t)N];
        int rc = nqs_ansatz_logpsi_gradient_complex(
            ansatz, spins, N, grad_re, grad_im);
        if (rc != 0) {
            free(batch); free(mean_re); free(mean_im); free(msq);
            free(grad_re); free(grad_im);
            return -1;
        }
        for (long k = 0; k < num_params; k++) {
            mean_re[k] += grad_re[k];
            mean_im[k] += grad_im[k];
            msq[k]     += grad_re[k] * grad_re[k]
                        + grad_im[k] * grad_im[k];
        }
    }

    /* Finalise means and compute Tr(S). */
    double inv_batch = 1.0 / (double)batch_size;
    double trace = 0.0;
    for (long k = 0; k < num_params; k++) {
        double mk_re = mean_re[k] * inv_batch;
        double mk_im = mean_im[k] * inv_batch;
        double mean_sq_magnitude = mk_re * mk_re + mk_im * mk_im;
        double mean_of_sq        = msq[k] * inv_batch;
        double variance_k        = mean_of_sq - mean_sq_magnitude;
        /* Clamp tiny-negative-due-to-floating-point to zero; the
         * theoretical value S_{kk} is non-negative. */
        if (variance_k < 0.0 && variance_k > -1e-12) variance_k = 0.0;
        trace += variance_k;
    }

    *out_trace_S = trace;
    if (out_per_param) *out_per_param = trace / (double)num_params;

    free(batch); free(mean_re); free(mean_im); free(msq);
    free(grad_re); free(grad_im);
    return 0;
}

/* Return the sublattice index (0=A, 1=B, 2=C) for a kagome site.
 * Sites are indexed by `3*(cx*Ly + cy) + sublattice` in this repo. */
static inline int kg_sublattice(int site_index) {
    return site_index % 3;
}

/* Bond class index for a kagome bond between sublattices (a, b):
 *   {0,1} → 0   (A-B)
 *   {0,2} → 1   (A-C)
 *   {1,2} → 2   (B-C)
 * Unordered — the mapping is symmetric in (a, b). */
static inline int kg_bond_class(int sub_a, int sub_b) {
    int lo = sub_a < sub_b ? sub_a : sub_b;
    int hi = sub_a < sub_b ? sub_b : sub_a;
    if (lo == 0 && hi == 1) return 0;
    if (lo == 0 && hi == 2) return 1;
    return 2;
}

int nqs_compute_kagome_bond_phase(const nqs_config_t *cfg,
                                   int Lx_cells, int Ly_cells,
                                   nqs_ansatz_t *ansatz,
                                   nqs_sampler_t *sampler,
                                   double *out_mean_re,
                                   double *out_mean_im,
                                   long   *out_counts) {
    if (!cfg || !ansatz || !sampler || !out_mean_re || !out_mean_im)
        return -1;
    if (cfg->hamiltonian != NQS_HAM_KAGOME_HEISENBERG) return -2;
    if (Lx_cells < 1 || Ly_cells < 1) return -1;
    int pbc = cfg->kagome_pbc;
    int N   = 3 * Lx_cells * Ly_cells;
    if (nqs_sampler_num_sites(sampler) != N) return -1;
    int batch_size = cfg->num_samples;
    if (batch_size <= 0) return -1;

    int *batch = malloc((size_t)batch_size * (size_t)N * sizeof(int));
    if (!batch) return -1;
    if (nqs_sampler_batch(sampler, batch_size, batch) != 0) {
        free(batch);
        return -1;
    }

    /* Accumulators: per-class sums of Re/Im of r_{ij}(s) = ψ(s_{ij})/ψ(s). */
    double sum_re[NQS_KAGOME_NUM_BOND_CLASSES] = {0};
    double sum_im[NQS_KAGOME_NUM_BOND_CLASSES] = {0};
    long   count [NQS_KAGOME_NUM_BOND_CLASSES] = {0};

    int *scratch = malloc((size_t)N * sizeof(int));
    if (!scratch) { free(batch); return -1; }

    for (int smp = 0; smp < batch_size; smp++) {
        const int *spins = &batch[(size_t)smp * (size_t)N];
        memcpy(scratch, spins, (size_t)N * sizeof(int));

        double log_abs0, arg0;
        nqs_ansatz_log_amp(spins, N, ansatz, &log_abs0, &arg0);

        for (int cx = 0; cx < Lx_cells; cx++) {
            for (int cy = 0; cy < Ly_cells; cy++) {
                int A = 3 * (cx * Ly_cells + cy) + 0;
                int B = 3 * (cx * Ly_cells + cy) + 1;
                int C = 3 * (cx * Ly_cells + cy) + 2;

                int bonds_up[3][2] = { {A, B}, {A, C}, {B, C} };

                int cxm, cym;
                int use_down = 1;
                if (pbc) {
                    cxm = (cx - 1 + Lx_cells) % Lx_cells;
                    cym = (cy - 1 + Ly_cells) % Ly_cells;
                } else if (cx == 0 || cy == 0) {
                    use_down = 0;
                    cxm = cym = 0;  /* unused */
                } else {
                    cxm = cx - 1;
                    cym = cy - 1;
                }
                int Bm = 3 * (cxm * Ly_cells + cy)  + 1;
                int Cm = 3 * (cx  * Ly_cells + cym) + 2;
                int bonds_dn[3][2] = { {A, Bm}, {A, Cm}, {Bm, Cm} };

                /* Walk up- then down-triangle bonds. */
                int (*groups[2])[2] = { bonds_up, bonds_dn };
                int group_active[2] = { 1, use_down };

                for (int g = 0; g < 2; g++) {
                    if (!group_active[g]) continue;
                    for (int b = 0; b < 3; b++) {
                        int u = groups[g][b][0];
                        int v = groups[g][b][1];
                        int su = spins[u], sv = spins[v];
                        if (su == sv) continue;  /* H_xy only flips opposite-spin bonds */

                        int cls = kg_bond_class(kg_sublattice(u),
                                                 kg_sublattice(v));
                        scratch[u] = -su; scratch[v] = -sv;
                        double log_abs1, arg1;
                        nqs_ansatz_log_amp(scratch, N, ansatz,
                                            &log_abs1, &arg1);
                        scratch[u] = su; scratch[v] = sv;

                        double mag = exp(log_abs1 - log_abs0);
                        double dth = arg1 - arg0;
                        sum_re[cls] += mag * cos(dth);
                        sum_im[cls] += mag * sin(dth);
                        count [cls] += 1;
                    }
                }
            }
        }
    }

    for (int c = 0; c < NQS_KAGOME_NUM_BOND_CLASSES; c++) {
        if (count[c] > 0) {
            double inv = 1.0 / (double)count[c];
            out_mean_re[c] = sum_re[c] * inv;
            out_mean_im[c] = sum_im[c] * inv;
        } else {
            out_mean_re[c] = 0.0;
            out_mean_im[c] = 0.0;
        }
        if (out_counts) out_counts[c] = count[c];
    }

    free(scratch);
    free(batch);
    return 0;
}
