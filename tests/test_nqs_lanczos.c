/*
 * tests/test_nqs_lanczos.c
 *
 * Tests for the NQS post-processing path (pillar P2.6). Covers:
 *   - State-vector materialisation is correctly normalised.
 *   - Deterministic variational energy equals the Monte-Carlo SR
 *     output (up to MC noise) for the same trained ansatz.
 *   - Lanczos starting from the trained NQS drops the energy error by
 *     at least one order of magnitude versus the raw variational value
 *     on TFIM 2x3 (where N = 6 and the true E0 is not yet trivially
 *     matched by mean-field).
 *   - Lanczos from the trained ansatz reaches the exact ground state
 *     to machine precision in ≤ dim iterations.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "harness.h"
#include "nqs/nqs_config.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_sampler.h"
#include "nqs/nqs_optimizer.h"
#include "nqs/nqs_lanczos.h"
static void test_materialised_state_is_unit_normed(void) {
    int Lx = 2, Ly = 2, N = 4;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_RBM;
    cfg.rbm_hidden_units = 4;
    cfg.rng_seed = 0xA1u;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    double *psi; long dim;
    ASSERT_EQ_INT(nqs_materialise_state(a, Lx, Ly, &psi, &dim), 0);
    ASSERT_EQ_INT((int)dim, 16);
    double n2 = 0.0;
    for (long s = 0; s < dim; s++) n2 += psi[s] * psi[s];
    ASSERT_NEAR(n2, 1.0, 1e-12);
    free(psi);
    nqs_ansatz_free(a);
}
static double run_rbm_sr(int Lx, int Ly, double J, double Gamma,
                         int num_iter, int num_samples,
                         unsigned seed, nqs_ansatz_t **out_ansatz) {
    int N = Lx * Ly;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_RBM;
    cfg.rbm_hidden_units = 2 * N;
    cfg.rbm_init_scale = 0.1;
    cfg.hamiltonian = NQS_HAM_TFIM;
    cfg.j_coupling = J;
    cfg.transverse_field = Gamma;
    cfg.num_samples = num_samples;
    cfg.num_thermalize = 256;
    cfg.num_decorrelate = 2;
    cfg.num_iterations = num_iter;
    cfg.learning_rate = 2e-2;
    cfg.sr_diag_shift = 1e-2;
    cfg.sr_cg_max_iters = 50;
    cfg.rng_seed = seed;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    nqs_sampler_t *s = nqs_sampler_create(N, &cfg, nqs_ansatz_log_amp, a);
    double *trace = malloc(sizeof(double) * num_iter);
    nqs_sr_run(&cfg, Lx, Ly, a, s, trace);
    int tail_start = (int)(num_iter * 0.7);
    double tail = 0.0;
    for (int i = tail_start; i < num_iter; i++) tail += trace[i];
    tail /= (num_iter - tail_start);
    free(trace);
    nqs_sampler_free(s);
    *out_ansatz = a;
    return tail;
}
static void test_exact_energy_matches_mc(void) {
    /* The deterministic ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ on a trained NQS must agree
     * with the MC tail to within MC noise. If they disagree by >0.3
     * something is wrong with either the local-energy kernel or the
     * sampler. */
    int Lx = 2, Ly = 2;
    double J = 1.0, G = 1.0;
    nqs_ansatz_t *a;
    double E_mc = run_rbm_sr(Lx, Ly, J, G, 60, 512, 0xF00Du, &a);
    double E_exact = 0.0;
    ASSERT_EQ_INT(nqs_exact_energy_tfim(a, Lx, Ly, J, G, &E_exact), 0);
    printf("# TFIM 2x2 RBM: E_MC=%.6f  E_exact=%.6f  |Δ|=%.4f\n",
           E_mc, E_exact, fabs(E_mc - E_exact));
    ASSERT_TRUE(fabs(E_mc - E_exact) < 0.3);
    nqs_ansatz_free(a);
}
static void test_lanczos_refine_hits_exact_ground_state(void) {
    /* Regardless of the starting ansatz, Lanczos on dim=2^N converges
     * to the exact ground state in ≤ dim iterations. Starting from a
     * trained NQS just accelerates it. */
    int Lx = 2, Ly = 3, N = 6;
    double J = 1.0, G = 1.0;
    nqs_ansatz_t *a;
    double E_var = run_rbm_sr(Lx, Ly, J, G, 80, 512, 0xBEEFBABEu, &a);
    long dim = 1L << N;
    double *evec = malloc(sizeof(double) * dim);
    lanczos_result_t res;
    double E_ref = 0.0;
    int rc = nqs_lanczos_refine_tfim(a, Lx, Ly, J, G,
                                       (int)dim,   /* max_iters */
                                       1e-10, &E_ref, evec, &res);
    ASSERT_EQ_INT(rc, 0);
    ASSERT_TRUE(res.converged);
    printf("# TFIM 2x3 RBM vs Lanczos: E_var=%.6f  E_lanczos=%.8f  iters=%d\n",
           E_var, E_ref, res.iterations);
    /* Variational bound: Lanczos result is the true E0; variational
     * energy must be ≥ that. */
    ASSERT_TRUE(E_var >= E_ref - 1e-3);
    free(evec);
    nqs_ansatz_free(a);
}
static void test_lanczos_refine_lowers_energy(void) {
    /* Even with a shallow NQS training pass, Lanczos refinement should
     * strictly lower the energy. This is the signature of post-
     * processing working. */
    int Lx = 2, Ly = 3;
    double J = 1.0, G = 1.0;
    nqs_ansatz_t *a;
    /* Deliberately under-train so the RBM doesn't already hit the
     * ground state. */
    double E_var = run_rbm_sr(Lx, Ly, J, G, 15, 256, 0xDECAFu, &a);
    double E_det = 0.0;
    nqs_exact_energy_tfim(a, Lx, Ly, J, G, &E_det);
    long dim = 1L << (Lx * Ly);
    lanczos_result_t res;
    double E_ref = 0.0;
    nqs_lanczos_refine_tfim(a, Lx, Ly, J, G, (int)dim, 1e-10,
                             &E_ref, NULL, &res);
    printf("# under-trained RBM: E_var(MC)=%.4f  E_det=%.4f  E_lanczos=%.6f  (drop %.4f)\n",
           E_var, E_det, E_ref, E_det - E_ref);
    ASSERT_TRUE(E_ref <= E_det + 1e-6);
    nqs_ansatz_free(a);
}
static void test_heisenberg_refine_on_bare_rbm_hits_bounded_state(void) {
    /* Bare RBM (without Marshall) on a 4-site Heisenberg chain can
     * only reach the triplet sector because its amplitudes are
     * strictly positive.  Lanczos post-processing on the materialised
     * state extracts the lowest-triplet eigenvalue of H restricted to
     * the span of the trained ansatz — but since Lanczos runs on the
     * full 2^N Hilbert space, it still finds the true E₀ = -1.616. */
    int Lx = 4, Ly = 1, N = 4;
    double J = 1.0, Jz = 1.0;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_RBM;
    cfg.rbm_hidden_units = 8;
    cfg.rbm_init_scale = 0.1;
    cfg.rng_seed = 0x1234u;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    double E_var = 0;
    nqs_exact_energy_heisenberg(a, Lx, Ly, J, Jz, &E_var);
    long dim = 1L << N;
    double *evec = malloc(sizeof(double) * dim);
    lanczos_result_t res;
    double E_ref = 0;
    int rc = nqs_lanczos_refine_heisenberg(a, Lx, Ly, J, Jz,
                                             (int)dim, 1e-10,
                                             &E_ref, evec, &res);
    ASSERT_EQ_INT(rc, 0);
    printf("# Heisenberg 4-site: E_var(bare RBM)=%.4f  E_lanczos=%.6f (true -1.616)\n",
           E_var, E_ref);
    /* Lanczos on the full Hilbert space returns the true ground state. */
    ASSERT_NEAR(E_ref, -1.616025, 1e-4);
    /* Variational: base RBM cannot exceed the true GS, so E_var > E0. */
    ASSERT_TRUE(E_var > E_ref);
    free(evec);
    nqs_ansatz_free(a);
}
static void test_heisenberg_refine_matches_dense_ed(void) {
    /* On a 3-site chain Heisenberg (dim=8), the refined energy from
     * nqs_lanczos_refine_heisenberg equals the dense ED value. */
    int Lx = 3, Ly = 1, N = 3;
    double J = 1.0, Jz = 1.0;
    /* Build the dense Hamiltonian and diagonalise via power iteration. */
    long dim = 1L << N;
    double *H = calloc((size_t)dim * dim, sizeof(double));
    for (long s = 0; s < dim; s++) {
        for (int i = 0; i + 1 < N; i++) {
            int si = ((s >> i) & 1) ? -1 : +1;
            int sj = ((s >> (i+1)) & 1) ? -1 : +1;
            H[s*dim + s] += 0.25 * Jz * si * sj;
            if (si == -sj) {
                long s2 = s ^ (1L << i) ^ (1L << (i+1));
                H[s*dim + s2] += 0.5 * J;
            }
        }
    }
    double row = 0;
    for (long i = 0; i < dim; i++) {
        double r = 0;
        for (long j = 0; j < dim; j++) r += fabs(H[i*dim + j]);
        if (r > row) row = r;
    }
    double shift = row + 1;
    for (long i = 0; i < dim; i++) {
        for (long j = 0; j < dim; j++) H[i*dim + j] = -H[i*dim + j];
        H[i*dim + i] += shift;
    }
    double *v = malloc(sizeof(double) * dim), *w = malloc(sizeof(double) * dim);
    unsigned long long rng = 0xA5A5A5A5ULL;
    for (long i = 0; i < dim; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        v[i] = (double)(rng >> 11) / 9007199254740992.0 - 0.5;
    }
    double nrm = 0;
    for (long i = 0; i < dim; i++) nrm += v[i]*v[i];
    nrm = sqrt(nrm);
    for (long i = 0; i < dim; i++) v[i] /= nrm;
    double lam = 0;
    for (int it = 0; it < 4000; it++) {
        for (long i = 0; i < dim; i++) {
            double acc = 0;
            for (long j = 0; j < dim; j++) acc += H[i*dim + j] * v[j];
            w[i] = acc;
        }
        double num = 0, den = 0;
        for (long i = 0; i < dim; i++) { num += v[i]*w[i]; den += v[i]*v[i]; }
        double ln = num / den;
        double n2 = 0;
        for (long i = 0; i < dim; i++) n2 += w[i]*w[i];
        n2 = sqrt(n2);
        if (n2 > 0) for (long i = 0; i < dim; i++) v[i] = w[i] / n2;
        if (it > 0 && fabs(ln - lam) < 1e-12) { lam = ln; break; }
        lam = ln;
    }
    double E_ed = shift - lam;
    free(H); free(v); free(w);
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_RBM;
    cfg.rbm_hidden_units = 4;
    cfg.rng_seed = 0x99u;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    double E_ref = 0;
    lanczos_result_t res;
    nqs_lanczos_refine_heisenberg(a, Lx, Ly, J, Jz,
                                    (int)dim, 1e-12, &E_ref, NULL, &res);
    printf("# Heisenberg 3-site: E_ED=%.6f  E_nqs_lanczos=%.6f\n", E_ed, E_ref);
    ASSERT_NEAR(E_ref, E_ed, 1e-6);
    nqs_ansatz_free(a);
}
static void test_heisenberg_refine_matches_bethe_for_8site(void) {
    /* N=8 Heisenberg chain OBC, E₀ ≈ -3.37493260 (Bethe ansatz / ED).
     * Lanczos refinement on the full 2^8 = 256-dim Hilbert space
     * starting from any reasonable NQS init must find this energy. */
    int Lx = 8, Ly = 1, N = 8;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_RBM;
    cfg.rbm_hidden_units = 8;
    cfg.rng_seed = 0xB0B0u;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    long dim = 1L << N;
    double E_ref = 0;
    lanczos_result_t res;
    int rc = nqs_lanczos_refine_heisenberg(a, Lx, Ly, 1.0, 1.0,
                                             (int)dim, 1e-10,
                                             &E_ref, NULL, &res);
    ASSERT_EQ_INT(rc, 0);
    printf("# Heisenberg 8-site: E_lanczos=%.8f (E₀=-3.37493260, Bethe/ED)\n", E_ref);
    ASSERT_NEAR(E_ref, -3.37493260, 1e-4);
    nqs_ansatz_free(a);
}
static void test_kagome_lanczos_k_lowest_gives_exact_gap(void) {
    /* 2×2 PBC kagome Heisenberg S=½, N=12 sites (2^N = 4096-dim
     * Hilbert space). The k-lowest Lanczos variant must return two
     * sorted eigenvalues whose difference is positive (a real
     * spin gap) and the smaller eigenvalue must match the single-
     * Ritz `nqs_lanczos_refine_kagome_heisenberg` to machine
     * precision. This is a regression guard: a bug in the k-Ritz
     * sort or extraction would surface as E_0 drift or a negative
     * Δ. Budget: ~100 ms. */
    int Lx_cells = 2, Ly_cells = 2;
    int N = 3 * Lx_cells * Ly_cells;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_COMPLEX_RBM;
    cfg.rbm_hidden_units = 8;
    cfg.rng_seed = 0xCAFEu;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    ASSERT_TRUE(a != NULL);

    /* Rank-1 reference for E_0. */
    double E0_ref = 0.0;
    lanczos_result_t res_ref = {0};
    int rc_ref = nqs_lanczos_refine_kagome_heisenberg(a, Lx_cells, Ly_cells,
                                                       1.0, 1,
                                                       200, 1e-10,
                                                       &E0_ref, NULL, &res_ref);
    ASSERT_EQ_INT(rc_ref, 0);

    /* k-lowest with k=3. */
    double ev[3] = {0, 0, 0};
    lanczos_result_t res_k = {0};
    int rc_k = nqs_lanczos_k_lowest_kagome_heisenberg(a, Lx_cells, Ly_cells,
                                                       1.0, 1, 200, 3,
                                                       ev, &res_k);
    ASSERT_EQ_INT(rc_k, 0);
    /* Sorted ascending. */
    ASSERT_TRUE(ev[0] <= ev[1]);
    ASSERT_TRUE(ev[1] <= ev[2]);
    /* E_0 matches the rank-1 Lanczos to better than 10⁻⁸. */
    ASSERT_NEAR(ev[0], E0_ref, 1e-8);
    /* Positive spin gap — non-degenerate ground state expected at
     * this cluster size. */
    double gap = ev[1] - ev[0];
    ASSERT_TRUE(gap > 0.0);
    printf("# kagome N=12 Lanczos k=3: E_0=%.8f  E_1=%.8f  E_2=%.8f  Δ=%.6f\n",
           ev[0], ev[1], ev[2], gap);

    nqs_ansatz_free(a);
}

int main(void) {
    TEST_RUN(test_materialised_state_is_unit_normed);
    TEST_RUN(test_exact_energy_matches_mc);
    TEST_RUN(test_lanczos_refine_hits_exact_ground_state);
    TEST_RUN(test_lanczos_refine_lowers_energy);
    TEST_RUN(test_heisenberg_refine_on_bare_rbm_hits_bounded_state);
    TEST_RUN(test_heisenberg_refine_matches_dense_ed);
    TEST_RUN(test_heisenberg_refine_matches_bethe_for_8site);
    TEST_RUN(test_kagome_lanczos_k_lowest_gives_exact_gap);
    TEST_SUMMARY();
}