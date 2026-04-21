/*
 * tests/test_nqs_marshall.c
 *
 * Marshall sign rule turns the Heisenberg antiferromagnet into a
 * stoquastic problem that a strictly-positive real RBM can solve.
 *
 * Scientific claim under test:
 *   - Without Marshall wrapping, the RBM cannot reach the Heisenberg
 *     AFM ground state (it converges to the triplet at E = +J/4).
 *   - With Marshall wrapping, the RBM reaches the singlet E₀ = -3J/4
 *     on 2 sites and -2J on the 4-site chain.
 *
 * These are the canonical textbook sanity checks for Marshall-rotated
 * NQS (Carleo & Troyer 2017 used it for the Heisenberg benchmark).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "harness.h"
#include "nqs/nqs_config.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_sampler.h"
#include "nqs/nqs_optimizer.h"
#include "nqs/nqs_marshall.h"
static void test_parity_detection(void) {
    /* 2x1 chain: sites 0 (A), 1 (B).
     *   (+,+): no down spins on A, parity 0
     *   (-,+): site 0 down, on A → parity 1
     *   (+,-): site 1 down, on B → parity 0
     *   (-,-): both down, one on A → parity 1 */
    int a[2] = {+1, +1};
    int b[2] = {-1, +1};
    int c[2] = {+1, -1};
    int d[2] = {-1, -1};
    ASSERT_EQ_INT(nqs_marshall_parity(a, 2, 1), 0);
    ASSERT_EQ_INT(nqs_marshall_parity(b, 2, 1), 1);
    ASSERT_EQ_INT(nqs_marshall_parity(c, 2, 1), 0);
    ASSERT_EQ_INT(nqs_marshall_parity(d, 2, 1), 1);
}
static void test_parity_on_2x2_square(void) {
    /* 2x2 lattice:
     *   (0,0) A  (0,1) B
     *   (1,0) B  (1,1) A
     * Index layout (x*Ly + y): (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3.
     * A sites: indices 0 and 3. */
    int s_neel[4] = {+1, -1, -1, +1};   /* AFM Néel (+, -, -, +) */
    /* Down spins on A: none (indices 0 and 3 are +1). Parity = 0. */
    ASSERT_EQ_INT(nqs_marshall_parity(s_neel, 2, 2), 0);
    int s_flipped[4] = {-1, +1, +1, -1};
    /* Now A sites (indices 0, 3) are both down → parity 0 (two downs). */
    ASSERT_EQ_INT(nqs_marshall_parity(s_flipped, 2, 2), 0);
    int s_single[4] = {-1, -1, -1, +1};
    /* A sites: index 0 down, index 3 up → one down on A → parity 1. */
    ASSERT_EQ_INT(nqs_marshall_parity(s_single, 2, 2), 1);
}
static double run_rbm_heisenberg(int Lx, int Ly, int use_marshall,
                                  int num_iter, int num_samples,
                                  unsigned seed) {
    int N = Lx * Ly;
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz = NQS_ANSATZ_RBM;
    cfg.rbm_hidden_units = 4 * N;
    cfg.rbm_init_scale = 0.1;
    cfg.hamiltonian = NQS_HAM_HEISENBERG;
    cfg.j_coupling = 1.0;
    cfg.num_samples = num_samples;
    cfg.num_thermalize = 256;
    cfg.num_decorrelate = 2;
    cfg.num_iterations = num_iter;
    cfg.learning_rate = 2e-2;
    cfg.sr_diag_shift = 1e-2;
    cfg.sr_cg_max_iters = 50;
    cfg.rng_seed = seed;
    nqs_ansatz_t *a = nqs_ansatz_create(&cfg, N);
    nqs_marshall_wrapper_t w = {
.base_log_amp = nqs_ansatz_log_amp,
.base_user    = a,
.size_x       = Lx,
.size_y       = Ly,
    };
    void *user = use_marshall ? (void *)&w : (void *)a;
    nqs_log_amp_fn_t fn = use_marshall ? nqs_marshall_log_amp
                                       : nqs_ansatz_log_amp;
    nqs_sampler_t *s = nqs_sampler_create(N, &cfg, fn, user);
    double *trace = malloc(sizeof(double) * num_iter);
    nqs_sr_run_custom(&cfg, Lx, Ly, a, s, fn, user, trace);
    int tail_start = (int)(num_iter * 0.7);
    double tail = 0.0;
    for (int i = tail_start; i < num_iter; i++) tail += trace[i];
    tail /= (double)(num_iter - tail_start);
    free(trace);
    nqs_sampler_free(s);
    nqs_ansatz_free(a);
    return tail;
}
static void test_heisenberg_2site_requires_marshall(void) {
    double E0 = -0.75;   /* singlet of -J · (3/4) with J=1 */
    double e_no  = run_rbm_heisenberg(2, 1, 0, 80, 512, 0x42u);
    double e_yes = run_rbm_heisenberg(2, 1, 1, 80, 512, 0x42u);
    printf("# Heisenberg 2-site: E0=%.4f  no-Marshall=%.4f  with-Marshall=%.4f\n",
           E0, e_no, e_yes);
    /* Without Marshall: stuck in the triplet sector (E = +0.25). */
    ASSERT_TRUE(e_no > -0.1);                  /* far from singlet */
    /* With Marshall: reaches the singlet to within MC noise. */
    ASSERT_NEAR(e_yes, E0, 0.10);
}
static void test_heisenberg_4site_chain_with_marshall(void) {
    /* 4-site Heisenberg chain open BC: exact E₀ = -2 (standard result
     * from two-singlet pairing on bipartite bonds; computable via ED).
     * Actually let us compute via ED here to be safe. */
    /* Use quick ED through our own power-iteration on shift-I−H. */
    int N = 4;
    int dim = 1 << N;
    double *H = calloc((size_t)dim * dim, sizeof(double));
    double J = 1.0;
    for (int s = 0; s < dim; s++) {
        /* Map bit 0 → +1, bit 1 → -1.  σ^z_i = ±1, spin = σ^z / 2 → ¼ σ^z σ^z. */
        for (int i = 0; i + 1 < N; i++) {
            int si = ((s >> i) & 1) ? -1 : +1;
            int sj = ((s >> (i+1)) & 1) ? -1 : +1;
            H[s*dim + s] += 0.25 * J * si * sj;
            if (si == -sj) {
                int sflip = s ^ (1 << i) ^ (1 << (i+1));
                H[s*dim + sflip] += 0.5 * J;
            }
        }
    }
    double row = 0; for (int i = 0; i < dim; i++) { double r=0; for (int j=0;j<dim;j++) r+=fabs(H[i*dim+j]); if (r>row) row=r; }
    double shift = row + 1;
    for (int i = 0; i < dim; i++) { for (int j = 0; j < dim; j++) H[i*dim+j] = -H[i*dim+j]; H[i*dim+i] += shift; }
    double *v = malloc(sizeof(double)*dim), *w = malloc(sizeof(double)*dim);
    unsigned long long rng = 0xC0DEFACEULL;
    for (int i = 0; i < dim; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        v[i] = (double)(rng >> 11) / 9007199254740992.0 - 0.5;
    }
    { double n = 0; for (int i = 0; i < dim; i++) n += v[i]*v[i]; n = sqrt(n);
      for (int i = 0; i < dim; i++) v[i] /= n; }
    double lam = 0;
    for (int it = 0; it < 4000; it++) {
        for (int i = 0; i < dim; i++) { double a=0; for (int j=0;j<dim;j++) a+=H[i*dim+j]*v[j]; w[i]=a; }
        double num=0, den=0; for (int i=0;i<dim;i++) { num+=v[i]*w[i]; den+=v[i]*v[i]; }
        double ln = num/den;
        double n2 = 0; for (int i=0;i<dim;i++) n2+=w[i]*w[i]; n2=sqrt(n2);
        if (n2 > 0) for (int i=0;i<dim;i++) v[i]=w[i]/n2;
        if (it > 0 && fabs(ln-lam) < 1e-12) { lam = ln; break; }
        lam = ln;
    }
    double E0 = shift - lam;
    free(H); free(v); free(w);
    double e_yes = run_rbm_heisenberg(4, 1, 1, 150, 1024, 0xCAFEu);
    printf("# Heisenberg 4-site chain: E0=%.6f (ED)  RBM+Marshall=%.4f\n", E0, e_yes);
    ASSERT_NEAR(e_yes, E0, 0.15);
}
int main(void) {
    TEST_RUN(test_parity_detection);
    TEST_RUN(test_parity_on_2x2_square);
    TEST_RUN(test_heisenberg_2site_requires_marshall);
    TEST_RUN(test_heisenberg_4site_chain_with_marshall);
    TEST_SUMMARY();
}