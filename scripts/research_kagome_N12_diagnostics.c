/*
 * scripts/research_kagome_N12_diagnostics.c
 *
 * End-to-end diagnostics driver on the open kagome Heisenberg S=1/2
 * ground-state problem. Pipelines three brand-new infrastructure
 * pieces into one research artefact:
 *
 *   Stage A — SR convergence of a complex-RBM variational ansatz
 *             (holomorphic SR; same machinery as the smoke tests).
 *   Stage B — χ_F = Tr(S)/2 from the converged wavefunction. The
 *             QGT trace is a Riemannian-metric signature on the
 *             variational manifold; tracks phase transitions and
 *             diverges at a gapless critical point in the
 *             thermodynamic limit.
 *   Stage C — Per-bond-class amplitude ratio ⟨ψ(s_{ij})/ψ(s)⟩_α
 *             for α ∈ {A-B, A-C, B-C}. Marshall-like structure
 *             manifests as |⟨r⟩| ≈ 1 with arg ≈ π on the
 *             singlet-favoured classes; frustrated / Dirac
 *             behaviour shows more uniform phases and reduced
 *             magnitudes.
 *   Stage D — Excited-state SR via orthogonal penalty. Produces an
 *             estimate of E₁ − E₀, the spin-gap diagnostic in the
 *             5-probe protocol.
 *
 * This is a research driver, NOT a unit test. Costs O(10 min) on an
 * M-series Mac. Wired behind `make research_kagome_N12_diagnostics`;
 * NOT part of `make test`. Output is a single TAP-style report so it
 * can be checked into benchmarks/results/kagome_N12_diagnostics.log
 * for longitudinal comparison.
 *
 * All four numbers together form the raw input to the 3-of-5 bar of
 * the 5-diagnostic protocol coordinated with libirrep (see
 * docs/research/kagome_KH_plan.md).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nqs/nqs_config.h"
#include "nqs/nqs_ansatz.h"
#include "nqs/nqs_sampler.h"
#include "nqs/nqs_optimizer.h"
#include "nqs/nqs_diagnostics.h"
#include "nqs/nqs_lanczos.h"
#include "mps/lanczos.h"

#define E0_PER_SITE_REF  (-0.4365)
#define E0_TOTAL_REF(N)  ((E0_PER_SITE_REF) * (double)(N))

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double tail_mean(const double *trace, int total, int tail_len) {
    double s = 0.0;
    for (int i = total - tail_len; i < total; i++) s += trace[i];
    return s / (double)tail_len;
}

static nqs_config_t kagome_cfg(int num_iter, int hidden, int samples,
                                unsigned seed) {
    nqs_config_t cfg = nqs_config_defaults();
    cfg.ansatz           = NQS_ANSATZ_COMPLEX_RBM;
    cfg.rbm_hidden_units = hidden;
    cfg.rbm_init_scale   = 0.02;
    cfg.hamiltonian      = NQS_HAM_KAGOME_HEISENBERG;
    cfg.j_coupling       = 1.0;
    cfg.kagome_pbc       = 1;
    cfg.num_samples      = samples;
    cfg.num_thermalize   = 512;
    cfg.num_decorrelate  = 2;
    cfg.num_iterations   = num_iter;
    cfg.learning_rate    = 0.02;
    cfg.sr_diag_shift    = 1e-3;
    cfg.sr_cg_max_iters  = 200;
    cfg.sr_cg_tol        = 1e-8;
    cfg.rng_seed         = seed;
    return cfg;
}

int main(int argc, char **argv) {
    /* Line-buffer stdout so each stage's summary appears as it
     * completes, not at the very end. Research drivers take minutes
     * and a silent stdout turns into a "is it alive?" problem. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    /* Default knobs — overridable for quick re-runs on a different
     * budget. Defaults target "reproduce the 3.66 % gap run plus
     * diagnostics in one shot". */
    int Lx = 2, Ly = 2;
    int N  = 3 * Lx * Ly;        /* 12 sites */
    int gs_iters      = 1000;
    int gs_hidden     = 48;
    int gs_samples    = 2048;
    unsigned gs_seed  = 0xC0AEEDu;

    int diag_samples  = 4096;    /* tighter MC for the frozen-state measurements */
    int exc_iters     = 400;
    int exc_hidden    = 48;
    int exc_samples   = 2048;
    double exc_mu     = 5.0;
    unsigned exc_seed = 0xD17AE5u;

    if (argc > 1) gs_iters    = atoi(argv[1]);
    if (argc > 2) gs_hidden   = atoi(argv[2]);
    if (argc > 3) gs_samples  = atoi(argv[3]);
    if (argc > 4) exc_iters   = atoi(argv[4]);

    printf("# kagome N=12 PBC Heisenberg — end-to-end diagnostics driver\n");
    printf("# GS  pre-train: iters=%d hidden=%d samples=%d seed=0x%x\n",
           gs_iters, gs_hidden, gs_samples, gs_seed);
    printf("# Diag sampling: samples=%d  (MC for frozen-state measurements)\n",
           diag_samples);
    printf("# Excited SR:    iters=%d hidden=%d samples=%d μ=%.2f seed=0x%x\n",
           exc_iters, exc_hidden, exc_samples, exc_mu, exc_seed);
    printf("# ED reference:  E_0 = %+.4f  (N=12, E_0/N = %+.4f)\n",
           E0_TOTAL_REF(N), E0_PER_SITE_REF);
    printf("# ============================================================\n");

    /* ============== Stage A — GS SR convergence ================= */
    double tA0 = now_seconds();
    nqs_config_t cfg_gs = kagome_cfg(gs_iters, gs_hidden, gs_samples, gs_seed);
    nqs_ansatz_t  *a_gs = nqs_ansatz_create(&cfg_gs, N);
    nqs_sampler_t *s_gs = nqs_sampler_create(N, &cfg_gs, nqs_ansatz_log_amp, a_gs);
    if (!a_gs || !s_gs) { fprintf(stderr, "error: GS ansatz/sampler alloc\n"); return 2; }
    double *trace_gs = malloc((size_t)gs_iters * sizeof(double));
    if (!trace_gs) { fprintf(stderr, "error: trace alloc\n"); return 2; }
    int rc = nqs_sr_run_holomorphic(&cfg_gs, Lx, Ly, a_gs, s_gs,
                                     nqs_ansatz_log_amp, a_gs, trace_gs);
    if (rc != 0) { fprintf(stderr, "error: GS SR rc=%d\n", rc); return 2; }
    double dtA = now_seconds() - tA0;

    int tail_len_gs = gs_iters / 10;
    if (tail_len_gs < 10) tail_len_gs = 10;
    double E_gs_tail = tail_mean(trace_gs, gs_iters, tail_len_gs);
    double E_gs_last10 = tail_mean(trace_gs, gs_iters, 10);
    double E_ref_total = E0_TOTAL_REF(N);
    double rel_gap_tail   = (E_gs_tail   - E_ref_total) / fabs(E_ref_total);
    double rel_gap_last10 = (E_gs_last10 - E_ref_total) / fabs(E_ref_total);

    printf("# Stage A — GS convergence  (wall-clock %.1f s)\n", dtA);
    printf("#   tail mean (last %d iters):  E = %+.6f   relative gap = %+.3f %%\n",
           tail_len_gs, E_gs_tail, rel_gap_tail * 100.0);
    printf("#   last 10 iters:              E = %+.6f   relative gap = %+.3f %%\n",
           E_gs_last10, rel_gap_last10 * 100.0);
    printf("#   ED reference:              E_0 = %+.6f\n", E_ref_total);

    /* ============== Stage A' — exact + Lanczos refinement ==========
     *
     * Lanczos on a 2^12 = 4096-dim space built from the full kagome
     * Heisenberg matvec is essentially exact diagonalization: with
     * full re-orthogonalisation and k ≤ dim Krylov steps it converges
     * to machine precision. Two uses:
     *   (a) Variational "exact" energy: evaluates ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ on
     *       the full basis (no MC noise) using Re(ψ) of the trained
     *       cRBM. Compare against Stage A's MC tail to see how much
     *       of the gap is noise vs residual optimization.
     *   (b) Lanczos-refined energy seeded from the trained state.
     *       With seed overlap > 10^{-6} (anything non-pathological),
     *       Lanczos finds the TRUE ground state — independent of the
     *       seed's variational quality. Gives us exact E₀ on our
     *       specific cluster, which differs from the Leung-Elser
     *       literature value because of PBC-wrap convention. */
    double tAp0 = now_seconds();
    double E_gs_exact = 0.0;
    int rc_exact = nqs_exact_energy_kagome_heisenberg(a_gs, Lx, Ly, 1.0, 1,
                                                        &E_gs_exact);
    double E_gs_lanczos = 0.0;
    lanczos_result_t lres = {0};
    int rc_lanczos = nqs_lanczos_refine_kagome_heisenberg(a_gs, Lx, Ly,
                                                           1.0, 1,
                                                           200, 1e-10,
                                                           &E_gs_lanczos,
                                                           NULL, &lres);
    double dtAp = now_seconds() - tAp0;
    /* "Ground truth" reference from Lanczos when available, falling
     * back to the plan's literature value otherwise. Downstream gaps
     * use the best reference we have. */
    double E_truth = (rc_lanczos == 0) ? E_gs_lanczos : E_ref_total;
    double rel_gap_exact   = (rc_exact   == 0)
        ? (E_gs_exact   - E_truth) / fabs(E_truth) : 0.0;
    double rel_gap_lanczos = (rc_lanczos == 0)
        ? (E_gs_lanczos - E_truth) / fabs(E_truth) : 0.0;
    /* MC tail gap relative to the *true* (Lanczos) GS, not the plan
     * literature value. Tells us how well the trainer actually did. */
    double rel_gap_tail_truth = (E_gs_tail - E_truth) / fabs(E_truth);
    printf("# Stage A' — exact + Lanczos refinement  (wall-clock %.1f s)\n", dtAp);
    if (rc_exact == 0) {
        printf("#   exact variational ⟨ψ|H|ψ⟩: E = %+.6f   (gap vs truth %+.4f %%)\n",
               E_gs_exact, rel_gap_exact * 100.0);
    } else {
        printf("#   exact variational:         FAILED rc=%d\n", rc_exact);
    }
    if (rc_lanczos == 0) {
        printf("#   Lanczos (k=%d, %sconv): E = %+.10f   (gap %+.2e %%)\n",
               lres.iterations, lres.converged ? "" : "NOT ",
               E_gs_lanczos, rel_gap_lanczos * 100.0);
        printf("#     residual norm           = %.3e\n", lres.residual_norm);
        printf("#     ⇒ adopting as ground truth for downstream gaps\n");
        printf("#   (plan literature value %+.4f is %+.3f %% off our cluster's true GS)\n",
               E_ref_total, (E_ref_total - E_truth) / fabs(E_truth) * 100.0);
        printf("#   corrected MC tail gap vs truth: %+.3f %% (was %+.3f %% vs literature)\n",
               rel_gap_tail_truth * 100.0, rel_gap_tail * 100.0);
    } else {
        printf("#   Lanczos:                  FAILED rc=%d\n", rc_lanczos);
    }

    /* ============== Stage B — χ_F = Tr(S) / 2 ==================== */
    double tB0 = now_seconds();
    /* Reconfigure the sampler/config for the measurement budget —
     * leave the ansatz parameters frozen. */
    cfg_gs.num_samples = diag_samples;
    double trace_S = 0.0, per_param = 0.0;
    rc = nqs_compute_chi_F(&cfg_gs, Lx, Ly, a_gs, s_gs, &trace_S, &per_param);
    if (rc != 0) { fprintf(stderr, "error: chi_F rc=%d\n", rc); return 2; }
    double dtB = now_seconds() - tB0;
    double chi_F = 0.5 * trace_S;
    long num_params = nqs_ansatz_num_params(a_gs);
    printf("# Stage B — χ_F = Tr(S)/2     (wall-clock %.1f s)\n", dtB);
    printf("#   num_params   = %ld\n", num_params);
    printf("#   Tr(S)        = %.6f\n", trace_S);
    printf("#   χ_F          = %.6f\n", chi_F);
    printf("#   Tr(S) / N_p  = %.6f   (per-parameter mean)\n", per_param);

    /* ============== Stage C — per-bond-class phase ================ */
    double tC0 = now_seconds();
    double r_re[NQS_KAGOME_NUM_BOND_CLASSES] = {0};
    double r_im[NQS_KAGOME_NUM_BOND_CLASSES] = {0};
    long   cnt [NQS_KAGOME_NUM_BOND_CLASSES] = {0};
    rc = nqs_compute_kagome_bond_phase(&cfg_gs, Lx, Ly, a_gs, s_gs,
                                         r_re, r_im, cnt);
    if (rc != 0) { fprintf(stderr, "error: bond_phase rc=%d\n", rc); return 2; }
    double dtC = now_seconds() - tC0;
    const char *class_name[3] = {"A-B", "A-C", "B-C"};
    printf("# Stage C — bond-phase probe  (wall-clock %.1f s)\n", dtC);
    printf("#   class   ⟨r⟩ Re         ⟨r⟩ Im         |⟨r⟩|         arg⟨r⟩/π       n\n");
    for (int c = 0; c < NQS_KAGOME_NUM_BOND_CLASSES; c++) {
        double mag = sqrt(r_re[c] * r_re[c] + r_im[c] * r_im[c]);
        double phase = atan2(r_im[c], r_re[c]) / M_PI;
        printf("#   %s     %+.6f     %+.6f     %.6f     %+.4f        %ld\n",
               class_name[c], r_re[c], r_im[c], mag, phase, cnt[c]);
    }
    /* Interpretation reminder baked in: |r|≈1 with arg≈π across all
     * classes ⇒ Marshall-like bipartite sign structure. |r|<<1 or
     * mixed phases ⇒ frustrated / Dirac-compatible. */

    /* Free the GS sampler — excited stage uses a fresh sampler of
     * |ψ_exc|². The GS ansatz stays alive as the reference. */
    nqs_sampler_free(s_gs); s_gs = NULL;

    /* ============== Stage D — excited-state SR gap ================ */
    double tD0 = now_seconds();
    nqs_config_t cfg_exc = kagome_cfg(exc_iters, exc_hidden, exc_samples, exc_seed);
    cfg_exc.learning_rate = 0.015;    /* smaller step; penalty adds drive */
    cfg_exc.sr_diag_shift  = 5e-3;
    nqs_ansatz_t  *a_exc = nqs_ansatz_create(&cfg_exc, N);
    nqs_sampler_t *s_exc = nqs_sampler_create(N, &cfg_exc, nqs_ansatz_log_amp, a_exc);
    if (!a_exc || !s_exc) { fprintf(stderr, "error: exc ansatz/sampler alloc\n"); return 2; }
    double *trace_exc = malloc((size_t)exc_iters * sizeof(double));
    if (!trace_exc) { fprintf(stderr, "error: exc trace alloc\n"); return 2; }
    rc = nqs_sr_run_excited(&cfg_exc, Lx, Ly, a_exc, s_exc,
                              nqs_ansatz_log_amp, a_exc,
                              nqs_ansatz_log_amp, a_gs,
                              exc_mu, trace_exc);
    if (rc != 0) { fprintf(stderr, "error: excited SR rc=%d\n", rc); return 2; }
    double dtD = now_seconds() - tD0;

    int tail_len_exc = exc_iters / 10;
    if (tail_len_exc < 10) tail_len_exc = 10;
    double E_exc_tail = tail_mean(trace_exc, exc_iters, tail_len_exc);
    double gap_est = E_exc_tail - E_gs_tail;
    printf("# Stage D — excited-state SR   (wall-clock %.1f s)\n", dtD);
    printf("#   tail mean (last %d iters):  E₁* = %+.6f\n", tail_len_exc, E_exc_tail);
    printf("#   variational gap estimate:  Δ = E₁* - E₀* = %+.6f\n", gap_est);

    /* k-lowest Lanczos from the excited seed. Extracts the k smallest
     * Ritz values from a full Krylov basis — E₀ and E₁ land together,
     * so the gap E₁ − E₀ is a single subtraction once both have
     * converged. Seeding from the excited trainer (rather than the GS
     * trainer) helps the Krylov subspace find the excited sector
     * faster, but with max_iters large enough both are recovered
     * regardless of seed. */
    int k_lanczos = 4;
    double exc_evs[4] = {0, 0, 0, 0};
    lanczos_result_t lres_exc = {0};
    int rc_exc_lanczos = nqs_lanczos_k_lowest_kagome_heisenberg(
        a_exc, Lx, Ly, 1.0, 1, 200, k_lanczos, exc_evs, &lres_exc);
    double E_exc_lanczos     = rc_exc_lanczos == 0 ? exc_evs[0] : 0.0;
    double E1_lanczos        = rc_exc_lanczos == 0 ? exc_evs[1] : 0.0;
    double gap_lanczos_exact = rc_exc_lanczos == 0 ? exc_evs[1] - exc_evs[0] : 0.0;

    /* =================== Final summary ========================== */
    printf("# ============================================================\n");
    printf("# SUMMARY (N=%d, 2x2 PBC kagome Heisenberg, J=1)\n", N);
    if (rc_lanczos == 0) {
        printf("#   E_0 (exact, Lanczos):      %+.10f  [ground truth for this cluster]\n",
               E_gs_lanczos);
        printf("#   E_0* (variational MC):     %+.6f   (gap vs truth %+.3f %%)\n",
               E_gs_tail, rel_gap_tail_truth * 100.0);
    } else {
        printf("#   E_0* (variational MC):     %+.6f   (ED lit: %+.6f, gap %+.3f %%)\n",
               E_gs_tail, E_ref_total, rel_gap_tail * 100.0);
    }
    if (rc_exact == 0) {
        printf("#   E_0 (exact ⟨ψ|H|ψ⟩):       %+.6f   (gap vs truth %+.4f %%)\n",
               E_gs_exact, rel_gap_exact * 100.0);
    }
    if (rc_exc_lanczos == 0) {
        printf("#   E_0 (Lanczos k=%d, seeded from excited): %+.10f\n",
               k_lanczos, E_exc_lanczos);
        printf("#   E_1 (Lanczos k=%d, seeded from excited): %+.10f\n",
               k_lanczos, E1_lanczos);
        printf("#   spin gap Δ = E_1 − E_0 (exact):         %+.6f\n",
               gap_lanczos_exact);
        /* Variational gap from the MC trainer, as a consistency
         * check against the exact Lanczos gap. */
        printf("#   spin gap Δ_var = E₁* − E_0* (MC trainer): %+.6f\n",
               gap_est);
    }
    printf("#   χ_F:                        %.6f\n", chi_F);
    printf("#   arg⟨r⟩/π per class {A-B, A-C, B-C}:  %+.4f  %+.4f  %+.4f\n",
           atan2(r_im[0], r_re[0]) / M_PI,
           atan2(r_im[1], r_re[1]) / M_PI,
           atan2(r_im[2], r_re[2]) / M_PI);
    printf("#   |⟨r⟩| per class {A-B, A-C, B-C}:     %.4f  %.4f  %.4f\n",
           sqrt(r_re[0]*r_re[0]+r_im[0]*r_im[0]),
           sqrt(r_re[1]*r_re[1]+r_im[1]*r_im[1]),
           sqrt(r_re[2]*r_re[2]+r_im[2]*r_im[2]));
    printf("#   E_1* (excited SR):         %+.6f\n", E_exc_tail);
    printf("#   spin-gap estimate Δ:        %+.6f\n", gap_est);
    printf("#   total wall-clock:          %.1f s\n", dtA + dtB + dtC + dtD);

    free(trace_exc); free(trace_gs);
    nqs_sampler_free(s_exc); nqs_ansatz_free(a_exc);
    nqs_ansatz_free(a_gs);
    /* Exit code: 0 if GS within 15% of ED (generous research bound),
     * 1 otherwise. Allows scripted regression detection. */
    return (fabs(rel_gap_tail) < 0.15) ? 0 : 1;
}
