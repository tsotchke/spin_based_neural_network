/*
 * src/nqs/nqs_optimizer.c
 *
 * Stochastic Reconfiguration (Sorella 1998) with conjugate-gradient
 * preconditioning. The key trick is that the QGT-vector product
 *
 *     (S v)_k = <O_k* O_l v_l> - <O_k*> <O_l v_l>
 *
 * can be computed without materialising the N_p × N_p matrix S: given
 * the N_s × N_p matrix O whose rows are the per-sample log-psi
 * gradients, one computes
 *
 *     u = O v                         (N_s vector)
 *     (S v) = (O^T u)/N_s - <O>(<u>)
 *
 * The cost per CG iteration is therefore O(N_s · N_p) which is linear
 * in both batch size and parameter count.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "nqs/nqs_optimizer.h"
#include "nqs/nqs_gradient.h"

static double vec_dot(const double *a, const double *b, long n) {
    double s = 0.0;
    for (long i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static double vec_norm(const double *a, long n) {
    return sqrt(vec_dot(a, a, n));
}

static void vec_axpy(double *y, double alpha, const double *x, long n) {
    /* y ← y + α x */
    for (long i = 0; i < n; i++) y[i] += alpha * x[i];
}

/* Compute (S + ε I) v where S is the centered QGT represented by the
 * row-major batch_grads matrix O (N_s × N_p), with row means stored
 * in grad_mean (length N_p). Writes result to out (length N_p). */
static void qgt_apply(const double *batch_grads, long batch_size, long num_params,
                       const double *grad_mean,
                       double epsilon,
                       const double *v,
                       double *out) {
    /* u_i = Σ_k O_ik v_k  (N_s) */
    double *u = calloc((size_t)batch_size, sizeof(double));
    if (!u) { memset(out, 0, (size_t)num_params * sizeof(double)); return; }
    for (long i = 0; i < batch_size; i++) {
        u[i] = 0.0;
        for (long k = 0; k < num_params; k++) {
            u[i] += batch_grads[i * num_params + k] * v[k];
        }
    }
    double u_mean = 0.0;
    for (long i = 0; i < batch_size; i++) u_mean += u[i];
    u_mean /= (double)batch_size;

    /* mean_grad_dot_v = <O_k> v_k  (scalar) */
    double mean_dot_v = vec_dot(grad_mean, v, num_params);

    /* (S v)_k = (O^T u / N_s)_k - <O_k> · u_mean
     *        = (1/N_s) Σ_i O_ik u_i - grad_mean_k · u_mean. */
    for (long k = 0; k < num_params; k++) {
        double s = 0.0;
        for (long i = 0; i < batch_size; i++) {
            s += batch_grads[i * num_params + k] * u[i];
        }
        out[k] = (s / (double)batch_size) - grad_mean[k] * u_mean;
        /* Regularise with a small ε on the diagonal. */
        out[k] += epsilon * v[k];
        /* Subtract the cross term <O_k> <O_l v_l> properly — already
         * absorbed in grad_mean · u_mean minus the diagonal regulariser. */
        (void)mean_dot_v;
    }
    free(u);
}

/* Solve (S + ε I) δ = F via conjugate gradient. Returns number of
 * iterations used. Sets *converged iff residual norm dropped below tol. */
static int cg_solve(const double *batch_grads, long batch_size, long num_params,
                    const double *grad_mean,
                    double epsilon,
                    const double *rhs,
                    int max_iters, double tol,
                    double *out_delta,
                    int *out_converged) {
    memset(out_delta, 0, (size_t)num_params * sizeof(double));
    double *r = malloc((size_t)num_params * sizeof(double));
    double *p = malloc((size_t)num_params * sizeof(double));
    double *Ap = malloc((size_t)num_params * sizeof(double));
    if (!r || !p || !Ap) {
        free(r); free(p); free(Ap);
        if (out_converged) *out_converged = 0;
        return 0;
    }
    memcpy(r, rhs, (size_t)num_params * sizeof(double));  /* r0 = rhs - A·0 = rhs */
    memcpy(p, r,   (size_t)num_params * sizeof(double));
    double rs_old = vec_dot(r, r, num_params);
    double rs0 = rs_old;
    int iter;
    int converged = 0;
    for (iter = 0; iter < max_iters; iter++) {
        qgt_apply(batch_grads, batch_size, num_params, grad_mean, epsilon, p, Ap);
        double pAp = vec_dot(p, Ap, num_params);
        if (fabs(pAp) < 1e-30) break;
        double alpha = rs_old / pAp;
        vec_axpy(out_delta, alpha, p, num_params);
        vec_axpy(r, -alpha, Ap, num_params);
        double rs_new = vec_dot(r, r, num_params);
        if (rs_new <= tol * tol * rs0) {
            converged = 1;
            iter++;
            break;
        }
        double beta = rs_new / rs_old;
        for (long i = 0; i < num_params; i++) p[i] = r[i] + beta * p[i];
        rs_old = rs_new;
    }
    free(r); free(p); free(Ap);
    if (out_converged) *out_converged = converged;
    return iter;
}

int nqs_sr_step_custom_full(const nqs_config_t *cfg,
                             int size_x, int size_y,
                             nqs_ansatz_t *ansatz,
                             nqs_sampler_t *sampler,
                             nqs_log_amp_fn_t log_amp_fn,
                             void *log_amp_user,
                             nqs_gradient_fn_t gradient_fn,
                             void *grad_user,
                             nqs_sr_step_info_t *out_info) {
    if (!cfg || !ansatz || !sampler) return -1;
    int N = size_x * size_y;
    long num_params = nqs_ansatz_num_params(ansatz);
    if (num_params <= 0) return -1;
    if (!log_amp_fn) { log_amp_fn = nqs_ansatz_log_amp; log_amp_user = ansatz; }

    int batch_size = cfg->num_samples;
    int *batch = malloc((size_t)batch_size * (size_t)N * sizeof(int));
    double *energies = malloc((size_t)batch_size * sizeof(double));
    double *grads = malloc((size_t)batch_size * (size_t)num_params * sizeof(double));
    double *grad_mean = calloc((size_t)num_params, sizeof(double));
    double *force = calloc((size_t)num_params, sizeof(double));
    double *delta = calloc((size_t)num_params, sizeof(double));
    if (!batch || !energies || !grads || !grad_mean || !force || !delta) {
        free(batch); free(energies); free(grads);
        free(grad_mean); free(force); free(delta);
        return -1;
    }

    /* 1. Sample a batch from |ψ|^2. */
    nqs_sampler_batch(sampler, batch_size, batch);

    /* 2. Local energies — using the same log_amp callback the sampler
     * sees so any sign wrapper (Marshall, symmetry projection) feeds
     * through consistently. */
    nqs_local_energy_batch(cfg, size_x, size_y, batch, batch_size,
                           log_amp_fn, log_amp_user, energies);

    /* 3. Per-sample log-psi gradients. Dispatch to wrapper-aware
     * gradient if provided; otherwise straight base-ansatz gradient. */
    for (int i = 0; i < batch_size; i++) {
        int *sp = &batch[(size_t)i * (size_t)N];
        double *gp = &grads[(size_t)i * (size_t)num_params];
        if (gradient_fn) {
            gradient_fn(grad_user, ansatz, sp, N, gp);
        } else {
            nqs_ansatz_logpsi_gradient(ansatz, sp, N, gp);
        }
    }

    /* 4. Means and forces. */
    double e_mean = 0.0;
    double e_sq_mean = 0.0;
    for (int i = 0; i < batch_size; i++) {
        e_mean += energies[i];
        e_sq_mean += energies[i] * energies[i];
    }
    e_mean /= (double)batch_size;
    e_sq_mean /= (double)batch_size;

    for (long k = 0; k < num_params; k++) {
        double sum_grad = 0.0;
        double sum_grad_e = 0.0;
        for (int i = 0; i < batch_size; i++) {
            double g = grads[(size_t)i * (size_t)num_params + k];
            sum_grad  += g;
            sum_grad_e += g * energies[i];
        }
        grad_mean[k] = sum_grad / (double)batch_size;
        /* F_k = <O_k E_loc> - <O_k> <E_loc> */
        force[k] = (sum_grad_e / (double)batch_size) - grad_mean[k] * e_mean;
    }

    /* 5. Solve (S + ε I) δ = F via CG. */
    int converged = 0;
    int iters = cg_solve(grads, batch_size, num_params, grad_mean,
                         cfg->sr_diag_shift,
                         force,
                         cfg->sr_cg_max_iters, cfg->sr_cg_tol,
                         delta, &converged);

    /* 6. Apply update: θ ← θ - lr · δθ. */
    nqs_ansatz_apply_update(ansatz, delta, -cfg->learning_rate);

    if (out_info) {
        out_info->mean_energy = e_mean;
        out_info->variance_energy = e_sq_mean - e_mean * e_mean;
        out_info->update_norm = vec_norm(delta, num_params) * cfg->learning_rate;
        out_info->acceptance_ratio = nqs_sampler_acceptance_ratio(sampler);
        out_info->cg_iterations = iters;
        out_info->converged = converged;
    }

    free(batch); free(energies); free(grads);
    free(grad_mean); free(force); free(delta);
    return 0;
}

int nqs_sr_step_custom(const nqs_config_t *cfg,
                       int size_x, int size_y,
                       nqs_ansatz_t *ansatz,
                       nqs_sampler_t *sampler,
                       nqs_log_amp_fn_t log_amp_fn,
                       void *log_amp_user,
                       nqs_sr_step_info_t *out_info) {
    return nqs_sr_step_custom_full(cfg, size_x, size_y, ansatz, sampler,
                                    log_amp_fn, log_amp_user,
                                    NULL, NULL, out_info);
}

int nqs_sr_step(const nqs_config_t *cfg,
                int size_x, int size_y,
                nqs_ansatz_t *ansatz,
                nqs_sampler_t *sampler,
                nqs_sr_step_info_t *out_info) {
    return nqs_sr_step_custom_full(cfg, size_x, size_y, ansatz, sampler,
                                    NULL, NULL, NULL, NULL, out_info);
}

int nqs_sr_run_custom_full(const nqs_config_t *cfg,
                            int size_x, int size_y,
                            nqs_ansatz_t *ansatz,
                            nqs_sampler_t *sampler,
                            nqs_log_amp_fn_t log_amp_fn,
                            void *log_amp_user,
                            nqs_gradient_fn_t gradient_fn,
                            void *grad_user,
                            double *out_energy_trace) {
    if (!cfg || !ansatz || !sampler) return -1;
    nqs_sampler_thermalize(sampler);
    for (int it = 0; it < cfg->num_iterations; it++) {
        nqs_sr_step_info_t info;
        int rc = nqs_sr_step_custom_full(cfg, size_x, size_y, ansatz, sampler,
                                          log_amp_fn, log_amp_user,
                                          gradient_fn, grad_user, &info);
        if (rc != 0) return rc;
        if (out_energy_trace) out_energy_trace[it] = info.mean_energy;
    }
    return 0;
}

int nqs_sr_run_custom(const nqs_config_t *cfg,
                      int size_x, int size_y,
                      nqs_ansatz_t *ansatz,
                      nqs_sampler_t *sampler,
                      nqs_log_amp_fn_t log_amp_fn,
                      void *log_amp_user,
                      double *out_energy_trace) {
    return nqs_sr_run_custom_full(cfg, size_x, size_y, ansatz, sampler,
                                   log_amp_fn, log_amp_user,
                                   NULL, NULL, out_energy_trace);
}

int nqs_sr_run(const nqs_config_t *cfg,
               int size_x, int size_y,
               nqs_ansatz_t *ansatz,
               nqs_sampler_t *sampler,
               double *out_energy_trace) {
    return nqs_sr_run_custom(cfg, size_x, size_y, ansatz, sampler,
                              NULL, NULL, out_energy_trace);
}

/* ===================== holomorphic SR ================================ */
/*
 * Complex G = 2 Re{⟨O* O⟩ - ⟨O*⟩⟨O⟩} with O = R + i I is
 *     G_kl = 2 [ ⟨R_k R_l + I_k I_l⟩ - ⟨R_k⟩⟨R_l⟩ - ⟨I_k⟩⟨I_l⟩ ]
 * Matrix-free product with a real vector v:
 *     u^R_i = Σ_k R_ik v_k,    u^I_i = Σ_k I_ik v_k
 *     (G v)_k = 2 [ (R^T u^R + I^T u^I)/N_s - ⟨R⟩_k ⟨u^R⟩ - ⟨I⟩_k ⟨u^I⟩ ]
 *             + ε v_k
 * The factor of 2 is kept for consistency with the force definition
 * below. */
static void qgt_complex_apply(const double *R, const double *I_mat,
                               long batch_size, long num_params,
                               const double *R_mean, const double *I_mean,
                               double epsilon,
                               const double *v,
                               double *out) {
    double *uR = calloc((size_t)batch_size, sizeof(double));
    double *uI = calloc((size_t)batch_size, sizeof(double));
    if (!uR || !uI) {
        free(uR); free(uI);
        memset(out, 0, (size_t)num_params * sizeof(double));
        return;
    }
    for (long i = 0; i < batch_size; i++) {
        double sR = 0.0, sI = 0.0;
        const double *rowR = &R[i * num_params];
        const double *rowI = &I_mat[i * num_params];
        for (long k = 0; k < num_params; k++) {
            sR += rowR[k] * v[k];
            sI += rowI[k] * v[k];
        }
        uR[i] = sR; uI[i] = sI;
    }
    double uR_mean = 0.0, uI_mean = 0.0;
    for (long i = 0; i < batch_size; i++) { uR_mean += uR[i]; uI_mean += uI[i]; }
    uR_mean /= (double)batch_size;
    uI_mean /= (double)batch_size;

    for (long k = 0; k < num_params; k++) {
        double accR = 0.0, accI = 0.0;
        for (long i = 0; i < batch_size; i++) {
            accR += R[i * num_params + k] * uR[i];
            accI += I_mat[i * num_params + k] * uI[i];
        }
        out[k] = 2.0 * ((accR + accI) / (double)batch_size
                        - R_mean[k] * uR_mean - I_mean[k] * uI_mean)
                 + epsilon * v[k];
    }
    free(uR); free(uI);
}

static int cg_solve_complex(const double *R, const double *I_mat,
                             long batch_size, long num_params,
                             const double *R_mean, const double *I_mean,
                             double epsilon,
                             const double *rhs,
                             int max_iters, double tol,
                             double *out_delta, int *out_converged) {
    memset(out_delta, 0, (size_t)num_params * sizeof(double));
    double *r = malloc((size_t)num_params * sizeof(double));
    double *p = malloc((size_t)num_params * sizeof(double));
    double *Ap = malloc((size_t)num_params * sizeof(double));
    if (!r || !p || !Ap) {
        free(r); free(p); free(Ap);
        if (out_converged) *out_converged = 0;
        return 0;
    }
    memcpy(r, rhs, (size_t)num_params * sizeof(double));
    memcpy(p, r,   (size_t)num_params * sizeof(double));
    double rs_old = vec_dot(r, r, num_params);
    double rs0 = rs_old;
    int iter;
    int converged = 0;
    for (iter = 0; iter < max_iters; iter++) {
        qgt_complex_apply(R, I_mat, batch_size, num_params,
                           R_mean, I_mean, epsilon, p, Ap);
        double pAp = vec_dot(p, Ap, num_params);
        if (fabs(pAp) < 1e-30) break;
        double alpha = rs_old / pAp;
        vec_axpy(out_delta, alpha, p, num_params);
        vec_axpy(r, -alpha, Ap, num_params);
        double rs_new = vec_dot(r, r, num_params);
        if (rs_new <= tol * tol * rs0) {
            converged = 1;
            iter++;
            break;
        }
        double beta = rs_new / rs_old;
        for (long i = 0; i < num_params; i++) p[i] = r[i] + beta * p[i];
        rs_old = rs_new;
    }
    free(r); free(p); free(Ap);
    if (out_converged) *out_converged = converged;
    return iter;
}

int nqs_sr_step_holomorphic(const nqs_config_t *cfg,
                             int size_x, int size_y,
                             nqs_ansatz_t *ansatz,
                             nqs_sampler_t *sampler,
                             nqs_log_amp_fn_t log_amp_fn,
                             void *log_amp_user,
                             nqs_sr_step_info_t *out_info) {
    if (!cfg || !ansatz || !sampler) return -1;
    int N = size_x * size_y;
    long num_params = nqs_ansatz_num_params(ansatz);
    if (num_params <= 0) return -1;
    if (!log_amp_fn) { log_amp_fn = nqs_ansatz_log_amp; log_amp_user = ansatz; }

    int batch_size = cfg->num_samples;
    int *batch = malloc((size_t)batch_size * (size_t)N * sizeof(int));
    double *E_re = malloc((size_t)batch_size * sizeof(double));
    double *E_im = malloc((size_t)batch_size * sizeof(double));
    double *R  = malloc((size_t)batch_size * (size_t)num_params * sizeof(double));
    double *Im = malloc((size_t)batch_size * (size_t)num_params * sizeof(double));
    double *R_mean = calloc((size_t)num_params, sizeof(double));
    double *I_mean = calloc((size_t)num_params, sizeof(double));
    double *F = calloc((size_t)num_params, sizeof(double));
    double *delta = calloc((size_t)num_params, sizeof(double));
    if (!batch || !E_re || !E_im || !R || !Im || !R_mean || !I_mean || !F || !delta) {
        free(batch); free(E_re); free(E_im); free(R); free(Im);
        free(R_mean); free(I_mean); free(F); free(delta);
        return -1;
    }

    /* 1. Sample batch from |ψ|². */
    nqs_sampler_batch(sampler, batch_size, batch);

    /* 2. Complex local energies per sample. */
    nqs_local_energy_batch_complex(cfg, size_x, size_y, batch, batch_size,
                                    log_amp_fn, log_amp_user, E_re, E_im);

    /* 3. Complex gradients per sample. */
    for (int i = 0; i < batch_size; i++) {
        nqs_ansatz_logpsi_gradient_complex(ansatz,
                                            &batch[(size_t)i * (size_t)N], N,
                                            &R[(size_t)i * (size_t)num_params],
                                            &Im[(size_t)i * (size_t)num_params]);
    }

    /* 4. Means + force. */
    double Er_mean = 0.0, Ei_mean = 0.0, Er_sq_mean = 0.0;
    for (int i = 0; i < batch_size; i++) {
        Er_mean    += E_re[i];
        Ei_mean    += E_im[i];
        Er_sq_mean += E_re[i] * E_re[i];
    }
    Er_mean    /= (double)batch_size;
    Ei_mean    /= (double)batch_size;
    Er_sq_mean /= (double)batch_size;

    for (long k = 0; k < num_params; k++) {
        double sR = 0.0, sI = 0.0;
        double sRe = 0.0, sIe = 0.0;
        for (int i = 0; i < batch_size; i++) {
            double rk = R [(size_t)i * (size_t)num_params + k];
            double ik = Im[(size_t)i * (size_t)num_params + k];
            sR  += rk;
            sI  += ik;
            sRe += rk * E_re[i];
            sIe += ik * E_im[i];
        }
        R_mean[k] = sR / (double)batch_size;
        I_mean[k] = sI / (double)batch_size;
        /* F_k = 2 [ (R^T E_re + I^T E_im)/N_s - ⟨R⟩ ⟨E_re⟩ - ⟨I⟩ ⟨E_im⟩ ] */
        F[k] = 2.0 * ((sRe + sIe) / (double)batch_size
                      - R_mean[k] * Er_mean - I_mean[k] * Ei_mean);
    }

    /* 5. CG solve (G + ε I) δ = F. */
    int converged = 0;
    int iters = cg_solve_complex(R, Im, batch_size, num_params,
                                   R_mean, I_mean,
                                   cfg->sr_diag_shift,
                                   F,
                                   cfg->sr_cg_max_iters, cfg->sr_cg_tol,
                                   delta, &converged);

    /* 6. Apply update: θ ← θ - lr · δ. */
    nqs_ansatz_apply_update(ansatz, delta, -cfg->learning_rate);

    if (out_info) {
        out_info->mean_energy = Er_mean;
        out_info->variance_energy = Er_sq_mean - Er_mean * Er_mean;
        out_info->update_norm = vec_norm(delta, num_params) * cfg->learning_rate;
        out_info->acceptance_ratio = nqs_sampler_acceptance_ratio(sampler);
        out_info->cg_iterations = iters;
        out_info->converged = converged;
    }

    free(batch); free(E_re); free(E_im); free(R); free(Im);
    free(R_mean); free(I_mean); free(F); free(delta);
    return 0;
}

int nqs_sr_run_holomorphic(const nqs_config_t *cfg,
                            int size_x, int size_y,
                            nqs_ansatz_t *ansatz,
                            nqs_sampler_t *sampler,
                            nqs_log_amp_fn_t log_amp_fn,
                            void *log_amp_user,
                            double *out_energy_trace) {
    if (!cfg || !ansatz || !sampler) return -1;
    nqs_sampler_thermalize(sampler);
    for (int it = 0; it < cfg->num_iterations; it++) {
        nqs_sr_step_info_t info;
        int rc = nqs_sr_step_holomorphic(cfg, size_x, size_y, ansatz, sampler,
                                           log_amp_fn, log_amp_user, &info);
        if (rc != 0) return rc;
        if (out_energy_trace) out_energy_trace[it] = info.mean_energy;
    }
    return 0;
}

/* ===================== real-time tVMC ================================ */
/* Compute Re(S)^{-1} Im(F) at the current ansatz params from a freshly
 * sampled batch. Does NOT apply the update. `out_delta` length = num_params.
 * Used by both Euler and Heun time-steppers. */
static int tvmc_compute_delta(const nqs_config_t *cfg,
                               int size_x, int size_y,
                               nqs_ansatz_t *ansatz,
                               nqs_sampler_t *sampler,
                               nqs_log_amp_fn_t log_amp_fn,
                               void *log_amp_user,
                               double *out_delta,
                               double *out_mean_energy,
                               double *out_variance_energy,
                               int    *out_cg_iters,
                               int    *out_cg_converged) {
    int N = size_x * size_y;
    long num_params = nqs_ansatz_num_params(ansatz);
    int batch_size = cfg->num_samples;
    int *batch = malloc((size_t)batch_size * (size_t)N * sizeof(int));
    double *E_re = malloc((size_t)batch_size * sizeof(double));
    double *E_im = malloc((size_t)batch_size * sizeof(double));
    double *R  = malloc((size_t)batch_size * (size_t)num_params * sizeof(double));
    double *Im = malloc((size_t)batch_size * (size_t)num_params * sizeof(double));
    double *R_mean = calloc((size_t)num_params, sizeof(double));
    double *I_mean = calloc((size_t)num_params, sizeof(double));
    double *F = calloc((size_t)num_params, sizeof(double));
    if (!batch || !E_re || !E_im || !R || !Im || !R_mean || !I_mean || !F) {
        free(batch); free(E_re); free(E_im); free(R); free(Im);
        free(R_mean); free(I_mean); free(F);
        return -1;
    }

    nqs_sampler_batch(sampler, batch_size, batch);
    nqs_local_energy_batch_complex(cfg, size_x, size_y, batch, batch_size,
                                    log_amp_fn, log_amp_user, E_re, E_im);
    for (int i = 0; i < batch_size; i++) {
        nqs_ansatz_logpsi_gradient_complex(ansatz,
                                            &batch[(size_t)i * (size_t)N], N,
                                            &R[(size_t)i * (size_t)num_params],
                                            &Im[(size_t)i * (size_t)num_params]);
    }

    double Er_mean = 0.0, Ei_mean = 0.0, Er_sq_mean = 0.0;
    for (int i = 0; i < batch_size; i++) {
        Er_mean    += E_re[i];
        Ei_mean    += E_im[i];
        Er_sq_mean += E_re[i] * E_re[i];
    }
    Er_mean    /= (double)batch_size;
    Ei_mean    /= (double)batch_size;
    Er_sq_mean /= (double)batch_size;

    for (long k = 0; k < num_params; k++) {
        double sR = 0.0, sI = 0.0;
        double sR_EI = 0.0, sI_ER = 0.0;
        for (int i = 0; i < batch_size; i++) {
            double rk = R [(size_t)i * (size_t)num_params + k];
            double ik = Im[(size_t)i * (size_t)num_params + k];
            sR    += rk;
            sI    += ik;
            sR_EI += rk * E_im[i];
            sI_ER += ik * E_re[i];
        }
        R_mean[k] = sR / (double)batch_size;
        I_mean[k] = sI / (double)batch_size;
        F[k] = 2.0 * ((sR_EI - sI_ER) / (double)batch_size
                      - R_mean[k] * Ei_mean + I_mean[k] * Er_mean);
    }

    int converged = 0;
    int iters = cg_solve_complex(R, Im, batch_size, num_params,
                                   R_mean, I_mean,
                                   cfg->sr_diag_shift, F,
                                   cfg->sr_cg_max_iters, cfg->sr_cg_tol,
                                   out_delta, &converged);

    if (out_mean_energy)     *out_mean_energy     = Er_mean;
    if (out_variance_energy) *out_variance_energy = Er_sq_mean - Er_mean * Er_mean;
    if (out_cg_iters)        *out_cg_iters        = iters;
    if (out_cg_converged)    *out_cg_converged    = converged;

    free(batch); free(E_re); free(E_im); free(R); free(Im);
    free(R_mean); free(I_mean); free(F);
    return 0;
}

int nqs_tvmc_step_heun(const nqs_config_t *cfg, double dt,
                        int size_x, int size_y,
                        nqs_ansatz_t *ansatz,
                        nqs_sampler_t *sampler,
                        nqs_log_amp_fn_t log_amp_fn,
                        void *log_amp_user,
                        nqs_sr_step_info_t *out_info) {
    /* Heun (improved Euler):
     *     k1 = δ(θ)
     *     k2 = δ(θ + dt · k1)
     *     θ_new = θ + (dt/2) · (k1 + k2)
     * For the tVMC real-time force, energy is conserved to O(dt³).
     * Requires two MC samplings per step vs one for Euler. */
    if (!cfg || !ansatz || !sampler) return -1;
    long num_params = nqs_ansatz_num_params(ansatz);
    if (num_params <= 0) return -1;
    if (!log_amp_fn) { log_amp_fn = nqs_ansatz_log_amp; log_amp_user = ansatz; }

    double *k1 = calloc((size_t)num_params, sizeof(double));
    double *k2 = calloc((size_t)num_params, sizeof(double));
    double *theta0 = malloc((size_t)num_params * sizeof(double));
    if (!k1 || !k2 || !theta0) { free(k1); free(k2); free(theta0); return -1; }
    /* Snapshot θ. */
    memcpy(theta0, nqs_ansatz_params_raw(ansatz),
           (size_t)num_params * sizeof(double));

    double E1 = 0, V1 = 0;
    int iters1 = 0, conv1 = 0;
    int rc = tvmc_compute_delta(cfg, size_x, size_y, ansatz, sampler,
                                 log_amp_fn, log_amp_user,
                                 k1, &E1, &V1, &iters1, &conv1);
    if (rc != 0) { free(k1); free(k2); free(theta0); return rc; }

    /* Probe at θ + dt · k1. */
    nqs_ansatz_apply_update(ansatz, k1, dt);
    int iters2 = 0, conv2 = 0;
    rc = tvmc_compute_delta(cfg, size_x, size_y, ansatz, sampler,
                             log_amp_fn, log_amp_user,
                             k2, NULL, NULL, &iters2, &conv2);
    if (rc != 0) {
        /* Restore θ. */
        double *p = nqs_ansatz_params_raw(ansatz);
        memcpy(p, theta0, (size_t)num_params * sizeof(double));
        free(k1); free(k2); free(theta0);
        return rc;
    }
    /* Restore θ, then apply (dt/2) · (k1 + k2). */
    double *p = nqs_ansatz_params_raw(ansatz);
    memcpy(p, theta0, (size_t)num_params * sizeof(double));
    for (long i = 0; i < num_params; i++) k1[i] = 0.5 * (k1[i] + k2[i]);
    nqs_ansatz_apply_update(ansatz, k1, dt);

    if (out_info) {
        out_info->mean_energy = E1;
        out_info->variance_energy = V1;
        out_info->update_norm = vec_norm(k1, num_params) * fabs(dt);
        out_info->acceptance_ratio = nqs_sampler_acceptance_ratio(sampler);
        out_info->cg_iterations = iters1 + iters2;
        out_info->converged = conv1 && conv2;
    }

    free(k1); free(k2); free(theta0);
    return 0;
}

int nqs_tvmc_step_real_time(const nqs_config_t *cfg, double dt,
                             int size_x, int size_y,
                             nqs_ansatz_t *ansatz,
                             nqs_sampler_t *sampler,
                             nqs_log_amp_fn_t log_amp_fn,
                             void *log_amp_user,
                             nqs_sr_step_info_t *out_info) {
    if (!cfg || !ansatz || !sampler) return -1;
    int N = size_x * size_y;
    long num_params = nqs_ansatz_num_params(ansatz);
    if (num_params <= 0) return -1;
    if (!log_amp_fn) { log_amp_fn = nqs_ansatz_log_amp; log_amp_user = ansatz; }

    int batch_size = cfg->num_samples;
    int *batch = malloc((size_t)batch_size * (size_t)N * sizeof(int));
    double *E_re = malloc((size_t)batch_size * sizeof(double));
    double *E_im = malloc((size_t)batch_size * sizeof(double));
    double *R  = malloc((size_t)batch_size * (size_t)num_params * sizeof(double));
    double *Im = malloc((size_t)batch_size * (size_t)num_params * sizeof(double));
    double *R_mean = calloc((size_t)num_params, sizeof(double));
    double *I_mean = calloc((size_t)num_params, sizeof(double));
    double *F = calloc((size_t)num_params, sizeof(double));
    double *delta = calloc((size_t)num_params, sizeof(double));
    if (!batch || !E_re || !E_im || !R || !Im || !R_mean || !I_mean || !F || !delta) {
        free(batch); free(E_re); free(E_im); free(R); free(Im);
        free(R_mean); free(I_mean); free(F); free(delta);
        return -1;
    }

    nqs_sampler_batch(sampler, batch_size, batch);
    nqs_local_energy_batch_complex(cfg, size_x, size_y, batch, batch_size,
                                    log_amp_fn, log_amp_user, E_re, E_im);

    for (int i = 0; i < batch_size; i++) {
        nqs_ansatz_logpsi_gradient_complex(ansatz,
                                            &batch[(size_t)i * (size_t)N], N,
                                            &R[(size_t)i * (size_t)num_params],
                                            &Im[(size_t)i * (size_t)num_params]);
    }

    double Er_mean = 0.0, Ei_mean = 0.0, Er_sq_mean = 0.0;
    for (int i = 0; i < batch_size; i++) {
        Er_mean    += E_re[i];
        Ei_mean    += E_im[i];
        Er_sq_mean += E_re[i] * E_re[i];
    }
    Er_mean    /= (double)batch_size;
    Ei_mean    /= (double)batch_size;
    Er_sq_mean /= (double)batch_size;

    /* Real-time force:  F_k = 2 [ Cov(R_k, E_im) - Cov(I_k, E_re) ]
     *                       = 2 [ (R^T E_im - I^T E_re)/N_s
     *                             - ⟨R⟩ ⟨E_im⟩ + ⟨I⟩ ⟨E_re⟩ ] */
    for (long k = 0; k < num_params; k++) {
        double sR = 0.0, sI = 0.0;
        double sR_EI = 0.0, sI_ER = 0.0;
        for (int i = 0; i < batch_size; i++) {
            double rk = R [(size_t)i * (size_t)num_params + k];
            double ik = Im[(size_t)i * (size_t)num_params + k];
            sR    += rk;
            sI    += ik;
            sR_EI += rk * E_im[i];
            sI_ER += ik * E_re[i];
        }
        R_mean[k] = sR / (double)batch_size;
        I_mean[k] = sI / (double)batch_size;
        F[k] = 2.0 * ((sR_EI - sI_ER) / (double)batch_size
                      - R_mean[k] * Ei_mean + I_mean[k] * Er_mean);
    }

    int converged = 0;
    int iters = cg_solve_complex(R, Im, batch_size, num_params,
                                   R_mean, I_mean,
                                   cfg->sr_diag_shift,
                                   F,
                                   cfg->sr_cg_max_iters, cfg->sr_cg_tol,
                                   delta, &converged);

    /* Forward Euler: θ ← θ + dt · δ. Energy is conserved to O(dt²). */
    nqs_ansatz_apply_update(ansatz, delta, dt);

    if (out_info) {
        out_info->mean_energy = Er_mean;
        out_info->variance_energy = Er_sq_mean - Er_mean * Er_mean;
        out_info->update_norm = vec_norm(delta, num_params) * fabs(dt);
        out_info->acceptance_ratio = nqs_sampler_acceptance_ratio(sampler);
        out_info->cg_iterations = iters;
        out_info->converged = converged;
    }

    free(batch); free(E_re); free(E_im); free(R); free(Im);
    free(R_mean); free(I_mean); free(F); free(delta);
    return 0;
}
