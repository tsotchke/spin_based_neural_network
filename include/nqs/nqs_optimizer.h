/*
 * include/nqs/nqs_optimizer.h
 *
 * Stochastic Reconfiguration (Sorella 1998) with conjugate-gradient
 * preconditioning on the quantum geometric tensor.
 *
 * One SR step:
 *
 *   1. Sample a batch {s_1, ..., s_N} from |ψ(s)|^2 via nqs_sampler.
 *   2. Compute per-sample local energies E_loc(s_i) via nqs_gradient.
 *   3. Compute per-sample log-psi gradients O_k(s_i) via
 *      nqs_ansatz_logpsi_gradient.
 *   4. Form the QGT   S_kl = <O_k* O_l> - <O_k*> <O_l>
 *      and the force  F_k  = <O_k* E_loc> - <O_k*> <E_loc>.
 *   5. Solve (S + ε I) δθ = F via preconditioned CG.
 *   6. Apply the update   θ ← θ - learning_rate · δθ.
 *
 * The implementation keeps memory linear in the parameter count by
 * never forming S explicitly: the QGT-vector product S v reduces to
 * two vector-vector products per sample (see Rende et al.,
 * Communications Physics 2024, for the identity this exploits).
 */
#ifndef NQS_OPTIMIZER_H
#define NQS_OPTIMIZER_H

#include "nqs_config.h"
#include "nqs_ansatz.h"
#include "nqs_sampler.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Per-step diagnostics returned by nqs_sr_step. */
typedef struct {
    double mean_energy;        /* <E_loc> over the batch */
    double variance_energy;    /* Var(E_loc) */
    double update_norm;        /* ||δθ|| */
    double acceptance_ratio;   /* sampler acceptance ratio */
    int    cg_iterations;      /* CG iterations used */
    int    converged;          /* 1 if CG met tolerance, 0 otherwise */
} nqs_sr_step_info_t;

/* Run a single stochastic-reconfiguration update. The sampler must
 * already be thermalised (call nqs_sampler_thermalize once before the
 * first step). Returns 0 on success. */
int nqs_sr_step(const nqs_config_t *cfg,
                int size_x, int size_y,
                nqs_ansatz_t *ansatz,
                nqs_sampler_t *sampler,
                nqs_sr_step_info_t *out_info);

/* Run `cfg->num_iterations` SR updates and accumulate a per-step
 * energy trajectory into `out_energy_trace` (length
 * `cfg->num_iterations`, caller-owned; NULL to disable).
 *
 * Returns 0 on success. */
int nqs_sr_run(const nqs_config_t *cfg,
               int size_x, int size_y,
               nqs_ansatz_t *ansatz,
               nqs_sampler_t *sampler,
               double *out_energy_trace);

/* Gradient callback for wrapped / symmetrised ansätze. When NULL the
 * optimizer computes ∂ log ψ / ∂θ directly from nqs_ansatz_logpsi_gradient.
 * A wrapper that modifies log_amp (e.g. translation projection) needs
 * its own gradient that accounts for the wrapper; such wrappers plug
 * in here. */
typedef int (*nqs_gradient_fn_t)(void *grad_user,
                                  nqs_ansatz_t *ansatz,
                                  const int *spins, int num_sites,
                                  double *out_grad);

/* SR step / run with an explicit log-amp callback. Lets a Marshall or
 * symmetry-projected wrapper feed the local-energy kernel the same
 * phase the sampler sees. When `log_amp_fn == NULL` these variants
 * degrade to the plain nqs_ansatz_log_amp path. */
int nqs_sr_step_custom(const nqs_config_t *cfg,
                       int size_x, int size_y,
                       nqs_ansatz_t *ansatz,
                       nqs_sampler_t *sampler,
                       nqs_log_amp_fn_t log_amp_fn,
                       void *log_amp_user,
                       nqs_sr_step_info_t *out_info);

int nqs_sr_run_custom(const nqs_config_t *cfg,
                      int size_x, int size_y,
                      nqs_ansatz_t *ansatz,
                      nqs_sampler_t *sampler,
                      nqs_log_amp_fn_t log_amp_fn,
                      void *log_amp_user,
                      double *out_energy_trace);

/* SR step / run that also takes a gradient callback for wrapped
 * ansätze. Pass NULL for both gradient_fn and grad_user to fall back
 * to the default ∂ log ψ_base / ∂θ path. */
int nqs_sr_step_custom_full(const nqs_config_t *cfg,
                             int size_x, int size_y,
                             nqs_ansatz_t *ansatz,
                             nqs_sampler_t *sampler,
                             nqs_log_amp_fn_t log_amp_fn,
                             void *log_amp_user,
                             nqs_gradient_fn_t gradient_fn,
                             void *grad_user,
                             nqs_sr_step_info_t *out_info);

int nqs_sr_run_custom_full(const nqs_config_t *cfg,
                            int size_x, int size_y,
                            nqs_ansatz_t *ansatz,
                            nqs_sampler_t *sampler,
                            nqs_log_amp_fn_t log_amp_fn,
                            void *log_amp_user,
                            nqs_gradient_fn_t gradient_fn,
                            void *grad_user,
                            double *out_energy_trace);

/* Holomorphic stochastic reconfiguration for complex-valued ansätze.
 *
 *   F_k = 2 Re{ ⟨O_k* E_loc⟩ - ⟨O_k*⟩⟨E_loc⟩ }
 *   G_kl = 2 Re{ ⟨O_k* O_l⟩ - ⟨O_k*⟩⟨O_l⟩ }
 *
 * where O_k = ∂ log ψ / ∂θ_k is complex (obtained via
 * nqs_ansatz_logpsi_gradient_complex) and E_loc may have a
 * non-vanishing imaginary part per sample (obtained via
 * nqs_local_energy_batch_complex). Compared to the real-projected
 * `nqs_sr_step_*` path, the holomorphic version picks up the
 * imaginary covariance that the real path silently drops, giving
 * the correct natural-gradient direction for non-stoquastic
 * Hamiltonians (e.g. frustrated J1-J2, Kitaev).
 *
 * For ansätze with `nqs_ansatz_is_complex(a) == 0`, the imaginary
 * part of O is zero and the holomorphic step reduces to the real
 * path up to the factor-of-2 normalisation (harmless: absorbed
 * into an effective learning rate). */
int nqs_sr_step_holomorphic(const nqs_config_t *cfg,
                             int size_x, int size_y,
                             nqs_ansatz_t *ansatz,
                             nqs_sampler_t *sampler,
                             nqs_log_amp_fn_t log_amp_fn,
                             void *log_amp_user,
                             nqs_sr_step_info_t *out_info);

int nqs_sr_run_holomorphic(const nqs_config_t *cfg,
                            int size_x, int size_y,
                            nqs_ansatz_t *ansatz,
                            nqs_sampler_t *sampler,
                            nqs_log_amp_fn_t log_amp_fn,
                            void *log_amp_user,
                            double *out_energy_trace);

/* Real-time tVMC step (Schrödinger evolution in the variational manifold).
 *
 * For real parameters θ, the complex tVMC projection equation
 *     S · θ̇ = -i · F
 * with F = ⟨O* E_loc⟩_c takes its real part to
 *     Re(S) · θ̇ = Im(F)
 * so the natural-gradient force for unitary evolution is
 *     F_k^{rt} = 2 [ Cov(R_k, E_im) − Cov(I_k, E_re) ]
 * while the metric Re(S) is the same Fubini–Study tensor used by
 * `nqs_sr_step_holomorphic`. The update is θ(t+dt) = θ(t) + dt · δ.
 *
 * Test: for any time-independent H, ⟨H⟩ is a conserved charge under
 * exact real-time evolution. A forward-Euler tVMC step conserves ⟨H⟩
 * up to O(dt²); the Heun (2nd-order) variant below improves this to
 * O(dt³) at the cost of one extra MC sampling per step.
 *
 * Out fields: `mean_energy`, `variance_energy`, and `update_norm` are
 * populated as with SR (update_norm uses dt instead of learning_rate). */
int nqs_tvmc_step_real_time(const nqs_config_t *cfg, double dt,
                             int size_x, int size_y,
                             nqs_ansatz_t *ansatz,
                             nqs_sampler_t *sampler,
                             nqs_log_amp_fn_t log_amp_fn,
                             void *log_amp_user,
                             nqs_sr_step_info_t *out_info);

/* Excited-state Stochastic Reconfiguration via orthogonal-ansatz
 * penalty (Choo-Neupert-Carleo 2018, arXiv:1810.10196 §II).
 *
 * Given a reference wavefunction ψ_ref (typically a converged ground-
 * state ansatz, passed as ref_log_amp_fn / ref_log_amp_user), trains
 * the ansatz to minimise
 *
 *     L[ψ]  =  ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩  +  μ |⟨r⟩_{|ψ|²}|²
 *
 * where r(s) = ψ_ref(s)/ψ(s) is the per-sample amplitude ratio,
 * ⟨r⟩_{|ψ|²} is the Monte Carlo estimator of the (unnormalised)
 * overlap, and μ = `penalty_mu` is the orthogonality strength.
 *
 * Implementation mirrors `nqs_sr_step_holomorphic`: same QGT metric,
 * same CG solve, same ansatz-update; the only modification is that
 * the per-sample local energy carries an additive penalty term
 *
 *     ΔE_loc(s)  =  μ · r(s) · conj(⟨r⟩_batch)
 *
 * whose expectation is μ|⟨r⟩|². Taking the gradient of L then slots
 * into the same natural-gradient formula as the ground-state SR step.
 *
 * `out_info->mean_energy` reports the *physical* energy ⟨H⟩ (not the
 * augmented loss); the penalty appears only as a gradient pressure
 * steering away from ψ_ref. For a well-trained ψ_ref ≈ |GS⟩ and large
 * μ, the trainer should converge to the first excited state.
 *
 * Returns 0 on success. */
int nqs_sr_step_excited(const nqs_config_t *cfg,
                         int size_x, int size_y,
                         nqs_ansatz_t *ansatz,
                         nqs_sampler_t *sampler,
                         nqs_log_amp_fn_t log_amp_fn,
                         void *log_amp_user,
                         nqs_log_amp_fn_t ref_log_amp_fn,
                         void *ref_log_amp_user,
                         double penalty_mu,
                         nqs_sr_step_info_t *out_info);

int nqs_sr_run_excited(const nqs_config_t *cfg,
                        int size_x, int size_y,
                        nqs_ansatz_t *ansatz,
                        nqs_sampler_t *sampler,
                        nqs_log_amp_fn_t log_amp_fn,
                        void *log_amp_user,
                        nqs_log_amp_fn_t ref_log_amp_fn,
                        void *ref_log_amp_user,
                        double penalty_mu,
                        double *out_energy_trace);

/* Second-order Heun integrator for real-time tVMC. One step costs two
 * MC samplings (one for k1 at θ, one for k2 at θ + dt·k1) but reduces
 * the energy-conservation error from O(dt²) to O(dt³). */
int nqs_tvmc_step_heun(const nqs_config_t *cfg, double dt,
                        int size_x, int size_y,
                        nqs_ansatz_t *ansatz,
                        nqs_sampler_t *sampler,
                        nqs_log_amp_fn_t log_amp_fn,
                        void *log_amp_user,
                        nqs_sr_step_info_t *out_info);

/*
 * MinSR (Chen & Heyl, Nat. Phys. 20:1476, 2024, arXiv:2302.01941;
 * Rende et al., Comm. Phys. 7:260, 2024, arXiv:2310.05715).
 *
 * Solves the same SR linear system   (S + ε I) δ = F   but in the
 * smaller N_s × N_s sample-space Gram matrix
 *
 *     T_{ij} = (1/N_s) Σ_k O_c(s_i)_k O_c(s_j)_k,
 *
 * via the push-through identity
 *
 *     (O_c^T O_c / N_s + ε I)⁻¹ O_c^T  =  O_c^T (T + ε I)⁻¹.
 *
 * Concretely:
 *   1.  ε_i = E_loc(s_i) − ⟨E_loc⟩
 *   2.  Form T (N_s × N_s) and add ε I   (cost O(N_s² N_p))
 *   3.  Cholesky-solve (T + ε I) y = ε for y   (cost O(N_s³))
 *   4.  δ_k = (1/N_s) Σ_i O_c(s_i)_k · y_i      (cost O(N_s N_p))
 *   5.  θ ← θ − lr · δ
 *
 * Vs. matrix-free CG SR: identical math, but Krylov dimension is
 * bounded by N_s rather than N_p, and the linear-system memory drops
 * from N_p × N_p to N_s × N_s.  This is the recipe that scales NQS to
 * the 10⁵–10⁶-parameter ansätze used at the 2024-2026 frontier.
 *
 * Use when N_p ≫ N_s (e.g. ResNet / ViT NQS).  For N_p ≲ N_s the
 * matrix-free CG path may be marginally cheaper.
 *
 * `out_info->cg_iterations` reports 0 (Cholesky is direct) and
 * `out_info->converged` is 1 on a successful Cholesky factorisation,
 * 0 when the regularised T failed to be PSD (numerical breakdown).
 */
int nqs_sr_step_minsr_full(const nqs_config_t *cfg,
                            int size_x, int size_y,
                            nqs_ansatz_t *ansatz,
                            nqs_sampler_t *sampler,
                            nqs_log_amp_fn_t log_amp_fn,
                            void *log_amp_user,
                            nqs_gradient_fn_t gradient_fn,
                            void *grad_user,
                            nqs_sr_step_info_t *out_info);

int nqs_sr_run_minsr(const nqs_config_t *cfg,
                      int size_x, int size_y,
                      nqs_ansatz_t *ansatz,
                      nqs_sampler_t *sampler,
                      nqs_log_amp_fn_t log_amp_fn,
                      void *log_amp_user,
                      double *out_energy_trace);

#ifdef __cplusplus
}
#endif

#endif /* NQS_OPTIMIZER_H */
