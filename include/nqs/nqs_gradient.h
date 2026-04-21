/*
 * include/nqs/nqs_gradient.h
 *
 * Local-energy estimator for Neural Network Quantum States. The local
 * energy
 *
 *     E_loc(s) = <s | Ĥ | ψ> / <s | ψ> = Σ_{s'} <s | Ĥ | s'> · ψ(s') / ψ(s)
 *
 * is the Monte Carlo estimator whose mean is the variational energy:
 *
 *     <E> = <E_loc>_{s ~ |ψ(s)|^2}
 *
 * The Hamiltonians supported in v0.4 are listed in
 * `nqs_hamiltonian_kind_t` (include/nqs/nqs_config.h). Each kernel
 * computes the local energy from a batch of sampled configurations by
 * enumerating the off-diagonal connections s → s' that the Hamiltonian
 * induces.
 */
#ifndef NQS_GRADIENT_H
#define NQS_GRADIENT_H

#include "nqs_config.h"
#include "nqs_sampler.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Compute the local energy for a single configuration `spins[num_sites]`
 * using the Hamiltonian + lattice shape described in `cfg`. The caller
 * supplies the ansatz via `log_amp` (see nqs_sampler.h) so the kernel
 * can evaluate the ψ(s') / ψ(s) ratios.
 *
 *   size_x, size_y: 2D lattice shape for local-neighbour connections
 *                   (total sites must equal size_x * size_y).
 *
 * Returns the scalar local energy (real part of ⟨s|Ĥ|ψ⟩/⟨s|ψ⟩; the
 * imaginary part cancels in expectation). */
double nqs_local_energy(const nqs_config_t *cfg,
                        int size_x, int size_y,
                        const int *spins,
                        nqs_log_amp_fn_t log_amp,
                        void *log_amp_user);

/* Batch variant: computes `E_loc[i]` for each `spins_batch[i * num_sites]`. */
void nqs_local_energy_batch(const nqs_config_t *cfg,
                            int size_x, int size_y,
                            const int *spins_batch, int batch_size,
                            nqs_log_amp_fn_t log_amp,
                            void *log_amp_user,
                            double *out_energies);

/* Complex local energy variant: returns real + imaginary parts of
 * E_loc(s) = Σ_{s'} H_{ss'} · ψ(s')/ψ(s). For Hermitian H, the
 * expectation ⟨E_loc⟩_{|ψ|²} is real, but per-sample values may be
 * complex when the wavefunction has non-trivial phase. Holomorphic SR
 * needs both parts (the force F_k = 2 Re{⟨O_k* E_loc⟩ - ⟨O_k*⟩⟨E_loc⟩}
 * picks up contributions from the imaginary correlation). */
void nqs_local_energy_complex(const nqs_config_t *cfg,
                               int size_x, int size_y,
                               const int *spins,
                               nqs_log_amp_fn_t log_amp,
                               void *log_amp_user,
                               double *out_re, double *out_im);

void nqs_local_energy_batch_complex(const nqs_config_t *cfg,
                                     int size_x, int size_y,
                                     const int *spins_batch, int batch_size,
                                     nqs_log_amp_fn_t log_amp,
                                     void *log_amp_user,
                                     double *out_re, double *out_im);

/* Running-mean and variance accumulator for Monte Carlo energy estimates. */
typedef struct {
    double sum;
    double sum_sq;
    long   count;
} nqs_energy_accumulator_t;

static inline void nqs_energy_accumulator_init(nqs_energy_accumulator_t *a) {
    a->sum = 0.0;
    a->sum_sq = 0.0;
    a->count = 0;
}

static inline void nqs_energy_accumulator_add(nqs_energy_accumulator_t *a,
                                               double sample) {
    a->sum    += sample;
    a->sum_sq += sample * sample;
    a->count  += 1;
}

static inline double nqs_energy_accumulator_mean(const nqs_energy_accumulator_t *a) {
    return a->count > 0 ? a->sum / (double)a->count : 0.0;
}

static inline double nqs_energy_accumulator_variance(const nqs_energy_accumulator_t *a) {
    if (a->count < 2) return 0.0;
    double m = nqs_energy_accumulator_mean(a);
    return (a->sum_sq / (double)a->count) - m * m;
}

#ifdef __cplusplus
}
#endif

#endif /* NQS_GRADIENT_H */
