/*
 * src/thqcp/coupling.c
 *
 * Reference implementation of the THQCP coupling scheduler. The
 * classical-annealing plane uses an Ising Metropolis sweep with a
 * geometric β-schedule; the quantum-window primitive is a stub that
 * applies a single-shot bias to the targeted p-bit's local field and
 * returns a ±1 "measurement outcome" drawn from a biased coin whose
 * weight is the sigmoid of the local-field magnitude. This is the
 * first-order model; v0.6 replaces the stub with a real defect-qubit
 * evolution sampled through nqs_sampler + nqs_ansatz.
 *
 * The scheduler shape is what matters at this stage: one THQCP cycle
 * runs exactly the expected PHASE_ANNEAL → PHASE_QUANTUM → PHASE_FEEDBACK
 * state machine, so every higher-level integration (eshkol autograd,
 * noesis proof-trace reasoning, QGTL parameter-manifold logging) can
 * be wired against a stable cycle protocol.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "thqcp/coupling.h"

struct thqcp_state {
    thqcp_config_t  cfg;
    int            *pbits;          /* length N, values ±1               */
    double         *J;              /* N×N, row-major (copy)             */
    double         *h;              /* N, local fields (copy + fb bias)  */
    double         *h_bias_q;       /* N, accumulated quantum feedback   */
    int             current_sweep;
    int             stagnation_count;
    int             last_qubit_idx;
    double          energy;
    double          best_energy;
    unsigned long long rng;
    int             windows_opened;
    int             feedbacks_applied;
};

static double xs_uniform(unsigned long long *st) {
    unsigned long long x = *st ? *st : 0x9E3779B97F4A7C15ULL;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *st = x;
    return (double)(x >> 11) / 9007199254740992.0;
}

static double compute_energy(const thqcp_state_t *s) {
    int N = s->cfg.num_pbits;
    double E = 0.0;
    for (int i = 0; i < N; i++) {
        E -= (s->h[i] + s->h_bias_q[i]) * (double)s->pbits[i];
        for (int j = 0; j < i; j++) {
            E -= s->J[i * N + j] * (double)s->pbits[i] * (double)s->pbits[j];
        }
    }
    return E;
}

static double local_field(const thqcp_state_t *s, int i) {
    int N = s->cfg.num_pbits;
    double f = s->h[i] + s->h_bias_q[i];
    const double *row = &s->J[i * N];
    for (int j = 0; j < N; j++) f += row[j] * (double)s->pbits[j];
    return f;
}

static double current_beta(const thqcp_state_t *s) {
    int T = s->cfg.num_sweeps > 1 ? s->cfg.num_sweeps - 1 : 1;
    double t = (double)s->current_sweep / (double)T;
    if (t > 1.0) t = 1.0;
    double b0 = s->cfg.beta_start;
    double b1 = s->cfg.beta_end;
    if (b0 <= 0.0 || b1 <= 0.0) return (1.0 - t) * b0 + t * b1;
    return b0 * pow(b1 / b0, t);              /* geometric schedule */
}

thqcp_state_t *thqcp_state_create(const thqcp_config_t *cfg,
                                   const double *J_pbit,
                                   const double *h_pbit) {
    if (!cfg || !J_pbit || !h_pbit) return NULL;
    if (cfg->num_pbits <= 0 || cfg->num_qubits < 0) return NULL;
    if (cfg->num_qubits > cfg->num_pbits) return NULL;
    int N = cfg->num_pbits;
    thqcp_state_t *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->cfg = *cfg;
    /* Explicit size_t casts on BOTH operands of any N*N product. Without
     * both casts the compiler still does the right thing on LP64 (int
     * promotes to size_t), but the explicit form is overflow-safe under
     * any platform model and makes the intent unambiguous. */
    s->pbits     = malloc((size_t)N * sizeof(int));
    s->J         = malloc((size_t)N * (size_t)N * sizeof(double));
    s->h         = malloc((size_t)N * sizeof(double));
    s->h_bias_q  = calloc((size_t)N, sizeof(double));
    if (!s->pbits || !s->J || !s->h || !s->h_bias_q) {
        thqcp_state_free(s); return NULL;
    }
    memcpy(s->J, J_pbit, (size_t)N * (size_t)N * sizeof(double));
    memcpy(s->h, h_pbit, (size_t)N * sizeof(double));
    s->rng = cfg->seed ? cfg->seed : 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < N; i++) {
        s->pbits[i] = (xs_uniform(&s->rng) < 0.5) ? +1 : -1;
    }
    s->energy = compute_energy(s);
    s->best_energy = s->energy;
    s->last_qubit_idx = -1;
    return s;
}

void thqcp_state_free(thqcp_state_t *s) {
    if (!s) return;
    free(s->pbits); free(s->J); free(s->h); free(s->h_bias_q);
    free(s);
}

/* PHASE_ANNEAL — one Metropolis sweep at current β. */
static void anneal_sweep(thqcp_state_t *s) {
    int N = s->cfg.num_pbits;
    double beta = current_beta(s);
    int changed = 0;
    for (int k = 0; k < N; k++) {
        int i = (int)((double)N * xs_uniform(&s->rng));
        if (i >= N) i = N - 1;
        double f = local_field(s, i);
        double dE = 2.0 * (double)s->pbits[i] * f;
        if (dE <= 0.0 || xs_uniform(&s->rng) < exp(-beta * dE)) {
            s->pbits[i] = -s->pbits[i];
            s->energy += dE;          /* E_new = E_old + ΔE */
            changed++;
        }
    }
    if (changed == 0) s->stagnation_count++;
    else               s->stagnation_count = 0;
    if (s->energy < s->best_energy) s->best_energy = s->energy;
    s->current_sweep++;
}

/* Decide whether to open a quantum window and, if so, which qubit. */
static int pick_qubit_site(thqcp_state_t *s) {
    switch (s->cfg.open_policy) {
        case THQCP_OPEN_NEVER:
            return -1;
        case THQCP_OPEN_PERIODIC:
            if (s->cfg.period_k > 0 &&
                (s->current_sweep % s->cfg.period_k) == 0 &&
                s->current_sweep > 0) {
                return (s->last_qubit_idx + 1) % s->cfg.num_qubits;
            }
            return -1;
        case THQCP_OPEN_STAGNATION:
            if (s->stagnation_count >= s->cfg.stagnation_threshold) {
                s->stagnation_count = 0;
                return (s->last_qubit_idx + 1) % s->cfg.num_qubits;
            }
            return -1;
    }
    return -1;
}

/* PHASE_QUANTUM stub — v0.5 deterministic biased-coin model. Kept as
 * ablation baseline for the Paper-2 benchmarks. */
static int quantum_window_sample_stub(thqcp_state_t *s, int qubit_idx) {
    int site = qubit_idx;
    if (site >= s->cfg.num_pbits) site = s->cfg.num_pbits - 1;
    double f = local_field(s, site);
    double tau = s->cfg.qubit_window_tau;
    double p_flip = 1.0 / (1.0 + exp(+2.0 * f * tau));
    return (xs_uniform(&s->rng) < p_flip) ? -1 : +1;
}

/* PHASE_QUANTUM v0.6 — exact single-qubit evolution under
 *     H_q = h_z σ_z + h_x σ_x
 * with h_z = f_i (local p-bit field at the qubit locus) and h_x
 * set by the tunneling strength (1/τ_coh). Initial state is |+⟩ — an
 * equal superposition, the natural "neutral" starting point for a
 * briefly-opened defect qubit before the bias field steers it. Evolves
 * for τ_coh time units; measures projectively in σ_z basis. Outcome
 * ±1 is Born-rule sampled from |⟨±z|ψ(τ)⟩|².
 *
 * Derivation (2-level system):
 *   Eigenenergies ±Ω where Ω = √(h_z² + h_x²).
 *   Starting |+⟩ = (|+z⟩ + |−z⟩)/√2, probability of measuring |+z⟩:
 *       P(+) = ½ + (h_x h_z / Ω²) · sin²(Ωτ)
 *   (see e.g. Cohen-Tannoudji §IV.C — the Rabi formula for |+⟩ start.)
 *
 * Physics regimes:
 *   - h_z ≫ h_x (strong classical bias):  P(+) → ½ (|+⟩ is orthogonal
 *     to the eigenbasis — measurement gives ~50/50 unless h_x mixes).
 *     Actually the formula gives P(+) ≈ ½; the classical-like
 *     behaviour emerges from the |0⟩ or |1⟩ initial-state variants.
 *     For coherent tunneling modelling, we want a different initial
 *     state that captures "qubit probes whether to flip" — see below.
 *
 * Adopted physics model: the qubit starts in the p-bit's current
 * classical value (|+z⟩ if s_i = +1, |−z⟩ if s_i = −1), evolves under
 * H_q for τ_coh, and is measured. Then:
 *     P(flip | start +z) = (h_x² / Ω²) · sin²(Ωτ)
 * This is the transverse-field-quantum-annealing tunneling probability
 * — the qubit starts aligned with the current classical state, the
 * transverse field provides tunneling amplitude to the opposite state,
 * and projective measurement samples the outcome.
 *
 * Regimes:
 *   - h_x = 0: no tunneling, outcome always matches starting state.
 *   - h_z = 0, h_x · τ = π/2: coherent 50/50 regardless of start.
 *   - h_z ≫ h_x: small tunneling, P(flip) ≈ (h_x/h_z)² sin²(h_z τ).
 *     This is the *quantum-tunneling-assisted anneal* regime.
 */
static int quantum_window_sample_coherent(thqcp_state_t *s, int qubit_idx) {
    int site = qubit_idx;
    if (site >= s->cfg.num_pbits) site = s->cfg.num_pbits - 1;
    double hz = local_field(s, site);
    double tau = s->cfg.qubit_window_tau > 0 ? s->cfg.qubit_window_tau : 1.0;
    /* Transverse-field strength parameterisation: a generic defect qubit
     * (SiC V_Si, NV) in a weak coupling regime has h_x ~ 1/τ_coh. We
     * take h_x = 1.0 in normalised units and fold all scaling into
     * the anneal schedule (β) and feedback_strength. */
    double hx = 1.0;
    double Omega = sqrt(hz * hz + hx * hx);
    double phase = Omega * tau;
    double sin_sq = sin(phase) * sin(phase);
    double p_flip = (hx * hx / (Omega * Omega)) * sin_sq;
    /* Born-rule sample. Outcome is "flip" (−s_i) with prob p_flip,
     * else "stay" (+s_i). */
    int current_spin = s->pbits[site];
    if (xs_uniform(&s->rng) < p_flip) return -current_spin;
    return current_spin;
}

static int quantum_window_sample(thqcp_state_t *s, int qubit_idx) {
    if (s->cfg.window_model == THQCP_WINDOW_COHERENT) {
        return quantum_window_sample_coherent(s, qubit_idx);
    }
    return quantum_window_sample_stub(s, qubit_idx);
}

/* PHASE_FEEDBACK — turn a measurement outcome into a bias on the
 * local field, biasing subsequent classical sweeps toward that spin
 * assignment. Strength = α · outcome. */
static void apply_feedback(thqcp_state_t *s, int site, int outcome) {
    s->h_bias_q[site] += s->cfg.feedback_strength * (double)outcome;
    s->energy = compute_energy(s);
    s->feedbacks_applied++;
}

thqcp_phase_t thqcp_cycle_step(thqcp_state_t *s) {
    if (!s) return THQCP_PHASE_ANNEAL;
    anneal_sweep(s);
    int qubit_idx = pick_qubit_site(s);
    if (qubit_idx < 0) return THQCP_PHASE_ANNEAL;

    int outcome = quantum_window_sample(s, qubit_idx);
    s->last_qubit_idx = qubit_idx;
    s->windows_opened++;

    apply_feedback(s, qubit_idx, outcome);
    return THQCP_PHASE_FEEDBACK;
}

int thqcp_run(thqcp_state_t *s, thqcp_run_info_t *out_info) {
    if (!s) return -1;
    while (s->current_sweep < s->cfg.num_sweeps) {
        thqcp_cycle_step(s);
    }
    if (out_info) {
        out_info->sweeps_run        = s->current_sweep;
        out_info->windows_opened    = s->windows_opened;
        out_info->feedbacks_applied = s->feedbacks_applied;
        out_info->final_energy      = s->energy;
        out_info->best_energy       = s->best_energy;
        out_info->converged         = 1;
    }
    return 0;
}

const int *thqcp_state_pbit_config(const thqcp_state_t *s) {
    return s ? s->pbits : NULL;
}
double thqcp_state_energy(const thqcp_state_t *s) {
    return s ? s->energy : 0.0;
}
int thqcp_state_sweep_count(const thqcp_state_t *s) {
    return s ? s->current_sweep : 0;
}
