/*
 * include/thqcp/coupling.h
 *
 * Thermodynamic Hybrid Quantum-Classical Processor — quantum-classical
 * coupling scheduler.
 *
 * A THQCP interleaves two dynamical processes on one silicon substrate:
 *   (1) a classical probabilistic-bit array (V-MTJ on CMOS) running a
 *       simulated-annealing trajectory through an Ising problem
 *       Hamiltonian H_p(s) over discrete configurations s ∈ {±1}^N;
 *   (2) an array of room-temperature defect qubits (SiC V_Si or NV) each
 *       evolving under a local Hamiltonian H_q(k) for a fixed window
 *       determined by its coherence time τ_coh.
 *
 * The coupling scheduler is the *clock* that decides when qubit windows
 * open, which p-bit neighbourhoods drive them, and how measurement
 * outcomes feed back into the anneal schedule. It is the central
 * architectural primitive — without a coupling scheduler, the THQCP is
 * two unrelated devices on one chip.
 *
 * The scheduler is implemented as a state machine over three phases:
 *
 *   PHASE_ANNEAL    p-bit array runs a Metropolis sweep at β(t). No
 *                   qubit activity. Advances classical trajectory
 *                   through the cooling schedule.
 *   PHASE_QUANTUM   A local defect-qubit window opens. The p-bit
 *                   configuration at a chosen locus biases the qubit
 *                   Hamiltonian (via dispersive coupling or ac-field).
 *                   The qubit evolves for ≤ τ_coh, then gets measured.
 *   PHASE_FEEDBACK  Measurement outcomes modify the p-bit energy
 *                   landscape — typically a shift in the local field
 *                   h_i reflecting the qubit's tunneling decision.
 *
 * One THQCP *cycle* is PHASE_ANNEAL → PHASE_QUANTUM → PHASE_FEEDBACK.
 * A run is a sequence of cycles at decreasing β. The scheduler decides:
 *   - when to open a quantum window (e.g., at level-crossing bottlenecks
 *     in the classical trajectory, detected via local energy stagnation),
 *   - which qubit to activate (one per chosen locus, round-robin, or
 *     triggered by a heuristic),
 *   - how long the window stays open (bounded by τ_coh).
 *
 * Theoretical basis:
 *   Sanchez-Forero et al., Quantum 8, 1486 (2024), "Stochastic
 *   Thermodynamics at the Quantum-Classical Boundary: A Self-Consistent
 *   Framework Based on Adiabatic-Response Theory."
 */
#ifndef THQCP_COUPLING_H
#define THQCP_COUPLING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    THQCP_PHASE_ANNEAL   = 0,
    THQCP_PHASE_QUANTUM  = 1,
    THQCP_PHASE_FEEDBACK = 2
} thqcp_phase_t;

typedef enum {
    /* Open a quantum window every `k` classical sweeps. Deterministic. */
    THQCP_OPEN_PERIODIC  = 0,
    /* Open a quantum window when the local p-bit configuration has not
     * changed for ≥ `stagnation_threshold` sweeps (level-crossing). */
    THQCP_OPEN_STAGNATION = 1,
    /* Never open quantum windows — pure classical annealer baseline. */
    THQCP_OPEN_NEVER     = 2
} thqcp_open_policy_t;

/* Quantum-window model. v0.5 stub (biased-coin) is the baseline; v0.6
 * coherent model is an exact single-qubit evolution under
 *     H_q = h_z σ_z  +  h_x σ_x
 * with h_z = f_i (local p-bit field at the qubit locus) and h_x =
 * τ_coh⁻¹ setting the tunneling strength. Measurement is projective
 * in the σ_z basis, Born-rule. This is the minimum physics-grounded
 * model of a room-T defect qubit (SiC V_Si or NV) driven briefly
 * through a coupling window. */
typedef enum {
    THQCP_WINDOW_STUB     = 0,    /* v0.5 biased-coin, kept for ablation    */
    THQCP_WINDOW_COHERENT = 1     /* v0.6 exact 2-level single-qubit evol.  */
} thqcp_window_model_t;

typedef struct {
    int    num_pbits;               /* N: classical Ising dimension        */
    int    num_qubits;              /* K: defect-qubit sites (K ≤ N)       */
    double beta_start;              /* anneal schedule β(0)                */
    double beta_end;                /* anneal schedule β(T_final)          */
    int    num_sweeps;              /* classical sweeps per run            */
    double qubit_window_tau;        /* τ_coh, in arbitrary time units      */
    thqcp_open_policy_t open_policy;
    thqcp_window_model_t window_model;
    int    period_k;                /* for PERIODIC: sweeps per window     */
    int    stagnation_threshold;    /* for STAGNATION: triggering value    */
    double feedback_strength;       /* α: scale of qubit → p-bit bias      */
    unsigned long long seed;
} thqcp_config_t;

static inline thqcp_config_t thqcp_config_defaults(void) {
    thqcp_config_t c;
    c.num_pbits             = 64;
    c.num_qubits            = 4;
    c.beta_start            = 0.1;
    c.beta_end              = 10.0;
    c.num_sweeps            = 1000;
    c.qubit_window_tau      = 1.0;
    c.open_policy           = THQCP_OPEN_PERIODIC;
    c.window_model          = THQCP_WINDOW_STUB;
    c.period_k              = 100;
    c.stagnation_threshold  = 20;
    c.feedback_strength     = 0.3;
    c.seed                  = 0xC0FFEEC0FFEEULL;
    return c;
}

/* Per-run diagnostics. */
typedef struct {
    int    sweeps_run;              /* actual classical sweeps executed   */
    int    windows_opened;          /* total PHASE_QUANTUM events          */
    int    feedbacks_applied;       /* total PHASE_FEEDBACK events         */
    double final_energy;            /* classical H_p(s_final)              */
    double best_energy;             /* min H_p observed over trajectory    */
    int    converged;               /* 1 = anneal completed cleanly        */
} thqcp_run_info_t;

/* Opaque state handle — holds p-bit configuration, qubit amplitudes,
 * coupling-map, and RNG state. Backed by the existing p-bit (`src/
 * neuromorphic/pbit.c`) and complex-RBM (`src/nqs/nqs_ansatz.c`)
 * modules; see src/thqcp/coupling.c for the wiring. */
typedef struct thqcp_state thqcp_state_t;

thqcp_state_t *thqcp_state_create(const thqcp_config_t *cfg,
                                   const double *J_pbit,   /* N×N Ising J */
                                   const double *h_pbit);  /* N field h   */
void           thqcp_state_free  (thqcp_state_t *s);

/* Run a full THQCP cycle: PHASE_ANNEAL → maybe PHASE_QUANTUM → maybe
 * PHASE_FEEDBACK. Returns the phase that most recently completed. */
thqcp_phase_t thqcp_cycle_step(thqcp_state_t *s);

/* Run the full anneal schedule for `cfg->num_sweeps` classical sweeps,
 * with coupling-scheduler-driven quantum windows. Writes diagnostics
 * to `*out_info`. Returns 0 on success. */
int thqcp_run(thqcp_state_t *s, thqcp_run_info_t *out_info);

/* Accessors. */
const int    *thqcp_state_pbit_config (const thqcp_state_t *s);
double        thqcp_state_energy      (const thqcp_state_t *s);
int           thqcp_state_sweep_count (const thqcp_state_t *s);

#ifdef __cplusplus
}
#endif

#endif /* THQCP_COUPLING_H */
