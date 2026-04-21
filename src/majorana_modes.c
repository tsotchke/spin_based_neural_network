#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include "majorana_modes.h"

#define PI 3.14159265358979323846

// Initialize a Majorana fermion chain
MajoranaChain* initialize_majorana_chain(int num_sites, KitaevWireParameters *params) {
    if (num_sites <= 0) {
        fprintf(stderr, "Error: Number of sites must be positive\n");
        return NULL;
    }

    MajoranaChain *chain = (MajoranaChain *)malloc(sizeof(MajoranaChain));
    if (!chain) {
        fprintf(stderr, "Error: Memory allocation failed for MajoranaChain\n");
        return NULL;
    }

    chain->num_sites = num_sites;
    chain->num_operators = 2 * num_sites;  // 2 Majorana operators per site
    
    // Initialize parameters from KitaevWireParameters
    if (params) {
        chain->mu = params->chemical_potential;
        chain->t = params->coupling_strength;
        chain->delta = params->superconducting_gap;
    } else {
        // Default values for topological phase
        chain->mu = 0.5;
        chain->t = 1.0;
        chain->delta = 1.0;
    }
    
    // Determine if we're in the topological phase (|μ| < 2|t|)
    chain->in_topological_phase = (fabs(chain->mu) < 2.0 * fabs(chain->t));
    
    // Allocate memory for Majorana operators
    chain->operators = (double _Complex *)malloc(chain->num_operators * sizeof(double _Complex));
    if (!chain->operators) {
        fprintf(stderr, "Error: Memory allocation failed for operators\n");
        free(chain);
        return NULL;
    }

    // Initialize operators with random values (will be normalized later)
    for (int i = 0; i < chain->num_operators; i++) {
        double real_part = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        double imag_part = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        chain->operators[i] = real_part + imag_part * _Complex_I;
    }

    // Create proper Majorana operators from the random initialization
    create_majorana_operators(chain);

    return chain;
}

// Free memory allocated for a Majorana chain
void free_majorana_chain(MajoranaChain *chain) {
    if (chain) {
        if (chain->operators) {
            free(chain->operators);
        }
        free(chain);
    }
}

// Create Majorana operators from fermionic operators
void create_majorana_operators(MajoranaChain *chain) {
    if (!chain) return;

    // For each site j, create two operators:
    // γ_{2j-1} = c_j† + c_j
    // γ_{2j} = i(c_j† - c_j)
    for (int j = 0; j < chain->num_sites; j++) {
        // Simulating the creation of Majorana operators
        // In a real implementation, this would involve complex matrix operations
        
        // First Majorana operator at site j: γ_{2j}
        chain->operators[2*j] = 1.0;  // Normalized operator
        
        // Second Majorana operator at site j: γ_{2j+1}
        chain->operators[2*j + 1] = _Complex_I;  // I represents the imaginary unit
    }

    // Ensure operators satisfy the Majorana condition: γ^2 = 1
    for (int i = 0; i < chain->num_operators; i++) {
        double norm = cabs(chain->operators[i]);
        if (norm > 0) {
            chain->operators[i] /= norm;
        }
    }
}

// Apply Majorana operators to lattice sites
void apply_majorana_operator(MajoranaChain *chain, int operator_index, KitaevLattice *lattice) {
    if (!chain || !lattice || operator_index < 0 || operator_index >= chain->num_operators) {
        fprintf(stderr, "Error: Invalid parameters for apply_majorana_operator\n");
        return;
    }

    // In a full implementation, this would apply a Majorana operator transformation
    // For simplicity, we'll just perform a spin flip at the corresponding site
    int site_index = operator_index / 2;
    int x = site_index % lattice->size_x;
    int y = (site_index / lattice->size_x) % lattice->size_y;
    int z = site_index / (lattice->size_x * lattice->size_y);

    if (x < lattice->size_x && y < lattice->size_y && z < lattice->size_z) {
        lattice->spins[x][y][z] *= -1;  // Flip the spin
    }
}

// Calculate the parity of a Majorana chain
int calculate_majorana_parity(MajoranaChain *chain) {
    if (!chain) return 0;

    // For a proper implementation, parity would be calculated as:
    // P = i^N * γ_1 * γ_2 * ... * γ_{2N}
    // For simplicity, we'll return a random parity value
    return (rand() % 2) * 2 - 1;  // -1 or 1
}

// Detect zero modes at the ends of a chain
double detect_majorana_zero_modes(MajoranaChain *chain, KitaevWireParameters *params) {
    if (!chain || !params) return 0.0;

    // In a real implementation, we would diagonalize the Hamiltonian and
    // look for zero-energy eigenstates localized at the chain ends
    
    // For demonstration, we'll return a value based on the parameters
    // In the topological phase (|μ| < 2|t|), zero modes exist
    double t = params->coupling_strength;
    double mu = params->chemical_potential;
    double delta = params->superconducting_gap;
    
    // Simplified condition for topological phase
    if (fabs(mu) < 2.0 * fabs(t) && fabs(delta) > 0.1) {
        // Strength of localization decays exponentially into the bulk
        return 1.0 - fabs(mu)/(2.0 * fabs(t));
    } else {
        // No zero modes
        return 0.0;
    }
}

/*
 * Legacy operator-permutation "braiding": kept for back-compat with older
 * demos / docs. This is NOT the true Ising-anyon braiding unitary — it just
 * permutes the operator array with a sign. For real physics use
 * apply_braid_unitary() on a MajoranaHilbertState.
 */
void braid_majorana_operators_legacy(MajoranaChain *chain, int mode1, int mode2) {
    if (!chain || mode1 < 0 || mode1 >= chain->num_operators ||
        mode2 < 0 || mode2 >= chain->num_operators) {
        fprintf(stderr, "Error: Invalid parameters for braiding\n");
        return;
    }

    double _Complex temp = chain->operators[mode1];
    chain->operators[mode1] = chain->operators[mode2];
    chain->operators[mode2] = -temp;
}

void braid_majorana_modes(MajoranaChain *chain, int mode1, int mode2) {
    braid_majorana_operators_legacy(chain, mode1, mode2);
}

/* ------------------------------------------------------------------ */
/* Real Majorana braiding on a fermionic Fock-space state vector.     */
/*                                                                    */
/* For N Majorana operators γ_0, ..., γ_{N-1} with N even, pair them  */
/* into N/2 complex fermions and work in the 2^(N/2)-dim occupation   */
/* basis. Jordan-Wigner:                                              */
/*   γ_{2k}   = Z_0 Z_1 ... Z_{k-1} X_k                               */
/*   γ_{2k+1} = Z_0 Z_1 ... Z_{k-1} Y_k                               */
/* Braiding unitary:                                                  */
/*   B_{ij}   = exp(π γ_i γ_j / 4) = (1 + γ_i γ_j) / √2               */
/* satisfying (γ_i γ_j)^2 = -1, hence B^4 = -I, B^8 = I (Ising anyon) */
/* ------------------------------------------------------------------ */

MajoranaHilbertState* initialize_majorana_hilbert_state(int num_majoranas) {
    if (num_majoranas <= 0 || (num_majoranas & 1)) {
        fprintf(stderr, "Error: num_majoranas must be positive and even (got %d)\n",
                num_majoranas);
        return NULL;
    }
    int num_fermion_modes = num_majoranas / 2;
    if (num_fermion_modes > 20) {
        fprintf(stderr, "Error: num_fermion_modes=%d too large (2^%d states)\n",
                num_fermion_modes, num_fermion_modes);
        return NULL;
    }

    MajoranaHilbertState *state = malloc(sizeof(*state));
    if (!state) return NULL;

    state->num_fermion_modes = num_fermion_modes;
    state->hilbert_dim = 1 << num_fermion_modes;
    state->amplitudes = calloc((size_t)state->hilbert_dim, sizeof(double _Complex));
    if (!state->amplitudes) { free(state); return NULL; }

    state->amplitudes[0] = 1.0 + 0.0 * _Complex_I; /* fermion vacuum |0..0> */
    return state;
}

void free_majorana_hilbert_state(MajoranaHilbertState *state) {
    if (!state) return;
    free(state->amplitudes);
    free(state);
}

void majorana_hilbert_state_set_vacuum(MajoranaHilbertState *state) {
    if (!state) return;
    memset(state->amplitudes, 0, (size_t)state->hilbert_dim * sizeof(double _Complex));
    state->amplitudes[0] = 1.0 + 0.0 * _Complex_I;
}

void majorana_state_copy(const MajoranaHilbertState *src, MajoranaHilbertState *dst) {
    if (!src || !dst || src->hilbert_dim != dst->hilbert_dim) return;
    memcpy(dst->amplitudes, src->amplitudes,
           (size_t)src->hilbert_dim * sizeof(double _Complex));
}

double majorana_state_norm_squared(const MajoranaHilbertState *state) {
    if (!state) return 0.0;
    double s = 0.0;
    for (int n = 0; n < state->hilbert_dim; n++) {
        double re = creal(state->amplitudes[n]);
        double im = cimag(state->amplitudes[n]);
        s += re * re + im * im;
    }
    return s;
}

double _Complex majorana_states_inner_product(const MajoranaHilbertState *a,
                                              const MajoranaHilbertState *b) {
    if (!a || !b || a->hilbert_dim != b->hilbert_dim) return 0.0;
    double _Complex s = 0.0;
    for (int n = 0; n < a->hilbert_dim; n++) {
        s += conj(a->amplitudes[n]) * b->amplitudes[n];
    }
    return s;
}

void apply_majorana_op_to_state(int op_index, MajoranaHilbertState *state) {
    if (!state) return;
    int num_majoranas = 2 * state->num_fermion_modes;
    if (op_index < 0 || op_index >= num_majoranas) {
        fprintf(stderr, "Error: op_index=%d out of range [0,%d)\n",
                op_index, num_majoranas);
        return;
    }

    int k = op_index >> 1;       /* fermion mode */
    int is_y = op_index & 1;     /* γ_{2k+1} = Z-string * Y_k; otherwise X_k */
    int dim = state->hilbert_dim;
    int mask_k = 1 << k;
    /* Mask of all modes < k (for Jordan-Wigner Z-string parity) */
    int mask_lower = mask_k - 1;

    double _Complex *psi = state->amplitudes;
    double _Complex *out = calloc((size_t)dim, sizeof(double _Complex));
    if (!out) return;

    /* Action on basis |b>:
     *   Z-string on modes < k: multiply by (-1)^(popcount(b & mask_lower))
     *   X_k : flip bit k, no extra phase
     *   Y_k : flip bit k; phase +i if bit k was 0, -i if bit k was 1.
     * Populate `out[b_new] += phase * psi[b_old]`.
     */
    for (int b = 0; b < dim; b++) {
        double _Complex amp = psi[b];
        /* Exact-zero skip is safe here (not an FP-tolerance shortcut):
         * γ_i acts on the occupation basis as a phase-tagged bit-flip —
         * a permutation of basis vectors — so it preserves the support
         * of a sparse state exactly. Fresh calloc'd state vectors and
         * vectors evolved by γ_i alone therefore reliably trip this
         * skip and avoid ~dim−K unneeded complex multiplies. */
        if (amp == 0.0) continue;

        int pop = __builtin_popcount(b & mask_lower);
        double _Complex phase = (pop & 1) ? -1.0 : 1.0;

        if (is_y) {
            int bit_k = (b >> k) & 1;
            phase *= bit_k ? (-_Complex_I) : _Complex_I;
        }

        int b_new = b ^ mask_k;
        out[b_new] += phase * amp;
    }

    memcpy(psi, out, (size_t)dim * sizeof(double _Complex));
    free(out);
}

void apply_braid_unitary(MajoranaHilbertState *state, int i, int j) {
    if (!state || i == j) {
        fprintf(stderr, "Error: braid requires distinct modes (i=%d, j=%d)\n", i, j);
        return;
    }
    int num_majoranas = 2 * state->num_fermion_modes;
    if (i < 0 || i >= num_majoranas || j < 0 || j >= num_majoranas) {
        fprintf(stderr, "Error: braid modes out of range\n");
        return;
    }

    /* B_{ij}|ψ⟩ = (|ψ⟩ + γ_i γ_j |ψ⟩) / √2
     * Apply γ_j then γ_i to a scratch copy, add to original, scale.
     */
    int dim = state->hilbert_dim;
    double _Complex *save = malloc((size_t)dim * sizeof(double _Complex));
    if (!save) return;
    memcpy(save, state->amplitudes, (size_t)dim * sizeof(double _Complex));

    apply_majorana_op_to_state(j, state);
    apply_majorana_op_to_state(i, state);

    const double inv_sqrt2 = 1.0 / sqrt(2.0);
    for (int n = 0; n < dim; n++) {
        state->amplitudes[n] = (save[n] + state->amplitudes[n]) * inv_sqrt2;
    }

    free(save);
}

// Compute the energy of a Kitaev wire
double compute_kitaev_wire_energy(MajoranaChain *chain, KitaevWireParameters *params) {
    if (!chain || !params) return 0.0;

    // The Kitaev wire Hamiltonian in terms of Majorana operators:
    // H = iΣ_j [(μ/2)γ_{2j-1}γ_{2j} + (t+Δ)/2 γ_{2j}γ_{2j+1} + (t-Δ)/2 γ_{2j-1}γ_{2j+2}]
    
    double energy = 0.0;
    double mu = params->chemical_potential;
    double t = params->coupling_strength;
    double delta = params->superconducting_gap;
    
    // On-site terms
    for (int j = 0; j < chain->num_sites; j++) {
        energy += mu/2.0;  // Simplified on-site energy contribution
    }
    
    // Nearest-neighbor terms
    for (int j = 0; j < chain->num_sites - 1; j++) {
        energy += (t + delta)/2.0;  // Simplified nearest-neighbor energy
    }
    
    return energy;
}

// Map Majorana chain onto Kitaev lattice
void map_chain_to_lattice(MajoranaChain *chain, KitaevLattice *lattice, 
                         int start_x, int start_y, int start_z, int direction) {
    if (!chain || !lattice) {
        fprintf(stderr, "Error: Invalid parameters for map_chain_to_lattice\n");
        return;
    }
    
    // Check for valid starting position
    if (start_x < 0 || start_x >= lattice->size_x ||
        start_y < 0 || start_y >= lattice->size_y ||
        start_z < 0 || start_z >= lattice->size_z) {
        fprintf(stderr, "Error: Starting position is outside lattice bounds\n");
        return;
    }
    
    // Direction: 0 = x, 1 = y, 2 = z
    int dx = (direction == 0) ? 1 : 0;
    int dy = (direction == 1) ? 1 : 0;
    int dz = (direction == 2) ? 1 : 0;
    
    // Map each site of the chain onto the lattice
    for (int j = 0; j < chain->num_sites; j++) {
        int x = start_x + j * dx;
        int y = start_y + j * dy;
        int z = start_z + j * dz;
        
        // Check if position is within lattice bounds
        if (x < lattice->size_x && y < lattice->size_y && z < lattice->size_z) {
            // Set the spin based on the parity of adjacent Majorana operators
            // This is a simplified mapping - a real implementation would be more complex
            if (j % 2 == 0) {
                lattice->spins[x][y][z] = 1;  // Up spin
            } else {
                lattice->spins[x][y][z] = -1;  // Down spin
            }
        }
    }
}
