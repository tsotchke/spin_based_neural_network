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

    /* Set operator labels to canonical Majorana basis tags:
     *   γ_{2j}   ↦ +1 (real "X-like" component label)
     *   γ_{2j+1} ↦ +i (imaginary "Y-like" component label)
     * These are placeholder labels used by the operator-array API; real
     * Majorana physics uses MajoranaHilbertState (apply_majorana_op_to_state). */
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

/*
 * Build the 2N × 2N real symmetric BdG Hamiltonian for the Kitaev p-wave
 * superconducting wire with open boundary conditions.
 *
 * Basis: Ψ = (c_1, ..., c_N, c_1†, ..., c_N†)^T.  The Hamiltonian is
 *
 *   H = -μ Σ_j c_j†c_j  -  t Σ_j (c_j†c_{j+1} + h.c.)
 *       -  Δ Σ_j (c_j c_{j+1} + c_{j+1}†c_j†)
 *
 * which in the Nambu form H = (1/2) Ψ† H_BdG Ψ + const has the block
 * structure
 *
 *   H_BdG = [[ -μI - t·NN  ,  -Δ·NN_anti  ],
 *            [ -Δ·NN_anti  ,  +μI + t·NN  ]]
 *
 * where NN is the symmetric nearest-neighbour shift and NN_anti is the
 * antisymmetric pairing operator (Δ_{j,j+1} = -Δ, Δ_{j+1,j} = +Δ).
 * Real Δ keeps the matrix real and symmetric.
 */
static void build_kitaev_bdg(double *H, int N, double mu, double t, double delta) {
    int dim = 2 * N;
    for (int i = 0; i < dim * dim; i++) H[i] = 0.0;

    /* Top-left N×N block (cc): -μI on diagonal, -t on neighbours */
    for (int i = 0; i < N; i++) H[i * dim + i] = -mu;
    for (int i = 0; i < N - 1; i++) {
        H[i * dim + (i + 1)] = -t;
        H[(i + 1) * dim + i] = -t;
    }
    /* Bottom-right N×N block (c†c†): +μI, +t on neighbours */
    for (int i = 0; i < N; i++) H[(N + i) * dim + (N + i)] = mu;
    for (int i = 0; i < N - 1; i++) {
        H[(N + i) * dim + (N + i + 1)] = t;
        H[(N + i + 1) * dim + (N + i)] = t;
    }
    /* Off-diagonal pairing block: antisymmetric in the bare indices */
    for (int i = 0; i < N - 1; i++) {
        H[i * dim + (N + i + 1)] = -delta;
        H[(i + 1) * dim + (N + i)] =  delta;
        /* Symmetric counterpart in the lower triangle */
        H[(N + i + 1) * dim + i] = -delta;
        H[(N + i) * dim + (i + 1)] =  delta;
    }
}

/*
 * Real-symmetric Jacobi eigenvalue solver.  In-place destruction of A;
 * eigenvalues placed on the diagonal at exit and copied to evals[].  V is
 * filled with eigenvectors as columns and is initialised to identity.
 *
 * Convergence: until off-diagonal Frobenius norm < 1e-12 or 100 sweeps.
 */
static void jacobi_eig_symmetric(double *A, int n, double *evals, double *V) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            V[i * n + j] = (i == j) ? 1.0 : 0.0;

    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                off += A[i * n + j] * A[i * n + j];
        if (off < 1e-24) break;

        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                double a_pq = A[p * n + q];
                if (fabs(a_pq) < 1e-15) continue;
                double a_pp = A[p * n + p], a_qq = A[q * n + q];
                double theta = (a_qq - a_pp) / (2.0 * a_pq);
                double tval = (theta >= 0.0)
                    ? 1.0 / (theta + sqrt(1.0 + theta * theta))
                    : 1.0 / (theta - sqrt(1.0 + theta * theta));
                double c = 1.0 / sqrt(1.0 + tval * tval);
                double s = tval * c;
                /* Rotate rows */
                for (int k = 0; k < n; k++) {
                    double a_pk = A[p * n + k], a_qk = A[q * n + k];
                    A[p * n + k] = c * a_pk - s * a_qk;
                    A[q * n + k] = s * a_pk + c * a_qk;
                }
                /* Rotate columns */
                for (int k = 0; k < n; k++) {
                    double a_kp = A[k * n + p], a_kq = A[k * n + q];
                    A[k * n + p] = c * a_kp - s * a_kq;
                    A[k * n + q] = s * a_kp + c * a_kq;
                }
                /* Accumulate eigenvectors */
                for (int k = 0; k < n; k++) {
                    double v_kp = V[k * n + p], v_kq = V[k * n + q];
                    V[k * n + p] = c * v_kp - s * v_kq;
                    V[k * n + q] = s * v_kp + c * v_kq;
                }
            }
        }
    }
    for (int i = 0; i < n; i++) evals[i] = A[i * n + i];

    /* Sort eigenvalues ascending and reorder V columns */
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++)
            if (evals[j] < evals[min_idx]) min_idx = j;
        if (min_idx != i) {
            double tmp = evals[i]; evals[i] = evals[min_idx]; evals[min_idx] = tmp;
            for (int k = 0; k < n; k++) {
                double v = V[k * n + i];
                V[k * n + i] = V[k * n + min_idx];
                V[k * n + min_idx] = v;
            }
        }
    }
}

/*
 * Ground-state fermion parity of the Kitaev wire.
 *
 * The BdG ground state |GS⟩ is the vacuum of all positive-energy Bogoliubov
 * quasiparticles.  In the topological phase (|μ| < 2|t|) the OBC chain has
 * a doubly-degenerate ground state with a localized zero-mode pair; we
 * return +1 (even-parity sector) by convention.  In the trivial phase
 * (|μ| > 2|t|) the ground state is unique and its parity for the special
 * Δ = t > 0 line of the Kitaev wire reduces to (-1)^N (Kitaev 2001,
 * Phys.-Usp. 44:131, eq. 9).
 *
 * The result is deterministic and ±1.
 */
int calculate_majorana_parity(MajoranaChain *chain) {
    if (!chain) return 0;
    double abs_mu = fabs(chain->mu);
    double abs_t  = fabs(chain->t);

    if (abs_mu < 2.0 * abs_t) return +1;          /* topological: even sector */
    return (chain->num_sites & 1) ? -1 : +1;      /* trivial: (-1)^N */
}

/*
 * Detect Majorana zero modes by diagonalising the BdG Hamiltonian and
 * checking whether the lowest-magnitude eigenvalue is gap-suppressed and
 * end-localised.  Returns an end-localisation measure in [0, 1] when both
 * criteria are satisfied, zero otherwise.
 *
 * Criteria (heuristic but physics-motivated):
 *   gap_ratio  = min |E_k| / max |E_k|       must be < 0.05 (zero mode)
 *   end_weight = Σ_{site<2 or site≥N-2} |v|² must exceed 0.5 (localized)
 *
 * For an N=5 OBC Kitaev wire at (μ=0.5, t=1.0, Δ=1.0) the lowest pair of
 * BdG eigenvalues is exponentially suppressed, giving gap_ratio ~10⁻³ and
 * end_weight ~0.95.  At (μ=3.0, t=1.0, Δ=1.0) the bulk gap is ~|μ|-2|t|=1
 * and gap_ratio ~0.2, returning 0.
 */
double detect_majorana_zero_modes(MajoranaChain *chain, KitaevWireParameters *params) {
    if (!chain || !params || chain->num_sites < 2) return 0.0;

    int N   = chain->num_sites;
    int dim = 2 * N;

    double *H = (double *)malloc((size_t)dim * (size_t)dim * sizeof(double));
    double *V = (double *)malloc((size_t)dim * (size_t)dim * sizeof(double));
    double *E = (double *)malloc((size_t)dim * sizeof(double));
    if (!H || !V || !E) { free(H); free(V); free(E); return 0.0; }

    build_kitaev_bdg(H, N,
                     params->chemical_potential,
                     params->coupling_strength,
                     params->superconducting_gap);
    jacobi_eig_symmetric(H, dim, E, V);

    double max_abs = 0.0;
    int    min_idx = 0;
    double min_abs = INFINITY;
    for (int i = 0; i < dim; i++) {
        double a = fabs(E[i]);
        if (a > max_abs) max_abs = a;
        if (a < min_abs) { min_abs = a; min_idx = i; }
    }
    double gap_ratio = (max_abs > 1e-12) ? min_abs / max_abs : 0.0;

    /* Sum |amplitude|² of the lowest-|E| eigenvector at sites near both ends.
     * Each row index 0..N-1 is c_j, row N..2N-1 is c_j†; both map to site j. */
    double end_weight = 0.0, total_weight = 0.0;
    for (int i = 0; i < dim; i++) {
        double v = V[i * dim + min_idx];
        double v2 = v * v;
        total_weight += v2;
        int site = (i < N) ? i : (i - N);
        if (site < 2 || site >= N - 2) end_weight += v2;
    }
    double loc = (total_weight > 1e-12) ? end_weight / total_weight : 0.0;

    free(H); free(V); free(E);

    if (gap_ratio < 0.05 && loc > 0.5) return loc;
    return 0.0;
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
