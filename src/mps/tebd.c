/*
 * src/mps/tebd.c
 *
 * Imaginary-time TEBD for a 1D XXZ / TFIM / Heisenberg chain. The
 * implementation mirrors the DMRG code's internal MPS representation
 * so the two modules interoperate (e.g. a DMRG ground state could be
 * handed to TEBD for real-time evolution, once the complex-tensor
 * extension lands).
 *
 * Trotter step: e^{-H τ} ≈ Π_even e^{-h_e τ/2} Π_odd e^{-h_o τ} Π_even e^{-h_e τ/2}
 * where h_i is the 2-site term on bond i.  For the second-order
 * symmetric split this is O(τ³) per-step error and O(τ²) total over
 * a fixed propagation time.
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "mps/tebd.h"
#include "mps/svd.h"

#define PHYS_D 2

typedef struct {
    int D_left, D_right;
    double *A;
} tebd_site_t;

/* Apply a single-site 2x2 gate to MPS site `i` (modifies physical
 * index in place). For a diagonal-in-|s⟩ gate diag(a, b):
 *   A[l, s, r] → gate(s) · A[l, s, r]
 * For a general 2x2 gate G:
 *   A[l, s', r] = Σ_s G[s', s] · A[l, s, r]  */
static void apply_site_gate(tebd_site_t *site, const double gate[4]) {
    int Dl = site->D_left, Dr = site->D_right;
    double *A = site->A;
    for (int l = 0; l < Dl; l++) for (int r = 0; r < Dr; r++) {
        double a0 = A[(l*PHYS_D + 0)*Dr + r];
        double a1 = A[(l*PHYS_D + 1)*Dr + r];
        A[(l*PHYS_D + 0)*Dr + r] = gate[0]*a0 + gate[1]*a1;
        A[(l*PHYS_D + 1)*Dr + r] = gate[2]*a0 + gate[3]*a1;
    }
}

/* Build the 4×4 bond gate e^{-h_bond · τ} for the cfg-selected
 * Hamiltonian. h_bond acts on a 4-dim 2-spin Hilbert space and is
 * diagonalised analytically — no matrix-exponential library required.
 *
 * We always express the gate in the basis |↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩:
 *   TFIM: h = -J σ^z σ^z - (Γ/2)(σ^x_i + σ^x_{i+1})
 *         (the Γ term is halved because each site participates in
 *          two bond gates when we Trotter-split; callers assemble it
 *          into the chain that way.)
 *   Heisenberg / XXZ: h = J/4 (σ^+ σ^- + σ^- σ^+) + Jz/4 σ^z σ^z
 *         wait: Heisenberg = J (S·S) = J/4 (σ·σ) so h/J = S·S.
 *         For XXZ: h = J/4 (σ^x σ^x + σ^y σ^y) + Jz/4 σ^z σ^z.
 *
 * Writes the real 4×4 gate into gate[4*4] row-major. */
static void build_bond_gate(const mps_config_t *cfg, double tau,
                             int edge_scale_field, double *gate) {
    memset(gate, 0, sizeof(double) * 16);
    double J  = cfg->J;
    double Jz = (cfg->ham == MPS_HAM_XXZ) ? cfg->Jz : cfg->J;
    double Gamma = cfg->Gamma;

    if (cfg->ham == MPS_HAM_TFIM) {
        /* For TFIM we use a split evolution:
         *     U(τ) = U_x(τ/2) · U_zz(τ) · U_x(τ/2)
         * with U_zz built here (diagonal 4×4) and U_x applied as
         * single-site gates outside. This avoids the boundary
         * double-counting inherent in stuffing σ^x into the bond
         * gate. The bond gate returned here is e^{+J σ^z σ^z τ}
         * (note sign: h_bond_zz = -J σ^z σ^z). */
        double a = exp(J * tau);
        double b = exp(-J * tau);
        gate[0*4+0]  = a;   /* ↑↑ */
        gate[1*4+1]  = b;   /* ↑↓ */
        gate[2*4+2]  = b;   /* ↓↑ */
        gate[3*4+3]  = a;   /* ↓↓ */
        (void)edge_scale_field;
        (void)Gamma;
        return;
    }


    /* Heisenberg / XXZ: diagonal pieces ±Jz/4 on (↑↑, ↓↓, ↑↓, ↓↑)
     * as respectively +Jz/4, +Jz/4, -Jz/4, -Jz/4. Off-diagonal J/2
     * couples the two antiparallel states. */
    double h[16];
    memset(h, 0, sizeof(h));
    h[0*4+0]   =  0.25 * Jz;
    h[3*4+3]   =  0.25 * Jz;
    h[1*4+1]   = -0.25 * Jz;
    h[2*4+2]   = -0.25 * Jz;
    h[1*4+2]   =  0.5  * J;
    h[2*4+1]   =  0.5  * J;
    /* Diagonalise: block-diagonal {|↑↑⟩}, {|↓↓⟩}, {|↑↓⟩, |↓↑⟩}. */
    double V[16];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) V[i*4+j] = (i==j)?1:0;
    /* |↓↓⟩ and |↑↑⟩ are eigenvectors; only the middle 2×2 needs rotating. */
    double a = h[1*4+1], b = h[2*4+2], c = h[1*4+2];
    double center = 0.5 * (a + b);
    double diff = 0.5 * (a - b);
    double radius = sqrt(diff * diff + c * c);
    double lam_m = center - radius;     /* singlet-like */
    double lam_p = center + radius;     /* triplet-m=0-like */
    double theta = 0.5 * atan2(2 * c, a - b);
    double cs = cos(theta), sn = sin(theta);
    /* Eigenvectors: V(:,1) = (cs, sn)^T for eigenvalue lam_? in the subspace. */
    V[1*4+1] = cs;  V[1*4+2] = -sn;
    V[2*4+1] = sn;  V[2*4+2] =  cs;
    double lam1 = cs*cs*a + sn*sn*b + 2*cs*sn*c;
    double lam2 = sn*sn*a + cs*cs*b - 2*cs*sn*c;
    (void)lam_p; (void)lam_m;
    double lam[4] = {h[0], lam1, lam2, h[15]};
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
        double v = 0;
        for (int k = 0; k < 4; k++) v += V[i*4+k] * exp(-lam[k] * tau) * V[j*4+k];
        gate[i*4+j] = v;
    }
    (void)edge_scale_field;
}

/* Apply a 4×4 gate to a pair of adjacent MPS sites. Merge →
 * contract with gate on the 4-dim physical space → SVD → split. */
static int apply_bond_gate(tebd_site_t *sites, int i, int D_max,
                            const double *gate) {
    int D_l = sites[i].D_left;
    int D_m = sites[i].D_right;  /* shared bond */
    int D_r = sites[i+1].D_right;
    int d = PHYS_D;

    /* Merge T[l, s1, s2, r] = Σ_m A[i][l, s1, m] A[i+1][m, s2, r]. */
    double *T = malloc(sizeof(double) * D_l * d * d * D_r);
    for (int l = 0; l < D_l; l++)
        for (int s1 = 0; s1 < d; s1++)
            for (int s2 = 0; s2 < d; s2++)
                for (int r = 0; r < D_r; r++) {
                    double v = 0;
                    for (int m = 0; m < D_m; m++)
                        v += sites[i  ].A[(l*d + s1)*D_m + m] *
                             sites[i+1].A[(m*d + s2)*D_r + r];
                    T[((l*d + s1)*d + s2)*D_r + r] = v;
                }
    /* Apply gate: T'[l, s1', s2', r] = Σ_{s1, s2} gate[(s1',s2'),(s1,s2)] T[l, s1, s2, r]. */
    double *Tg = malloc(sizeof(double) * D_l * d * d * D_r);
    for (int l = 0; l < D_l; l++)
        for (int s1p = 0; s1p < d; s1p++)
            for (int s2p = 0; s2p < d; s2p++)
                for (int r = 0; r < D_r; r++) {
                    double v = 0;
                    for (int s1 = 0; s1 < d; s1++)
                        for (int s2 = 0; s2 < d; s2++) {
                            int row = s1p * d + s2p;
                            int col = s1  * d + s2;
                            v += gate[row * (d*d) + col] *
                                 T[((l*d + s1)*d + s2)*D_r + r];
                        }
                    Tg[((l*d + s1p)*d + s2p)*D_r + r] = v;
                }
    free(T);
    /* Reshape Tg (D_l, d, d, D_r) → (D_l·d, d·D_r) matrix for SVD. */
    int rows = D_l * d;
    int cols = d * D_r;
    int n = rows < cols ? rows : cols;
    double *U, *sv, *Vt;
    double *M_tmp = NULL;
    const double *M_in = Tg;
    int Mrows = rows, Mcols = cols;
    if (rows < cols) {
        M_tmp = malloc(sizeof(double) * cols * rows);
        for (int i2 = 0; i2 < rows; i2++) for (int j2 = 0; j2 < cols; j2++)
            M_tmp[j2 * rows + i2] = Tg[i2 * cols + j2];
        M_in = M_tmp;
        Mrows = cols; Mcols = rows;
    }
    U  = malloc(sizeof(double) * Mrows * Mcols);
    sv = malloc(sizeof(double) * Mcols);
    Vt = malloc(sizeof(double) * Mcols * Mcols);
    svd_jacobi(M_in, Mrows, Mcols, U, sv, Vt, 1e-14);

    int D_new = n < D_max ? n : D_max;
    double *A1_new = calloc((size_t)D_l * d * D_new, sizeof(double));
    double *A2_new = calloc((size_t)D_new * d * D_r, sizeof(double));
    if (rows >= cols) {
        for (int ii = 0; ii < rows; ii++) for (int m = 0; m < D_new; m++)
            A1_new[(size_t)ii * D_new + m] = U[(size_t)ii * Mcols + m];
        for (int m = 0; m < D_new; m++) for (int j = 0; j < cols; j++)
            A2_new[(size_t)m * cols + j] = sv[m] * Vt[(size_t)m * Mcols + j];
    } else {
        double *VtT = malloc(sizeof(double) * Mcols * Mcols);
        for (int ii = 0; ii < Mcols; ii++) for (int j = 0; j < Mcols; j++)
            VtT[ii * Mcols + j] = Vt[j * Mcols + ii];
        double *UT = malloc(sizeof(double) * Mcols * Mrows);
        for (int ii = 0; ii < Mrows; ii++) for (int j = 0; j < Mcols; j++)
            UT[j * Mrows + ii] = U[ii * Mcols + j];
        for (int ii = 0; ii < rows; ii++) for (int m = 0; m < D_new; m++)
            A1_new[(size_t)ii * D_new + m] = VtT[(size_t)ii * Mcols + m];
        for (int m = 0; m < D_new; m++) for (int j = 0; j < cols; j++)
            A2_new[(size_t)m * cols + j] = sv[m] * UT[(size_t)m * Mrows + j];
        free(VtT); free(UT);
    }
    free(U); free(sv); free(Vt); free(M_tmp); free(Tg);

    free(sites[i  ].A); sites[i  ].A = A1_new;
    sites[i  ].D_left = D_l; sites[i  ].D_right = D_new;
    free(sites[i+1].A); sites[i+1].A = A2_new;
    sites[i+1].D_left = D_new; sites[i+1].D_right = D_r;
    return 0;
}

static int init_mps(tebd_site_t *sites, int N, int D_max, unsigned long long *rng) {
    int *dims = calloc((size_t)(N + 1), sizeof(int));
    dims[0] = 1; dims[N] = 1;
    for (int i = 1; i < N; i++) {
        int l = 1; for (int j = 0; j < i && l < D_max; j++) l *= 2;
        int r = 1; for (int j = 0; j < N - i && r < D_max; j++) r *= 2;
        dims[i] = l < r ? l : r;
        if (dims[i] > D_max) dims[i] = D_max;
        if (dims[i] < 1) dims[i] = 1;
    }
    for (int i = 0; i < N; i++) {
        sites[i].D_left = dims[i];
        sites[i].D_right = dims[i+1];
        sites[i].A = calloc((size_t)dims[i] * PHYS_D * dims[i+1], sizeof(double));
        for (int k = 0; k < dims[i] * PHYS_D * dims[i+1]; k++) {
            *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
            double u = (double)(*rng >> 11) / 9007199254740992.0;
            sites[i].A[k] = 0.1 * (u - 0.5);
        }
    }
    free(dims);
    return 0;
}

static double mps_norm2(const tebd_site_t *sites, int N) {
    int Dl = sites[0].D_left;
    double *L = calloc((size_t)Dl * Dl, sizeof(double));
    L[0] = 1.0;
    for (int i = 0; i < N; i++) {
        int dl = sites[i].D_left, dr = sites[i].D_right;
        double *Lp = calloc((size_t)dr * dr, sizeof(double));
        for (int r = 0; r < dr; r++) for (int rp = 0; rp < dr; rp++) {
            double v = 0;
            for (int l = 0; l < dl; l++) for (int lp = 0; lp < dl; lp++)
                for (int s = 0; s < PHYS_D; s++)
                    v += L[l*dl + lp] * sites[i].A[(l*PHYS_D + s)*dr + r]
                                       * sites[i].A[(lp*PHYS_D + s)*dr + rp];
            Lp[r*dr + rp] = v;
        }
        free(L); L = Lp;
    }
    double n2 = L[0];
    free(L);
    return n2;
}

static void mps_normalise(tebd_site_t *sites, int N) {
    double n2 = mps_norm2(sites, N);
    if (n2 <= 0) return;
    double scale = 1.0 / sqrt(n2);
    /* Distribute scale: put it all on A[0]. */
    int len = sites[0].D_left * PHYS_D * sites[0].D_right;
    for (int k = 0; k < len; k++) sites[0].A[k] *= scale;
}

/* Compute ⟨ψ|h_bond_i|ψ⟩ by exact contraction (valid for any MPS).
 * `h` is the 4×4 two-site Hamiltonian matrix (same basis as gate).
 * Used only inside the legacy per-bond energy path below (#if 0); marked
 * as unused to suppress -Wunused-function when that block is disabled. */
__attribute__((unused))
static double bond_expectation(const tebd_site_t *sites, int N,
                                int bond_i, const double *h) {
    (void)N;
    int i = bond_i;
    int D_l = sites[i].D_left;
    int D_m = sites[i].D_right;
    int D_r = sites[i+1].D_right;
    double E = 0.0;
    /* For each (s1, s2), (s1', s2'), reconstruct the local two-site
     * amplitudes on the whole chain and compute the bra-ket
     * inner product restricted to the two-site replacement. */
    for (int s1 = 0; s1 < PHYS_D; s1++)
    for (int s2 = 0; s2 < PHYS_D; s2++)
    for (int s1p = 0; s1p < PHYS_D; s1p++)
    for (int s2p = 0; s2p < PHYS_D; s2p++) {
        int row = s1p * PHYS_D + s2p;
        int col = s1  * PHYS_D + s2;
        double coeff = h[row * 4 + col];
        if (coeff == 0) continue;
        /* ⟨s1', s2'| ψ ... | s1, s2⟩ contraction: for an MPS with
         * left-env L and right-env R (both identity if we're exactly
         * normalised), this reduces to
         *   Σ_{l, r} conj(A[i][l, s1p, m]) conj(A[i+1][m, s2p, r])
         *           · A[i][l, s1, m'] A[i+1][m', s2, r]
         * Since our tensors are real, conjugation is identity. */
        double local = 0.0;
        for (int l = 0; l < D_l; l++) for (int r = 0; r < D_r; r++) {
            double ket = 0, bra = 0;
            for (int m = 0; m < D_m; m++) {
                ket += sites[i].A[(l*PHYS_D + s1)*D_m + m]
                     * sites[i+1].A[(m*PHYS_D + s2)*D_r + r];
                bra += sites[i].A[(l*PHYS_D + s1p)*D_m + m]
                     * sites[i+1].A[(m*PHYS_D + s2p)*D_r + r];
            }
            local += bra * ket;
        }
        E += coeff * local;
    }
    return E;
}

/* Contract the MPS into its full 2^N state vector (real). For N ≤ 20
 * this is always feasible and lets us compute ⟨H⟩ exactly without
 * relying on canonical form. */
static double *mps_to_dense(const tebd_site_t *sites, int N, long *out_dim) {
    int Dl = sites[0].D_left;
    int Dr = sites[0].D_right;
    long block_states = 1;
    long nblock = block_states * PHYS_D * Dr;
    double *block = malloc(sizeof(double) * nblock);
    for (int s = 0; s < PHYS_D; s++) for (int r = 0; r < Dr; r++)
        block[s * Dr + r] = sites[0].A[(0 * PHYS_D + s) * Dr + r];
    block_states = PHYS_D;
    int D_in = Dr;
    for (int i = 1; i < N; i++) {
        int D_out = sites[i].D_right;
        long new_states = block_states * PHYS_D;
        double *nb = calloc((size_t)new_states * D_out, sizeof(double));
        for (long st = 0; st < block_states; st++)
            for (int s = 0; s < PHYS_D; s++)
                for (int r = 0; r < D_out; r++) {
                    double v = 0;
                    for (int m = 0; m < D_in; m++)
                        v += block[st * D_in + m] * sites[i].A[(m*PHYS_D + s)*D_out + r];
                    nb[(st * PHYS_D + s) * D_out + r] = v;
                }
        free(block);
        block = nb;
        block_states = new_states;
        D_in = D_out;
    }
    double *psi = malloc(sizeof(double) * block_states);
    for (long st = 0; st < block_states; st++) psi[st] = block[st];
    free(block);
    (void)Dl;
    *out_dim = block_states;
    return psi;
}

/* Sum-of-bonds Hamiltonian expectation value. */
static double total_energy(const tebd_site_t *sites, int N,
                           const mps_config_t *cfg) {
    /* Dense exact energy: materialise ψ, build H sparsely, compute
     * <ψ|H|ψ>/<ψ|ψ>. Valid for N ≤ 20. Avoids any MPS-canonical-form
     * assumptions. */
    long dim;
    double *psi = mps_to_dense(sites, N, &dim);
    /* Compute ⟨ψ|H|ψ⟩ by enumerating connections.
     *   TFIM:        H = -J Σ σ^z σ^z - Γ Σ σ^x
     *   Heisenberg/XXZ: H = J/4 Σ (S+ S- + S- S+) · (2) + Jz/4 Σ σ^z σ^z
     * We iterate over every basis state s and evaluate (Hψ)[s]. */
    double num = 0.0, den = 0.0;
    double J = cfg->J;
    double Jz = (cfg->ham == MPS_HAM_XXZ) ? cfg->Jz : cfg->J;
    double G = cfg->Gamma;
    for (long s = 0; s < dim; s++) {
        double diag = 0;
        for (int i = 0; i + 1 < N; i++) {
            int si = ((s >> i) & 1) ? -1 : +1;
            int sj = ((s >> (i+1)) & 1) ? -1 : +1;
            if (cfg->ham == MPS_HAM_TFIM)        diag += -J * si * sj;
            else                                  diag += 0.25 * Jz * si * sj;
        }
        double Hpsi = diag * psi[s];
        if (cfg->ham == MPS_HAM_TFIM) {
            for (int i = 0; i < N; i++) {
                long s2 = s ^ (1L << i);
                Hpsi += -G * psi[s2];
            }
        } else {
            for (int i = 0; i + 1 < N; i++) {
                int si = ((s >> i) & 1) ? -1 : +1;
                int sj = ((s >> (i+1)) & 1) ? -1 : +1;
                if (si == -sj) {
                    long s2 = s ^ (1L << i) ^ (1L << (i+1));
                    Hpsi += 0.5 * J * psi[s2];
                }
            }
        }
        num += psi[s] * Hpsi;
        den += psi[s] * psi[s];
    }
    free(psi);
    return (den > 0) ? num / den : 0.0;
}

#if 0
/* (Legacy per-bond path kept for reference — no longer used.) */
static double total_energy_legacy(const tebd_site_t *sites, int N,
                                   const mps_config_t *cfg) {
    /* Build h for τ=0 (so exp(-h·0) = I, but we want h itself).
     * Re-use bond-gate eigendecomposition: h is encoded inside
     * build_bond_gate via the λ[] and V[] intermediate variables,
     * not directly exported. For correctness we rebuild the 4×4 h
     * matrix here. */
    double h[16]; memset(h, 0, sizeof(h));
    double J = cfg->J;
    double Jz = (cfg->ham == MPS_HAM_XXZ) ? cfg->Jz : cfg->J;
    if (cfg->ham == MPS_HAM_TFIM) {
        h[0 ] = -J; h[5 ] =  J; h[10] =  J; h[15] = -J;
        double g2 = -0.5 * cfg->Gamma;
        for (int i = 0; i < 4; i++) { int j = i ^ 2; if (j > i) { h[i*4+j] += g2; h[j*4+i] += g2; } }
        for (int i = 0; i < 4; i++) { int j = i ^ 1; if (j > i) { h[i*4+j] += g2; h[j*4+i] += g2; } }
    } else {
        h[0 ] =  0.25 * Jz; h[5 ] = -0.25 * Jz; h[10] = -0.25 * Jz; h[15] =  0.25 * Jz;
        h[1*4+2] = 0.5 * J; h[2*4+1] = 0.5 * J;
    }
    double E = 0.0;
    for (int i = 0; i + 1 < N; i++) E += bond_expectation(sites, N, i, h);
    double n2 = mps_norm2(sites, N);
    return n2 > 0 ? E / n2 : 0.0;
}
#endif

int mps_tebd_imaginary_run(const mps_config_t *cfg,
                            double tau, int num_sweeps,
                            double *out_energy_trace,
                            double *out_final_energy) {
    if (!cfg || cfg->num_sites < 2) return -1;
    int N = cfg->num_sites;
    int D_max = cfg->max_bond_dim > 0 ? cfg->max_bond_dim : 16;
    tebd_site_t *sites = calloc((size_t)N, sizeof(tebd_site_t));
    unsigned long long rng = 0xABCDEF123456ULL;
    init_mps(sites, N, D_max, &rng);
    mps_normalise(sites, N);

    double gate_full[16], gate_half[16];
    build_bond_gate(cfg, tau,        1, gate_full);
    build_bond_gate(cfg, tau * 0.5,  1, gate_half);

    /* Single-site σ^x gate for TFIM: e^{+Γ σ^x τ/2} in the {↑, ↓}
     * basis is [[cosh, sinh], [sinh, cosh]] with argument Γτ/2. */
    double site_gate_half[4] = {1, 0, 0, 1};
    int is_tfim = (cfg->ham == MPS_HAM_TFIM);
    if (is_tfim) {
        double ch = cosh(cfg->Gamma * tau * 0.5);
        double sh = sinh(cfg->Gamma * tau * 0.5);
        site_gate_half[0] = ch; site_gate_half[1] = sh;
        site_gate_half[2] = sh; site_gate_half[3] = ch;
    }

    for (int sw = 0; sw < num_sweeps; sw++) {
        /* 2nd-order split:
         *   TFIM:  U_x(τ/2) · U_zz(τ) · U_x(τ/2)
         *          with U_zz assembled from even/odd bonds and
         *          itself 2nd-order-Trotter split.
         *   XXZ/Heisenberg: just the bond gates, since the single-
         *   site piece is empty. */
        if (is_tfim) {
            for (int i = 0; i < N; i++) {
                apply_site_gate(&sites[i], site_gate_half);
            }
            mps_normalise(sites, N);
        }
        /* Even bonds at τ/2. */
        for (int i = 0; i + 1 < N; i += 2) {
            apply_bond_gate(sites, i, D_max, gate_half);
            mps_normalise(sites, N);
        }
        /* Odd bonds at τ. */
        for (int i = 1; i + 1 < N; i += 2) {
            apply_bond_gate(sites, i, D_max, gate_full);
            mps_normalise(sites, N);
        }
        /* Even bonds at τ/2 again. */
        for (int i = 0; i + 1 < N; i += 2) {
            apply_bond_gate(sites, i, D_max, gate_half);
            mps_normalise(sites, N);
        }
        if (is_tfim) {
            for (int i = 0; i < N; i++) {
                apply_site_gate(&sites[i], site_gate_half);
            }
            mps_normalise(sites, N);
        }
        if (out_energy_trace) out_energy_trace[sw] = total_energy(sites, N, cfg);
    }
    double E_final = total_energy(sites, N, cfg);
    if (out_final_energy) *out_final_energy = E_final;

    for (int i = 0; i < N; i++) free(sites[i].A);
    free(sites);
    return 0;
}

int mps_tebd_imaginary_step(const mps_config_t *cfg,
                             double tau, int num_sites,
                             double *out_energy) {
    mps_config_t c = *cfg;
    c.num_sites = num_sites;
    return mps_tebd_imaginary_run(&c, tau, 1, NULL, out_energy);
}

/* ======================= real-time evolution ======================== */

/* Build the product state |ψ⟩ = ⊗_i |s_i⟩ where each single-site
 * ket is parameterised by its Bloch vector (bx, by, bz):
 *   |s⟩ = cos(θ/2)|↑⟩ + e^{iφ} sin(θ/2)|↓⟩,  (θ, φ) = (acos(bz), atan2(by, bx))
 * Returns allocated complex vector of length 2^N. */
static double _Complex *build_product_state(const double *xyz, int N) {
    long dim = 1L << N;
    double _Complex *psi = calloc((size_t)dim, sizeof(double _Complex));
    psi[0] = 1.0 + 0.0 * _Complex_I;
    /* Build up site by site. For site i with Bloch (bx, by, bz): the
     * 2×2 amplitudes are (cos(θ/2), e^{iφ} sin(θ/2)) for (↑, ↓). */
    long cur_dim = 1;
    for (int i = 0; i < N; i++) {
        double bx = xyz[3*i], by = xyz[3*i+1], bz = xyz[3*i+2];
        double theta = acos(bz > 1.0 ? 1.0 : (bz < -1.0 ? -1.0 : bz));
        double phi   = atan2(by, bx);
        double _Complex amp_up   = cos(0.5 * theta) + 0.0 * _Complex_I;
        double _Complex amp_down = sin(0.5 * theta) * (cos(phi) + sin(phi) * _Complex_I);
        /* New psi: amp_{new_state}_{bit i=0} = amp_up   · psi_old[state]
         *          amp_{new_state}_{bit i=1} = amp_down · psi_old[state]  */
        long new_dim = cur_dim * 2;
        double _Complex *new_psi = calloc((size_t)new_dim, sizeof(double _Complex));
        for (long st = 0; st < cur_dim; st++) {
            new_psi[st]               = amp_up   * psi[st];
            new_psi[st | (1L << i)]   = amp_down * psi[st];
        }
        free(psi);
        psi = new_psi;
        cur_dim = new_dim;
    }
    return psi;
}

/* Compute H · ψ for the cfg-selected Hamiltonian (complex ψ). */
static void H_matvec_complex(const mps_config_t *cfg, int N,
                              const double _Complex *in, double _Complex *out) {
    long dim = 1L << N;
    double J = cfg->J;
    double Jz = (cfg->ham == MPS_HAM_XXZ) ? cfg->Jz : cfg->J;
    double G = cfg->Gamma;
    for (long s = 0; s < dim; s++) {
        double diag = 0;
        for (int i = 0; i + 1 < N; i++) {
            int si = ((s >> i) & 1) ? -1 : +1;
            int sj = ((s >> (i+1)) & 1) ? -1 : +1;
            if (cfg->ham == MPS_HAM_TFIM) diag += -J * si * sj;
            else                           diag += 0.25 * Jz * si * sj;
        }
        double _Complex y = diag * in[s];
        if (cfg->ham == MPS_HAM_TFIM) {
            for (int i = 0; i < N; i++) {
                long s2 = s ^ (1L << i);
                y += -G * in[s2];
            }
        } else {
            for (int i = 0; i + 1 < N; i++) {
                int si = ((s >> i) & 1) ? -1 : +1;
                int sj = ((s >> (i+1)) & 1) ? -1 : +1;
                if (si == -sj) {
                    long s2 = s ^ (1L << i) ^ (1L << (i+1));
                    y += 0.5 * J * in[s2];
                }
            }
        }
        out[s] = y;
    }
}

/* One 4th-order Runge-Kutta step of dψ/dt = -i H ψ. */
static void rk4_step(const mps_config_t *cfg, int N, double dt,
                      double _Complex *psi, double _Complex *k1,
                      double _Complex *k2, double _Complex *k3,
                      double _Complex *k4, double _Complex *tmp) {
    long dim = 1L << N;
    double _Complex factor = -_Complex_I;
    H_matvec_complex(cfg, N, psi, k1);
    for (long i = 0; i < dim; i++) k1[i] *= factor;
    for (long i = 0; i < dim; i++) tmp[i] = psi[i] + 0.5 * dt * k1[i];
    H_matvec_complex(cfg, N, tmp, k2);
    for (long i = 0; i < dim; i++) k2[i] *= factor;
    for (long i = 0; i < dim; i++) tmp[i] = psi[i] + 0.5 * dt * k2[i];
    H_matvec_complex(cfg, N, tmp, k3);
    for (long i = 0; i < dim; i++) k3[i] *= factor;
    for (long i = 0; i < dim; i++) tmp[i] = psi[i] + dt * k3[i];
    H_matvec_complex(cfg, N, tmp, k4);
    for (long i = 0; i < dim; i++) k4[i] *= factor;
    for (long i = 0; i < dim; i++) {
        psi[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

int mps_tebd_real_time_run(const mps_config_t *cfg,
                            const double *initial_sites_xyz,
                            double dt, int num_steps,
                            double *out_mz_trace,
                            double *out_loschmidt) {
    if (!cfg || !initial_sites_xyz || num_steps <= 0) return -1;
    int N = cfg->num_sites;
    if (N <= 0 || N > 20) return -1;
    long dim = 1L << N;
    double _Complex *psi  = build_product_state(initial_sites_xyz, N);
    double _Complex *psi0 = malloc((size_t)dim * sizeof(double _Complex));
    memcpy(psi0, psi, (size_t)dim * sizeof(double _Complex));
    double _Complex *k1  = malloc((size_t)dim * sizeof(double _Complex));
    double _Complex *k2  = malloc((size_t)dim * sizeof(double _Complex));
    double _Complex *k3  = malloc((size_t)dim * sizeof(double _Complex));
    double _Complex *k4  = malloc((size_t)dim * sizeof(double _Complex));
    double _Complex *tmp = malloc((size_t)dim * sizeof(double _Complex));

    for (int step = 0; step < num_steps; step++) {
        if (step > 0) rk4_step(cfg, N, dt, psi, k1, k2, k3, k4, tmp);
        if (out_mz_trace) {
            for (int i = 0; i < N; i++) {
                double mz = 0;
                for (long s = 0; s < dim; s++) {
                    double p2 = creal(psi[s]) * creal(psi[s]) + cimag(psi[s]) * cimag(psi[s]);
                    mz += (((s >> i) & 1) ? -1.0 : 1.0) * p2;
                }
                out_mz_trace[step * N + i] = mz;
            }
        }
        if (out_loschmidt) {
            double _Complex ovlp = 0;
            for (long s = 0; s < dim; s++) ovlp += conj(psi0[s]) * psi[s];
            out_loschmidt[step] = creal(ovlp * conj(ovlp));
        }
    }
    free(psi); free(psi0); free(k1); free(k2); free(k3); free(k4); free(tmp);
    return 0;
}
