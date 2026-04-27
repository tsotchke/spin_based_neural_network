/*
 * include/nqs/nqs_symproj.h
 *
 * Generic finite-group symmetry-projection wrapper for NQS ansätze.
 *
 * Given a base log-amplitude callback ψ_base(s), a permutation table
 * for a finite group G acting on the lattice sites, and a 1-D character
 * χ : G → ℝ (typically ±1 for trivial / sign irreps), the symmetrised
 * wavefunction is
 *
 *     ψ_sym(s) = (1/√|G|) · Σ_g χ(g) · ψ_base(g·s).
 *
 * Each group element g is represented as a site-permutation π_g : N → N
 * such that (g·s)_i = s_{π_g(i)}.  The wrapper evaluates ψ_base on every
 * orbit member, log-sum-exps the contributions, and returns
 *
 *     log|ψ_sym(s)|, arg(ψ_sym(s))
 *
 * with the same numerical-stability discipline as nqs_translation.
 *
 * Cost per log_amp evaluation: |G| base-ansatz calls.
 *
 * This wrapper supersedes nqs_translation for kagome and any other
 * lattice with non-trivial point-group symmetry.  For pure translation
 * on a square lattice, nqs_translation remains marginally faster (less
 * indexing overhead).  Both can be stacked as base_log_amp callbacks.
 */
#ifndef NQS_SYMPROJ_H
#define NQS_SYMPROJ_H

#include "nqs/nqs_sampler.h"
#include "nqs/nqs_ansatz.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    nqs_log_amp_fn_t base_log_amp;
    void            *base_user;
    int              num_sites;          /* lattice size N */
    int              num_group_elements; /* |G| */
    const int       *perm;               /* [|G| * N], row-major:
                                          *   perm[g * N + i] = π_g(i) */
    const double    *characters;         /* [|G|], χ(g); typically ±1 */
} nqs_symproj_wrapper_t;

/* log_amp callback for the generic-symmetry-projected wavefunction. */
void nqs_symproj_log_amp(const int *spins, int num_sites,
                         void *user,
                         double *out_log_abs, double *out_arg);

/* Gradient callback. ∂ log ψ_sym / ∂θ = Σ_g w_g · ∂ log ψ_base(g·s) / ∂θ
 * with w_g = χ(g) · ψ_base(g·s) / Σ_g' χ(g') · ψ_base(g'·s). */
int nqs_symproj_gradient(void *grad_user,
                         nqs_ansatz_t *ansatz,
                         const int *spins, int num_sites,
                         double *out_grad);

/* Holomorphic gradient through the projection wrapper.  Returns
 *   ∂ log ψ_sym / ∂ θ_k = Σ_g w_g · ∂ log ψ_base(g·s) / ∂ θ_k     (complex)
 * with w_g = χ(g) · ψ_base(g·s) / Σ_g' χ(g') · ψ_base(g'·s)        (complex).
 *
 * For a real-valued character χ ∈ {-1, +1} (the kagome p6m 1D irreps)
 * and a complex-amplitude base ansatz, both numerator and denominator
 * carry phases and the weights are themselves complex.  This is the
 * gradient required by `nqs_sr_step_holomorphic_full` to train a
 * symmetry-projected complex ansatz against a non-stoquastic target. */
int nqs_symproj_gradient_complex(void *grad_user,
                                  nqs_ansatz_t *ansatz,
                                  const int *spins, int num_sites,
                                  double *out_grad_re,
                                  double *out_grad_im);

/* ------------------------------------------------------------------ */
/* Permutation builders for standard kagome subgroups.                */
/*                                                                    */
/* Site convention (must match nqs_gradient.c kagome kernels):        */
/*                                                                    */
/*   site index = 3 * (cx * Ly + cy) + sub,  sub ∈ {A=0, B=1, C=2}    */
/*                                                                    */
/* Sublattice positions inside each unit cell:                        */
/*   A at (0, 0), B at a₁/2, C at a₂/2                                */
/* with primitive vectors a₁ = (1, 0), a₂ = (½, √3/2).                */
/* ------------------------------------------------------------------ */

/* Build the |G| = Lx·Ly translation permutation table for a kagome
 * lattice of Lx × Ly unit cells with PBC.
 *
 * Allocates *out_perm (caller frees). Populates *out_num_elements.
 * Returns 0 on success, -1 on bad arguments / allocation failure. */
int nqs_kagome_translation_perm(int Lx, int Ly,
                                int **out_perm,
                                int *out_num_elements);

/* Build the |G| = 2·Lx·Ly  p2 = (translations) ⋊ (C₂ inversion)
 * permutation table for kagome with PBC.  The C₂ acts as
 * (cx, cy, sub) → (-cx + δx(sub), -cy + δy(sub), sub) mod (Lx, Ly)
 * with δ_A = (0,0), δ_B = (-1, 0), δ_C = (0, -1).  Sublattices stay
 * fixed under inversion through the unit-cell origin (sublattice A
 * site).  Combined with translations, the full p2 orbit has order
 * 2·Lx·Ly.
 *
 * Allocates *out_perm of length 2·Lx·Ly·N (N = 3·Lx·Ly), and
 * *out_characters of length 2·Lx·Ly (all +1 for the A₁ trivial
 * irrep).  Caller frees both. */
int nqs_kagome_p2_perm(int Lx, int Ly,
                        int **out_perm,
                        double **out_characters,
                        int *out_num_elements);

/*
 * Build the |G| = 3·L²  p3 = (translations) ⋊ (C₃ around up-triangle
 * centroid) permutation table for kagome on an L × L torus with PBC.
 *
 * C₃ rotation: 120° around the centroid of up-triangle (0, 0), located
 * at (1/4, √3/12) in Cartesian coordinates.  The rotation cycles
 * sublattices A → B → C → A and shifts cell coordinates by an amount
 * that depends on the original site (computed numerically; the cell
 * shift per site is non-trivial because the rotation centre is offset
 * from the lattice origin).
 *
 * Requires Lx = Ly = L; rectangular tori cannot host C₃.  Returns -1
 * with an explanatory stderr message on rectangular input.
 *
 * Allocates *out_perm of length 3·L²·N (N = 3·L²) and *out_characters
 * of length 3·L² (all +1 for the A₁ trivial irrep).  Caller frees both.
 */
int nqs_kagome_p3_perm(int L,
                        int **out_perm,
                        double **out_characters,
                        int *out_num_elements);

/*
 * Build the |G| = 6·L²  p6 = (translations) ⋊ (C₆ around hexagon
 * centroid) permutation table for kagome on an L × L torus with PBC.
 *
 * C₆ rotation: 60° around the hexagon centroid at (a₁ + a₂)/2 in
 * Cartesian = (3/4, √3/4) in our (1, 0), (1/2, √3/2) basis.  This
 * point is the unique 6-fold rotation centre per unit cell (Wyckoff
 * 1a in the standard p6m setting, shifted by −(a₁ + a₂)/2 from our
 * convention's unit-cell origin).  Verified by tools/find_kagome_p6_centre.
 *
 * Sublattices cycle in a non-trivial pattern under R(60°) — the cell
 * shift depends on both the original cell and sublattice — so the
 * permutation is built numerically (Cartesian rotate + lattice-basis
 * inverse + PBC reduction) following the same approach as
 * nqs_kagome_p3_perm.
 *
 * Requires Lx = Ly = L; rectangular tori cannot host C₆.  Returns -1
 * with an explanatory stderr message on rectangular input.
 *
 * Allocates *out_perm of length 6·L²·N (N = 3·L²) and *out_characters
 * of length 6·L² (all +1 for the A₁ trivial irrep).  Caller frees both.
 */
int nqs_kagome_p6_perm(int L,
                        int **out_perm,
                        double **out_characters,
                        int *out_num_elements);

/*
 * Build the |G| = 12·L²  p6m = (translations) ⋊ (C₆ × {1, M})
 * permutation table for kagome on an L × L torus with PBC.
 *
 * The 12 point operations are:
 *   • 6 rotations  R(60° · k)    for k = 0..5
 *   • 6 mirror-rotations  M ∘ R(60° · k)
 *
 * where M is the mirror through the line y = √3/4 (horizontal line
 * passing through the hexagon centroid).  This mirror swaps
 * sublattices A ↔ B and fixes sublattice C; combined with C₆ it
 * generates the full point group of the kagome lattice (D_6h in 3-D /
 * 6mm in 2-D), hence the wallpaper group p6m.
 *
 * |G| = 12·L².  All characters are +1 (A₁ trivial irrep).  Caller
 * frees both arrays.
 */
int nqs_kagome_p6m_perm(int L,
                         int **out_perm,
                         double **out_characters,
                         int *out_num_elements);

/* Named 1D irrep of C_6v at Γ for the kagome p6m wallpaper group.
 * Mirrors `nqs_kspace_irrep_t` from nqs_kspace_ed.h but kept here for
 * dependency hygiene (nqs_symproj must build with or without libirrep). */
typedef enum {
    NQS_SYMPROJ_KAGOME_GAMMA_A1 = 0, /* trivial — what _p6m_perm builds  */
    NQS_SYMPROJ_KAGOME_GAMMA_A2 = 1, /* sign on all 6 mirrors            */
    NQS_SYMPROJ_KAGOME_GAMMA_B1 = 2, /* sign on C_6, σ_d                 */
    NQS_SYMPROJ_KAGOME_GAMMA_B2 = 3  /* sign on C_6, σ_v                 */
} nqs_symproj_kagome_irrep_t;

/* Same |G| = 12·L² p6m permutation as nqs_kagome_p6m_perm, but the
 * out_characters vector is the C_6v character at Γ for the named
 * 1D irrep, tiled across translations (which all carry +1 at k=0).
 *
 * Group-element ordering matches build_p6m_perm_row in
 * src/nqs/nqs_symproj.c:
 *   ops 0..5  = pure rotation R(60° · k)  for k = 0..5
 *   ops 6..11 = mirror M ∘ R(60° · k)     for k = 0..5
 *
 * The mirror-after-rotation ordering above produces alternating
 * σ_v / σ_d axes (σ_v through A-vertex bonds at op 6, then 30°-
 * offset σ_d at op 7, etc.).
 *
 * Characters per element (tiled across L² translations):
 *
 *   irrep | E  C6  C3  C2  C3²  C6⁵  σ_v  σ_d  σ_v  σ_d  σ_v  σ_d
 *   A_1   |  1   1   1   1   1    1    1    1    1    1    1    1
 *   A_2   |  1   1   1   1   1    1   -1   -1   -1   -1   -1   -1
 *   B_1   |  1  -1   1  -1   1   -1    1   -1    1   -1    1   -1
 *   B_2   |  1  -1   1  -1   1   -1   -1    1   -1    1   -1    1
 *
 * For irrep_name = NQS_SYMPROJ_KAGOME_GAMMA_A1 this is identical to
 * nqs_kagome_p6m_perm.  Caller frees both arrays. */
int nqs_kagome_p6m_perm_irrep(int L,
                               nqs_symproj_kagome_irrep_t irrep_name,
                               int **out_perm,
                               double **out_characters,
                               int *out_num_elements);

#ifdef __cplusplus
}
#endif

#endif /* NQS_SYMPROJ_H */
