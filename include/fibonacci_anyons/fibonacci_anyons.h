/*
 * include/fibonacci_anyons/fibonacci_anyons.h
 *
 * Fibonacci-anyon universal topological quantum computation (v0.5
 * pillar P1.3b). Complements the Majorana module — Majorana anyons
 * give non-Abelian statistics but are not computationally universal;
 * Fibonacci anyons are universal via braiding alone [Freedman, Kitaev,
 * Larsen, Wang, "Topological quantum computation," 2003].
 *
 * The anyon theory has two particle types: the vacuum 1 and the
 * non-trivial Fibonacci anyon τ. Fusion rules:
 *
 *     1 × 1 = 1
 *     1 × τ = τ × 1 = τ
 *     τ × τ = 1 + τ     (the crucial non-abelian rule)
 *
 * The quantum dimension of τ is the golden ratio φ = (1 + √5) / 2.
 * An n-anyon system with fixed total charge has a Hilbert space of
 * dimension equal to Fibonacci(n).
 *
 * Operations supported in v0.4-local:
 *   - Fusion-tree basis enumeration for n τ anyons.
 *   - F-matrix F^τττ_τ  (the single non-trivial 2×2 F-symbol).
 *   - R-matrices R^1_ττ and R^τ_ττ (Fibonacci braiding phases).
 *   - Elementary braid operators B_i as unitaries in the fusion basis.
 *   - Solovay–Kitaev compilation of an arbitrary single-qubit unitary
 *     U ∈ SU(2) into a braid word on a 4-anyon qubit.
 */
#ifndef FIBONACCI_ANYONS_H
#define FIBONACCI_ANYONS_H

#include <complex.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Golden ratio φ = (1+√5)/2.  Use this constant rather than recomputing. */
extern const double FIBO_PHI;

/* Anyon particle types. */
typedef enum {
    FIBO_ONE = 0,  /* vacuum */
    FIBO_TAU = 1   /* Fibonacci anyon */
} fibo_type_t;

/* -------- F-matrix ---------------------------------------------------
 * The only non-trivial F-symbol is F^τττ_τ, a 2×2 unitary acting on
 *     { |(τ τ)_1 τ ; τ⟩ , |(τ τ)_τ τ ; τ⟩ }.
 *                                                                   */
typedef struct {
    double _Complex m[2][2];
} fibo_fmatrix_t;

/* Returns the 2×2 F-matrix F^τττ_τ. All other F-symbols are either
 * trivial (1×1 identity) or zero. */
fibo_fmatrix_t fibo_f_matrix(void);

/* -------- R-matrix ---------------------------------------------------
 * R^c_ab is the phase acquired when two anyons a, b with total fusion
 * charge c are exchanged. For Fibonacci:
 *     R^1_ττ = e^{-4πi/5}
 *     R^τ_ττ = e^{+3πi/5}
 */
double _Complex fibo_r_one(void);   /* R^1_ττ */
double _Complex fibo_r_tau(void);   /* R^τ_ττ */

/* -------- Fibonacci numbers ------------------------------------------ */

/* Fibonacci(n) with Fibonacci(1) = 1, Fibonacci(2) = 2, Fibonacci(3) = 3, ... */
long fibo_number(int n);

/* Hilbert-space dimension for n τ anyons with total charge τ. */
long fibo_hilbert_dim(int num_anyons);

/* -------- Single-qubit encoding on 4 τ anyons ------------------------
 * The standard encoding: 4 Fibonacci τ anyons with total fusion charge
 * τ give a 2-dimensional Hilbert space spanned by
 *     |0⟩ = |((ττ)_1 τ)_τ τ⟩
 *     |1⟩ = |((ττ)_τ τ)_τ τ⟩
 * Single-qubit gates are built from two generators:
 *     σ₁ = R-move on the first pair of anyons
 *     σ₂ = F (R-move on adjacent-basis) R (F^{-1})
 */

/* Return the 2×2 braid matrix B_1 (exchange of anyons 1, 2). */
void fibo_braid_b1(double _Complex out[2][2]);

/* Return the 2×2 braid matrix B_2 (exchange of anyons 2, 3), constructed
 * via the F-gauge-change identity B_2 = F · diag(R^1, R^τ) · F^{-1}. */
void fibo_braid_b2(double _Complex out[2][2]);

/* -------- Solovay–Kitaev compilation --------------------------------- */

typedef struct {
    int   *sigmas;   /* each element is 1 or 2, meaning σ_1 or σ_2 */
    int    length;
    int    capacity;
} fibo_braid_word_t;

fibo_braid_word_t *fibo_braid_word_create(int initial_capacity);
void               fibo_braid_word_free  (fibo_braid_word_t *w);
int                fibo_braid_word_push  (fibo_braid_word_t *w, int sigma);

/* Evaluate a braid word into a 2×2 unitary. */
void fibo_braid_word_eval(const fibo_braid_word_t *w, double _Complex out[2][2]);

/* Compile a target single-qubit unitary U ∈ SU(2) into a braid word
 * approximation. Uses a greedy Solovay–Kitaev-style search over words
 * of bounded length. Returns operator-norm distance between the
 * compiled and target unitaries in *out_err (NULL to disable).
 *
 * depth: maximum braid-word length (larger = more accurate, slower).
 * The search is O(3^depth · 4) — keep depth ≤ 20 for reasonable time. */
fibo_braid_word_t *fibo_compile_unitary(const double _Complex target[2][2],
                                         int depth,
                                         double *out_err);

/* Operator-norm distance between two 2×2 matrices (largest singular
 * value of the difference). */
double fibo_operator_norm_distance(const double _Complex a[2][2],
                                    const double _Complex b[2][2]);

/* Multiply 2×2 complex matrices: out = a · b. */
void fibo_mat2_mul(const double _Complex a[2][2],
                   const double _Complex b[2][2],
                   double _Complex out[2][2]);

/* Identity 2×2 matrix. */
void fibo_mat2_identity(double _Complex out[2][2]);

/* Conjugate-transpose (Hermitian adjoint) of a 2×2 matrix. */
void fibo_mat2_dagger(const double _Complex a[2][2],
                       double _Complex out[2][2]);

/* -------- Proper Solovay–Kitaev recursion ---------------------------- */

/*
 * Recursive Solovay–Kitaev algorithm: any single-qubit unitary can be
 * approximated by a product of elementary gates with error scaling as
 * ε ~ c · (ε_0)^{log 5 / log 3} where ε_0 is the base-level accuracy
 * and the depth of recursion is L (so ε shrinks super-polynomially).
 *
 * fibo_compile_unitary_sk(target, recursion_depth, base_search_depth,
 *                          out_err)
 *
 *   recursion_depth    — number of SK recursions; typical value 2–3
 *   base_search_depth  — depth of the DFS used at the base case
 *
 * Each recursion step writes U = V W V† W† · U_base where U_base is
 * the depth-0 approximation and (V, W) are the balanced group
 * commutator that fixes U·U_base^{-1}. This produces much shorter
 * braid words than a flat DFS at comparable accuracy.
 *
 * Returns a freshly-allocated braid word (caller frees). */
fibo_braid_word_t *fibo_compile_unitary_sk(const double _Complex target[2][2],
                                             int recursion_depth,
                                             int base_search_depth,
                                             double *out_err);

#ifdef __cplusplus
}
#endif

#endif /* FIBONACCI_ANYONS_H */
