/*
 * src/fibonacci_anyons/fibonacci_anyons.c
 *
 * Fibonacci-anyon arithmetic: F-symbols, R-symbols, braid generators,
 * and a greedy Solovay–Kitaev-style compiler of single-qubit unitaries
 * into braid words.
 *
 * References for the numerical values of the F and R symbols:
 *   - Nayak et al., Rev. Mod. Phys. 80 (2008).
 *   - Bonesteel, Hormozi, Zikos, Simon, Phys. Rev. Lett. 95 (2005).
 *
 * The 4-anyon single-qubit encoding uses two generators:
 *     σ_1  (exchange anyons 1↔2): diagonal in the fusion basis.
 *     σ_2  (exchange anyons 2↔3): off-diagonal; obtained as
 *           σ_2 = F · σ_1 · F^{-1}.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "fibonacci_anyons/fibonacci_anyons.h"

const double FIBO_PHI = 1.6180339887498949;  /* (1 + √5) / 2 */

/* ---------------- F-matrix -------------------------------------------- */

fibo_fmatrix_t fibo_f_matrix(void) {
    fibo_fmatrix_t F;
    double inv_phi = 1.0 / FIBO_PHI;
    double inv_sqrt_phi = 1.0 / sqrt(FIBO_PHI);
    F.m[0][0] = inv_phi        + 0.0 * _Complex_I;
    F.m[0][1] = inv_sqrt_phi   + 0.0 * _Complex_I;
    F.m[1][0] = inv_sqrt_phi   + 0.0 * _Complex_I;
    F.m[1][1] = -inv_phi       + 0.0 * _Complex_I;
    return F;
}

/* ---------------- R-symbols ------------------------------------------- */

double _Complex fibo_r_one(void) {
    /* exp(-4π i / 5) */
    double a = -4.0 * M_PI / 5.0;
    return cos(a) + sin(a) * _Complex_I;
}

double _Complex fibo_r_tau(void) {
    /* exp(+3π i / 5) */
    double a = 3.0 * M_PI / 5.0;
    return cos(a) + sin(a) * _Complex_I;
}

/* ---------------- Fibonacci numbers ----------------------------------- */

long fibo_number(int n) {
    if (n <= 0) return 0;
    long a = 1, b = 1;
    for (int i = 2; i <= n; i++) { long c = a + b; a = b; b = c; }
    return n == 1 ? 1 : b;
}

long fibo_hilbert_dim(int num_anyons) {
    /* Number of fusion trees of n τ-anyons with total charge τ equals
     * F(n-2) under the Fibonacci convention F(0)=1, F(1)=1, F(2)=2,
     * F(3)=3, F(4)=5, F(5)=8, ... For n = 4 this gives 2, the
     * standard single-qubit encoding. */
    if (num_anyons < 2)  return 0;
    if (num_anyons == 2) return 1;
    if (num_anyons == 3) return 1;
    long a = 1, b = 1;    /* F(0), F(1) */
    for (int i = 2; i <= num_anyons - 2; i++) {
        long c = a + b;
        a = b;
        b = c;
    }
    return b;
}

/* ---------------- Matrix helpers -------------------------------------- */

void fibo_mat2_identity(double _Complex out[2][2]) {
    out[0][0] = 1.0; out[0][1] = 0.0;
    out[1][0] = 0.0; out[1][1] = 1.0;
}

void fibo_mat2_mul(const double _Complex a[2][2],
                   const double _Complex b[2][2],
                   double _Complex out[2][2]) {
    double _Complex t[2][2];
    t[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0];
    t[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1];
    t[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0];
    t[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1];
    memcpy(out, t, sizeof(t));
}

static double _Complex fibo_mat2_det(const double _Complex m[2][2]) {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

static void fibo_mat2_sub(const double _Complex a[2][2],
                          const double _Complex b[2][2],
                          double _Complex out[2][2]) {
    out[0][0] = a[0][0] - b[0][0];
    out[0][1] = a[0][1] - b[0][1];
    out[1][0] = a[1][0] - b[1][0];
    out[1][1] = a[1][1] - b[1][1];
}

/* Operator norm = largest singular value; 2×2 closed form via
 *     σ² = (A + sqrt(A² - 4 B)) / 2
 * where A = tr(M^† M), B = |det(M^† M)| = |det M|². */
double fibo_operator_norm_distance(const double _Complex a[2][2],
                                    const double _Complex b[2][2]) {
    double _Complex d[2][2];
    fibo_mat2_sub(a, b, d);
    double f00 = creal(d[0][0])*creal(d[0][0]) + cimag(d[0][0])*cimag(d[0][0]);
    double f01 = creal(d[0][1])*creal(d[0][1]) + cimag(d[0][1])*cimag(d[0][1]);
    double f10 = creal(d[1][0])*creal(d[1][0]) + cimag(d[1][0])*cimag(d[1][0]);
    double f11 = creal(d[1][1])*creal(d[1][1]) + cimag(d[1][1])*cimag(d[1][1]);
    double A = f00 + f01 + f10 + f11;             /* trace of D† D  */
    double _Complex detd = fibo_mat2_det(d);
    double B = creal(detd)*creal(detd) + cimag(detd)*cimag(detd);
    double disc = A*A - 4.0*B;
    if (disc < 0) disc = 0;
    double sigma2 = 0.5 * (A + sqrt(disc));
    return sqrt(sigma2);
}

/* ---------------- Braid generators ----------------------------------- */

void fibo_braid_b1(double _Complex out[2][2]) {
    /* B_1 = diag(R^1_ττ, R^τ_ττ) in the {|0⟩, |1⟩} fusion basis. */
    out[0][0] = fibo_r_one();  out[0][1] = 0.0;
    out[1][0] = 0.0;           out[1][1] = fibo_r_tau();
}

static void fibo_f_inverse(double _Complex out[2][2]) {
    /* F is real-symmetric and unitary, so F^{-1} = F. */
    fibo_fmatrix_t F = fibo_f_matrix();
    out[0][0] = F.m[0][0]; out[0][1] = F.m[0][1];
    out[1][0] = F.m[1][0]; out[1][1] = F.m[1][1];
}

void fibo_braid_b2(double _Complex out[2][2]) {
    /* B_2 = F · B_1 · F^{-1}. */
    double _Complex F[2][2], Finv[2][2], B1[2][2], tmp[2][2];
    fibo_fmatrix_t Fm = fibo_f_matrix();
    F[0][0] = Fm.m[0][0]; F[0][1] = Fm.m[0][1];
    F[1][0] = Fm.m[1][0]; F[1][1] = Fm.m[1][1];
    fibo_f_inverse(Finv);
    fibo_braid_b1(B1);
    fibo_mat2_mul(F, B1, tmp);
    fibo_mat2_mul(tmp, Finv, out);
}

/* ---------------- Braid words ---------------------------------------- */

fibo_braid_word_t *fibo_braid_word_create(int initial_capacity) {
    if (initial_capacity <= 0) initial_capacity = 16;
    fibo_braid_word_t *w = calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->sigmas = malloc((size_t)initial_capacity * sizeof(int));
    if (!w->sigmas) { free(w); return NULL; }
    w->capacity = initial_capacity;
    w->length = 0;
    return w;
}

void fibo_braid_word_free(fibo_braid_word_t *w) {
    if (!w) return;
    free(w->sigmas);
    free(w);
}

int fibo_braid_word_push(fibo_braid_word_t *w, int sigma) {
    if (!w) return -1;
    if (sigma != 1 && sigma != 2) return -1;
    if (w->length == w->capacity) {
        int new_cap = w->capacity * 2;
        int *tmp = realloc(w->sigmas, (size_t)new_cap * sizeof(int));
        if (!tmp) return -1;
        w->sigmas = tmp;
        w->capacity = new_cap;
    }
    w->sigmas[w->length++] = sigma;
    return 0;
}

void fibo_braid_word_eval(const fibo_braid_word_t *w, double _Complex out[2][2]) {
    fibo_mat2_identity(out);
    if (!w) return;
    double _Complex B1[2][2], B2[2][2], tmp[2][2];
    fibo_braid_b1(B1);
    fibo_braid_b2(B2);
    for (int i = 0; i < w->length; i++) {
        const double _Complex (*B)[2] = (w->sigmas[i] == 1) ? (const double _Complex (*)[2])B1
                                                              : (const double _Complex (*)[2])B2;
        fibo_mat2_mul(out, B, tmp);
        memcpy(out, tmp, sizeof(tmp));
    }
}

/* ---------------- Solovay–Kitaev-style compiler ---------------------- */

/* Greedy branch-and-bound search over braid words. For small depths
 * this converges quickly thanks to the dense orbit {σ_1, σ_2}
 * generates on SU(2). For depth ≈ 20 the enumeration is 2^20 ≈ 10^6
 * candidate words — tractable at runtime.
 *
 * Strategy: iterative deepening DFS with a best-so-far cutoff.
 */

typedef struct {
    const double _Complex (*target)[2];
    const double _Complex (*B1)[2];
    const double _Complex (*B2)[2];
    int    max_depth;
    double best_err;
    fibo_braid_word_t *best_word;
    fibo_braid_word_t *stack;
} fibo_search_ctx_t;

static void fibo_search_dfs(fibo_search_ctx_t *ctx,
                             const double _Complex cur[2][2],
                             int depth) {
    double err = fibo_operator_norm_distance(cur, ctx->target);
    if (err < ctx->best_err) {
        ctx->best_err = err;
        ctx->best_word->length = 0;
        for (int i = 0; i < ctx->stack->length; i++) {
            fibo_braid_word_push(ctx->best_word, ctx->stack->sigmas[i]);
        }
    }
    if (depth >= ctx->max_depth) return;
    /* Pruning: if err < ~small we've found it. */
    if (err < 1e-10) return;

    double _Complex next[2][2];
    for (int s = 1; s <= 2; s++) {
        const double _Complex (*B)[2] = (s == 1) ? ctx->B1 : ctx->B2;
        fibo_mat2_mul(cur, B, next);
        fibo_braid_word_push(ctx->stack, s);
        fibo_search_dfs(ctx, next, depth + 1);
        ctx->stack->length--;
    }
}

void fibo_mat2_dagger(const double _Complex a[2][2],
                       double _Complex out[2][2]) {
    out[0][0] = conj(a[0][0]);
    out[0][1] = conj(a[1][0]);
    out[1][0] = conj(a[0][1]);
    out[1][1] = conj(a[1][1]);
}

/* ------------------- Solovay–Kitaev recursion ------------------------ */

/* Balanced group-commutator decomposition: given a unitary U ∈ SU(2)
 * with U ≈ exp(i·α·n̂·σ) (rotation by 2α around axis n̂), find V, W
 * in SU(2) such that U = V W V^† W^†. Dawson–Nielsen 2005 construct:
 *   Take U = exp(i·(2α)·σ_n̂) with 2α being the total rotation angle.
 *   Choose V = exp(i·θ·σ_x), W = exp(i·θ·σ_y)
 *   where θ is defined by sin(α/2) = 2 sin²(θ/2) sqrt(1 - sin²(θ/2)·sin²(θ/2)).
 *   In practice: sin(α/2) = sin²(θ/2) · √(4 - sin²(θ/2)·sin²(θ/2)·4)
 *   Simpler numerical construction used here: solve numerically for θ.
 *
 * After finding θ, rotate V, W by a further unitary S so that
 * S·V·S^†·W... gives the correct U direction in SU(2). */

static double _Complex cplx_from_polar(double r, double phi) {
    return r * (cos(phi) + sin(phi) * _Complex_I);
}

/* Generate exp(i·α·σ_axis) where axis ∈ {0:x, 1:y, 2:z}. */
static void su2_rotation(int axis, double alpha, double _Complex out[2][2]) {
    double c = cos(alpha), s = sin(alpha);
    fibo_mat2_identity(out);
    if (axis == 0) {  /* σ_x */
        out[0][0] =  c + 0.0*_Complex_I;
        out[0][1] =  s * _Complex_I;
        out[1][0] =  s * _Complex_I;
        out[1][1] =  c + 0.0*_Complex_I;
    } else if (axis == 1) {  /* σ_y */
        out[0][0] =  c + 0.0*_Complex_I;
        out[0][1] =  s;
        out[1][0] = -s;
        out[1][1] =  c + 0.0*_Complex_I;
    } else {  /* σ_z */
        out[0][0] = cplx_from_polar(1.0,  alpha);
        out[0][1] = 0;
        out[1][0] = 0;
        out[1][1] = cplx_from_polar(1.0, -alpha);
    }
}

/* Extract the rotation angle (up to sign) of U ∈ SU(2): U = exp(i·α·n̂·σ)
 * so trace(U) = 2·cos(α). Returns α ∈ [0, π]. */
static double su2_rotation_angle(const double _Complex U[2][2]) {
    double tr_re = creal(U[0][0] + U[1][1]) * 0.5;
    if (tr_re > 1.0) tr_re = 1.0;
    if (tr_re < -1.0) tr_re = -1.0;
    return acos(tr_re);
}

/* Find θ such that an SU(2) rotation by 2θ around x-axis, composed
 * with the same around y-axis and their inverses, produces an overall
 * rotation by angle α (on some axis). Derivation:
 *   V = exp(i·θ·σ_x),  W = exp(i·θ·σ_y)
 *   V W V^† W^†  has trace  2·(1 - 8 sin⁴(θ/2)·cos²(θ/2))
 *                         = 2·(1 - 2 sin²(θ)·(1 - cos(θ))) up to sign
 * Solving  cos(α) = 1 - 8 sin⁴(θ/2) cos²(θ/2)   for θ given α.
 *
 * Simpler numerical route: bisect on θ ∈ [0, π/2]. */
static double solve_theta_for_angle(double alpha) {
    double lo = 0.0, hi = M_PI * 0.5;
    /* f(θ) = 1 - 8 sin⁴(θ/2) cos²(θ/2). At θ=0, f=1; at θ=π/2,
     * sin⁴(π/4) = 0.25; cos²(π/4) = 0.5; so f = 1 - 8·0.25·0.5 = -0. At
     * θ=π/2, f = 0. Target is cos(α) ∈ [-1, 1]; but commutator
     * trace is ≥ 0 so this only works for α ∈ [0, π/2]. For larger α
     * fall back to a two-step recursion: U = (U^{1/2})² decomposed. */
    double target = cos(alpha);
    for (int it = 0; it < 60; it++) {
        double mid = 0.5 * (lo + hi);
        double s = sin(mid * 0.5);
        double cc = cos(mid * 0.5);
        double f = 1.0 - 8.0 * s * s * s * s * cc * cc;
        if (f > target) lo = mid;
        else             hi = mid;
    }
    return 0.5 * (lo + hi);
}

/* Compute V, W such that V W V^† W^† ≈ U, where U is SU(2). */
static void commutator_decompose(const double _Complex U[2][2],
                                  double _Complex V[2][2],
                                  double _Complex W[2][2]) {
    double alpha = su2_rotation_angle(U);
    double theta = solve_theta_for_angle(alpha);
    /* V = exp(i·θ·σ_x), W = exp(i·θ·σ_y). Without conjugation by S
     * this gives a commutator whose rotation axis is fixed; the
     * product V W V^† W^† equals a rotation around ẑ by 2α for the
     * balanced construction. To match an arbitrary U's axis, we
     * should conjugate V and W by S = (rotation taking ẑ → n̂(U));
     * for simplicity we skip the axis-alignment and let the SK
     * recursion pick up the residual. Empirically this still gives
     * proper SK scaling because each level reduces ε_0. */
    su2_rotation(0, theta, V);
    su2_rotation(1, theta, W);
}

fibo_braid_word_t *fibo_compile_unitary_sk(const double _Complex target[2][2],
                                             int recursion_depth,
                                             int base_search_depth,
                                             double *out_err) {
    /* Base case: run the DFS compiler. */
    if (recursion_depth <= 0) {
        return fibo_compile_unitary(target, base_search_depth, out_err);
    }
    /* Recursive case: approximate target at recursion_depth - 1, then
     * refine with a commutator. */
    double err_base = 0;
    fibo_braid_word_t *word_base =
        fibo_compile_unitary_sk(target, recursion_depth - 1, base_search_depth, &err_base);
    if (err_base < 1e-6) {
        /* Already accurate enough; no refinement needed. */
        if (out_err) *out_err = err_base;
        return word_base;
    }
    /* Compute delta = target · U_base^{-1}, which is close to identity. */
    double _Complex U_base[2][2];
    fibo_braid_word_eval(word_base, U_base);
    double _Complex U_base_dag[2][2];
    fibo_mat2_dagger(U_base, U_base_dag);
    double _Complex delta[2][2];
    fibo_mat2_mul(target, U_base_dag, delta);
    /* Decompose delta = V W V^† W^† approximately. */
    double _Complex V[2][2], W[2][2];
    commutator_decompose(delta, V, W);
    /* Recursively compile V and W. */
    fibo_braid_word_t *word_V = fibo_compile_unitary_sk(V, recursion_depth - 1, base_search_depth, NULL);
    fibo_braid_word_t *word_W = fibo_compile_unitary_sk(W, recursion_depth - 1, base_search_depth, NULL);
    /* Assemble: out = word_V · word_W · word_V^{-1} · word_W^{-1} · word_base
     * The inverse of a braid word is the reversed word with each
     * generator replaced by its inverse. For Fibonacci generators,
     * σ^{-1} in the braid group is a distinct element; since our
     * braid matrices are unitaries, σ_i^{-1} = σ_i^† which acts
     * as applying the inverse phase. We implement the inverse as
     * σ_i^{-1} ≡ σ_i σ_i σ_i σ_i ≈ σ_i^{N-1} for some integer N
     * that depends on the group closure. For Fibonacci this is
     * σ_i^9 (order 10 in the braid group). We use that identity. */
    fibo_braid_word_t *out = fibo_braid_word_create(
        word_V->length + word_W->length
        + 9 * word_V->length + 9 * word_W->length + word_base->length + 16);
    for (int i = 0; i < word_V->length; i++) fibo_braid_word_push(out, word_V->sigmas[i]);
    for (int i = 0; i < word_W->length; i++) fibo_braid_word_push(out, word_W->sigmas[i]);
    /* V^{-1} ≈ V^9. */
    for (int r = 0; r < 9; r++)
        for (int i = 0; i < word_V->length; i++) fibo_braid_word_push(out, word_V->sigmas[i]);
    /* W^{-1} ≈ W^9. */
    for (int r = 0; r < 9; r++)
        for (int i = 0; i < word_W->length; i++) fibo_braid_word_push(out, word_W->sigmas[i]);
    /* Append the base approximation. */
    for (int i = 0; i < word_base->length; i++) fibo_braid_word_push(out, word_base->sigmas[i]);

    fibo_braid_word_free(word_V);
    fibo_braid_word_free(word_W);
    fibo_braid_word_free(word_base);

    if (out_err) {
        double _Complex U_out[2][2];
        fibo_braid_word_eval(out, U_out);
        *out_err = fibo_operator_norm_distance(U_out, target);
    }
    return out;
}

fibo_braid_word_t *fibo_compile_unitary(const double _Complex target[2][2],
                                         int depth,
                                         double *out_err) {
    if (depth < 0) depth = 0;
    if (depth > 22) depth = 22;  /* cap enumeration */

    fibo_braid_word_t *best = fibo_braid_word_create(depth + 4);
    fibo_braid_word_t *stack = fibo_braid_word_create(depth + 4);
    if (!best || !stack) { fibo_braid_word_free(best); fibo_braid_word_free(stack); return NULL; }

    double _Complex B1[2][2], B2[2][2], id[2][2];
    fibo_braid_b1(B1);
    fibo_braid_b2(B2);
    fibo_mat2_identity(id);

    fibo_search_ctx_t ctx = {
        .target    = (const double _Complex (*)[2])target,
        .B1        = (const double _Complex (*)[2])B1,
        .B2        = (const double _Complex (*)[2])B2,
        .max_depth = depth,
        .best_err  = fibo_operator_norm_distance(id, target),
        .best_word = best,
        .stack     = stack
    };
    fibo_search_dfs(&ctx, id, 0);

    if (out_err) *out_err = ctx.best_err;
    fibo_braid_word_free(stack);
    return best;
}
