/*
 * src/nqs/nqs_ansatz.c
 *
 * Wavefunction ansatz implementations. Three concrete options ship:
 *
 *   (1) Mean-field (NQS_ANSATZ_LEGACY_MLP)
 *         log ψ(s) = Σ_i θ_i · s_i              (N parameters)
 *
 *   (2) Real RBM (NQS_ANSATZ_RBM), Carleo & Troyer 2017
 *         log ψ(s) = Σ_i a_i s_i
 *                  + Σ_h log(2 cosh(b_h + Σ_i W_hi s_i))
 *       N + M + M·N real parameters. Stoquastic-only (strictly
 *       positive amplitudes); handles TFIM directly and bipartite
 *       Heisenberg with the Marshall sign rule.
 *
 *   (3) Complex RBM (NQS_ANSATZ_COMPLEX_RBM), for non-stoquastic
 *       Hamiltonians (frustrated J1-J2, Kitaev). Same functional
 *       form but parameters are complex: a_i = aR_i + i·aI_i,
 *       likewise b_h, W_hi. Storage is 2·(N + M + M·N) real
 *       doubles: first the real parts in the same layout as the
 *       real RBM, then the imaginary parts in the same layout.
 *       log ψ is now complex; |ψ(s)| and arg ψ(s) both depend on
 *       the spin configuration and can carry arbitrary signs.
 *
 * The ViT / factored-ViT / autoregressive / KAN kinds return NULL
 * until the external NN engine lands. Samplers and optimizers
 * operate through the opaque handle and do not know which concrete
 * kind is behind it.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "nqs/nqs_ansatz.h"

/* ----------------------------------------------------------------- */
/* Internal state. All concrete ansätze share the same struct and     */
/* dispatch on `kind`. Parameter layouts:                             */
/*   LEGACY_MLP:  [θ_0 .. θ_{N-1}]                                    */
/*   RBM:         [a_0 .. a_{N-1}, b_0 .. b_{M-1},                    */
/*                  W_00 .. W_0,N-1, W_10 .. W_1,N-1, ...]            */
/* ----------------------------------------------------------------- */
struct nqs_ansatz {
    nqs_ansatz_kind_t kind;
    int               num_sites;     /* N */
    int               num_hidden;    /* M (RBM only, else 0)           */
    long              num_params;
    double           *params;        /* length = num_params            */
};

static double xorshift_uniform(unsigned long long *state) {
    unsigned long long x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x;
    return (double)(x >> 11) / 9007199254740992.0;
}

/* Box-Muller, one call → one normal. */
static double xorshift_gauss(unsigned long long *state) {
    double u1 = xorshift_uniform(state);
    double u2 = xorshift_uniform(state);
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

nqs_ansatz_t *nqs_ansatz_create(const nqs_config_t *cfg, int num_sites) {
    if (!cfg || num_sites <= 0) return NULL;
    nqs_ansatz_t *a = calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->kind = cfg->ansatz;
    a->num_sites = num_sites;

    unsigned long long seed = cfg->rng_seed
        ? (unsigned long long)cfg->rng_seed
        : 0xDEADBEEFCAFEBABEULL;

    if (cfg->ansatz == NQS_ANSATZ_LEGACY_MLP) {
        a->num_hidden = 0;
        a->num_params = num_sites;
        a->params = calloc((size_t)a->num_params, sizeof(double));
        if (!a->params) { free(a); return NULL; }
        for (long i = 0; i < a->num_params; i++) {
            double u = xorshift_uniform(&seed);
            a->params[i] = 0.01 * (u - 0.5);
        }
        return a;
    }

    if (cfg->ansatz == NQS_ANSATZ_RBM) {
        int M = cfg->rbm_hidden_units > 0 ? cfg->rbm_hidden_units
                                          : 2 * num_sites;
        a->num_hidden = M;
        a->num_params = (long)num_sites + (long)M + (long)M * num_sites;
        a->params = calloc((size_t)a->num_params, sizeof(double));
        if (!a->params) { free(a); return NULL; }
        double scale = cfg->rbm_init_scale > 0.0 ? cfg->rbm_init_scale : 0.01;
        /* visible biases a_i — small uniform noise */
        for (int i = 0; i < num_sites; i++) {
            a->params[i] = scale * (xorshift_uniform(&seed) - 0.5);
        }
        /* hidden biases b_h — small uniform noise */
        for (int h = 0; h < M; h++) {
            a->params[num_sites + h] = scale * (xorshift_uniform(&seed) - 0.5);
        }
        /* weights W_hi — Gaussian with std = scale / sqrt(N) */
        double wstd = scale / sqrt((double)num_sites);
        long W_off = (long)num_sites + (long)M;
        for (long i = 0; i < (long)M * num_sites; i++) {
            a->params[W_off + i] = wstd * xorshift_gauss(&seed);
        }
        return a;
    }

    if (cfg->ansatz == NQS_ANSATZ_COMPLEX_RBM) {
        int M = cfg->rbm_hidden_units > 0 ? cfg->rbm_hidden_units
                                          : 2 * num_sites;
        a->num_hidden = M;
        long P_real = (long)num_sites + (long)M + (long)M * num_sites;
        a->num_params = 2 * P_real;    /* real part + imaginary part */
        a->params = calloc((size_t)a->num_params, sizeof(double));
        if (!a->params) { free(a); return NULL; }
        double scale = cfg->rbm_init_scale > 0.0 ? cfg->rbm_init_scale : 0.01;
        double wstd = scale / sqrt((double)num_sites);
        /* Real parts: same layout and init as real RBM. */
        for (int i = 0; i < num_sites; i++)
            a->params[i] = scale * (xorshift_uniform(&seed) - 0.5);
        for (int h = 0; h < M; h++)
            a->params[num_sites + h] = scale * (xorshift_uniform(&seed) - 0.5);
        long W_off = (long)num_sites + (long)M;
        for (long i = 0; i < (long)M * num_sites; i++)
            a->params[W_off + i] = wstd * xorshift_gauss(&seed);
        /* Imaginary parts: initialised at a smaller scale so the
         * initial phase is close to zero (≈ real RBM at init). */
        double im_scale = 0.1 * scale;
        double im_wstd  = im_scale / sqrt((double)num_sites);
        for (int i = 0; i < num_sites; i++)
            a->params[P_real + i] = im_scale * (xorshift_uniform(&seed) - 0.5);
        for (int h = 0; h < M; h++)
            a->params[P_real + num_sites + h] = im_scale * (xorshift_uniform(&seed) - 0.5);
        for (long i = 0; i < (long)M * num_sites; i++)
            a->params[P_real + W_off + i] = im_wstd * xorshift_gauss(&seed);
        return a;
    }

    free(a);
    return NULL;
}

void nqs_ansatz_free(nqs_ansatz_t *a) {
    if (!a) return;
    free(a->params);
    free(a);
}

long nqs_ansatz_num_params(const nqs_ansatz_t *a) {
    return a ? a->num_params : 0;
}

/* log ψ for the mean-field ansatz */
static double mf_log_amp(const nqs_ansatz_t *a, const int *spins) {
    double acc = 0.0;
    for (int i = 0; i < a->num_sites; i++)
        acc += a->params[i] * (double)spins[i];
    return acc;
}

/* log ψ for the RBM ansatz.
 * log ψ = Σ_i a_i s_i + Σ_h log(2 cosh(x_h)),  x_h = b_h + Σ_i W_hi s_i
 * Numerically stable form for each term:
 *     log(2 cosh(x)) = |x| + log(1 + exp(-2|x|))
 */
static double rbm_log_amp(const nqs_ansatz_t *a, const int *spins) {
    int N = a->num_sites;
    int M = a->num_hidden;
    const double *av = &a->params[0];
    const double *bh = &a->params[N];
    const double *Wh = &a->params[N + M];

    double acc = 0.0;
    for (int i = 0; i < N; i++) acc += av[i] * (double)spins[i];
    for (int h = 0; h < M; h++) {
        double x = bh[h];
        const double *Wrow = &Wh[(long)h * N];
        for (int i = 0; i < N; i++) x += Wrow[i] * (double)spins[i];
        double ax = fabs(x);
        acc += ax + log1p(exp(-2.0 * ax));
    }
    return acc;
}

/* Complex RBM log ψ. For each hidden unit h compute
 *   x_h = (b_h_R + i b_h_I) + Σ_i (W_hi_R + i W_hi_I) · s_i
 * and accumulate log(2 cosh(x_h)). Visible term Σ_i (a_i_R + i a_i_I) s_i
 * contributes directly. Return (log|ψ|, arg ψ).
 *
 * Stable form: for complex z = x + iy,
 *   log(2 cosh z) = log(cosh(x) cos(y) + i sinh(x) sin(y)) + log 2
 * Use hypot to get the magnitude and atan2 for the phase.
 * For large |x|, subtract |x| before hypot to avoid overflow:
 *   cosh(x) = e^{|x|}/2 · (1 + e^{-2|x|})
 *   sinh(x) = sign(x) · e^{|x|}/2 · (1 - e^{-2|x|})
 * so 2 cosh(z) · e^{-|x|} = (1 + e^{-2|x|}) cos(y) + i sign(x) (1 - e^{-2|x|}) sin(y)
 * whose magnitude is stable. log|2 cosh(z)| = |x| + log|...stable...|. */
static void complex_cosh_log(double x, double y, double *out_log_abs, double *out_arg) {
    double ax = fabs(x);
    double expo = exp(-2.0 * ax);
    double re = (1.0 + expo) * cos(y);
    double im = (x >= 0.0 ? 1.0 : -1.0) * (1.0 - expo) * sin(y);
    double mag = hypot(re, im);
    /* 2 cosh(z) = e^{|x|} · (re + i im),  so log|2 cosh(z)| = |x| + log mag */
    if (out_log_abs) *out_log_abs = ax + (mag > 0 ? log(mag) : -1e300);
    if (out_arg)     *out_arg     = atan2(im, re);
}

static void crbm_log_amp(const nqs_ansatz_t *a, const int *spins,
                          double *out_log_abs, double *out_arg) {
    int N = a->num_sites;
    int M = a->num_hidden;
    long P_real = (long)N + (long)M + (long)M * N;
    const double *aR = &a->params[0];
    const double *bR = &a->params[N];
    const double *WR = &a->params[N + M];
    const double *aI = &a->params[P_real];
    const double *bI = &a->params[P_real + N];
    const double *WI = &a->params[P_real + N + M];

    double logabs = 0.0, arg = 0.0;
    /* Visible term. */
    for (int i = 0; i < N; i++) {
        logabs += aR[i] * (double)spins[i];
        arg    += aI[i] * (double)spins[i];
    }
    /* Hidden terms. */
    for (int h = 0; h < M; h++) {
        double xR = bR[h];
        double xI = bI[h];
        const double *WrowR = &WR[(long)h * N];
        const double *WrowI = &WI[(long)h * N];
        for (int i = 0; i < N; i++) {
            xR += WrowR[i] * (double)spins[i];
            xI += WrowI[i] * (double)spins[i];
        }
        double la, ar;
        complex_cosh_log(xR, xI, &la, &ar);
        logabs += la;
        arg    += ar;
    }
    if (out_log_abs) *out_log_abs = logabs;
    if (out_arg)     *out_arg     = arg;
}

void nqs_ansatz_log_amp(const int *spins, int num_sites,
                        void *ansatz_user,
                        double *out_log_abs,
                        double *out_arg) {
    nqs_ansatz_t *a = (nqs_ansatz_t *)ansatz_user;
    double acc = 0.0, arg = 0.0;
    if (a && spins && num_sites == a->num_sites) {
        switch (a->kind) {
            case NQS_ANSATZ_COMPLEX_RBM:
                crbm_log_amp(a, spins, &acc, &arg);
                break;
            case NQS_ANSATZ_RBM:
                acc = rbm_log_amp(a, spins);
                break;
            default:
                acc = mf_log_amp(a, spins);
                break;
        }
    }
    if (out_log_abs) *out_log_abs = acc;
    if (out_arg)     *out_arg     = arg;
}

static int mf_gradient(const nqs_ansatz_t *a,
                       const int *spins, double *out) {
    for (int i = 0; i < a->num_sites; i++) out[i] = (double)spins[i];
    return 0;
}

/* ∂ log ψ / ∂a_i = s_i
 * ∂ log ψ / ∂b_h = tanh(x_h)
 * ∂ log ψ / ∂W_hi = tanh(x_h) · s_i
 */
static int rbm_gradient(const nqs_ansatz_t *a,
                        const int *spins, double *out) {
    int N = a->num_sites;
    int M = a->num_hidden;
    const double *bh = &a->params[N];
    const double *Wh = &a->params[N + M];

    for (int i = 0; i < N; i++) out[i] = (double)spins[i];

    double *tanh_x = &out[N];  /* temporarily store tanh(x_h) here */
    for (int h = 0; h < M; h++) {
        double x = bh[h];
        const double *Wrow = &Wh[(long)h * N];
        for (int i = 0; i < N; i++) x += Wrow[i] * (double)spins[i];
        tanh_x[h] = tanh(x);
    }
    /* Now write W gradient from the tanh values. Visit in parameter
     * order so we do not overwrite tanh_x[] prematurely — we walk
     * h = 0..M-1 and append the N W-gradients after the hidden block. */
    long W_off = (long)N + M;
    for (int h = 0; h < M; h++) {
        double th = tanh_x[h];   /* safe: only read before we write its slot */
        double *Wout = &out[W_off + (long)h * N];
        for (int i = 0; i < N; i++) Wout[i] = th * (double)spins[i];
    }
    return 0;
}

/* Complex RBM gradient of log|ψ|:
 *   ∂ log|ψ| / ∂aR_i  = s_i
 *   ∂ log|ψ| / ∂aI_i  = 0            (visible-imag only shifts phase)
 *   ∂ log|ψ| / ∂bR_h  = Re tanh(z_h)  where z_h = xR_h + i xI_h
 *   ∂ log|ψ| / ∂bI_h  = -Im tanh(z_h)  (chain rule: d|cosh|/d(Im z))
 *   ∂ log|ψ| / ∂WR_hi = s_i · Re tanh(z_h)
 *   ∂ log|ψ| / ∂WI_hi = -s_i · Im tanh(z_h)
 *
 * Derivation: log|ψ| = Re(log ψ). For log(2 cosh(z)),
 *   d/dz log(2 cosh z) = tanh(z) = tR + i tI  (with real z = xR + i xI).
 * By chain rule with complex composition,
 *   ∂ Re log(cosh(xR + i xI)) / ∂ xR = Re tanh(z) = tR
 *   ∂ Re log(cosh(xR + i xI)) / ∂ xI = -Im tanh(z) = -tI
 * (the minus sign on the imag-axis derivative comes from the Cauchy-
 *  Riemann equations for the holomorphic log(cosh z)).
 *
 * A full complex SR would use the holomorphic ∂ log ψ / ∂θ = tR + i tI
 * directly; v0.4 returns only the real part, which is sufficient for a
 * real-projected natural-gradient step and matches what the current
 * optimizer expects. */
static int crbm_gradient(const nqs_ansatz_t *a,
                          const int *spins, double *out) {
    int N = a->num_sites;
    int M = a->num_hidden;
    long P_real = (long)N + (long)M + (long)M * N;
    const double *bR = &a->params[N];
    const double *WR = &a->params[N + M];
    const double *bI = &a->params[P_real + N];
    const double *WI = &a->params[P_real + N + M];

    /* Visible gradients. */
    for (int i = 0; i < N; i++) {
        out[i]            = (double)spins[i];
        out[P_real + i]   = 0.0;
    }

    /* For each hidden unit, compute tanh(z) = (tR, tI). */
    double *tR = &out[N];               /* reuse hidden-bias slots */
    double *tI = &out[P_real + N];
    for (int h = 0; h < M; h++) {
        double xR = bR[h], xI = bI[h];
        const double *WrowR = &WR[(long)h * N];
        const double *WrowI = &WI[(long)h * N];
        for (int i = 0; i < N; i++) {
            xR += WrowR[i] * (double)spins[i];
            xI += WrowI[i] * (double)spins[i];
        }
        /* tanh(xR + i xI) — use stable formula via sinh/cosh. */
        double cxr = cosh(xR), sxr = sinh(xR);
        double cy  = cos(xI),  sy  = sin(xI);
        /* sinh(z) = sxr cy + i cxr sy,  cosh(z) = cxr cy + i sxr sy */
        double nr = sxr * cy, ni = cxr * sy;
        double dr = cxr * cy, di = sxr * sy;
        double denom = dr * dr + di * di;
        if (denom < 1e-300) denom = 1e-300;
        /* tanh = sinh/cosh = (nr + i ni)(dr - i di) / denom */
        double trh = (nr * dr + ni * di) / denom;
        double tih = (ni * dr - nr * di) / denom;
        tR[h] = trh;
        tI[h] = -tih;
    }
    /* W gradients: read tR/tI from the slots above BEFORE overwriting. */
    long W_off_R = (long)N + (long)M;
    long W_off_I = P_real + (long)N + (long)M;
    /* Copy tR/tI out since we will overwrite them below. */
    double *tR_buf = malloc(sizeof(double) * M);
    double *tI_buf = malloc(sizeof(double) * M);
    for (int h = 0; h < M; h++) { tR_buf[h] = tR[h]; tI_buf[h] = tI[h]; }
    for (int h = 0; h < M; h++) {
        double trh = tR_buf[h];
        double tih = tI_buf[h];
        double *woutR = &out[W_off_R + (long)h * N];
        double *woutI = &out[W_off_I + (long)h * N];
        for (int i = 0; i < N; i++) {
            woutR[i] =  trh * (double)spins[i];
            woutI[i] =  tih * (double)spins[i];   /* sign already flipped */
        }
    }
    /* Restore hidden-bias gradient slots. */
    for (int h = 0; h < M; h++) { tR[h] = tR_buf[h]; tI[h] = tI_buf[h]; }
    free(tR_buf); free(tI_buf);
    return 0;
}

int nqs_ansatz_logpsi_gradient(nqs_ansatz_t *a,
                               const int *spins, int num_sites,
                               double *out_grad) {
    if (!a || !spins || !out_grad || num_sites != a->num_sites) return -1;
    switch (a->kind) {
        case NQS_ANSATZ_COMPLEX_RBM: return crbm_gradient(a, spins, out_grad);
        case NQS_ANSATZ_RBM:         return rbm_gradient(a, spins, out_grad);
        default:                     return mf_gradient(a, spins, out_grad);
    }
}

int nqs_ansatz_is_complex(const nqs_ansatz_t *a) {
    return (a && a->kind == NQS_ANSATZ_COMPLEX_RBM) ? 1 : 0;
}

/* Holomorphic gradient for the complex RBM:
 *     ∂ log ψ / ∂ aR_i   =  s_i
 *     ∂ log ψ / ∂ aI_i   =  i · s_i
 *     ∂ log ψ / ∂ bR_h   =  tanh(z_h)       (z_h = xR_h + i xI_h)
 *     ∂ log ψ / ∂ bI_h   =  i · tanh(z_h)
 *     ∂ log ψ / ∂ WR_hi  =  s_i · tanh(z_h)
 *     ∂ log ψ / ∂ WI_hi  =  i · s_i · tanh(z_h)
 * Real / imag components are extracted and written into the two
 * output buffers. */
static int crbm_gradient_complex(const nqs_ansatz_t *a,
                                  const int *spins,
                                  double *out_re, double *out_im) {
    int N = a->num_sites;
    int M = a->num_hidden;
    long P_real = (long)N + (long)M + (long)M * N;
    const double *bR = &a->params[N];
    const double *WR = &a->params[N + M];
    const double *bI = &a->params[P_real + N];
    const double *WI = &a->params[P_real + N + M];

    /* Visible gradients. */
    for (int i = 0; i < N; i++) {
        out_re[i]          = (double)spins[i];   /* aR_i */
        out_im[i]          = 0.0;
        out_re[P_real + i] = 0.0;                /* aI_i */
        out_im[P_real + i] = (double)spins[i];
    }
    /* Hidden + weight gradients: first compute tanh(z_h) for each h. */
    double *trh = malloc(sizeof(double) * M);
    double *tih = malloc(sizeof(double) * M);
    for (int h = 0; h < M; h++) {
        double xR = bR[h], xI = bI[h];
        const double *WrowR = &WR[(long)h * N];
        const double *WrowI = &WI[(long)h * N];
        for (int i = 0; i < N; i++) {
            xR += WrowR[i] * (double)spins[i];
            xI += WrowI[i] * (double)spins[i];
        }
        double cxr = cosh(xR), sxr = sinh(xR);
        double cy  = cos(xI),  sy  = sin(xI);
        double nr = sxr * cy, ni = cxr * sy;
        double dr = cxr * cy, di = sxr * sy;
        double denom = dr * dr + di * di;
        if (denom < 1e-300) denom = 1e-300;
        trh[h] = (nr * dr + ni * di) / denom;
        tih[h] = (ni * dr - nr * di) / denom;
    }
    /* Hidden bias slots. */
    for (int h = 0; h < M; h++) {
        out_re[N + h]             = trh[h];       /* bR_h */
        out_im[N + h]             = tih[h];
        out_re[P_real + N + h]    = -tih[h];      /* bI_h: ∂ψ/∂bI = i·tanh(z), so Re(i·t) = -Im(t) */
        out_im[P_real + N + h]    = trh[h];       /* Im(i·t) = Re(t) */
    }
    /* Weight slots. */
    long W_off_R = (long)N + (long)M;
    long W_off_I = P_real + (long)N + (long)M;
    for (int h = 0; h < M; h++) {
        double tr = trh[h], ti = tih[h];
        double *wReR = &out_re[W_off_R + (long)h * N];
        double *wImR = &out_im[W_off_R + (long)h * N];
        double *wReI = &out_re[W_off_I + (long)h * N];
        double *wImI = &out_im[W_off_I + (long)h * N];
        for (int i = 0; i < N; i++) {
            double si = (double)spins[i];
            /* ∂ψ/∂WR_hi = s_i · tanh(z_h) */
            wReR[i] =  tr * si;
            wImR[i] =  ti * si;
            /* ∂ψ/∂WI_hi = i · s_i · tanh(z_h) */
            wReI[i] = -ti * si;
            wImI[i] =  tr * si;
        }
    }
    free(trh); free(tih);
    return 0;
}

int nqs_ansatz_logpsi_gradient_complex(nqs_ansatz_t *a,
                                        const int *spins, int num_sites,
                                        double *out_re, double *out_im) {
    if (!a || !spins || !out_re || !out_im || num_sites != a->num_sites) return -1;
    long P = a->num_params;
    if (a->kind == NQS_ANSATZ_COMPLEX_RBM) {
        return crbm_gradient_complex(a, spins, out_re, out_im);
    }
    /* Real ansätze: imaginary part is zero; real part = ∂ log|ψ|/∂θ. */
    int rc = nqs_ansatz_logpsi_gradient(a, spins, num_sites, out_re);
    if (rc == 0) for (long k = 0; k < P; k++) out_im[k] = 0.0;
    return rc;
}

int nqs_ansatz_apply_update(nqs_ansatz_t *a,
                            const double *delta, double step) {
    if (!a || !delta) return -1;
    for (long i = 0; i < a->num_params; i++) {
        a->params[i] += step * delta[i];
    }
    return 0;
}

double *nqs_ansatz_params_raw(nqs_ansatz_t *a) {
    return a ? a->params : NULL;
}
