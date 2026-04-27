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

    if (cfg->ansatz == NQS_ANSATZ_FACTORED_VIT) {
        /* Factored-attention ViT NQS, v0 (Rende et al. 2024,
         * arXiv:2310.05715, single-head + patch_size=1 + real-amplitude
         * slice).  Parameter layout, total 3·d + d² + N + 1:
         *
         *   [0       , d         )  w_emb        (d)
         *   [d       , 2d        )  b_emb        (d)
         *   [2d      , 2d + d²   )  V (row-major) (d × d)
         *   [2d + d² , 2d + d² + N)  attn bias a  (N relative-position slots)
         *   [2d+d²+N , 2d+d²+N+d)   W_out         (d)
         *   [2d+d²+N+d              ]  b_out      (1 scalar)
         */
        int d = cfg->width > 0 ? cfg->width : 4;
        a->num_hidden = d;             /* repurpose: stash d in num_hidden */
        long N = num_sites;
        long P = 3L * d + (long)d * d + N + 1L;
        a->num_params = P;
        a->params = calloc((size_t)P, sizeof(double));
        if (!a->params) { free(a); return NULL; }
        double scale = cfg->rbm_init_scale > 0.0 ? cfg->rbm_init_scale : 0.01;

        /* w_emb: small Gaussian.  V: Gaussian / √d (Glorot-ish).
         * b_emb, attn bias, W_out: small uniform.  b_out: zero. */
        double w_std = scale;
        double v_std = scale / sqrt((double)d);
        for (int i = 0; i < d; i++)
            a->params[i] = w_std * xorshift_gauss(&seed);                 /* w_emb */
        for (int i = 0; i < d; i++)
            a->params[d + i] = scale * (xorshift_uniform(&seed) - 0.5);    /* b_emb */
        long V_off = 2L * d;
        for (long k = 0; k < (long)d * d; k++)
            a->params[V_off + k] = v_std * xorshift_gauss(&seed);          /* V */
        long a_off = V_off + (long)d * d;
        for (long k = 0; k < N; k++)
            a->params[a_off + k] = scale * (xorshift_uniform(&seed) - 0.5);/* attn */
        long Wout_off = a_off + N;
        for (int i = 0; i < d; i++)
            a->params[Wout_off + i] = scale * (xorshift_uniform(&seed) - 0.5);
        a->params[Wout_off + d] = 0.0;                                     /* b_out */
        return a;
    }

    if (cfg->ansatz == NQS_ANSATZ_FACTORED_VIT_COMPLEX) {
        /* Complex-amplitude factored-attention ViT NQS.  Every embed /
         * value / output weight has a real and imaginary part (the
         * factored attention bias stays real — we only soft-max real
         * scores).  Required for non-stoquastic ground states.
         *
         * Parameter layout, total 6·d + 2·d² + N + 2:
         *   [0          , d           )  w_emb_R
         *   [d          , 2d          )  w_emb_I
         *   [2d         , 3d          )  b_emb_R
         *   [3d         , 4d          )  b_emb_I
         *   [4d         , 4d + d²     )  V_R   (row-major)
         *   [4d + d²    , 4d + 2d²    )  V_I   (row-major)
         *   [4d + 2d²   , 4d + 2d² + N)  attn bias (real)
         *   [4d+2d²+N   , 4d+2d²+N+d  )  W_out_R
         *   [4d+2d²+N+d , 4d+2d²+N+2d )  W_out_I
         *   [4d+2d²+N+2d                ]  b_out_R
         *   [4d+2d²+N+2d+1              ]  b_out_I
         */
        int d = cfg->width > 0 ? cfg->width : 4;
        a->num_hidden = d;
        long N = num_sites;
        long P = 6L * d + 2L * d * d + N + 2L;
        a->num_params = P;
        a->params = calloc((size_t)P, sizeof(double));
        if (!a->params) { free(a); return NULL; }
        double scale = cfg->rbm_init_scale > 0.0 ? cfg->rbm_init_scale : 0.01;
        double w_std = scale;
        double v_std = scale / sqrt((double)d);
        double im_scale = 0.1 * scale;          /* small imaginary init */
        /* w_emb_R, w_emb_I */
        for (int i = 0; i < d; i++)
            a->params[i] = w_std * xorshift_gauss(&seed);
        for (int i = 0; i < d; i++)
            a->params[d + i] = im_scale * xorshift_gauss(&seed);
        /* b_emb_R, b_emb_I */
        for (int i = 0; i < d; i++)
            a->params[2*d + i] = scale * (xorshift_uniform(&seed) - 0.5);
        for (int i = 0; i < d; i++)
            a->params[3*d + i] = im_scale * (xorshift_uniform(&seed) - 0.5);
        /* V_R, V_I */
        long VR_off = 4L * d;
        long VI_off = VR_off + (long)d * d;
        for (long k = 0; k < (long)d * d; k++)
            a->params[VR_off + k] = v_std * xorshift_gauss(&seed);
        for (long k = 0; k < (long)d * d; k++)
            a->params[VI_off + k] = (0.1 * v_std) * xorshift_gauss(&seed);
        /* attn (real) */
        long attn_off = VI_off + (long)d * d;
        for (long k = 0; k < N; k++)
            a->params[attn_off + k] = scale * (xorshift_uniform(&seed) - 0.5);
        /* W_out_R, W_out_I */
        long WoutR_off = attn_off + N;
        long WoutI_off = WoutR_off + d;
        for (int i = 0; i < d; i++)
            a->params[WoutR_off + i] = scale * (xorshift_uniform(&seed) - 0.5);
        for (int i = 0; i < d; i++)
            a->params[WoutI_off + i] = im_scale * (xorshift_uniform(&seed) - 0.5);
        /* b_out_R, b_out_I */
        a->params[WoutI_off + d]     = 0.0;
        a->params[WoutI_off + d + 1] = 0.0;
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

/* --------------------------------------------------------------------- */
/*  Factored-attention ViT NQS — v0                                      */
/*                                                                        */
/*  Single-head, patch_size=1, real-amplitude slice of Rende et al. 2024  */
/*  (arXiv:2310.05715).  Forward:                                         */
/*    e_i = w_emb · s_i + b_emb                       (d-vector / site)   */
/*    v_i = V · e_i                                                       */
/*    S_{ij} = a[(i − j + N) mod N]                   (factored bias)     */
/*    P_{ij} = softmax_j(S_{ij})                                          */
/*    o_i = Σ_j P_{ij} · v_j                                              */
/*    log|ψ| = Σ_i (W_out · o_i) + b_out                                  */
/* --------------------------------------------------------------------- */
typedef struct {
    int    N, d;
    long   V_off, a_off, Wout_off, bout_off;
    /* Per-evaluation scratch (caller-allocated, length-prefixed). */
    double *e;        /* [N · d] embeddings                              */
    double *v;        /* [N · d] values                                  */
    double *P;        /* [N · N] softmax weights                         */
    double *o;        /* [N · d] attention outputs                       */
} vit_workspace_t;

static void vit_offsets(const nqs_ansatz_t *a, int *out_d,
                        long *out_V, long *out_a, long *out_W, long *out_b) {
    int d = a->num_hidden;
    long N = a->num_sites;
    *out_d = d;
    *out_V = 2L * d;
    *out_a = *out_V + (long)d * d;
    *out_W = *out_a + N;
    *out_b = *out_W + d;
    (void)N;
}

/* Allocate scratch buffers used by both forward and gradient. */
static int vit_workspace_init(vit_workspace_t *ws, int N, int d) {
    ws->N = N; ws->d = d;
    ws->e = malloc((size_t)N * d * sizeof(double));
    ws->v = malloc((size_t)N * d * sizeof(double));
    ws->P = malloc((size_t)N * N * sizeof(double));
    ws->o = malloc((size_t)N * d * sizeof(double));
    if (!ws->e || !ws->v || !ws->P || !ws->o) {
        free(ws->e); free(ws->v); free(ws->P); free(ws->o);
        return -1;
    }
    return 0;
}
static void vit_workspace_free(vit_workspace_t *ws) {
    free(ws->e); free(ws->v); free(ws->P); free(ws->o);
}

static double vit_forward(const nqs_ansatz_t *a, const int *spins,
                          vit_workspace_t *ws) {
    int d; long V_off, a_off, Wout_off, bout_off;
    vit_offsets(a, &d, &V_off, &a_off, &Wout_off, &bout_off);
    int N = a->num_sites;
    const double *w_emb  = &a->params[0];
    const double *b_emb  = &a->params[d];
    const double *V      = &a->params[V_off];
    const double *attn   = &a->params[a_off];
    const double *W_out  = &a->params[Wout_off];
    double        b_out  = a->params[bout_off];

    /* e_i, v_i */
    for (int i = 0; i < N; i++) {
        double si = (double)spins[i];
        for (int k = 0; k < d; k++) ws->e[i * d + k] = w_emb[k] * si + b_emb[k];
    }
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < d; k++) {
            double s = 0.0;
            for (int l = 0; l < d; l++) s += V[k * d + l] * ws->e[i * d + l];
            ws->v[i * d + k] = s;
        }
    }
    /* P_{ij} via log-sum-exp */
    for (int i = 0; i < N; i++) {
        double max_S = -1e300;
        for (int j = 0; j < N; j++) {
            int rel = ((i - j) % N + N) % N;
            double S = attn[rel];
            if (S > max_S) max_S = S;
        }
        double Z = 0.0;
        for (int j = 0; j < N; j++) {
            int rel = ((i - j) % N + N) % N;
            double S = attn[rel];
            double e = exp(S - max_S);
            ws->P[i * N + j] = e;
            Z += e;
        }
        double invZ = 1.0 / Z;
        for (int j = 0; j < N; j++) ws->P[i * N + j] *= invZ;
    }
    /* o_i = Σ_j P_ij · v_j */
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < d; k++) {
            double s = 0.0;
            for (int j = 0; j < N; j++) s += ws->P[i * N + j] * ws->v[j * d + k];
            ws->o[i * d + k] = s;
        }
    }
    /* log|ψ| = Σ_i W_out · o_i + b_out */
    double acc = b_out;
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < d; k++) acc += W_out[k] * ws->o[i * d + k];
    }
    return acc;
}

static double vit_log_amp(const nqs_ansatz_t *a, const int *spins) {
    vit_workspace_t ws;
    if (vit_workspace_init(&ws, a->num_sites, a->num_hidden) != 0) return 0.0;
    double v = vit_forward(a, spins, &ws);
    vit_workspace_free(&ws);
    return v;
}

/* Analytic gradient via chain rule on the forward graph.  Layout matches
 * the parameter offsets used by vit_offsets / vit_forward. */
static int vit_gradient(const nqs_ansatz_t *a,
                        const int *spins, double *out) {
    int d; long V_off, a_off, Wout_off, bout_off;
    vit_offsets(a, &d, &V_off, &a_off, &Wout_off, &bout_off);
    int N = a->num_sites;
    const double *V      = &a->params[V_off];
    const double *W_out  = &a->params[Wout_off];

    vit_workspace_t ws;
    if (vit_workspace_init(&ws, N, d) != 0) return -1;
    (void)vit_forward(a, spins, &ws);

    memset(out, 0, (size_t)a->num_params * sizeof(double));

    /* dL/db_out = 1 */
    out[bout_off] = 1.0;
    /* dL/dW_out[k] = Σ_i o_i[k] */
    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++) out[Wout_off + k] += ws.o[i * d + k];

    /* s_pred[i] = Σ_i' P_{i'i}  (column sum of attention) */
    double *s_pred = calloc((size_t)N, sizeof(double));
    /* W_out · v_j  (scalar per j; equals dL/dP_{ij} for any i) */
    double *Wv = calloc((size_t)N, sizeof(double));
    if (!s_pred || !Wv) {
        free(s_pred); free(Wv); vit_workspace_free(&ws); return -1;
    }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) s_pred[j] += ws.P[i * N + j];
    for (int j = 0; j < N; j++) {
        double s = 0.0;
        for (int k = 0; k < d; k++) s += W_out[k] * ws.v[j * d + k];
        Wv[j] = s;
    }
    /* W_out · o_i (scalar per i) */
    double *Wo = calloc((size_t)N, sizeof(double));
    if (!Wo) { free(s_pred); free(Wv); vit_workspace_free(&ws); return -1; }
    for (int i = 0; i < N; i++) {
        double s = 0.0;
        for (int k = 0; k < d; k++) s += W_out[k] * ws.o[i * d + k];
        Wo[i] = s;
    }

    /* dL/dS_{ij} = P_{ij} · (Wv[j] − Wo[i]).  Multiple (i, j) pairs hit
     * the same a[(i-j+N)%N]; accumulate. */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double dS = ws.P[i * N + j] * (Wv[j] - Wo[i]);
            int rel = ((i - j) % N + N) % N;
            out[a_off + rel] += dS;
        }
    }

    /* dL/dV[k][l] = W_out[k] · Σ_i s_pred[i] · e_i[l]
     *             = W_out[k] · ev[l]
     */
    double *ev = calloc((size_t)d, sizeof(double));
    if (!ev) { free(s_pred); free(Wv); free(Wo); vit_workspace_free(&ws); return -1; }
    for (int i = 0; i < N; i++)
        for (int l = 0; l < d; l++) ev[l] += s_pred[i] * ws.e[i * d + l];
    for (int k = 0; k < d; k++)
        for (int l = 0; l < d; l++) out[V_off + k * d + l] = W_out[k] * ev[l];

    /* dL/de_i[l] = s_pred[i] · (V^T W_out)[l]
     * dL/dw_emb[l] = Σ_i dL/de_i[l] · s_i
     * dL/db_emb[l] = Σ_i dL/de_i[l] */
    double *VtW = calloc((size_t)d, sizeof(double));
    if (!VtW) {
        free(s_pred); free(Wv); free(Wo); free(ev);
        vit_workspace_free(&ws); return -1;
    }
    for (int l = 0; l < d; l++) {
        double s = 0.0;
        for (int k = 0; k < d; k++) s += V[k * d + l] * W_out[k];
        VtW[l] = s;
    }
    for (int l = 0; l < d; l++) {
        double dw = 0.0, db = 0.0;
        for (int i = 0; i < N; i++) {
            double dei = s_pred[i] * VtW[l];
            dw += dei * (double)spins[i];
            db += dei;
        }
        out[l]     = dw;          /* dL/dw_emb[l] */
        out[d + l] = db;          /* dL/db_emb[l] */
    }

    free(s_pred); free(Wv); free(Wo); free(ev); free(VtW);
    vit_workspace_free(&ws);
    return 0;
}

/* --------------------------------------------------------------------- */
/*  Complex-amplitude factored-attention ViT NQS                          */
/*                                                                        */
/*  Same architecture as the real ViT but every embed / value / output    */
/*  weight has a real and imaginary part; the attention bias is real      */
/*  (so the softmax operates on real scores).  Forward returns a complex  */
/*  log ψ; nqs_ansatz_log_amp returns (log|ψ| = Re(log ψ),                */
/*  arg ψ = Im(log ψ)).                                                   */
/* --------------------------------------------------------------------- */
typedef struct {
    int N, d;
    long w_emb_R, w_emb_I, b_emb_R, b_emb_I;
    long V_R, V_I, attn;
    long W_out_R, W_out_I, b_out_R, b_out_I;
} vit_c_offsets_t;

static void vit_c_offsets(const nqs_ansatz_t *a, vit_c_offsets_t *o) {
    int d = a->num_hidden;
    long N = a->num_sites;
    o->N = (int)N; o->d = d;
    o->w_emb_R = 0;
    o->w_emb_I = d;
    o->b_emb_R = 2L * d;
    o->b_emb_I = 3L * d;
    o->V_R     = 4L * d;
    o->V_I     = o->V_R + (long)d * d;
    o->attn    = o->V_I + (long)d * d;
    o->W_out_R = o->attn + N;
    o->W_out_I = o->W_out_R + d;
    o->b_out_R = o->W_out_I + d;
    o->b_out_I = o->b_out_R + 1;
}

typedef struct {
    int N, d;
    /* Real and imag parts of intermediate quantities (Cartesian basis). */
    double *eR, *eI;     /* [N · d] embeddings   */
    double *vR, *vI;     /* [N · d] values       */
    double *P;           /* [N · N] real softmax */
    double *oR, *oI;     /* [N · d] outputs      */
} vit_c_workspace_t;

static int vit_c_workspace_init(vit_c_workspace_t *ws, int N, int d) {
    ws->N = N; ws->d = d;
    ws->eR = calloc((size_t)N * d, sizeof(double));
    ws->eI = calloc((size_t)N * d, sizeof(double));
    ws->vR = calloc((size_t)N * d, sizeof(double));
    ws->vI = calloc((size_t)N * d, sizeof(double));
    ws->P  = calloc((size_t)N * N, sizeof(double));
    ws->oR = calloc((size_t)N * d, sizeof(double));
    ws->oI = calloc((size_t)N * d, sizeof(double));
    if (!ws->eR || !ws->eI || !ws->vR || !ws->vI ||
        !ws->P  || !ws->oR || !ws->oI) {
        free(ws->eR); free(ws->eI); free(ws->vR); free(ws->vI);
        free(ws->P);  free(ws->oR); free(ws->oI);
        return -1;
    }
    return 0;
}
static void vit_c_workspace_free(vit_c_workspace_t *ws) {
    free(ws->eR); free(ws->eI); free(ws->vR); free(ws->vI);
    free(ws->P);  free(ws->oR); free(ws->oI);
}

/* Forward: writes intermediates into `ws`, returns log ψ as
 * (out_log_abs = Re(log ψ), out_arg = Im(log ψ)). */
static void vit_c_forward(const nqs_ansatz_t *a, const int *spins,
                          vit_c_workspace_t *ws,
                          double *out_log_abs, double *out_arg) {
    vit_c_offsets_t o; vit_c_offsets(a, &o);
    int N = o.N, d = o.d;
    const double *wR = &a->params[o.w_emb_R];
    const double *wI = &a->params[o.w_emb_I];
    const double *bR = &a->params[o.b_emb_R];
    const double *bI = &a->params[o.b_emb_I];
    const double *VR = &a->params[o.V_R];
    const double *VI = &a->params[o.V_I];
    const double *attn = &a->params[o.attn];
    const double *WR = &a->params[o.W_out_R];
    const double *WI = &a->params[o.W_out_I];
    double  bout_R = a->params[o.b_out_R];
    double  bout_I = a->params[o.b_out_I];

    /* e_i = (w_R + i w_I) s_i + (b_R + i b_I) */
    for (int i = 0; i < N; i++) {
        double si = (double)spins[i];
        for (int k = 0; k < d; k++) {
            ws->eR[i*d + k] = wR[k] * si + bR[k];
            ws->eI[i*d + k] = wI[k] * si + bI[k];
        }
    }
    /* v_i = (V_R + i V_I) e_i */
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < d; k++) {
            double sR = 0.0, sI = 0.0;
            for (int l = 0; l < d; l++) {
                double vrr = VR[k*d + l], vii = VI[k*d + l];
                double er  = ws->eR[i*d + l], ei = ws->eI[i*d + l];
                sR += vrr * er - vii * ei;
                sI += vrr * ei + vii * er;
            }
            ws->vR[i*d + k] = sR;
            ws->vI[i*d + k] = sI;
        }
    }
    /* Real softmax of attn on (i-j+N) % N */
    for (int i = 0; i < N; i++) {
        double max_S = -1e300;
        for (int j = 0; j < N; j++) {
            int rel = ((i - j) % N + N) % N;
            double S = attn[rel];
            if (S > max_S) max_S = S;
        }
        double Z = 0.0;
        for (int j = 0; j < N; j++) {
            int rel = ((i - j) % N + N) % N;
            double e = exp(attn[rel] - max_S);
            ws->P[i*N + j] = e;
            Z += e;
        }
        double invZ = 1.0 / Z;
        for (int j = 0; j < N; j++) ws->P[i*N + j] *= invZ;
    }
    /* o_i = sum_j P_ij v_j (complex) */
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < d; k++) {
            double sR = 0.0, sI = 0.0;
            for (int j = 0; j < N; j++) {
                sR += ws->P[i*N + j] * ws->vR[j*d + k];
                sI += ws->P[i*N + j] * ws->vI[j*d + k];
            }
            ws->oR[i*d + k] = sR;
            ws->oI[i*d + k] = sI;
        }
    }
    /* log ψ = Σ_i (W_R + i W_I) · o_i + (b_R + i b_I) */
    double logabs = bout_R, arg = bout_I;
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < d; k++) {
            double or_ = ws->oR[i*d + k], oi_ = ws->oI[i*d + k];
            logabs += WR[k] * or_ - WI[k] * oi_;
            arg    += WR[k] * oi_ + WI[k] * or_;
        }
    }
    if (out_log_abs) *out_log_abs = logabs;
    if (out_arg)     *out_arg     = arg;
}

static void vit_c_log_amp(const nqs_ansatz_t *a, const int *spins,
                          double *out_log_abs, double *out_arg) {
    vit_c_workspace_t ws;
    if (vit_c_workspace_init(&ws, a->num_sites, a->num_hidden) != 0) {
        if (out_log_abs) *out_log_abs = 0.0;
        if (out_arg)     *out_arg     = 0.0;
        return;
    }
    vit_c_forward(a, spins, &ws, out_log_abs, out_arg);
    vit_c_workspace_free(&ws);
}

/* Real-projected gradient ∂ Re(log ψ) / ∂ θ for every parameter.
 * Wires into nqs_sr_step (real-projected SR path). */
static int vit_c_gradient(const nqs_ansatz_t *a,
                          const int *spins, double *out) {
    vit_c_offsets_t o; vit_c_offsets(a, &o);
    int N = o.N, d = o.d;
    const double *VR = &a->params[o.V_R];
    const double *VI = &a->params[o.V_I];
    const double *WR = &a->params[o.W_out_R];
    const double *WI = &a->params[o.W_out_I];

    vit_c_workspace_t ws;
    if (vit_c_workspace_init(&ws, N, d) != 0) return -1;
    double dummy_la, dummy_arg;
    vit_c_forward(a, spins, &ws, &dummy_la, &dummy_arg);

    memset(out, 0, (size_t)a->num_params * sizeof(double));
    /* ∂ Re(L) / ∂ b_out_R = 1 ; ∂ Re(L) / ∂ b_out_I = 0 (already zeroed) */
    out[o.b_out_R] = 1.0;
    /* ∂ Re(L) / ∂ W_out_R[k] = Σ_i Re(o_i)[k]
     * ∂ Re(L) / ∂ W_out_I[k] = - Σ_i Im(o_i)[k]            */
    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++) {
            out[o.W_out_R + k] += ws.oR[i*d + k];
            out[o.W_out_I + k] -= ws.oI[i*d + k];
        }

    /* s_pred[j] = Σ_i P_ij  (column sum of attention) */
    double *s_pred = calloc((size_t)N, sizeof(double));
    /* Re(W_out · v_j) and Re(W_out · o_i) */
    double *Wv_re = calloc((size_t)N, sizeof(double));
    double *Wo_re = calloc((size_t)N, sizeof(double));
    if (!s_pred || !Wv_re || !Wo_re) {
        free(s_pred); free(Wv_re); free(Wo_re);
        vit_c_workspace_free(&ws); return -1;
    }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) s_pred[j] += ws.P[i*N + j];
    for (int j = 0; j < N; j++) {
        double s = 0.0;
        for (int k = 0; k < d; k++) s += WR[k] * ws.vR[j*d + k] - WI[k] * ws.vI[j*d + k];
        Wv_re[j] = s;
    }
    for (int i = 0; i < N; i++) {
        double s = 0.0;
        for (int k = 0; k < d; k++) s += WR[k] * ws.oR[i*d + k] - WI[k] * ws.oI[i*d + k];
        Wo_re[i] = s;
    }

    /* ∂ Re(L) / ∂ S_{ij} = P_ij · (Wv_re[j] − Wo_re[i])   */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int rel = ((i - j) % N + N) % N;
            out[o.attn + rel] += ws.P[i*N + j] * (Wv_re[j] - Wo_re[i]);
        }
    }

    /* ev_R[l] = Σ_j s_pred[j] e_j_R[l],  ev_I[l] = Σ_j s_pred[j] e_j_I[l] */
    double *ev_R = calloc((size_t)d, sizeof(double));
    double *ev_I = calloc((size_t)d, sizeof(double));
    if (!ev_R || !ev_I) {
        free(s_pred); free(Wv_re); free(Wo_re); free(ev_R); free(ev_I);
        vit_c_workspace_free(&ws); return -1;
    }
    for (int j = 0; j < N; j++)
        for (int l = 0; l < d; l++) {
            ev_R[l] += s_pred[j] * ws.eR[j*d + l];
            ev_I[l] += s_pred[j] * ws.eI[j*d + l];
        }
    /* ∂ Re(L) / ∂ V_R[k][l] =  W_out_R[k] · ev_R[l] − W_out_I[k] · ev_I[l]
     * ∂ Re(L) / ∂ V_I[k][l] = −W_out_R[k] · ev_I[l] − W_out_I[k] · ev_R[l] */
    for (int k = 0; k < d; k++) {
        for (int l = 0; l < d; l++) {
            out[o.V_R + k*d + l] =  WR[k] * ev_R[l] - WI[k] * ev_I[l];
            out[o.V_I + k*d + l] = -WR[k] * ev_I[l] - WI[k] * ev_R[l];
        }
    }

    /* alpha[l] = (V_R^T W_out_R - V_I^T W_out_I)[l]
     * beta[l]  = (V_I^T W_out_R + V_R^T W_out_I)[l]                       */
    double *alpha = calloc((size_t)d, sizeof(double));
    double *beta  = calloc((size_t)d, sizeof(double));
    if (!alpha || !beta) {
        free(s_pred); free(Wv_re); free(Wo_re); free(ev_R); free(ev_I);
        free(alpha); free(beta);
        vit_c_workspace_free(&ws); return -1;
    }
    for (int l = 0; l < d; l++) {
        for (int k = 0; k < d; k++) {
            alpha[l] += VR[k*d + l] * WR[k] - VI[k*d + l] * WI[k];
            beta[l]  += VI[k*d + l] * WR[k] + VR[k*d + l] * WI[k];
        }
    }
    /* ∂ Re(L) / ∂ w_emb_R[l] =  alpha[l] · Σ_j s_pred[j] · s_j
     * ∂ Re(L) / ∂ w_emb_I[l] = −beta[l]  · Σ_j s_pred[j] · s_j
     * ∂ Re(L) / ∂ b_emb_R[l] =  alpha[l] · Σ_j s_pred[j]
     * ∂ Re(L) / ∂ b_emb_I[l] = −beta[l]  · Σ_j s_pred[j]                   */
    double sum_sp_s = 0.0, sum_sp = 0.0;
    for (int j = 0; j < N; j++) {
        sum_sp_s += s_pred[j] * (double)spins[j];
        sum_sp   += s_pred[j];
    }
    for (int l = 0; l < d; l++) {
        out[o.w_emb_R + l] =  alpha[l] * sum_sp_s;
        out[o.w_emb_I + l] = -beta[l]  * sum_sp_s;
        out[o.b_emb_R + l] =  alpha[l] * sum_sp;
        out[o.b_emb_I + l] = -beta[l]  * sum_sp;
    }

    free(s_pred); free(Wv_re); free(Wo_re); free(ev_R); free(ev_I);
    free(alpha); free(beta);
    vit_c_workspace_free(&ws);
    return 0;
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
            case NQS_ANSATZ_FACTORED_VIT:
                acc = vit_log_amp(a, spins);
                break;
            case NQS_ANSATZ_FACTORED_VIT_COMPLEX:
                vit_c_log_amp(a, spins, &acc, &arg);
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
        case NQS_ANSATZ_FACTORED_VIT: return vit_gradient(a, spins, out_grad);
        case NQS_ANSATZ_FACTORED_VIT_COMPLEX: return vit_c_gradient(a, spins, out_grad);
        default:                     return mf_gradient(a, spins, out_grad);
    }
}

int nqs_ansatz_is_complex(const nqs_ansatz_t *a) {
    if (!a) return 0;
    return (a->kind == NQS_ANSATZ_COMPLEX_RBM ||
            a->kind == NQS_ANSATZ_FACTORED_VIT_COMPLEX) ? 1 : 0;
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
