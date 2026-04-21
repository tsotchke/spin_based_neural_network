/*
 * src/nqs/nqs_gradient.c
 *
 * Local-energy estimators. For each supported Hamiltonian we enumerate
 * the off-diagonal connections s → s' that the Hamiltonian matrix
 * elements induce, evaluate the ψ(s') / ψ(s) ratio via the ansatz, and
 * sum them into E_loc(s).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "nqs/nqs_gradient.h"

/* Map 2D (x, y) → flat index, row-major. */
static inline int flat_idx(int x, int y, int size_y) {
    return x * size_y + y;
}

/* ψ(s') / ψ(s) as a real number. For a strictly real-valued ansatz the
 * phase is 0 everywhere and this reduces to |ψ'|/|ψ|. When a sign-
 * structure wrapper (e.g. Marshall for Heisenberg AFM) is installed,
 * the ansatz reports arg = 0 or π per configuration and the ratio
 * picks up the cos(Δ arg) = ±1 factor accordingly. */
static double amplitude_ratio(const int *spins_new, int num_sites,
                              double current_log_abs, double current_arg,
                              nqs_log_amp_fn_t log_amp, void *user) {
    double new_log_abs, new_arg;
    log_amp(spins_new, num_sites, user, &new_log_abs, &new_arg);
    return exp(new_log_abs - current_log_abs) * cos(new_arg - current_arg);
}

/* Full complex ψ(s')/ψ(s) as (re, im) pair. For real wavefunctions
 * this reduces to (cos(Δarg) · |ψ'|/|ψ|, 0); for complex wavefunctions
 * both components are populated. Needed by holomorphic SR. */
static void amplitude_ratio_complex(const int *spins_new, int num_sites,
                                     double current_log_abs, double current_arg,
                                     nqs_log_amp_fn_t log_amp, void *user,
                                     double *out_re, double *out_im) {
    double new_log_abs, new_arg;
    log_amp(spins_new, num_sites, user, &new_log_abs, &new_arg);
    double mag = exp(new_log_abs - current_log_abs);
    double dth = new_arg - current_arg;
    *out_re = mag * cos(dth);
    *out_im = mag * sin(dth);
}

/* TFIM: H = -J Σ_<ij> σ^z_i σ^z_j - Γ Σ_i σ^x_i.
 * Diagonal: -J × (sum of neighbor products on the lattice).
 * Off-diagonal (single flip at each site): -Γ per flipped connection. */
static double local_energy_tfim(const nqs_config_t *cfg,
                                int size_x, int size_y,
                                const int *spins,
                                nqs_log_amp_fn_t log_amp, void *user,
                                double current_log_abs, double current_arg) {
    double J = cfg->j_coupling;
    double G = cfg->transverse_field;
    int N = size_x * size_y;
    double diag = 0.0;

    /* Diagonal ZZ term, open boundary. */
    for (int x = 0; x < size_x; x++) {
        for (int y = 0; y < size_y; y++) {
            int s = spins[flat_idx(x, y, size_y)];
            if (x + 1 < size_x) diag += -J * (double)(s * spins[flat_idx(x + 1, y, size_y)]);
            if (y + 1 < size_y) diag += -J * (double)(s * spins[flat_idx(x, y + 1, size_y)]);
        }
    }

    /* Off-diagonal X term: single-site flips, each contributes -Γ · ψ(s_flipped) / ψ(s). */
    double off = 0.0;
    int *scratch = malloc((size_t)N * sizeof(int));
    if (!scratch) return diag;  /* graceful degradation */
    memcpy(scratch, spins, (size_t)N * sizeof(int));
    for (int i = 0; i < N; i++) {
        scratch[i] = -scratch[i];
        double r = amplitude_ratio(scratch, N, current_log_abs, current_arg, log_amp, user);
        off += -G * r;
        scratch[i] = -scratch[i];
    }
    free(scratch);
    return diag + off;
}

/* Heisenberg: H = J Σ_<ij> S_i · S_j = J Σ [ (1/2)(S+_i S-_j + S-_i S+_j) + S^z_i S^z_j ].
 * Diagonal (ZZ):   J/4 × Σ_<ij> s_i s_j  (since S^z = ±1/2 for S=1/2 spins).
 * Off-diagonal (XY): for each antiparallel pair, flip both; ratio contribution J/2. */
static double local_energy_xxz(const nqs_config_t *cfg,
                                int size_x, int size_y,
                                const int *spins,
                                nqs_log_amp_fn_t log_amp, void *user,
                                double current_log_abs, double current_arg,
                                double Jxy, double Jz) {
    (void)cfg;  /* reserved for future symmetry/kernel switches */
    int N = size_x * size_y;
    double diag = 0.0;
    double off  = 0.0;
    int *scratch = malloc((size_t)N * sizeof(int));
    if (!scratch) return 0.0;
    memcpy(scratch, spins, (size_t)N * sizeof(int));

    for (int x = 0; x < size_x; x++) {
        for (int y = 0; y < size_y; y++) {
            int a = flat_idx(x, y, size_y);
            int sa = spins[a];
            int neighbors[2];
            int nb = 0;
            if (x + 1 < size_x) neighbors[nb++] = flat_idx(x + 1, y, size_y);
            if (y + 1 < size_y) neighbors[nb++] = flat_idx(x, y + 1, size_y);
            for (int k = 0; k < nb; k++) {
                int b = neighbors[k];
                int sb = spins[b];
                /* ZZ term: Jz/4 · s_a s_b */
                diag += 0.25 * Jz * (double)(sa * sb);
                if (sa == -sb) {
                    /* Flip both — generates the XY off-diagonal
                     * contribution Jxy/2 · (S+S- + S-S+). */
                    scratch[a] = -sa;
                    scratch[b] = -sb;
                    double r = amplitude_ratio(scratch, N, current_log_abs, current_arg, log_amp, user);
                    off += 0.5 * Jxy * r;
                    scratch[a] = sa;
                    scratch[b] = sb;
                }
            }
        }
    }
    free(scratch);
    return diag + off;
}

static double local_energy_heisenberg(const nqs_config_t *cfg,
                                      int size_x, int size_y,
                                      const int *spins,
                                      nqs_log_amp_fn_t log_amp, void *user,
                                      double current_log_abs, double current_arg) {
    double J = cfg->j_coupling;
    return local_energy_xxz(cfg, size_x, size_y, spins, log_amp, user,
                             current_log_abs, current_arg, J, J);
}

/* Walk both diagonal next-nearest pairs:
 *   (x,y)–(x+1,y+1)    (positive diagonal)
 *   (x+1,y)–(x,y+1)    (negative diagonal, i.e. (x,y+1)–(x+1,y))
 * Runs `body` for each pair; used to share code between real and
 * complex J1-J2 kernels. */
static void j1j2_iterate_diagonals(int size_x, int size_y,
                                    void (*body)(int a, int b, void *ud),
                                    void *ud) {
    for (int x = 0; x + 1 < size_x; x++) {
        for (int y = 0; y + 1 < size_y; y++) {
            int a = x * size_y + y;
            int b = (x + 1) * size_y + (y + 1);
            body(a, b, ud);
            int c = x * size_y + (y + 1);
            int d = (x + 1) * size_y + y;
            body(c, d, ud);
        }
    }
}

/* J1-J2: H = J1 Σ_<ij> S_i·S_j + J2 Σ_<<ij>> S_i·S_j.
 * Iterates BOTH diagonals: (x,y)-(x+1,y+1) and (x,y+1)-(x+1,y). */
typedef struct {
    const int *spins;
    int *scratch;
    int N;
    double current_log_abs, current_arg;
    nqs_log_amp_fn_t log_amp;
    void *user;
    double J2;
    double *h2_diag;
    double *h2_off;
} j1j2_real_ctx_t;

static void j1j2_real_body(int a, int b, void *ud) {
    j1j2_real_ctx_t *c = (j1j2_real_ctx_t *)ud;
    int sa = c->spins[a], sb = c->spins[b];
    *c->h2_diag += 0.25 * c->J2 * (double)(sa * sb);
    if (sa == -sb) {
        c->scratch[a] = -sa; c->scratch[b] = -sb;
        double r = amplitude_ratio(c->scratch, c->N, c->current_log_abs,
                                    c->current_arg, c->log_amp, c->user);
        *c->h2_off += 0.5 * c->J2 * r;
        c->scratch[a] = sa; c->scratch[b] = sb;
    }
}

static double local_energy_j1j2(const nqs_config_t *cfg,
                                int size_x, int size_y,
                                const int *spins,
                                nqs_log_amp_fn_t log_amp, void *user,
                                double current_log_abs, double current_arg) {
    double h1 = local_energy_heisenberg(cfg, size_x, size_y, spins,
                                        log_amp, user, current_log_abs, current_arg);
    int N = size_x * size_y;
    double h2_diag = 0.0, h2_off = 0.0;
    int *scratch = malloc((size_t)N * sizeof(int));
    if (!scratch) return h1;
    memcpy(scratch, spins, (size_t)N * sizeof(int));
    j1j2_real_ctx_t ctx = {spins, scratch, N, current_log_abs, current_arg,
                            log_amp, user, cfg->j2_coupling, &h2_diag, &h2_off};
    j1j2_iterate_diagonals(size_x, size_y, j1j2_real_body, &ctx);
    free(scratch);
    return h1 + h2_diag + h2_off;
}

/* Kitaev honeycomb on a brick-wall representation of the Lx × Ly
 * lattice. Bonds are assigned by parity of (x+y):
 *   horizontal link (x,y)-(x+1,y):   x-bond if (x+y) even, y-bond if odd
 *   vertical link   (x,y)-(x,y+1):   z-bond always
 * σ^x σ^x connects |s_a, s_b⟩ → |-s_a, -s_b⟩ (full flip), coefficient -J_x.
 * σ^y σ^y also flips both but with a phase -s_a s_b, coefficient -J_y·(-s_a s_b).
 *   → combined sign (-s_a s_b) from σ^y. Net: -J_y · (-s_a s_b) · ⟨s'|σ^y σ^y|s⟩
 * Actually ⟨s'|σ^y_i σ^y_j|s⟩ = (i s_i)(i s_j) · δ_{s'_i, -s_i} δ_{s'_j, -s_j}
 *   = -s_i s_j  ·  indicator of double flip.
 * So H^y_{s,s'} = -J_y · (-s_i s_j) = +J_y s_i s_j when s' = s ⊕ {i,j}.
 * σ^z σ^z is diagonal: -J_z s_i s_j.
 *
 * Config: cfg->j_coupling = J_x, cfg->transverse_field = J_y,
 *         cfg->j2_coupling = J_z. */
static double local_energy_kitaev(const nqs_config_t *cfg,
                                   int size_x, int size_y,
                                   const int *spins,
                                   nqs_log_amp_fn_t log_amp, void *user,
                                   double current_log_abs, double current_arg) {
    double Jx = cfg->j_coupling;
    double Jy = cfg->transverse_field;
    double Jz = cfg->j2_coupling;
    int N = size_x * size_y;
    double diag = 0.0, off = 0.0;
    int *scratch = malloc((size_t)N * sizeof(int));
    if (!scratch) return 0.0;
    memcpy(scratch, spins, (size_t)N * sizeof(int));

    for (int x = 0; x < size_x; x++) for (int y = 0; y < size_y; y++) {
        int a = flat_idx(x, y, size_y);
        int sa = spins[a];
        /* Horizontal bond (x,y)-(x+1,y): x-bond if (x+y) even, else y-bond. */
        if (x + 1 < size_x) {
            int b = flat_idx(x + 1, y, size_y);
            int sb = spins[b];
            int is_x_bond = (((x + y) & 1) == 0);
            double J = is_x_bond ? Jx : Jy;
            scratch[a] = -sa; scratch[b] = -sb;
            double r = amplitude_ratio(scratch, N, current_log_abs,
                                        current_arg, log_amp, user);
            scratch[a] = sa; scratch[b] = sb;
            if (is_x_bond) off += -J * r;
            else           off += +J * (double)(sa * sb) * r;
        }
        /* Vertical bond (x,y)-(x,y+1): z-bond. */
        if (y + 1 < size_y) {
            int b = flat_idx(x, y + 1, size_y);
            int sb = spins[b];
            diag += -Jz * (double)(sa * sb);
        }
    }
    free(scratch);
    return diag + off;
}

double nqs_local_energy(const nqs_config_t *cfg,
                        int size_x, int size_y,
                        const int *spins,
                        nqs_log_amp_fn_t log_amp,
                        void *log_amp_user) {
    if (!cfg || !spins || !log_amp) return 0.0;
    int N = size_x * size_y;

    /* Seed ψ(s) for the ratio denominator once per call. */
    double cur_log_abs, cur_arg;
    log_amp(spins, N, log_amp_user, &cur_log_abs, &cur_arg);

    switch (cfg->hamiltonian) {
        case NQS_HAM_TFIM:
            return local_energy_tfim(cfg, size_x, size_y, spins,
                                     log_amp, log_amp_user, cur_log_abs, cur_arg);
        case NQS_HAM_HEISENBERG:
            return local_energy_heisenberg(cfg, size_x, size_y, spins,
                                           log_amp, log_amp_user, cur_log_abs, cur_arg);
        case NQS_HAM_XXZ:
            return local_energy_xxz(cfg, size_x, size_y, spins,
                                     log_amp, log_amp_user, cur_log_abs, cur_arg,
                                     cfg->j_coupling, cfg->j_z_coupling);
        case NQS_HAM_J1_J2:
            return local_energy_j1j2(cfg, size_x, size_y, spins,
                                     log_amp, log_amp_user, cur_log_abs, cur_arg);
        case NQS_HAM_KITAEV_HONEYCOMB:
            return local_energy_kitaev(cfg, size_x, size_y, spins,
                                        log_amp, log_amp_user, cur_log_abs, cur_arg);
        default:
            return local_energy_tfim(cfg, size_x, size_y, spins,
                                     log_amp, log_amp_user, cur_log_abs, cur_arg);
    }
}

/* -------------------- complex local-energy path ---------------------- */

static void local_energy_tfim_complex(const nqs_config_t *cfg,
                                      int size_x, int size_y,
                                      const int *spins,
                                      nqs_log_amp_fn_t log_amp, void *user,
                                      double cur_log_abs, double cur_arg,
                                      double *out_re, double *out_im) {
    double J = cfg->j_coupling, G = cfg->transverse_field;
    int N = size_x * size_y;
    double diag = 0.0;
    for (int x = 0; x < size_x; x++) for (int y = 0; y < size_y; y++) {
        int s = spins[flat_idx(x, y, size_y)];
        if (x + 1 < size_x) diag += -J * (double)(s * spins[flat_idx(x + 1, y, size_y)]);
        if (y + 1 < size_y) diag += -J * (double)(s * spins[flat_idx(x, y + 1, size_y)]);
    }
    double off_re = 0.0, off_im = 0.0;
    int *scratch = malloc((size_t)N * sizeof(int));
    memcpy(scratch, spins, (size_t)N * sizeof(int));
    for (int i = 0; i < N; i++) {
        scratch[i] = -scratch[i];
        double r, im;
        amplitude_ratio_complex(scratch, N, cur_log_abs, cur_arg, log_amp, user, &r, &im);
        off_re += -G * r;
        off_im += -G * im;
        scratch[i] = -scratch[i];
    }
    free(scratch);
    *out_re = diag + off_re;
    *out_im = off_im;
}

static void local_energy_xxz_complex(const nqs_config_t *cfg,
                                      int size_x, int size_y,
                                      const int *spins,
                                      nqs_log_amp_fn_t log_amp, void *user,
                                      double cur_log_abs, double cur_arg,
                                      double Jxy, double Jz,
                                      double *out_re, double *out_im) {
    (void)cfg;  /* reserved for future symmetry/kernel switches */
    int N = size_x * size_y;
    double diag = 0.0;
    double off_re = 0.0, off_im = 0.0;
    int *scratch = malloc((size_t)N * sizeof(int));
    memcpy(scratch, spins, (size_t)N * sizeof(int));
    for (int x = 0; x < size_x; x++) for (int y = 0; y < size_y; y++) {
        int a = flat_idx(x, y, size_y);
        int sa = spins[a];
        int neighbors[2];
        int nb = 0;
        if (x + 1 < size_x) neighbors[nb++] = flat_idx(x + 1, y, size_y);
        if (y + 1 < size_y) neighbors[nb++] = flat_idx(x, y + 1, size_y);
        for (int k = 0; k < nb; k++) {
            int b = neighbors[k];
            int sb = spins[b];
            diag += 0.25 * Jz * (double)(sa * sb);
            if (sa == -sb) {
                scratch[a] = -sa; scratch[b] = -sb;
                double r, im;
                amplitude_ratio_complex(scratch, N, cur_log_abs, cur_arg, log_amp, user, &r, &im);
                off_re += 0.5 * Jxy * r;
                off_im += 0.5 * Jxy * im;
                scratch[a] = sa; scratch[b] = sb;
            }
        }
    }
    free(scratch);
    *out_re = diag + off_re;
    *out_im = off_im;
}

static void local_energy_heisenberg_complex(const nqs_config_t *cfg,
                                            int size_x, int size_y,
                                            const int *spins,
                                            nqs_log_amp_fn_t log_amp, void *user,
                                            double cur_log_abs, double cur_arg,
                                            double *out_re, double *out_im) {
    double J = cfg->j_coupling;
    local_energy_xxz_complex(cfg, size_x, size_y, spins, log_amp, user,
                              cur_log_abs, cur_arg, J, J, out_re, out_im);
}

typedef struct {
    const int *spins;
    int *scratch;
    int N;
    double cur_log_abs, cur_arg;
    nqs_log_amp_fn_t log_amp;
    void *user;
    double J2;
    double *h2_diag;
    double *h2_re;
    double *h2_im;
} j1j2_cplx_ctx_t;

static void j1j2_cplx_body(int a, int b, void *ud) {
    j1j2_cplx_ctx_t *c = (j1j2_cplx_ctx_t *)ud;
    int sa = c->spins[a], sb = c->spins[b];
    *c->h2_diag += 0.25 * c->J2 * (double)(sa * sb);
    if (sa == -sb) {
        c->scratch[a] = -sa; c->scratch[b] = -sb;
        double r, im;
        amplitude_ratio_complex(c->scratch, c->N, c->cur_log_abs, c->cur_arg,
                                 c->log_amp, c->user, &r, &im);
        *c->h2_re += 0.5 * c->J2 * r;
        *c->h2_im += 0.5 * c->J2 * im;
        c->scratch[a] = sa; c->scratch[b] = sb;
    }
}

static void local_energy_j1j2_complex(const nqs_config_t *cfg,
                                       int size_x, int size_y,
                                       const int *spins,
                                       nqs_log_amp_fn_t log_amp, void *user,
                                       double cur_log_abs, double cur_arg,
                                       double *out_re, double *out_im) {
    double h1_re, h1_im;
    local_energy_heisenberg_complex(cfg, size_x, size_y, spins,
                                     log_amp, user, cur_log_abs, cur_arg,
                                     &h1_re, &h1_im);
    int N = size_x * size_y;
    double h2_diag = 0.0, h2_re = 0.0, h2_im = 0.0;
    int *scratch = malloc((size_t)N * sizeof(int));
    memcpy(scratch, spins, (size_t)N * sizeof(int));
    j1j2_cplx_ctx_t ctx = {spins, scratch, N, cur_log_abs, cur_arg,
                            log_amp, user, cfg->j2_coupling,
                            &h2_diag, &h2_re, &h2_im};
    j1j2_iterate_diagonals(size_x, size_y, (void (*)(int, int, void *))j1j2_cplx_body, &ctx);
    free(scratch);
    *out_re = h1_re + h2_diag + h2_re;
    *out_im = h1_im + h2_im;
}

static void local_energy_kitaev_complex(const nqs_config_t *cfg,
                                         int size_x, int size_y,
                                         const int *spins,
                                         nqs_log_amp_fn_t log_amp, void *user,
                                         double cur_log_abs, double cur_arg,
                                         double *out_re, double *out_im) {
    double Jx = cfg->j_coupling, Jy = cfg->transverse_field, Jz = cfg->j2_coupling;
    int N = size_x * size_y;
    double diag = 0.0, off_re = 0.0, off_im = 0.0;
    int *scratch = malloc((size_t)N * sizeof(int));
    memcpy(scratch, spins, (size_t)N * sizeof(int));
    for (int x = 0; x < size_x; x++) for (int y = 0; y < size_y; y++) {
        int a = flat_idx(x, y, size_y);
        int sa = spins[a];
        if (x + 1 < size_x) {
            int b = flat_idx(x + 1, y, size_y);
            int sb = spins[b];
            int is_x_bond = (((x + y) & 1) == 0);
            double J = is_x_bond ? Jx : Jy;
            scratch[a] = -sa; scratch[b] = -sb;
            double r, im;
            amplitude_ratio_complex(scratch, N, cur_log_abs, cur_arg,
                                     log_amp, user, &r, &im);
            scratch[a] = sa; scratch[b] = sb;
            double coef = is_x_bond ? (-J) : (+J * (double)(sa * sb));
            off_re += coef * r;
            off_im += coef * im;
        }
        if (y + 1 < size_y) {
            int b = flat_idx(x, y + 1, size_y);
            int sb = spins[b];
            diag += -Jz * (double)(sa * sb);
        }
    }
    free(scratch);
    *out_re = diag + off_re;
    *out_im = off_im;
}

void nqs_local_energy_complex(const nqs_config_t *cfg,
                               int size_x, int size_y,
                               const int *spins,
                               nqs_log_amp_fn_t log_amp,
                               void *log_amp_user,
                               double *out_re, double *out_im) {
    if (!cfg || !spins || !log_amp || !out_re || !out_im) {
        if (out_re) *out_re = 0; if (out_im) *out_im = 0;
        return;
    }
    int N = size_x * size_y;
    double cur_log_abs, cur_arg;
    log_amp(spins, N, log_amp_user, &cur_log_abs, &cur_arg);
    switch (cfg->hamiltonian) {
        case NQS_HAM_TFIM:
            local_energy_tfim_complex(cfg, size_x, size_y, spins,
                                       log_amp, log_amp_user,
                                       cur_log_abs, cur_arg,
                                       out_re, out_im);
            return;
        case NQS_HAM_HEISENBERG:
            local_energy_heisenberg_complex(cfg, size_x, size_y, spins,
                                             log_amp, log_amp_user,
                                             cur_log_abs, cur_arg,
                                             out_re, out_im);
            return;
        case NQS_HAM_XXZ:
            local_energy_xxz_complex(cfg, size_x, size_y, spins,
                                      log_amp, log_amp_user,
                                      cur_log_abs, cur_arg,
                                      cfg->j_coupling, cfg->j_z_coupling,
                                      out_re, out_im);
            return;
        case NQS_HAM_J1_J2:
            local_energy_j1j2_complex(cfg, size_x, size_y, spins,
                                       log_amp, log_amp_user,
                                       cur_log_abs, cur_arg,
                                       out_re, out_im);
            return;
        case NQS_HAM_KITAEV_HONEYCOMB:
            local_energy_kitaev_complex(cfg, size_x, size_y, spins,
                                         log_amp, log_amp_user,
                                         cur_log_abs, cur_arg,
                                         out_re, out_im);
            return;
        default:
            local_energy_tfim_complex(cfg, size_x, size_y, spins,
                                       log_amp, log_amp_user,
                                       cur_log_abs, cur_arg,
                                       out_re, out_im);
            return;
    }
}

void nqs_local_energy_batch_complex(const nqs_config_t *cfg,
                                     int size_x, int size_y,
                                     const int *spins_batch, int batch_size,
                                     nqs_log_amp_fn_t log_amp,
                                     void *log_amp_user,
                                     double *out_re, double *out_im) {
    int N = size_x * size_y;
    for (int i = 0; i < batch_size; i++) {
        nqs_local_energy_complex(cfg, size_x, size_y,
                                  &spins_batch[(size_t)i * (size_t)N],
                                  log_amp, log_amp_user,
                                  &out_re[i], &out_im[i]);
    }
}

void nqs_local_energy_batch(const nqs_config_t *cfg,
                            int size_x, int size_y,
                            const int *spins_batch, int batch_size,
                            nqs_log_amp_fn_t log_amp,
                            void *log_amp_user,
                            double *out_energies) {
    if (!cfg || !spins_batch || !out_energies || batch_size <= 0) return;
    int N = size_x * size_y;
    for (int i = 0; i < batch_size; i++) {
        out_energies[i] = nqs_local_energy(cfg, size_x, size_y,
                                           &spins_batch[(size_t)i * (size_t)N],
                                           log_amp, log_amp_user);
    }
}
