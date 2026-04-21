/*
 * src/nqs/nqs_marshall.c
 *
 * Bipartite-lattice Marshall sign rule. Sublattice A = sites where
 * (x + y) is even. Works for any L_x × L_y — for bipartite-incompatible
 * lattices (triangular, kagome) the rule doesn't produce a stoquastic
 * rotation and the wrapper should not be used.
 */
#include <math.h>
#include "nqs/nqs_marshall.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int nqs_marshall_parity(const int *spins, int size_x, int size_y) {
    int n_down_A = 0;
    for (int x = 0; x < size_x; x++) {
        for (int y = 0; y < size_y; y++) {
            if (((x + y) & 1) == 0) {
                int s = spins[x * size_y + y];
                if (s < 0) n_down_A++;
            }
        }
    }
    return n_down_A & 1;
}

void nqs_marshall_log_amp(const int *spins, int num_sites,
                           void *user,
                           double *out_log_abs,
                           double *out_arg) {
    nqs_marshall_wrapper_t *w = (nqs_marshall_wrapper_t *)user;
    double base_log_abs = 0.0, base_arg = 0.0;
    w->base_log_amp(spins, num_sites, w->base_user,
                    &base_log_abs, &base_arg);
    int parity = nqs_marshall_parity(spins, w->size_x, w->size_y);
    if (out_log_abs) *out_log_abs = base_log_abs;
    if (out_arg)     *out_arg     = base_arg + (parity ? M_PI : 0.0);
}
