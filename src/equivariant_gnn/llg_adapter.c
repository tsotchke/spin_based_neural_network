/*
 * src/equivariant_gnn/llg_adapter.c
 */
#include "equivariant_gnn/llg_adapter.h"

void llg_torque_field_fn(const double *m, double *b_eff,
                          long num_sites, void *user_data) {
    const llg_torque_user_t *u = (const llg_torque_user_t *)user_data;
    if (!u || !u->graph || !u->params) return;
    (void)num_sites;
    torque_net_forward(u->graph, m, u->params, b_eff);
}
