/*
 * include/equivariant_gnn/llg_adapter.h
 *
 * Bridges the equivariant torque-net into the LLG integrator's
 * `field_fn` callback slot. Given a graph + learnable parameters, the
 * adapter writes the torque per site into the llg_effective_field_fn
 * buffer at each integrator step.
 *
 * Because both the torque network and the LLG equation treat their
 * magnetization argument as an SO(3) vector, swapping the analytic
 * B_eff for a learned torque preserves the rotational symmetry of the
 * equations of motion. Smoke test: starting from a ferromagnetic
 * state, a torque ∝ (m × ẑ) alone should drive uniform precession at
 * a frequency determined by the radial and vector weights.
 */
#ifndef EQUIVARIANT_GNN_LLG_ADAPTER_H
#define EQUIVARIANT_GNN_LLG_ADAPTER_H

#include "equivariant_gnn/torque_net.h"
#include "llg/llg.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const torque_net_graph_t  *graph;
    const torque_net_params_t *params;
} llg_torque_user_t;

/* llg_effective_field_fn_t-compatible: writes τ(m) into b_eff. */
void llg_torque_field_fn(const double *m, double *b_eff,
                          long num_sites, void *user_data);

#ifdef __cplusplus
}
#endif

#endif /* EQUIVARIANT_GNN_LLG_ADAPTER_H */
