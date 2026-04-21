/*
 * include/qgtl_bridge.h
 *
 * Lazy bridge to the Quantum Geometric Tensor Library (QGTL), our
 * geometric-quantum-computing framework at
 * github.com/tsotchke/quantum_geometric_tensor.
 *
 * Primary use in this framework: extending stochastic reconfiguration
 * to use QGTL's high-order Fubini–Study metric on the variational
 * manifold. The existing `nqs_sr_step_holomorphic` implements an
 * O(N_p·N_s) matrix-free QGT-vector product; QGTL supplies the
 * higher-order hierarchical-tensor compression that keeps memory
 * sub-quadratic at larger network scales. Dormant by default;
 * enabled via -DSPIN_NN_HAS_QGTL=1.
 *
 * Bridge also provides a route to QGTL's hardware-backend abstraction
 * (IBM / Rigetti / D-Wave interfaces), so experimentally-obtained
 * device noise models can be ingested directly into our QEC decoder
 * training loop.
 */
#ifndef QGTL_BRIDGE_H
#define QGTL_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    QGTL_BRIDGE_OK         =  0,
    QGTL_BRIDGE_EDISABLED  = -1,
    QGTL_BRIDGE_ENOT_READY = -2,
    QGTL_BRIDGE_ELIB       = -3,
    QGTL_BRIDGE_EARG       = -4
} qgtl_bridge_status_t;

/* --- availability ------------------------------------------------- */

int         qgtl_bridge_is_available(void);
const char *qgtl_bridge_version(void);

int qgtl_bridge_init(void);
int qgtl_bridge_shutdown(void);

/* --- QGT extraction ---------------------------------------------- */

/* Compute the quantum geometric tensor G_kl = Re⟨∂_k ψ | ∂_l ψ⟩ for
 * a variational ansatz given its per-sample log-ψ gradients.
 *
 *   batch_grads: (N_samples × N_params), row-major — (Re O, Im O)
 *                concatenated for complex ansätze.
 *   batch_size : N_samples.
 *   num_params : N_params.
 *   out_G      : (N_params × N_params), row-major — writes the real
 *                Fubini–Study metric. Caller-allocated.
 *
 * When enabled, delegates to QGTL's hierarchical-tensor path which
 * returns the same numeric result but with O(N_p · log N_p) memory
 * instead of the dense O(N_p²). When disabled, returns EDISABLED —
 * caller should fall back to the existing matrix-free QGT-vector
 * product in nqs_sr_step_holomorphic. */
int qgtl_bridge_compute_qgt(const double *batch_grads,
                             int batch_size, int num_params,
                             double *out_G);

/* --- device backend descriptor ------------------------------------ */

/* Opaque handle referring to a specific hardware backend (IBM,
 * Rigetti, D-Wave). Used to fetch device-specific noise / gate /
 * connectivity data that drives QEC decoder training. */
typedef struct qgtl_device_backend qgtl_device_backend_t;

qgtl_device_backend_t *qgtl_bridge_device_backend_open(const char *backend_name);
void                    qgtl_bridge_device_backend_close(qgtl_device_backend_t *b);

/* Fetch the measured single-qubit depolarising error rate from the
 * device backend. When disabled, writes -1.0 and returns EDISABLED. */
int qgtl_bridge_device_single_qubit_p(qgtl_device_backend_t *b, double *out_p);

#ifdef __cplusplus
}
#endif

#endif /* QGTL_BRIDGE_H */
