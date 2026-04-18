/*
 * include/nn_backend.h
 *
 * Backend-agnostic neural-network handle for spin_based_neural_network.
 *
 * Two backends are supported:
 *   NN_BACKEND_LEGACY — the in-tree MLP (src/neural_network.c). Zero
 *                       external dependencies. This is v0.3 behavior.
 *   NN_BACKEND_ENGINE — transformer / KAN / reasoning models served by an
 *                       external engine via include/engine_adapter.h. The
 *                       concrete engine (Eshkol-native NN engine; or
 *                       Noesis once released) is chosen at build time via
 *                       SPIN_NN_HAS_ENGINE. Without that macro,
 *                       spin_nn_create(ENGINE) falls back to legacy with
 *                       a diagnostic.
 *
 * The two backends share a small API surface so main.c and pillar code can
 * swap implementations with a CLI flag (--nn-backend={legacy,engine}).
 */
#ifndef NN_BACKEND_H
#define NN_BACKEND_H

#include "neural_network.h"  /* legacy NeuralNetwork type */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NN_BACKEND_LEGACY = 0,
    NN_BACKEND_ENGINE = 1
} nn_backend_kind_t;

typedef struct spin_nn spin_nn_t;

/* Parse a backend string. Accepts "legacy", "engine" (case-insensitive).
 * Unknown strings return NN_BACKEND_LEGACY with *ok = 0. */
nn_backend_kind_t nn_backend_parse(const char *name, int *ok);
const char       *nn_backend_name (nn_backend_kind_t kind);

/* Create a network on the requested backend. Parameters mirror
 * create_neural_network() in neural_network.h. Returns NULL on failure. */
spin_nn_t *spin_nn_create(nn_backend_kind_t backend,
                          int input_size,
                          int num_hidden_layers,
                          int neurons_per_layer,
                          int output_size,
                          int activation_function);

void spin_nn_free(spin_nn_t *nn);

/* Forward pass. Returns NULL on error; returned pointer is owned by the
 * network and valid until the next forward/train call. */
double *spin_nn_forward(spin_nn_t *nn, double *input);

/* Single-step train. Returns 0 on success. */
int spin_nn_train(spin_nn_t *nn, double *input, double *target, double learning_rate);

/* Which backend a network is actually using (may differ from the request
 * if the engine was unavailable and the bridge fell back to legacy). */
nn_backend_kind_t spin_nn_backend(const spin_nn_t *nn);

/* Direct access to the underlying legacy MLP, or NULL if this isn't a
 * legacy-backed network. Exposed so callers that haven't been ported to
 * the polymorphic API yet keep working. */
NeuralNetwork *spin_nn_legacy_handle(spin_nn_t *nn);

#ifdef __cplusplus
}
#endif

#endif /* NN_BACKEND_H */
