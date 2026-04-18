/*
 * src/nn_backend.c
 *
 * Polymorphic wrapper over the legacy in-tree MLP and an engine-backed
 * path (transformer / KAN / reasoning). In v0.4 the engine path is a stub
 * that either falls back to legacy (if no engine is wired in) or reports
 * "not yet implemented" (engine present but pillar work in v0.5 hasn't
 * populated the forward/train kernels).
 *
 * Pillar P1.1 (NQS, v0.5) replaces the engine-path stubs with real
 * transformer forward/backward passes via include/engine_adapter.h.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>  /* strcasecmp */

#include "nn_backend.h"

#ifdef SPIN_NN_HAS_ENGINE
#include "engine_adapter.h"
#endif

struct spin_nn {
    nn_backend_kind_t backend;
    NeuralNetwork    *legacy;   /* used when backend == NN_BACKEND_LEGACY */
    /* engine state is added in v0.5 (P1.1) — handle to the plugged-in
     * engine's model object (Eshkol-native NN-engine handle, Noesis
     * handle once released, etc.). */
};

const char *nn_backend_name(nn_backend_kind_t kind) {
    switch (kind) {
        case NN_BACKEND_LEGACY: return "legacy";
        case NN_BACKEND_ENGINE: return "engine";
        default:                return "unknown";
    }
}

nn_backend_kind_t nn_backend_parse(const char *name, int *ok) {
    if (ok) *ok = 1;
    if (!name) { if (ok) *ok = 0; return NN_BACKEND_LEGACY; }
    if (strcasecmp(name, "legacy") == 0) return NN_BACKEND_LEGACY;
    if (strcasecmp(name, "engine") == 0) return NN_BACKEND_ENGINE;
    if (ok) *ok = 0;
    return NN_BACKEND_LEGACY;
}

spin_nn_t *spin_nn_create(nn_backend_kind_t backend,
                          int input_size,
                          int num_hidden_layers,
                          int neurons_per_layer,
                          int output_size,
                          int activation_function) {
    spin_nn_t *nn = calloc(1, sizeof(*nn));
    if (!nn) return NULL;

    if (backend == NN_BACKEND_ENGINE) {
#ifdef SPIN_NN_HAS_ENGINE
        /* v0.4: engine backend forward/train not yet wired. Warn and fall
         * back to legacy so the pipeline still runs. Pillar P1.1 lands the
         * real implementation and this branch becomes authoritative. */
        fprintf(stderr,
                "nn_backend: engine backend stubbed in v0.4 (build tag %s); "
                "falling back to legacy MLP. Real training arrives with "
                "pillar P1.1 in v0.5.\n",
                engine_adapter_build_version());
#else
        fprintf(stderr,
                "nn_backend: built without SPIN_NN_HAS_ENGINE; "
                "falling back to legacy MLP.\n");
#endif
        backend = NN_BACKEND_LEGACY;
    }

    nn->backend = backend;
    if (backend == NN_BACKEND_LEGACY) {
        nn->legacy = create_neural_network(input_size,
                                           num_hidden_layers,
                                           neurons_per_layer,
                                           output_size,
                                           activation_function);
        if (!nn->legacy) { free(nn); return NULL; }
    }
    return nn;
}

void spin_nn_free(spin_nn_t *nn) {
    if (!nn) return;
    if (nn->legacy) free_neural_network(nn->legacy);
    free(nn);
}

double *spin_nn_forward(spin_nn_t *nn, double *input) {
    if (!nn) return NULL;
    switch (nn->backend) {
        case NN_BACKEND_LEGACY: return nn->legacy ? forward(nn->legacy, input) : NULL;
        case NN_BACKEND_ENGINE: return NULL; /* unreachable in v0.4 (fell back) */
    }
    return NULL;
}

int spin_nn_train(spin_nn_t *nn, double *input, double *target, double learning_rate) {
    if (!nn) return -1;
    switch (nn->backend) {
        case NN_BACKEND_LEGACY:
            if (!nn->legacy) return -1;
            train(nn->legacy, input, target, learning_rate);
            return 0;
        case NN_BACKEND_ENGINE:
            return -1;
    }
    return -1;
}

nn_backend_kind_t spin_nn_backend(const spin_nn_t *nn) {
    return nn ? nn->backend : NN_BACKEND_LEGACY;
}

NeuralNetwork *spin_nn_legacy_handle(spin_nn_t *nn) {
    if (!nn || nn->backend != NN_BACKEND_LEGACY) return NULL;
    return nn->legacy;
}
