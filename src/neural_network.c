#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "neural_network.h"

NeuralNetwork* create_neural_network(int input_size, int num_hidden_layers, int neurons_per_layer, int output_size, int activation_function) {
    if (input_size <= 0 || num_hidden_layers < 0 || neurons_per_layer <= 0 || output_size <= 0) {
        fprintf(stderr, "Error: create_neural_network requires positive layer sizes (got in=%d hidden=%d neurons=%d out=%d)\n",
                input_size, num_hidden_layers, neurons_per_layer, output_size);
        return NULL;
    }

    /* calloc zero-inits so free_neural_network can safely walk the struct
     * after any partial-allocation failure on the err: path. */
    NeuralNetwork *nn = calloc(1, sizeof(NeuralNetwork));
    if (!nn) {
        fprintf(stderr, "Failed to allocate memory for NeuralNetwork.\n");
        return NULL;
    }

    nn->num_layers = num_hidden_layers + 2;
    nn->activation_function = activation_function;

    nn->layer_sizes = calloc((size_t)nn->num_layers,      sizeof(int));
    nn->W           = calloc((size_t)nn->num_layers - 1,  sizeof(double *));
    nn->b           = calloc((size_t)nn->num_layers - 1,  sizeof(double *));
    nn->mW          = calloc((size_t)nn->num_layers - 1,  sizeof(double *));
    nn->vW          = calloc((size_t)nn->num_layers - 1,  sizeof(double *));
    nn->mb          = calloc((size_t)nn->num_layers - 1,  sizeof(double *));
    nn->vb          = calloc((size_t)nn->num_layers - 1,  sizeof(double *));
    nn->a           = calloc((size_t)nn->num_layers,      sizeof(double *));
    nn->z           = calloc((size_t)nn->num_layers,      sizeof(double *));
    if (!nn->layer_sizes || !nn->W || !nn->b || !nn->mW || !nn->vW
                         || !nn->mb || !nn->vb || !nn->a || !nn->z) {
        fprintf(stderr, "Error: create_neural_network failed to allocate layer arrays\n");
        goto err;
    }

    nn->layer_sizes[0] = input_size;
    for (int i = 1; i < nn->num_layers - 1; i++) {
        nn->layer_sizes[i] = neurons_per_layer;
    }
    nn->layer_sizes[nn->num_layers - 1] = output_size;

    for (int i = 0; i < nn->num_layers; i++) {
        nn->a[i] = calloc((size_t)nn->layer_sizes[i], sizeof(double));
        nn->z[i] = calloc((size_t)nn->layer_sizes[i], sizeof(double));
        if (!nn->a[i] || !nn->z[i]) {
            fprintf(stderr, "Error: create_neural_network failed to allocate activation/pre-activation layer %d\n", i);
            goto err;
        }
    }

    for (int i = 0; i < nn->num_layers - 1; i++) {
        /* size_t cast before multiplication avoids int-overflow for wide layers
         * (int mult overflows silently at fan_in * fan_out ≳ 2·10⁹). */
        int fan_in  = nn->layer_sizes[i];
        int fan_out = nn->layer_sizes[i + 1];
        size_t w_elems = (size_t)fan_in * (size_t)fan_out;

        nn->W[i]  = malloc(w_elems * sizeof(double));
        nn->b[i]  = calloc((size_t)fan_out, sizeof(double));
        nn->mW[i] = calloc(w_elems, sizeof(double));
        nn->vW[i] = calloc(w_elems, sizeof(double));
        nn->mb[i] = calloc((size_t)fan_out, sizeof(double));
        nn->vb[i] = calloc((size_t)fan_out, sizeof(double));
        if (!nn->W[i] || !nn->b[i] || !nn->mW[i] || !nn->vW[i] || !nn->mb[i] || !nn->vb[i]) {
            fprintf(stderr, "Error: create_neural_network failed to allocate weights/biases at layer %d\n", i);
            goto err;
        }

        double limit;
        if (activation_function == ACTIVATION_SIREN) {
            /* SIREN init (Sitzmann et al. 2020). First layer uses
             * U[-1/fan_in, 1/fan_in]; deeper layers U[-√6/fan_in / ω, √6/fan_in / ω]
             * so that pre-activations stay in the linear regime of sin(·). */
            if (i == 0) {
                limit = 1.0 / (double)fan_in;
            } else {
                limit = sqrt(6.0 / (double)fan_in) / SIREN_OMEGA;
            }
        } else {
            limit = sqrt(6.0 / (fan_in + fan_out));
        }
        for (size_t j = 0; j < w_elems; j++) {
            nn->W[i][j] = ((double)rand() / RAND_MAX) * 2 * limit - limit;
        }
    }

    return nn;

err:
    free_neural_network(nn);
    return NULL;
}

/* Tolerates partial initialisation: the err: path in create_neural_network
 * calls this on a struct that may have any subset of the array slots
 * (nn->W / nn->a / ...) still NULL and any per-layer entry still NULL. */
void free_neural_network(NeuralNetwork *nn) {
    if (!nn) return;

    if (nn->a) {
        for (int i = 0; i < nn->num_layers; i++) free(nn->a[i]);
    }
    if (nn->z) {
        for (int i = 0; i < nn->num_layers; i++) free(nn->z[i]);
    }
    for (int i = 0; i < nn->num_layers - 1; i++) {
        if (nn->W)  free(nn->W[i]);
        if (nn->b)  free(nn->b[i]);
        if (nn->mW) free(nn->mW[i]);
        if (nn->vW) free(nn->vW[i]);
        if (nn->mb) free(nn->mb[i]);
        if (nn->vb) free(nn->vb[i]);
    }
    free(nn->W);
    free(nn->b);
    free(nn->mW);
    free(nn->vW);
    free(nn->mb);
    free(nn->vb);
    free(nn->a);
    free(nn->z);
    free(nn->layer_sizes);
    free(nn);
}

void batch_normalize(double* layer, int size) {
    double mean = 0.0, var = 0.0;
    for (int i = 0; i < size; i++) mean += layer[i];
    mean /= size;
    for (int i = 0; i < size; i++) var += (layer[i] - mean) * (layer[i] - mean);
    var /= size;
    for (int i = 0; i < size; i++) {
        layer[i] = (layer[i] - mean) / sqrt(var + 1e-8);
    }
}

double* forward(NeuralNetwork *nn, double *input) {
    for (int i = 0; i < nn->layer_sizes[0]; i++) {
        nn->a[0][i] = input[i];
    }

    for (int l = 0; l < nn->num_layers - 1; l++) {
        for (int j = 0; j < nn->layer_sizes[l + 1]; j++) {
            nn->z[l + 1][j] = nn->b[l][j];
            for (int i = 0; i < nn->layer_sizes[l]; i++) {
                nn->z[l + 1][j] += nn->W[l][i * nn->layer_sizes[l + 1] + j] * nn->a[l][i];
            }
        }
        batch_normalize(nn->z[l + 1], nn->layer_sizes[l + 1]);
        for (int j = 0; j < nn->layer_sizes[l + 1]; j++) {
            nn->a[l + 1][j] = activation_function(nn->z[l + 1][j], nn->activation_function);
        }
    }

    // Ensure the output is not exactly zero
    for (int i = 0; i < nn->layer_sizes[nn->num_layers - 1]; i++) {
        if (nn->a[nn->num_layers - 1][i] == 0) {
            nn->a[nn->num_layers - 1][i] = 1e-10;
        }
    }

    return nn->a[nn->num_layers - 1];
}

/*
 * train: one forward + one backward pass of the legacy MLP.
 *
 * Allocation pattern: two transient buffers per call. `delta` holds the
 * current layer's error vector; `prev_delta` holds the upstream layer's.
 * On each loop iteration we free the old `delta` and promote `prev_delta`
 * to `delta`, then at loop exit free the last `delta`. Net allocation
 * across one train() call is 2*num_layers mallocs and frees. If either
 * malloc fails, we free whatever buffers we already hold and return so
 * the caller sees a no-op rather than a NULL-deref crash.
 *
 * Preallocating these buffers on the `nn` struct would eliminate the
 * per-call alloc churn and is a natural v0.5 optimisation, but changes
 * the public ABI of NeuralNetwork. Left as-is for v0.4 to avoid breaking
 * out-of-tree consumers.
 */
void train(NeuralNetwork *nn, double *input, double *target, double learning_rate) {
    double* output = forward(nn, input);

    int output_layer = nn->num_layers - 1;
    double* delta = malloc((size_t)nn->layer_sizes[output_layer] * sizeof(double));
    if (!delta) {
        fprintf(stderr, "Error: train() malloc failed for output-layer delta\n");
        return;
    }

    for (int j = 0; j < nn->layer_sizes[output_layer]; j++) {
        delta[j] = (output[j] - target[j]) * activation_derivative(nn->z[output_layer][j], nn->activation_function);
    }

    for (int l = output_layer - 1; l >= 0; l--) {
        double* prev_delta = malloc((size_t)nn->layer_sizes[l] * sizeof(double));
        if (!prev_delta) {
            fprintf(stderr, "Error: train() malloc failed for layer %d prev_delta\n", l);
            free(delta);
            return;
        }
        for (int i = 0; i < nn->layer_sizes[l]; i++) {
            prev_delta[i] = 0;
            for (int j = 0; j < nn->layer_sizes[l + 1]; j++) {
                prev_delta[i] += delta[j] * nn->W[l][i * nn->layer_sizes[l + 1] + j];
            }
            prev_delta[i] *= activation_derivative(nn->z[l][i], nn->activation_function);
        }

        for (int i = 0; i < nn->layer_sizes[l]; i++) {
            for (int j = 0; j < nn->layer_sizes[l + 1]; j++) {
                int index = i * nn->layer_sizes[l + 1] + j;
                nn->mW[l][index] = ADAM_BETA_1 * nn->mW[l][index] + (1 - ADAM_BETA_1) * delta[j] * nn->a[l][i];
                nn->vW[l][index] = ADAM_BETA_2 * nn->vW[l][index] + (1 - ADAM_BETA_2) * pow(delta[j] * nn->a[l][i], 2);
                nn->W[l][index] -= learning_rate * nn->mW[l][index] / (sqrt(nn->vW[l][index]) + ADAM_EPSILON);

                // Add L2 regularization
                nn->W[l][index] -= learning_rate * L2_REG * nn->W[l][index];
            }
        }
        for (int j = 0; j < nn->layer_sizes[l + 1]; j++) {
            nn->mb[l][j] = ADAM_BETA_1 * nn->mb[l][j] + (1 - ADAM_BETA_1) * delta[j];
            nn->vb[l][j] = ADAM_BETA_2 * nn->vb[l][j] + (1 - ADAM_BETA_2) * pow(delta[j], 2);
            nn->b[l][j] -= learning_rate * nn->mb[l][j] / (sqrt(nn->vb[l][j]) + ADAM_EPSILON);
        }

        free(delta);
        delta = prev_delta;
    }

    free(delta);
}

double activation_function(double x, int type) {
    switch (type) {
        case ACTIVATION_RELU:
            return fmax(0, x);
        case ACTIVATION_SIGMOID:
            return 1.0 / (1.0 + exp(-x));
        case ACTIVATION_TANH:
            return tanh(x);
        case ACTIVATION_SIREN:
            return sin(SIREN_OMEGA * x);
        default:
            return x;
    }
}

double activation_derivative(double x, int type) {
    switch (type) {
        case ACTIVATION_RELU:
            return x > 0 ? 1 : 0;
        case ACTIVATION_SIGMOID:
            return x * (1.0 - x);
        case ACTIVATION_TANH:
            return 1.0 - x * x;
        case ACTIVATION_SIREN:
            /* d/dz sin(ω·z) = ω · cos(ω·z). Called with the
             * pre-activation z throughout the backward pass. */
            return SIREN_OMEGA * cos(SIREN_OMEGA * x);
        default:
            return 1;
    }
}

void reset_network_if_needed(NeuralNetwork *nn, double avg_error) {
    static int poor_performance_count = 0;
    if (avg_error > 1.0) {  // If average error is more than 100%
        poor_performance_count++;
        if (poor_performance_count > 10) {  // Reset after 10 consecutive poor performances
            for (int l = 0; l < nn->num_layers - 1; l++) {
                int fan_in = nn->layer_sizes[l];
                int fan_out = nn->layer_sizes[l + 1];
                double limit = sqrt(6.0 / (fan_in + fan_out));
                for (int i = 0; i < fan_in * fan_out; i++) {
                    nn->W[l][i] = ((double)rand() / RAND_MAX) * 2 * limit - limit;
                }
                for (int i = 0; i < fan_out; i++) {
                    nn->b[l][i] = 0;
                }
            }
            poor_performance_count = 0;
        }
    } else {
        poor_performance_count = 0;
    }
}