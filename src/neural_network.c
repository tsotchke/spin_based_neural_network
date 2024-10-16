#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "neural_network.h"

NeuralNetwork* create_neural_network(int input_size, int num_hidden_layers, int neurons_per_layer, int output_size, int activation_function) {
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    if (!nn) {
        fprintf(stderr, "Failed to allocate memory for NeuralNetwork.\n");
        return NULL;
    }

    nn->num_layers = num_hidden_layers + 2;
    nn->layer_sizes = malloc(nn->num_layers * sizeof(int));
    nn->W = malloc((nn->num_layers - 1) * sizeof(double *));
    nn->b = malloc((nn->num_layers - 1) * sizeof(double *));
    nn->mW = malloc((nn->num_layers - 1) * sizeof(double *));
    nn->vW = malloc((nn->num_layers - 1) * sizeof(double *));
    nn->mb = malloc((nn->num_layers - 1) * sizeof(double *));
    nn->vb = malloc((nn->num_layers - 1) * sizeof(double *));
    nn->a = malloc(nn->num_layers * sizeof(double *));
    nn->z = malloc(nn->num_layers * sizeof(double *));

    nn->layer_sizes[0] = input_size;
    for (int i = 1; i < nn->num_layers - 1; i++) {
        nn->layer_sizes[i] = neurons_per_layer;
    }
    nn->layer_sizes[nn->num_layers - 1] = output_size;

    for (int i = 0; i < nn->num_layers; i++) {
        nn->a[i] = calloc(nn->layer_sizes[i], sizeof(double));
        nn->z[i] = calloc(nn->layer_sizes[i], sizeof(double));
    }

    for (int i = 0; i < nn->num_layers - 1; i++) {
        int fan_in = nn->layer_sizes[i];
        int fan_out = nn->layer_sizes[i + 1];
        nn->W[i] = malloc(fan_in * fan_out * sizeof(double));
        nn->b[i] = calloc(fan_out, sizeof(double));
        nn->mW[i] = calloc(fan_in * fan_out, sizeof(double));
        nn->vW[i] = calloc(fan_in * fan_out, sizeof(double));
        nn->mb[i] = calloc(fan_out, sizeof(double));
        nn->vb[i] = calloc(fan_out, sizeof(double));

        double limit = sqrt(6.0 / (fan_in + fan_out));
        for (int j = 0; j < fan_in * fan_out; j++) {
            nn->W[i][j] = ((double)rand() / RAND_MAX) * 2 * limit - limit;
        }
    }

    nn->activation_function = activation_function;

    return nn;
}

void free_neural_network(NeuralNetwork *nn) {
    if (!nn) return;

    for (int i = 0; i < nn->num_layers; i++) {
        free(nn->a[i]);
        free(nn->z[i]);
    }
    for (int i = 0; i < nn->num_layers - 1; i++) {
        free(nn->W[i]);
        free(nn->b[i]);
        free(nn->mW[i]);
        free(nn->vW[i]);
        free(nn->mb[i]);
        free(nn->vb[i]);
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

void train(NeuralNetwork *nn, double *input, double *target, double learning_rate) {
    double* output = forward(nn, input);
    
    int output_layer = nn->num_layers - 1;
    double* delta = malloc(nn->layer_sizes[output_layer] * sizeof(double));

    for (int j = 0; j < nn->layer_sizes[output_layer]; j++) {
        delta[j] = (output[j] - target[j]) * activation_derivative(nn->z[output_layer][j], nn->activation_function);
    }

    for (int l = output_layer - 1; l >= 0; l--) {
        double* prev_delta = malloc(nn->layer_sizes[l] * sizeof(double));
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