#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#define ADAM_BETA_1 0.9
#define ADAM_BETA_2 0.999
#define ADAM_EPSILON 1e-8

#define BATCH_SIZE 32
#define EPOCHS 100

#define LEARNING_RATE 1e-4
#define TRAINING_ITERATIONS 1000
#define PREDICTION_INTERVAL 10

#define L2_REG 1e-5

// Activation function types
#define ACTIVATION_RELU 0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_TANH 2

typedef struct NeuralNetwork {
    int num_layers;
    int *layer_sizes;
    double **W;
    double **b;
    double **a;
    double **z;
    double **mW, **vW;
    double **mb, **vb;
    int activation_function;
} NeuralNetwork;

NeuralNetwork* create_neural_network(int input_size, int num_hidden_layers, int neurons_per_layer, int output_size, int activation_function);
void reset_network_if_needed(NeuralNetwork *nn, double avg_error);
void free_neural_network(NeuralNetwork *nn);
double* forward(NeuralNetwork *nn, double *input);
void train(NeuralNetwork *nn, double *input, double *target, double learning_rate);
double activation_function(double x, int function_type);
double activation_derivative(double x, int function_type);
void batch_normalize(double* layer, int size);

#endif