#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include "ising_model.h"
#include "kitaev_model.h"
#include "quantum_mechanics.h"
#include "reinforcement_learning.h"
#include "spin_models.h"
#include "physics_loss.h"
#include "disordered_model.h"
#include "neural_network.h"
#include "energy_utils.h"

#define TRAINING_ITERATIONS 1000
#define PREDICTION_INTERVAL 10
#define ENERGY_SCALE 1000.0
#define RANDOM_FACTOR 0.1

void print_usage() {
    printf("Usage: ./spin_based_neural_computation [OPTIONS]\n");
    printf("Options:\n");
    printf("  -i, --iterations N     Number of iterations to run (default: 100)\n");
    printf("  -v, --verbose          Verbose output (optional)\n");
    printf("  --lattice-size X Y Z   Set lattice size (default: 10 10 10)\n");
    printf("  --jx JX                Set Jx coupling constant (default: 1.0)\n");
    printf("  --jy JY                Set Jy coupling constant (default: 1.0)\n");
    printf("  --jz JZ                Set Jz coupling constant (default: -1.0)\n");
    printf("  --initial-state STATE  Specify initial state (random, all-up, all-down)\n");
    printf("  --dx DX                Spatial step size for simulation (default: 0.1)\n");
    printf("  --dt DT                Time step size for simulation (default: 0.1)\n");
    printf("  --loss-type TYPE       Specify loss type (heat, schrodinger, maxwell, navier_stokes, wave)\n");
    printf("  --activation FUNC      Specify activation function type (relu, tanh, sigmoid)\n");
    printf("  --log LOG_FILE         Specify log file name\n");
    printf("  -h, --help             Display this help message\n");
}

int parse_activation_function(const char *activation_str) {
    if (strcmp(activation_str, "relu") == 0) return ACTIVATION_RELU;
    else if (strcmp(activation_str, "tanh") == 0) return ACTIVATION_TANH;
    else if (strcmp(activation_str, "sigmoid") == 0) return ACTIVATION_SIGMOID;
    else return ACTIVATION_RELU;  // Default to ReLU
}

void normalize_input(double *input, int size) {
    for (int i = 0; i < size; i++) {
        input[i] = (input[i] + 1.0) / 2.0;  // Map from [-1, 1] to [0, 1]
    }
}

int main(int argc, char *argv[]) {
    srand(time(NULL));
    
    int iterations = 100;
    int lattice_size_x = 10, lattice_size_y = 10, lattice_size_z = 10;
    double jx = 1.0, jy = 1.0, jz = -1.0;
    int verbose = 0;
    char log_filename[256] = "simulation.log";
    char initial_state[10] = "random";
    double dx = 0.1, dt = 0.1;
    char loss_type[30] = "heat";
    char activation_function_str[30] = "relu";

    static struct option long_options[] = {
        {"iterations", required_argument, 0, 'i'},
        {"verbose", no_argument, 0, 'v'},
        {"lattice-size", required_argument, 0, 0},
        {"jx", required_argument, 0, 0},
        {"jy", required_argument, 0, 0},
        {"jz", required_argument, 0, 0},
        {"initial-state", required_argument, 0, 0},
        {"dx", required_argument, 0, 0},
        {"dt", required_argument, 0, 0},
        {"loss-type", required_argument, 0, 0},
        {"activation", required_argument, 0, 0},
        {"log", required_argument, 0, 0},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    int opt;
    while ((opt = getopt_long(argc, argv, "i:vh", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i': iterations = atoi(optarg); break;
            case 'v': verbose = 1; break;
            case 'h': print_usage(); return 0;
            case 0:
                if (strcmp("lattice-size", long_options[option_index].name) == 0) {
                    sscanf(optarg, "%d %d %d", &lattice_size_x, &lattice_size_y, &lattice_size_z);
                } else if (strcmp("jx", long_options[option_index].name) == 0) {
                    jx = atof(optarg);
                } else if (strcmp("jy", long_options[option_index].name) == 0) {
                    jy = atof(optarg);
                } else if (strcmp("jz", long_options[option_index].name) == 0) {
                    jz = atof(optarg);
                } else if (strcmp("initial-state", long_options[option_index].name) == 0) {
                    strncpy(initial_state, optarg, sizeof(initial_state) - 1);
                } else if (strcmp("dx", long_options[option_index].name) == 0) {
                    dx = atof(optarg);
                } else if (strcmp("dt", long_options[option_index].name) == 0) {
                    dt = atof(optarg);
                } else if (strcmp("loss-type", long_options[option_index].name) == 0) {
                    strncpy(loss_type, optarg, sizeof(loss_type) - 1);
                } else if (strcmp("activation", long_options[option_index].name) == 0) {
                    strncpy(activation_function_str, optarg, sizeof(activation_function_str) - 1);
                } else if (strcmp("log", long_options[option_index].name) == 0) {
                    strncpy(log_filename, optarg, sizeof(log_filename) - 1);
                }
                break;
            default: print_usage(); return 1;
        }
    }

    if (verbose) {
        printf("Starting simulation with %d iterations\n", iterations);
        printf("Lattice size: %d x %d x %d\n", lattice_size_x, lattice_size_y, lattice_size_z);
        printf("Coupling constants - Jx: %.2f, Jy: %.2f, Jz: %.2f\n", jx, jy, jz);
        printf("Initial state: %s\n", initial_state);
        printf("dx: %.2f, dt: %.2f\n", dx, dt);
        printf("Loss type: %s\n", loss_type);
        printf("Activation function: %s\n", activation_function_str);
        printf("Log file: %s\n", log_filename);
    }

    FILE *log_file = fopen(log_filename, "w");
    if (!log_file) {
        fprintf(stderr, "Error opening log file %s!\n", log_filename);
        return 1;
    }

    // Initialize lattices
    IsingLattice *ising_lattice = initialize_ising_lattice(lattice_size_x, lattice_size_y, lattice_size_z, initial_state);
    KitaevLattice *kitaev_lattice = initialize_kitaev_lattice(lattice_size_x, lattice_size_y, lattice_size_z, jx, jy, jz, initial_state);
    SpinLattice *spin_lattice = initialize_spin_lattice(lattice_size_x, lattice_size_y, lattice_size_z, initial_state);

    // Set parameters for quantum effects and reinforcement learning
    double noise_level = 0.2;
    double entanglement_prob = 0.1;
    double disorder_strength = 0.1;

    // Initialize the neural network
    int input_size = lattice_size_x * lattice_size_y * lattice_size_z * 3; // 3 components per spin
    int hidden_layers = 3;
    int neurons_per_layer = 256; // Increased from 128
    int output_size = 1; // Predicting total energy
    int activation_function = parse_activation_function(activation_function_str);
    NeuralNetwork *nn = create_neural_network(input_size, hidden_layers, neurons_per_layer, output_size, activation_function);

    // Training data arrays
    double **training_inputs = malloc(TRAINING_ITERATIONS * sizeof(double*));
    double **training_outputs = malloc(TRAINING_ITERATIONS * sizeof(double*));
    for (int i = 0; i < TRAINING_ITERATIONS; i++) {
        training_inputs[i] = malloc(input_size * sizeof(double));
        training_outputs[i] = malloc(output_size * sizeof(double));
    }

    double max_energy = 0.0;
    double previous_energy = 0.0;
    double cumulative_absolute_error = 0.0;
    double cumulative_relative_error = 0.0;
    double cumulative_squared_error = 0.0;
    double smoothed_error = 0.0;
    int error_count = 0;

    for (int iter = 0; iter < iterations; iter++) {
        // Flip random spins and apply quantum effects
        flip_random_spin_ising(ising_lattice);
        flip_random_spin_kitaev(kitaev_lattice);
        add_disorder_to_ising_lattice(ising_lattice, disorder_strength);
        add_disorder_to_kitaev_lattice(kitaev_lattice, disorder_strength);
        apply_quantum_effects(ising_lattice, kitaev_lattice, spin_lattice, noise_level);
        simulate_entanglement(ising_lattice, kitaev_lattice, entanglement_prob);

        // Compute energies
        double ising_energy = compute_ising_energy(ising_lattice);
        double kitaev_energy = compute_kitaev_energy(kitaev_lattice);
        double spin_energy = compute_spin_energy(spin_lattice);
        double total_energy = ising_energy + kitaev_energy + spin_energy;

        printf("Iteration %d: Raw energies - Ising: %e, Kitaev: %e, Spin: %e, Total: %e\n", 
               iter, ising_energy, kitaev_energy, spin_energy, total_energy);

        // Update max_energy if necessary
        if (fabs(total_energy) > max_energy) {
            max_energy = fabs(total_energy);
        }

        // Scale the energy for neural network input
        double scaled_energy = scale_energy(total_energy);

        printf("Scaled energy: %e\n", scaled_energy);

        // Prepare input for neural network
        double *nn_input = malloc(input_size * sizeof(double));
        int idx = 0;
        for (int x = 0; x < lattice_size_x; x++) {
            for (int y = 0; y < lattice_size_y; y++) {
                for (int z = 0; z < lattice_size_z; z++) {
                    nn_input[idx++] = spin_lattice->spins[x][y][z].sx;
                    nn_input[idx++] = spin_lattice->spins[x][y][z].sy;
                    nn_input[idx++] = spin_lattice->spins[x][y][z].sz;
                }
            }
        }

        // Normalize input data
        normalize_input(nn_input, input_size);

        // Store data for training
        int training_idx = iter % TRAINING_ITERATIONS;
        memcpy(training_inputs[training_idx], nn_input, input_size * sizeof(double));
        training_outputs[training_idx][0] = scaled_energy;

        // Train the network every PREDICTION_INTERVAL iterations
        if (iter % PREDICTION_INTERVAL == 0 && iter > 0) {
            double avg_error = cumulative_relative_error / error_count;
            reset_network_if_needed(nn, avg_error);
            cumulative_relative_error = 0.0;
            error_count = 0;
            
            for (int i = 0; i < TRAINING_ITERATIONS; i++) {
                train(nn, training_inputs[i], training_outputs[i], LEARNING_RATE);
            }
        }

        // Make a prediction
        double *nn_output = forward(nn, nn_input);
        double predicted_scaled_energy = nn_output[0];
        double predicted_energy = unscale_energy(predicted_scaled_energy);

        // Clip the predicted energy to a reasonable range
        double max_possible_energy = fabs(lattice_size_x * lattice_size_y * lattice_size_z * (fabs(jx) + fabs(jy) + fabs(jz)));
        predicted_energy = fmax(-max_possible_energy, fmin(max_possible_energy, predicted_energy));

        // Compute physics loss
        double physics_loss = compute_physics_loss(ising_energy, kitaev_energy, spin_energy, ising_lattice, kitaev_lattice, spin_lattice, dt, dx, loss_type);

        // Apply physics-based correction to the prediction
        double physics_correction_factor = 1.0 / (1.0 + physics_loss);
        predicted_energy *= physics_correction_factor;

        // Compute prediction errors
        double absolute_error = fabs(predicted_energy - total_energy);
        double relative_error = absolute_error / (fabs(total_energy) + 1e-10);
        double squared_error = pow(predicted_energy - total_energy, 2);

        // Update error metrics
        cumulative_absolute_error += absolute_error;
        cumulative_relative_error += relative_error;
        cumulative_squared_error += squared_error;
        error_count++;

        // Compute running averages
        double mean_absolute_error = cumulative_absolute_error / error_count;
        double mean_relative_error = cumulative_relative_error / error_count;
        double root_mean_squared_error = sqrt(cumulative_squared_error / error_count);

        // Exponential moving average for smoothed error tracking
        double alpha = 0.1; // Smoothing factor
        smoothed_error = alpha * relative_error + (1 - alpha) * smoothed_error;

        // Log or print the results
        if (verbose) {
            printf("Iteration %d:\n", iter);
            printf("  Predicted Energy: %e, Actual Energy: %e\n", predicted_energy, total_energy);
            printf("  Absolute Error: %e, Relative Error: %e\n", absolute_error, relative_error);
            printf("  Mean Absolute Error: %e, Mean Relative Error: %e, RMSE: %e\n", 
                   mean_absolute_error, mean_relative_error, root_mean_squared_error);
            printf("  Smoothed Error: %e\n", smoothed_error);
            printf("  Physics Loss: %e\n", physics_loss);

            char *ising_state_str = get_ising_state_string(ising_lattice);
            char *kitaev_state_str = get_kitaev_state_string(kitaev_lattice);
            
            printf("RL Iteration %d: \nIsing State: \n%s\nKitaev State: \n%s\n", iter, ising_state_str, kitaev_state_str);

            free(ising_state_str);
            free(kitaev_state_str);
        }

        // Log energies, predictions, and losses
        fprintf(log_file, "%d %e %e %e %e %e %e %e %e %e %e\n", 
                iter, ising_energy, kitaev_energy, spin_energy, total_energy, 
                predicted_energy, physics_loss, absolute_error, relative_error, 
                root_mean_squared_error, smoothed_error);

        // Reinforcement Learning and Optimization
        double reward = reinforce_learning(ising_lattice, kitaev_lattice, total_energy, previous_energy);
        optimize_spins_with_rl(ising_lattice, kitaev_lattice, reward);

        previous_energy = total_energy;
        free(nn_input);
    }

    // Free resources
    for (int i = 0; i < TRAINING_ITERATIONS; i++) {
        free(training_inputs[i]);
        free(training_outputs[i]);
    }
    
    free(training_inputs);
    free(training_outputs);
    free_ising_lattice(ising_lattice);
    free_kitaev_lattice(kitaev_lattice);
    free_spin_lattice(spin_lattice);
    free_neural_network(nn);
    fclose(log_file);

    return 0;
}