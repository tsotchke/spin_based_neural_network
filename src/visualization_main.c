 #include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <SDL2/SDL.h>

#define FPS 60
#include "visualization.h"
#include "berry_phase.h"
#include "toric_code.h"
#include "majorana_modes.h"
#include "topological_entropy.h"

// Simulate Berry curvature data
double* generate_sample_berry_curvature(int resolution) {
    double* curvature = malloc(resolution * resolution * sizeof(double));
    if (!curvature) return NULL;
    
    // Generate a simple pattern with a vortex
    double center_x = resolution / 2.0;
    double center_y = resolution / 2.0;
    double radius = resolution / 4.0;
    
    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            double dx = j - center_x;
            double dy = i - center_y;
            double dist = sqrt(dx*dx + dy*dy);
            double angle = atan2(dy, dx);
            
            if (dist < radius) {
                // Inside the vortex: positive curvature with angle-dependent pattern
                curvature[i * resolution + j] = 0.5 * (1 - dist/radius) * (1 + 0.3 * sin(3 * angle));
            } else if (dist < 2*radius) {
                // Outside the vortex: negative curvature with angle-dependent oscillation
                curvature[i * resolution + j] = -0.3 * (1 - (dist-radius)/radius) * (1 + 0.2 * cos(2 * angle));
            } else {
                // Background: subtle angle-dependent waves
                curvature[i * resolution + j] = 0.05 * sin(8 * angle) * exp(-(dist-2*radius)/(radius/2));
            }
            
            // Add some noise
            curvature[i * resolution + j] += ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }
    
    return curvature;
}

// Simulate toric code stabilizers
double* generate_sample_toric_code(int size_x, int size_y) {
    int total_size = 2 * size_x * size_y;  // Plaquette + vertex operators
    double* stabilizers = malloc(total_size * sizeof(double));
    if (!stabilizers) return NULL;
    
    // Initialize all stabilizers to +1 (no errors)
    for (int i = 0; i < total_size; i++) {
        stabilizers[i] = 1.0;
    }
    
    // Add some random errors (flip to -1)
    int num_errors = (size_x * size_y) / 5;  // About 20% error rate
    for (int i = 0; i < num_errors; i++) {
        int idx = rand() % total_size;
        stabilizers[idx] = -1.0;
    }
    
    return stabilizers;
}

// Simulate Majorana modes
double* generate_sample_majorana_modes(int num_modes) {
    double* positions = malloc(num_modes * sizeof(double));
    if (!positions) return NULL;
    
    // Initialize random occupation values
    for (int i = 0; i < num_modes; i++) {
        positions[i] = (double)rand() / RAND_MAX;
    }
    
    return positions;
}

int main(int argc, char* argv[]) {
    // Seed random number generator
    srand(time(NULL));
    
    // Initialize SDL and visualization
    VisualizationState* vis = init_visualization();
    if (!vis) {
        fprintf(stderr, "Visualization initialization failed\n");
        return 1;
    }
    
    printf("Topological Quantum Visualization\n");
    printf("----------------------------------\n");
    printf("Controls:\n");
    printf("  1-4: Switch visualization modes\n");
    printf("  Esc: Exit\n\n");
    
    // Generate sample data for visualization
    
    // Berry curvature data
    int k_resolution = 30;
    double* curvature = generate_sample_berry_curvature(k_resolution);
    if (curvature) {
        set_berry_curvature_data(vis, curvature, k_resolution, 1.0);  // Chern number = 1
    }
    
    // Toric code data
    int toric_size = 8;
    double* stabilizers = generate_sample_toric_code(toric_size, toric_size);
    if (stabilizers) {
        set_toric_code_data(vis, stabilizers, toric_size, toric_size);
    }
    
    // Majorana modes data
    int num_majoranas = 12;  // 6 sites with 2 Majorana modes each
    double* majorana_positions = generate_sample_majorana_modes(num_majoranas);
    if (majorana_positions) {
        set_majorana_data(vis, majorana_positions, num_majoranas);
    }
    
    // Set topological entropy
    set_topological_entropy(vis, 0.693);  // log(2) is typical for Z2 topological order
    
    // Main loop
    Uint32 frame_time;
    Uint32 last_time = SDL_GetTicks();
    
    while (is_visualization_running(vis)) {
        // Update and render visualization
        update_visualization(vis);
        
        // Cap frame rate
        frame_time = SDL_GetTicks() - last_time;
        if (frame_time < 1000 / FPS) {
            SDL_Delay(1000 / FPS - frame_time);
        }
        last_time = SDL_GetTicks();
    }
    
    // Clean up
    cleanup_visualization(vis);
    
    if (curvature) free(curvature);
    if (stabilizers) free(stabilizers);
    if (majorana_positions) free(majorana_positions);
    
    return 0;
}
