#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <SDL2/SDL.h>

#include "visualization.h"
#include "topological_entropy.h"
#include "berry_phase.h"
#include "majorana_modes.h"
#include "toric_code.h"

#define WINDOW_WIDTH 1200
#define WINDOW_HEIGHT 800
#define FPS 60
#define MAX_PARTICLES 1000

typedef struct {
    float x, y;
    float vx, vy;
    float size;
    float hue;
    float lifetime;
    int active;
} Particle;

struct VisualizationState {
    SDL_Window *window;
    SDL_Renderer *renderer;
    Particle particles[MAX_PARTICLES];
    int view_mode;  // 0 = Berry phase, 1 = Toric code, 2 = Majorana modes, 3 = Entropy
    int running;
    // Visualization data
    double *berry_curvature;
    int k_resolution;
    double *toric_stabilizers;
    int toric_size_x, toric_size_y;
    double *majorana_positions;
    int num_majoranas;
    double topological_entropy;
    double topological_invariant;
};

// Function to initialize SDL and visualization state
VisualizationState* init_visualization() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return NULL;
    }

    VisualizationState *state = malloc(sizeof(VisualizationState));
    if (!state) {
        fprintf(stderr, "Memory allocation failed for visualization state\n");
        SDL_Quit();
        return NULL;
    }

    state->window = SDL_CreateWindow("Topological Quantum Visualization", 
                                    SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                    WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    if (!state->window) {
        fprintf(stderr, "Window could not be created! SDL_Error: %s\n", SDL_GetError());
        free(state);
        SDL_Quit();
        return NULL;
    }

    state->renderer = SDL_CreateRenderer(state->window, -1, SDL_RENDERER_ACCELERATED);
    if (!state->renderer) {
        fprintf(stderr, "Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(state->window);
        free(state);
        SDL_Quit();
        return NULL;
    }

    // Initialize particles
    for (int i = 0; i < MAX_PARTICLES; i++) {
        state->particles[i].active = 0;
    }

    state->view_mode = 0;
    state->running = 1;
    
    state->berry_curvature = NULL;
    state->toric_stabilizers = NULL;
    state->majorana_positions = NULL;
    state->k_resolution = 0;
    state->toric_size_x = 0;
    state->toric_size_y = 0;
    state->num_majoranas = 0;
    state->topological_entropy = 0.0;
    state->topological_invariant = 0.0;

    return state;
}

// Function to create a particle at a specific position
void spawn_particle(VisualizationState *state, float x, float y, float size, float hue) {
    for (int i = 0; i < MAX_PARTICLES; i++) {
        if (!state->particles[i].active) {
            state->particles[i].x = x;
            state->particles[i].y = y;
            state->particles[i].vx = ((float)rand() / RAND_MAX * 2 - 1) * 0.5f;
            state->particles[i].vy = ((float)rand() / RAND_MAX * 2 - 1) * 0.5f;
            state->particles[i].size = size;
            state->particles[i].hue = hue;
            state->particles[i].lifetime = 1.0f;
            state->particles[i].active = 1;
            break;
        }
    }
}

// Convert HSL to RGB
void hsl_to_rgb(float h, float s, float l, Uint8 *r, Uint8 *g, Uint8 *b) {
    float c = (1 - fabsf(2 * l - 1)) * s;
    float x = c * (1 - fabsf(fmodf(h / 60.0f, 2) - 1));
    float m = l - c / 2;
    float r_temp, g_temp, b_temp;

    if (h >= 0 && h < 60) {
        r_temp = c; g_temp = x; b_temp = 0;
    } else if (h >= 60 && h < 120) {
        r_temp = x; g_temp = c; b_temp = 0;
    } else if (h >= 120 && h < 180) {
        r_temp = 0; g_temp = c; b_temp = x;
    } else if (h >= 180 && h < 240) {
        r_temp = 0; g_temp = x; b_temp = c;
    } else if (h >= 240 && h < 300) {
        r_temp = x; g_temp = 0; b_temp = c;
    } else {
        r_temp = c; g_temp = 0; b_temp = x;
    }

    *r = (Uint8)((r_temp + m) * 255);
    *g = (Uint8)((g_temp + m) * 255);
    *b = (Uint8)((b_temp + m) * 255);
}

// Update and render particles
void update_particles(VisualizationState *state) {
    for (int i = 0; i < MAX_PARTICLES; i++) {
        if (state->particles[i].active) {
            // Update position
            state->particles[i].x += state->particles[i].vx;
            state->particles[i].y += state->particles[i].vy;
            
            // Update lifetime
            state->particles[i].lifetime -= 0.01f;
            
            // Check if particle should be deactivated
            if (state->particles[i].lifetime <= 0 || 
                state->particles[i].x < 0 || state->particles[i].x > WINDOW_WIDTH ||
                state->particles[i].y < 0 || state->particles[i].y > WINDOW_HEIGHT) {
                state->particles[i].active = 0;
                continue;
            }
            
            // Render particle
            Uint8 r, g, b;
            float alpha = state->particles[i].lifetime;
            hsl_to_rgb(state->particles[i].hue, 0.8f, 0.5f, &r, &g, &b);
            
            SDL_SetRenderDrawBlendMode(state->renderer, SDL_BLENDMODE_BLEND);
            SDL_SetRenderDrawColor(state->renderer, r, g, b, (Uint8)(alpha * 255));
            
            int size = (int)(state->particles[i].size * state->particles[i].lifetime);
            SDL_Rect rect = {
                (int)state->particles[i].x - size/2,
                (int)state->particles[i].y - size/2,
                size,
                size
            };
            SDL_RenderFillRect(state->renderer, &rect);
        }
    }
}

// Draw the Berry curvature visualization
void draw_berry_curvature(VisualizationState *state) {
    if (!state->berry_curvature || state->k_resolution <= 0) {
        // Draw placeholder if no data
        SDL_SetRenderDrawColor(state->renderer, 100, 100, 200, 255);
        SDL_Rect rect = {WINDOW_WIDTH/4, WINDOW_HEIGHT/4, WINDOW_WIDTH/2, WINDOW_HEIGHT/2};
        SDL_RenderFillRect(state->renderer, &rect);
        
        // Display message
        char buffer[100];
        sprintf(buffer, "Berry Curvature Visualization - Chern Number: %.2f", state->topological_invariant);
        // Render text would go here (requires SDL_ttf)
        
        return;
    }
    
    int grid_size = state->k_resolution;
    int cell_width = (WINDOW_WIDTH * 3/4) / grid_size;
    int cell_height = (WINDOW_HEIGHT * 3/4) / grid_size;
    int grid_x_offset = WINDOW_WIDTH/8;
    int grid_y_offset = WINDOW_HEIGHT/8;
    
    // Find max curvature for normalization
    double max_curvature = 0.0;
    for (int i = 0; i < grid_size*grid_size; i++) {
        double abs_value = fabs(state->berry_curvature[i]);
        if (abs_value > max_curvature) {
            max_curvature = abs_value;
        }
    }
    
    if (max_curvature < 1e-10) max_curvature = 1.0;  // Avoid division by zero
    
    // Draw grid cells
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            int idx = i * grid_size + j;
            double value = state->berry_curvature[idx] / max_curvature;  // Normalize to [-1, 1]
            
            // Calculate color (blue for negative, red for positive)
            Uint8 r, g, b;
            float hue = (value > 0) ? 0.0f : 240.0f;  // Red for positive, blue for negative
            float saturation = 0.8f;
            float lightness = 0.5f * (0.5f + 0.5f * fabs(value));  // Brighter for stronger values
            
            hsl_to_rgb(hue, saturation, lightness, &r, &g, &b);
            
            SDL_SetRenderDrawColor(state->renderer, r, g, b, 255);
            SDL_Rect rect = {
                grid_x_offset + j * cell_width,
                grid_y_offset + i * cell_height,
                cell_width,
                cell_height
            };
            SDL_RenderFillRect(state->renderer, &rect);
            
            // Spawn particles for strong curvature
            if (rand() % 100 < fabs(value) * 30) {
                float particle_x = grid_x_offset + (j + 0.5f) * cell_width;
                float particle_y = grid_y_offset + (i + 0.5f) * cell_height;
                float size = 5.0f + fabs(value) * 15.0f;
                spawn_particle(state, particle_x, particle_y, size, hue);
            }
        }
    }
    
    // Display Chern number
    char buffer[100];
    sprintf(buffer, "Berry Curvature Visualization - Chern Number: %.2f", state->topological_invariant);
    // Render text would go here (requires SDL_ttf)
}

// Draw the Toric code visualization
void draw_toric_code(VisualizationState *state) {
    if (!state->toric_stabilizers || state->toric_size_x <= 0 || state->toric_size_y <= 0) {
        // Draw placeholder
        SDL_SetRenderDrawColor(state->renderer, 200, 100, 100, 255);
        SDL_Rect rect = {WINDOW_WIDTH/4, WINDOW_HEIGHT/4, WINDOW_WIDTH/2, WINDOW_HEIGHT/2};
        SDL_RenderFillRect(state->renderer, &rect);
        return;
    }
    
    int grid_width = state->toric_size_x;
    int grid_height = state->toric_size_y;
    int cell_size = fmin((WINDOW_WIDTH * 3/4) / grid_width, (WINDOW_HEIGHT * 3/4) / grid_height);
    int grid_x_offset = (WINDOW_WIDTH - grid_width * cell_size) / 2;
    int grid_y_offset = (WINDOW_HEIGHT - grid_height * cell_size) / 2;
    
    // Draw grid
    SDL_SetRenderDrawColor(state->renderer, 50, 50, 50, 255);
    for (int i = 0; i <= grid_height; i++) {
        SDL_RenderDrawLine(
            state->renderer,
            grid_x_offset,
            grid_y_offset + i * cell_size,
            grid_x_offset + grid_width * cell_size,
            grid_y_offset + i * cell_size
        );
    }
    
    for (int i = 0; i <= grid_width; i++) {
        SDL_RenderDrawLine(
            state->renderer,
            grid_x_offset + i * cell_size,
            grid_y_offset,
            grid_x_offset + i * cell_size,
            grid_y_offset + grid_height * cell_size
        );
    }
    
    // Draw stabilizers
    for (int i = 0; i < grid_height; i++) {
        for (int j = 0; j < grid_width; j++) {
            int plaquette_idx = i * grid_width + j;
            int vertex_idx = plaquette_idx + grid_width * grid_height;
            
            double plaquette_value = state->toric_stabilizers[plaquette_idx];
            double vertex_value = state->toric_stabilizers[vertex_idx];
            
            // Draw plaquette operator (center of cell)
            Uint8 r, g, b;
            float plaquette_hue = (plaquette_value > 0) ? 120.0f : 0.0f;  // Green for +1, red for -1
            hsl_to_rgb(plaquette_hue, 0.8f, 0.5f, &r, &g, &b);
            
            SDL_SetRenderDrawColor(state->renderer, r, g, b, 255);
            SDL_Rect plaquette_rect = {
                grid_x_offset + j * cell_size + cell_size/4,
                grid_y_offset + i * cell_size + cell_size/4,
                cell_size/2,
                cell_size/2
            };
            SDL_RenderFillRect(state->renderer, &plaquette_rect);
            
            // Draw vertex operator (at corners)
            float vertex_hue = (vertex_value > 0) ? 240.0f : 0.0f;  // Blue for +1, red for -1
            hsl_to_rgb(vertex_hue, 0.8f, 0.5f, &r, &g, &b);
            
            SDL_SetRenderDrawColor(state->renderer, r, g, b, 255);
            SDL_Rect vertex_rect = {
                grid_x_offset + j * cell_size - cell_size/8,
                grid_y_offset + i * cell_size - cell_size/8,
                cell_size/4,
                cell_size/4
            };
            SDL_RenderFillRect(state->renderer, &vertex_rect);
            
            // Spawn particles for error syndromes
            if (plaquette_value < 0 && rand() % 100 < 10) {
                float particle_x = grid_x_offset + (j + 0.5f) * cell_size;
                float particle_y = grid_y_offset + (i + 0.5f) * cell_size;
                spawn_particle(state, particle_x, particle_y, 10.0f, 0.0f);  // Red particles
            }
            
            if (vertex_value < 0 && rand() % 100 < 10) {
                float particle_x = grid_x_offset + j * cell_size;
                float particle_y = grid_y_offset + i * cell_size;
                spawn_particle(state, particle_x, particle_y, 10.0f, 240.0f);  // Blue particles
            }
        }
    }
    
    // Display information
    char buffer[100];
    sprintf(buffer, "Toric Code Visualization - %dx%d Lattice", grid_width, grid_height);
    // Render text would go here
}

// Draw Majorana zero modes
void draw_majorana_modes(VisualizationState *state) {
    if (!state->majorana_positions || state->num_majoranas <= 0) {
        // Draw placeholder
        SDL_SetRenderDrawColor(state->renderer, 100, 200, 100, 255);
        SDL_Rect rect = {WINDOW_WIDTH/4, WINDOW_HEIGHT/4, WINDOW_WIDTH/2, WINDOW_HEIGHT/2};
        SDL_RenderFillRect(state->renderer, &rect);
        return;
    }
    
    // Calculate parameters for drawing the chain
    int chain_length = state->num_majoranas / 2;  // Each site has 2 Majorana modes
    int circle_radius = fmin(WINDOW_WIDTH, WINDOW_HEIGHT) * 3/8;
    int center_x = WINDOW_WIDTH / 2;
    int center_y = WINDOW_HEIGHT / 2;
    
    // Draw circle representing the chain
    SDL_SetRenderDrawColor(state->renderer, 80, 80, 80, 255);
    for (int i = 0; i < 360; i++) {
        double angle = i * M_PI / 180.0;
        SDL_RenderDrawPoint(
            state->renderer,
            center_x + (int)(circle_radius * cos(angle)),
            center_y + (int)(circle_radius * sin(angle))
        );
    }
    
    // Draw Majorana modes
    for (int i = 0; i < state->num_majoranas; i++) {
        double angle = (2.0 * M_PI * i) / state->num_majoranas;
        int x = center_x + (int)(circle_radius * cos(angle));
        int y = center_y + (int)(circle_radius * sin(angle));
        
        double occupation = state->majorana_positions[i];
        
        // Calculate color based on occupation
        Uint8 r, g, b;
        float hue = (i % 2 == 0) ? 60.0f : 280.0f;  // Yellow for γ_A, purple for γ_B
        float saturation = 0.8f;
        float lightness = 0.4f + 0.4f * occupation;  // Brighter for higher occupation
        
        hsl_to_rgb(hue, saturation, lightness, &r, &g, &b);
        
        // Draw Majorana operator
        SDL_SetRenderDrawColor(state->renderer, r, g, b, 255);
        int size = 15;
        SDL_Rect rect = {x - size/2, y - size/2, size, size};
        SDL_RenderFillRect(state->renderer, &rect);
        
        // Draw connections between paired Majoranas
        if (i % 2 == 0 && i+1 < state->num_majoranas) {
            double next_angle = (2.0 * M_PI * (i+1)) / state->num_majoranas;
            int next_x = center_x + (int)(circle_radius * cos(next_angle));
            int next_y = center_y + (int)(circle_radius * sin(next_angle));
            
            SDL_SetRenderDrawColor(state->renderer, 200, 200, 200, 180);
            SDL_RenderDrawLine(state->renderer, x, y, next_x, next_y);
        }
        
        // Spawn particles
        if (rand() % 100 < 5) {
            spawn_particle(state, x, y, 8.0f, hue);
        }
    }
    
    // Display information
    char buffer[100];
    sprintf(buffer, "Majorana Zero Modes - Chain Length: %d", chain_length);
    // Render text would go here
}

// Draw topological entropy visualization
void draw_topological_entropy(VisualizationState *state) {
    // Calculate parameters
    int center_x = WINDOW_WIDTH / 2;
    int center_y = WINDOW_HEIGHT / 2;
    int outer_radius = fmin(WINDOW_WIDTH, WINDOW_HEIGHT) * 3/8;
    
    // Draw background circular regions for Kitaev-Preskill calculation
    SDL_SetRenderDrawColor(state->renderer, 40, 40, 40, 255);
    SDL_Rect background = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
    SDL_RenderFillRect(state->renderer, &background);
    
    // Draw regions A, B, C
    // Region A (red)
    Uint8 r, g, b;
    hsl_to_rgb(0.0f, 0.7f, 0.5f, &r, &g, &b);
    SDL_SetRenderDrawColor(state->renderer, r, g, b, 180);
    for (int y = 0; y < WINDOW_HEIGHT; y++) {
        for (int x = 0; x < WINDOW_WIDTH; x++) {
            double dx = x - center_x;
            double dy = y - center_y;
            double dist = sqrt(dx*dx + dy*dy);
            double angle = atan2(dy, dx);
            
            if (dist < outer_radius && angle >= 0 && angle < 2.0*M_PI/3.0) {
                SDL_RenderDrawPoint(state->renderer, x, y);
            }
        }
    }
    
    // Region B (green)
    hsl_to_rgb(120.0f, 0.7f, 0.5f, &r, &g, &b);
    SDL_SetRenderDrawColor(state->renderer, r, g, b, 180);
    for (int y = 0; y < WINDOW_HEIGHT; y++) {
        for (int x = 0; x < WINDOW_WIDTH; x++) {
            double dx = x - center_x;
            double dy = y - center_y;
            double dist = sqrt(dx*dx + dy*dy);
            double angle = atan2(dy, dx);
            
            if (dist < outer_radius && angle >= 2.0*M_PI/3.0 && angle < 4.0*M_PI/3.0) {
                SDL_RenderDrawPoint(state->renderer, x, y);
            }
        }
    }
    
    // Region C (blue)
    hsl_to_rgb(240.0f, 0.7f, 0.5f, &r, &g, &b);
    SDL_SetRenderDrawColor(state->renderer, r, g, b, 180);
    for (int y = 0; y < WINDOW_HEIGHT; y++) {
        for (int x = 0; x < WINDOW_WIDTH; x++) {
            double dx = x - center_x;
            double dy = y - center_y;
            double dist = sqrt(dx*dx + dy*dy);
            double angle = atan2(dy, dx);
            
            if (dist < outer_radius && (angle >= 4.0*M_PI/3.0 || angle < 0)) {
                SDL_RenderDrawPoint(state->renderer, x, y);
            }
        }
    }
    
    // Draw particles based on topological entropy value
    double entropy_abs = fabs(state->topological_entropy);
    int num_particles = (int)(entropy_abs * 30);
    for (int i = 0; i < num_particles; i++) {
        double angle = ((double)rand() / RAND_MAX) * 2.0 * M_PI;
        double distance = ((double)rand() / RAND_MAX) * outer_radius;
        float x = center_x + distance * cos(angle);
        float y = center_y + distance * sin(angle);
        
        // Different particle colors for different regions
        float hue = 0.0f;
        if (angle >= 0 && angle < 2.0*M_PI/3.0) {
            hue = 0.0f;  // Red for region A
        } else if (angle >= 2.0*M_PI/3.0 && angle < 4.0*M_PI/3.0) {
            hue = 120.0f;  // Green for region B
        } else {
            hue = 240.0f;  // Blue for region C
        }
        
        spawn_particle(state, x, y, 8.0f, hue);
    }
    
    // Display information
    char buffer[100];
    sprintf(buffer, "Topological Entanglement Entropy: %.4f", state->topological_entropy);
    // Render text would go here
    
    // If entropy is non-zero, add some "quantum fluctuation" effects
    if (entropy_abs > 0.1) {
        for (int i = 0; i < 5; i++) {
            double angle = ((double)rand() / RAND_MAX) * 2.0 * M_PI;
            double distance = outer_radius * 0.5;
            float x = center_x + distance * cos(angle);
            float y = center_y + distance * sin(angle);
            float size = 20.0f + ((float)rand() / RAND_MAX) * 10.0f;
            float hue = ((float)rand() / RAND_MAX) * 360.0f;
            spawn_particle(state, x, y, size, hue);
        }
    }
}

// Update visualization state
void update_visualization(VisualizationState *state) {
    if (!state) return;
    
    // Process events
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            state->running = 0;
        } else if (event.type == SDL_KEYDOWN) {
            switch (event.key.keysym.sym) {
                case SDLK_ESCAPE:
                    state->running = 0;
                    break;
                case SDLK_1:
                    state->view_mode = 0;  // Berry phase
                    break;
                case SDLK_2:
                    state->view_mode = 1;  // Toric code
                    break;
                case SDLK_3:
                    state->view_mode = 2;  // Majorana modes
                    break;
                case SDLK_4:
                    state->view_mode = 3;  // Entropy
                    break;
            }
        }
    }
    
    // Clear screen
    SDL_SetRenderDrawColor(state->renderer, 0, 0, 0, 255);
    SDL_RenderClear(state->renderer);
    
    // Draw current visualization mode
    switch (state->view_mode) {
        case 0:
            draw_berry_curvature(state);
            break;
        case 1:
            draw_toric_code(state);
            break;
        case 2:
            draw_majorana_modes(state);
            break;
        case 3:
            draw_topological_entropy(state);
            break;
    }
    
    // Update and render particles
    update_particles(state);
    
    // Update screen
    SDL_RenderPresent(state->renderer);
}

// Clean up and shut down visualization
void cleanup_visualization(VisualizationState *state) {
    if (state) {
        if (state->renderer) {
            SDL_DestroyRenderer(state->renderer);
        }
        if (state->window) {
            SDL_DestroyWindow(state->window);
        }
        
        // Free visualization data
        if (state->berry_curvature) {
            free(state->berry_curvature);
        }
        if (state->toric_stabilizers) {
            free(state->toric_stabilizers);
        }
        if (state->majorana_positions) {
            free(state->majorana_positions);
        }
        
        free(state);
    }
    
    SDL_Quit();
}

// Set Berry curvature data
void set_berry_curvature_data(VisualizationState *state, double *curvature, int resolution, double invariant) {
    if (!state) return;
    
    if (state->berry_curvature) {
        free(state->berry_curvature);
        state->berry_curvature = NULL;
    }
    
    state->k_resolution = resolution;
    state->topological_invariant = invariant;
    
    int size = resolution * resolution;
    state->berry_curvature = malloc(size * sizeof(double));
    if (state->berry_curvature && curvature) {
        memcpy(state->berry_curvature, curvature, size * sizeof(double));
    }
}

// Set toric code data
void set_toric_code_data(VisualizationState *state, double *stabilizers, int size_x, int size_y) {
    if (!state) return;
    
    if (state->toric_stabilizers) {
        free(state->toric_stabilizers);
        state->toric_stabilizers = NULL;
    }
    
    state->toric_size_x = size_x;
    state->toric_size_y = size_y;
    
    int size = 2 * size_x * size_y;  // Plaquette + vertex operators
    state->toric_stabilizers = malloc(size * sizeof(double));
    if (state->toric_stabilizers && stabilizers) {
        memcpy(state->toric_stabilizers, stabilizers, size * sizeof(double));
    }
}

// Set Majorana modes data
void set_majorana_data(VisualizationState *state, double *positions, int num_modes) {
    if (!state) return;
    
    if (state->majorana_positions) {
        free(state->majorana_positions);
        state->majorana_positions = NULL;
    }
    
    state->num_majoranas = num_modes;
    
    state->majorana_positions = malloc(num_modes * sizeof(double));
    if (state->majorana_positions && positions) {
        memcpy(state->majorana_positions, positions, num_modes * sizeof(double));
    }
}

// Set topological entropy value
void set_topological_entropy(VisualizationState *state, double entropy) {
    if (!state) return;
    state->topological_entropy = entropy;
}

// Check if visualization is still running
int is_visualization_running(VisualizationState *state) {
    if (!state) return 0;
    return state->running;
}
