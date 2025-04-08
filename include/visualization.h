#ifndef VISUALIZATION_H
#define VISUALIZATION_H

typedef struct VisualizationState VisualizationState;

// Initialize the visualization system
VisualizationState* init_visualization();

// Clean up and shut down visualization
void cleanup_visualization(VisualizationState *state);

// Update visualization state
void update_visualization(VisualizationState *state);

// Set Berry curvature data
void set_berry_curvature_data(VisualizationState *state, double *curvature, int resolution, double invariant);

// Set toric code data
void set_toric_code_data(VisualizationState *state, double *stabilizers, int size_x, int size_y);

// Set Majorana modes data
void set_majorana_data(VisualizationState *state, double *positions, int num_modes);

// Set topological entropy value
void set_topological_entropy(VisualizationState *state, double entropy);

// Check if visualization is still running
int is_visualization_running(VisualizationState *state);

#endif // VISUALIZATION_H
