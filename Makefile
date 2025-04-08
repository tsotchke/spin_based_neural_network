CC = gcc
CFLAGS_COMMON = -Wall -std=c11 -Iinclude
LDFLAGS = -lm

# Binary directory
BIN_DIR = bin

# Create bin directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Original source files
ORIGINAL_SRCS = src/main.c src/ising_model.c src/kitaev_model.c src/disordered_model.c \
               src/reinforcement_learning.c src/neural_network.c src/physics_loss.c \
               src/quantum_mechanics.c src/spin_models.c src/energy_utils.c

# New quantum computing files
QUANTUM_SRCS = src/majorana_modes.c src/topological_entropy.c src/toric_code.c \
              src/berry_phase.c src/ising_chain_qubits.c src/matrix_neon.c

SRCS = $(ORIGINAL_SRCS) $(QUANTUM_SRCS)

# Default target builds both ARM and non-ARM versions
all: arm non_arm

# ARM version with NEON explicitly enabled
arm: CFLAGS = $(CFLAGS_COMMON) -O2 -DUSE_NEON -march=armv8-a+simd
arm: $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/spin_based_neural_computation_arm $(SRCS) $(LDFLAGS)

# Non-ARM version with NEON disabled
non_arm: CFLAGS = $(CFLAGS_COMMON) -O2 -DDISABLE_NEON
non_arm: $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/spin_based_neural_computation_generic $(SRCS) $(LDFLAGS)

# Universal version with runtime detection of NEON support
universal: CFLAGS = $(CFLAGS_COMMON) -O2 -DUSE_NEON_IF_AVAILABLE
universal: $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/spin_based_neural_computation $(SRCS) $(LDFLAGS)

# Build everything
all_versions: arm non_arm universal

# SDL flags for visualization (using pkg-config for cross-platform compatibility)
SDL_CFLAGS = $(shell pkg-config --cflags sdl2 2>/dev/null || echo "-I/usr/include/SDL2 -I/usr/local/include/SDL2 -I/opt/homebrew/include/SDL2")
SDL_LDFLAGS = $(shell pkg-config --libs sdl2 2>/dev/null || echo "-lSDL2")

# Build topo example
topo_example: src/topological_example.c $(BIN_DIR)
	$(CC) $(CFLAGS_COMMON) -O2 -DUSE_NEON -march=armv8-a+simd -o $(BIN_DIR)/topo_example src/topological_example.c src/kitaev_model.c $(QUANTUM_SRCS) $(LDFLAGS)

# Build visualization
visualization: src/visualization.c src/visualization_main.c $(BIN_DIR)
	$(CC) $(CFLAGS_COMMON) $(SDL_CFLAGS) -O2 -o $(BIN_DIR)/visualization src/visualization.c src/visualization_main.c $(LDFLAGS) $(SDL_LDFLAGS)

# Test sources
TEST_SRC = tests/test_topological_entropy.c
TEST_DEPS = src/topological_entropy.c src/kitaev_model.c src/matrix_neon.c

# Build and run tests
test: $(BIN_DIR)
	$(CC) $(CFLAGS_COMMON) -O2 -DUSE_NEON_IF_AVAILABLE -o $(BIN_DIR)/test_topological_entropy $(TEST_SRC) $(TEST_DEPS) $(LDFLAGS)
	$(BIN_DIR)/test_topological_entropy

clean:
	rm -rf $(BIN_DIR)
