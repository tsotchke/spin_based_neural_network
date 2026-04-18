CC = gcc
CFLAGS_COMMON = -Wall -std=c11 -Iinclude
LDFLAGS = -lm

# Local build output directory
BIN_DIR = build

# ---- External engine integration -----------------------------------------
# The adapter (src/engine_adapter.c, src/nn_backend.c) compiles with
# `#ifdef SPIN_NN_HAS_ENGINE` guards. Scaffolding lives in the tree but
# stays dormant until an engine is wired in. Planned backends for v0.5+:
# an Eshkol-native NN engine (working title `eshkol-transformers`, built
# on https://github.com/tsotchke/eshkol) and Noesis (reasoning engine,
# not yet publicly released). Enable by pointing ENGINE_ROOT at the
# engine install and building with `make ENGINE_ENABLE=1`.
ENGINE_ENABLE ?= 0
ifeq ($(ENGINE_ENABLE),1)
  ENGINE_ROOT    ?= /usr/local/engine
  ENGINE_INCLUDE := $(ENGINE_ROOT)/include
  ENGINE_LIBDIR  := $(ENGINE_ROOT)/lib
  ENGINE_LIB     ?= spin_engine
  ENGINE_VERSION ?= dev
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Darwin)
    ENGINE_RPATH_FLAGS := -Wl,-rpath,$(ENGINE_LIBDIR)
  else
    ENGINE_RPATH_FLAGS := -Wl,-rpath,$(ENGINE_LIBDIR)
  endif
  ENGINE_CFLAGS  := -I$(ENGINE_INCLUDE) -DSPIN_NN_HAS_ENGINE=1 -DSPIN_NN_ENGINE_VERSION='"$(ENGINE_VERSION)"'
  ENGINE_LDFLAGS := -L$(ENGINE_LIBDIR) -l$(ENGINE_LIB) $(ENGINE_RPATH_FLAGS)
endif

# Create the local build directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Original source files
ORIGINAL_SRCS = src/main.c src/ising_model.c src/kitaev_model.c src/disordered_model.c \
               src/reinforcement_learning.c src/neural_network.c src/physics_loss.c \
               src/quantum_mechanics.c src/spin_models.c src/energy_utils.c

# New quantum computing files
QUANTUM_SRCS = src/majorana_modes.c src/topological_entropy.c src/toric_code.c \
              src/berry_phase.c src/ising_chain_qubits.c src/matrix_neon.c

# v0.4 foundation additions: backend-agnostic NN wrapper + Eshkol bridge
# (engine_adapter is linked only into engine-aware targets via ENGINE_LDFLAGS)
FOUNDATION_SRCS = src/nn_backend.c src/eshkol_bridge.c

SRCS = $(ORIGINAL_SRCS) $(QUANTUM_SRCS) $(FOUNDATION_SRCS)

# Default target builds both ARM and non-ARM versions
all: arm non_arm

# ARM version with NEON explicitly enabled
arm: CFLAGS = $(CFLAGS_COMMON) -O2 -DUSE_NEON -march=armv8-a+simd
arm: $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/spin_based_neural_computation_arm $(SRCS) $(LDFLAGS)

# Testing build: ARM + NEON + the SPIN_NN_TESTING gate that enables
# environment-variable overrides (CHERN_NUMBER, WINDING_NUMBER) used by the
# example scripts and integration tests.
arm_testing: CFLAGS = $(CFLAGS_COMMON) -O2 -DUSE_NEON -march=armv8-a+simd -DSPIN_NN_TESTING
arm_testing: $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/spin_based_neural_computation_arm_testing $(SRCS) $(LDFLAGS)

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

# ---- Test harness (v0.4 foundation) --------------------------------------
TEST_CFLAGS = $(CFLAGS_COMMON) -Itests -O2 -DUSE_NEON_IF_AVAILABLE

# Legacy entropy test (pre-v0.4)
test_topological_entropy: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_topological_entropy \
	    tests/test_topological_entropy.c \
	    src/topological_entropy.c src/kitaev_model.c src/matrix_neon.c \
	    $(LDFLAGS)

test_majorana: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_majorana \
	    tests/test_majorana.c src/majorana_modes.c src/kitaev_model.c $(LDFLAGS)

test_toric_code: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_toric_code \
	    tests/test_toric_code.c src/toric_code.c src/kitaev_model.c $(LDFLAGS)

test_ising: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_ising \
	    tests/test_ising.c src/ising_model.c $(LDFLAGS)

test_kitaev: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_kitaev \
	    tests/test_kitaev.c src/kitaev_model.c $(LDFLAGS)

test_engine_adapter: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_engine_adapter \
	    tests/test_engine_adapter.c src/engine_adapter.c \
	    src/ising_model.c src/kitaev_model.c src/spin_models.c $(LDFLAGS)

test_nn_backend: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_nn_backend \
	    tests/test_nn_backend.c src/nn_backend.c src/neural_network.c $(LDFLAGS)

test_spin_models: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_spin_models \
	    tests/test_spin_models.c src/spin_models.c $(LDFLAGS)

test_energy_utils: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_energy_utils \
	    tests/test_energy_utils.c src/energy_utils.c $(LDFLAGS)

test_disordered_model: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_disordered_model \
	    tests/test_disordered_model.c src/disordered_model.c \
	    src/ising_model.c src/kitaev_model.c $(LDFLAGS)

test_eshkol_bridge: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_eshkol_bridge \
	    tests/test_eshkol_bridge.c src/eshkol_bridge.c $(LDFLAGS)

test_physics_loss: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_physics_loss \
	    tests/test_physics_loss.c src/physics_loss.c \
	    src/ising_model.c src/kitaev_model.c src/spin_models.c $(LDFLAGS)

test_berry_phase: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_berry_phase \
	    tests/test_berry_phase.c src/berry_phase.c \
	    src/kitaev_model.c src/majorana_modes.c src/matrix_neon.c $(LDFLAGS)

test_reinforcement_learning: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_reinforcement_learning \
	    tests/test_reinforcement_learning.c src/reinforcement_learning.c \
	    src/ising_model.c src/kitaev_model.c $(LDFLAGS)

test_quantum_mechanics: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_quantum_mechanics \
	    tests/test_quantum_mechanics.c src/quantum_mechanics.c \
	    src/ising_model.c src/kitaev_model.c src/spin_models.c $(LDFLAGS)

test_ising_chain_qubits: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_ising_chain_qubits \
	    tests/test_ising_chain_qubits.c src/ising_chain_qubits.c \
	    src/majorana_modes.c src/kitaev_model.c $(LDFLAGS)

test_matrix_neon: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_matrix_neon \
	    tests/test_matrix_neon.c src/matrix_neon.c $(LDFLAGS)

# Aggregate: build and run every v0.4 test.  First failure propagates exit code.
test: test_majorana test_toric_code test_ising test_kitaev test_topological_entropy \
      test_engine_adapter test_nn_backend test_spin_models test_energy_utils \
      test_disordered_model test_eshkol_bridge test_physics_loss test_berry_phase \
      test_reinforcement_learning test_quantum_mechanics test_ising_chain_qubits \
      test_matrix_neon
	@echo "=== Running v0.4 foundation test suite ==="
	@$(BIN_DIR)/test_majorana
	@$(BIN_DIR)/test_toric_code
	@$(BIN_DIR)/test_ising
	@$(BIN_DIR)/test_kitaev
	@$(BIN_DIR)/test_topological_entropy
	@$(BIN_DIR)/test_engine_adapter
	@$(BIN_DIR)/test_nn_backend
	@$(BIN_DIR)/test_spin_models
	@$(BIN_DIR)/test_energy_utils
	@$(BIN_DIR)/test_disordered_model
	@$(BIN_DIR)/test_eshkol_bridge
	@$(BIN_DIR)/test_physics_loss
	@$(BIN_DIR)/test_berry_phase
	@$(BIN_DIR)/test_reinforcement_learning
	@$(BIN_DIR)/test_quantum_mechanics
	@$(BIN_DIR)/test_ising_chain_qubits
	@$(BIN_DIR)/test_matrix_neon

# ---- Benchmark harness (v0.4 foundation) ---------------------------------
BENCH_CFLAGS = $(CFLAGS_COMMON) -Ibenchmarks -O2 -DUSE_NEON_IF_AVAILABLE

bench_ising_bin: $(BIN_DIR)
	$(CC) $(BENCH_CFLAGS) -o $(BIN_DIR)/bench_ising \
	    benchmarks/bench_ising.c src/ising_model.c $(LDFLAGS)

bench_kitaev_bin: $(BIN_DIR)
	$(CC) $(BENCH_CFLAGS) -o $(BIN_DIR)/bench_kitaev \
	    benchmarks/bench_kitaev.c src/kitaev_model.c $(LDFLAGS)

bench_majorana_braid_bin: $(BIN_DIR)
	$(CC) $(BENCH_CFLAGS) -o $(BIN_DIR)/bench_majorana_braid \
	    benchmarks/bench_majorana_braid.c src/majorana_modes.c $(LDFLAGS)

bench_toric_decoder_bin: $(BIN_DIR)
	$(CC) $(BENCH_CFLAGS) -o $(BIN_DIR)/bench_toric_decoder \
	    benchmarks/bench_toric_decoder.c src/toric_code.c $(LDFLAGS)

bench: bench_ising_bin bench_kitaev_bin bench_majorana_braid_bin bench_toric_decoder_bin
	@echo "=== v0.4 benchmarks built. Run: ./scripts/run_benchmarks.sh ==="

# ---- Stack probe --------------------------------------------------------
check_stack:
	@sh scripts/check_stack.sh

clean:
	rm -rf $(BIN_DIR)
