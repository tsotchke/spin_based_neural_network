CC = gcc
# -Wall -Wextra is the default warning baseline. Additional strictness
# (-Wpedantic, -Wshadow, -Wconversion) can be layered on for targeted
# hardening runs via `make CFLAGS_EXTRA='-Wshadow -Wconversion' test`.
#
# -D_GNU_SOURCE exposes M_PI and friends via glibc's <math.h>. macOS's
# libc ships them under its default feature-test macros, but glibc
# with -std=c11 (strict ISO) hides them unless _GNU_SOURCE (or
# _DEFAULT_SOURCE / _XOPEN_SOURCE=600) is set. Using _GNU_SOURCE here
# keeps the same code compiling on both platforms without per-file
# guards.
CFLAGS_COMMON = -Wall -Wextra -std=c11 -D_GNU_SOURCE -Iinclude $(CFLAGS_EXTRA)
LDFLAGS = -lm

# Host architecture detection. Targets that unconditionally used
# `-march=armv8-a+simd` (topo_example, arm, arm_testing) broke on
# x86_64 CI runners. The topo_example target now picks an arch-
# appropriate flag; the explicitly-named `arm*` targets still force
# ARM and are guarded by being renamed / skipped on non-ARM hosts.
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
  ARCH_NEON_FLAGS := -DUSE_NEON -march=armv8-a+simd
else ifeq ($(UNAME_M),aarch64)
  ARCH_NEON_FLAGS := -DUSE_NEON -march=armv8-a+simd
else
  ARCH_NEON_FLAGS := -DDISABLE_NEON
endif

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
# ---- libirrep integration (dormant by default) ---------------------------
# Enable with: make IRREP_ENABLE=1 IRREP_ROOT=/path/to/libirrep/install
# The libirrep_bridge.c translation unit links against the library's
# public C ABI. Version support starts at libirrep 1.0.
IRREP_ENABLE ?= 0
ifeq ($(IRREP_ENABLE),1)
  IRREP_ROOT    ?= /usr/local/libirrep
  IRREP_INCLUDE := $(IRREP_ROOT)/include
  IRREP_LIBDIR  := $(IRREP_ROOT)/lib
  IRREP_LIB     ?= libirrep
  IRREP_CFLAGS  := -I$(IRREP_INCLUDE) -DSPIN_NN_HAS_IRREP=1
  IRREP_LDFLAGS := -L$(IRREP_LIBDIR) -l$(IRREP_LIB) -Wl,-rpath,$(IRREP_LIBDIR)
endif

MOONLAB_ENABLE ?= 0
ifeq ($(MOONLAB_ENABLE),1)
  ifndef MOONLAB_ROOT
    $(error MOONLAB_ENABLE=1 requires MOONLAB_ROOT to be set explicitly. \
            Example: make MOONLAB_ENABLE=1 MOONLAB_ROOT=/path/to/quantum_simulator)
  endif
  MOONLAB_INCLUDE := $(MOONLAB_ROOT)/src
  MOONLAB_LIBDIR  := $(MOONLAB_ROOT)/build
  MOONLAB_LIB     ?= quantumsim
  MOONLAB_CFLAGS  := -I$(MOONLAB_INCLUDE) -DSPIN_NN_HAS_MOONLAB=1
  MOONLAB_LDFLAGS := -L$(MOONLAB_LIBDIR) -l$(MOONLAB_LIB) -Wl,-rpath,$(MOONLAB_LIBDIR)
endif

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

# Pillar source groups. Defined here (before any rule references them) so the
# Makefile is order-independent: new test rules can consume any pillar's
# sources regardless of which pillar section they land in below.
NQS_SRCS  = src/nqs/nqs_sampler.c src/nqs/nqs_gradient.c \
            src/nqs/nqs_ansatz.c  src/nqs/nqs_optimizer.c \
            src/nqs/nqs_marshall.c src/nqs/nqs_translation.c \
            src/nqs/nqs_diagnostics.c
MPS_SRCS  = src/mps/lanczos.c src/mps/mps.c src/mps/svd.c src/mps/dmrg.c src/mps/tebd.c
LLG_SRCS  = src/llg/llg.c src/llg/exchange_field.c src/llg/demag.c
QEC_SRCS  = src/qec_decoder/qec_decoder.c
FNO_SRCS  = src/neural_operator/neural_operator.c src/neural_operator/fft.c
FLOW_SRCS = src/flow_matching/flow_matching.c
FIBO_SRCS = src/fibonacci_anyons/fibonacci_anyons.c

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

# SDL flags for visualization. pkg-config is the primary (cross-distro) path;
# the hardcoded fallback covers common install locations for systems without
# pkg-config (Linux stock, /usr/local, Homebrew, MacPorts, Nix common prefix).
# Visualization-free builds never consume these, so failures here only affect
# `make visualization`.
SDL_PKGCONFIG_CFLAGS := $(shell pkg-config --cflags sdl2 2>/dev/null)
SDL_PKGCONFIG_LDFLAGS := $(shell pkg-config --libs sdl2 2>/dev/null)
SDL_FALLBACK_CFLAGS  = -I/usr/include/SDL2 -I/usr/local/include/SDL2 \
                       -I/opt/homebrew/include/SDL2 -I/opt/local/include/SDL2
SDL_FALLBACK_LDFLAGS = -lSDL2
SDL_CFLAGS  = $(if $(SDL_PKGCONFIG_CFLAGS),$(SDL_PKGCONFIG_CFLAGS),$(SDL_FALLBACK_CFLAGS))
SDL_LDFLAGS = $(if $(SDL_PKGCONFIG_LDFLAGS),$(SDL_PKGCONFIG_LDFLAGS),$(SDL_FALLBACK_LDFLAGS))

# One-line diagnostic target so users can verify what the SDL resolution picked.
sdl_check:
	@echo "SDL_CFLAGS  = $(SDL_CFLAGS)"
	@echo "SDL_LDFLAGS = $(SDL_LDFLAGS)"
	@if [ -z "$(SDL_PKGCONFIG_LDFLAGS)" ]; then \
	    echo "note: pkg-config did not resolve sdl2; using hardcoded fallback paths."; \
	    echo "      If visualization fails to compile, install SDL2 development"; \
	    echo "      headers (e.g. 'brew install sdl2', 'apt install libsdl2-dev',"; \
	    echo "      'port install libsdl2') or install pkg-config."; \
	fi

# Build topo example. On ARM64 hosts this picks up NEON via
# ARCH_NEON_FLAGS; on x86_64 it falls back to the disable-NEON
# path so the target builds cleanly on every CI runner.
topo_example: src/topological_example.c $(BIN_DIR)
	$(CC) $(CFLAGS_COMMON) -O2 $(ARCH_NEON_FLAGS) -o $(BIN_DIR)/topo_example src/topological_example.c src/kitaev_model.c $(QUANTUM_SRCS) $(LDFLAGS)

# Build visualization
visualization: src/visualization.c src/visualization_main.c $(BIN_DIR)
	$(CC) $(CFLAGS_COMMON) $(SDL_CFLAGS) -O2 -o $(BIN_DIR)/visualization src/visualization.c src/visualization_main.c $(LDFLAGS) $(SDL_LDFLAGS)

# ---- Test harness (v0.4 foundation) --------------------------------------
# SANITIZE=1 turns on AddressSanitizer + UndefinedBehaviorSanitizer for
# tests. Use for local hardening runs and in CI's "sanitize" job. Adds
# ~2x build time and slows tests ~2-3x, so not the default.
SANITIZE ?= 0
ifeq ($(SANITIZE),1)
  SANITIZE_FLAGS := -fsanitize=address,undefined -fno-omit-frame-pointer
  # -O1 for better sanitizer trace quality; still fast enough for tests.
  TEST_OPT_LEVEL := -O1 -g
else
  SANITIZE_FLAGS :=
  TEST_OPT_LEVEL := -O2
endif

TEST_CFLAGS = $(CFLAGS_COMMON) -Itests $(TEST_OPT_LEVEL) -DUSE_NEON_IF_AVAILABLE $(SANITIZE_FLAGS)

# `make test_asan` — convenience alias for a full sanitizer-instrumented
# run. Equivalent to `make clean && make SANITIZE=1 test`. Catches use-after-
# free, buffer overruns, signed-integer overflow, shift-out-of-range,
# alignment, and null dereferences. Leak detection is Linux-only (macOS
# ASan runtime aborts at startup with detect_leaks=1), so the default
# ASAN_OPTIONS here toggle it by platform. UBSan is configured to halt on
# the first finding so the exit code reflects correctness.
ASAN_LEAK_FLAG := $(if $(filter Darwin,$(shell uname -s)),0,1)
test_asan:
	@$(MAKE) --no-print-directory clean >/dev/null 2>&1
	@UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1" \
	 ASAN_OPTIONS="detect_leaks=$(ASAN_LEAK_FLAG):abort_on_error=0" \
	 $(MAKE) --no-print-directory SANITIZE=1 test

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

test_toric_code_mwpm: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_toric_code_mwpm \
	    tests/test_toric_code_mwpm.c src/toric_code.c src/kitaev_model.c $(LDFLAGS)

test_ising: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_ising \
	    tests/test_ising.c src/ising_model.c $(LDFLAGS)

test_ising_sw: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_ising_sw \
	    tests/test_ising_sw.c src/ising_model.c $(LDFLAGS)

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

# v0.5 pillar P1.1 — Neural Network Quantum States scaffold.
# (NQS_SRCS is defined in the pillar source-group block near the top.)
test_nqs: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_nqs \
	    tests/test_nqs.c $(NQS_SRCS) $(LDFLAGS)

test_nqs_convergence: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_nqs_convergence \
	    tests/test_nqs_convergence.c $(NQS_SRCS) $(LDFLAGS)

test_nqs_rbm: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_nqs_rbm \
	    tests/test_nqs_rbm.c $(NQS_SRCS) $(LDFLAGS)

test_nqs_lanczos: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_nqs_lanczos \
	    tests/test_nqs_lanczos.c $(NQS_SRCS) src/nqs/nqs_lanczos.c \
	    src/mps/lanczos.c $(LDFLAGS)

test_nqs_marshall: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_nqs_marshall \
	    tests/test_nqs_marshall.c $(NQS_SRCS) $(LDFLAGS)

test_nqs_complex_rbm: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_nqs_complex_rbm \
	    tests/test_nqs_complex_rbm.c $(NQS_SRCS) $(LDFLAGS)

test_nqs_holomorphic_sr: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_nqs_holomorphic_sr \
	    tests/test_nqs_holomorphic_sr.c $(NQS_SRCS) $(LDFLAGS)

test_nqs_kitaev: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_nqs_kitaev \
	    tests/test_nqs_kitaev.c $(NQS_SRCS) $(LDFLAGS)

test_nqs_kagome: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_nqs_kagome \
	    tests/test_nqs_kagome.c $(NQS_SRCS) $(LDFLAGS)

test_nqs_chi_F: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_nqs_chi_F \
	    tests/test_nqs_chi_F.c $(NQS_SRCS) $(LDFLAGS)

test_nqs_excited: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_nqs_excited \
	    tests/test_nqs_excited.c $(NQS_SRCS) $(LDFLAGS)

test_nqs_minsr: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_nqs_minsr \
	    tests/test_nqs_minsr.c $(NQS_SRCS) $(LDFLAGS)

# Research-scale convergence runner for the kagome Heisenberg S=1/2
# open ground-state question. Takes minutes, NOT wired into `make test`.
# Run manually: `make research_kagome_N12 && ./build/research_kagome_N12`.
research_kagome_N12: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/research_kagome_N12 \
	    scripts/research_kagome_N12_convergence.c $(NQS_SRCS) $(LDFLAGS)

# End-to-end diagnostics driver: GS SR → χ_F → per-bond-class phase →
# excited-state SR → spin-gap estimate, all on the same N=12 PBC kagome
# cluster. O(10 min) on an M-series Mac. NOT wired into `make test`.
research_kagome_N12_diagnostics: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/research_kagome_N12_diagnostics \
	    scripts/research_kagome_N12_diagnostics.c \
	    $(NQS_SRCS) src/nqs/nqs_lanczos.c src/mps/lanczos.c $(LDFLAGS)

test_nqs_translation: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_nqs_translation \
	    tests/test_nqs_translation.c $(NQS_SRCS) $(LDFLAGS)

test_nqs_tvmc: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_nqs_tvmc \
	    tests/test_nqs_tvmc.c $(NQS_SRCS) $(LDFLAGS)

test_hopfield: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_hopfield \
	    tests/test_hopfield.c src/thermodynamic/hopfield.c $(LDFLAGS)

test_noesis_bridge: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_noesis_bridge \
	    tests/test_noesis_bridge.c src/noesis_bridge.c $(LDFLAGS)

test_qgtl_bridge: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_qgtl_bridge \
	    tests/test_qgtl_bridge.c src/qgtl_bridge.c $(LDFLAGS)

test_qllm_bridge: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_qllm_bridge \
	    tests/test_qllm_bridge.c src/qllm_bridge.c $(LDFLAGS)

test_rbm_cd: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_rbm_cd \
	    tests/test_rbm_cd.c src/thermodynamic/rbm_cd.c $(LDFLAGS)

test_torque_net: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_torque_net \
	    tests/test_torque_net.c src/equivariant_gnn/torque_net.c $(LDFLAGS)

test_torque_net_llg: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_torque_net_llg \
	    tests/test_torque_net_llg.c \
	    src/equivariant_gnn/torque_net.c \
	    src/equivariant_gnn/llg_adapter.c \
	    src/llg/llg.c $(LDFLAGS)

test_torque_net_golden: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_torque_net_golden \
	    tests/test_downstream_compat/test_torque_net_golden.c \
	    src/equivariant_gnn/torque_net.c $(LDFLAGS)

test_torque_net_heisenberg_fit: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_torque_net_heisenberg_fit \
	    tests/test_torque_net_heisenberg_fit.c \
	    src/equivariant_gnn/torque_net.c $(LDFLAGS)

test_siren: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_siren \
	    tests/test_siren.c src/neural_network.c $(LDFLAGS)

test_nqs_xxz: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_nqs_xxz \
	    tests/test_nqs_xxz.c $(NQS_SRCS) $(MPS_SRCS) $(LDFLAGS)

test_pillar_integration: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_pillar_integration \
	    tests/test_pillar_integration.c $(NQS_SRCS) $(MPS_SRCS) $(LDFLAGS)

test_libirrep_bridge: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) $(IRREP_CFLAGS) -o $(BIN_DIR)/test_libirrep_bridge \
	    tests/test_libirrep_bridge.c src/libirrep_bridge.c \
	    $(LDFLAGS) $(IRREP_LDFLAGS)

test_torque_net_irrep: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) $(IRREP_CFLAGS) -o $(BIN_DIR)/test_torque_net_irrep \
	    tests/test_torque_net_irrep.c src/libirrep_bridge.c \
	    $(LDFLAGS) $(IRREP_LDFLAGS)

test_moonlab_bridge: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) $(MOONLAB_CFLAGS) -o $(BIN_DIR)/test_moonlab_bridge \
	    tests/test_moonlab_bridge.c src/moonlab_bridge.c \
	    $(LDFLAGS) $(MOONLAB_LDFLAGS)

test_thqcp_coupling: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_thqcp_coupling \
	    tests/test_thqcp_coupling.c src/thqcp/coupling.c $(LDFLAGS)

# v0.5 pillar P1.3b — Fibonacci anyons (full landing).
# (FIBO_SRCS is defined in the pillar source-group block near the top.)
test_fibonacci_anyons: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_fibonacci_anyons \
	    tests/test_fibonacci_anyons.c $(FIBO_SRCS) $(LDFLAGS)

# v0.6+ pillars P2.2 + P2.6 — MPS + DMRG + Lanczos substrate.
# (MPS_SRCS is defined in the pillar source-group block near the top.)
test_mps: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_mps \
	    tests/test_mps.c $(MPS_SRCS) $(LDFLAGS)

test_mps_svd: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_mps_svd \
	    tests/test_mps_svd.c src/mps/svd.c $(LDFLAGS)

test_mps_dmrg: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_mps_dmrg \
	    tests/test_mps_dmrg.c $(MPS_SRCS) $(LDFLAGS)

test_mps_tebd: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_mps_tebd \
	    tests/test_mps_tebd.c $(MPS_SRCS) $(LDFLAGS)

test_pbit: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_pbit \
	    tests/test_pbit.c src/neuromorphic/pbit.c $(LDFLAGS)

# v0.5 pillar P1.2 — Landau-Lifshitz-Gilbert dynamics scaffold.
# (LLG_SRCS is defined in the pillar source-group block near the top.)
test_llg: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_llg \
	    tests/test_llg.c $(LLG_SRCS) $(FNO_SRCS) $(LDFLAGS)

test_llg_spin_wave: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_llg_spin_wave \
	    tests/test_llg_spin_wave.c $(LLG_SRCS) $(FNO_SRCS) $(LDFLAGS)

test_llg_demag: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_llg_demag \
	    tests/test_llg_demag.c $(LLG_SRCS) $(FNO_SRCS) $(LDFLAGS)

test_llg_2d: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_llg_2d \
	    tests/test_llg_2d.c $(LLG_SRCS) $(FNO_SRCS) $(LDFLAGS)

# v0.5 pillar P1.3 — Learned QEC decoder scaffold (greedy fall-back).
# (QEC_SRCS is defined in the pillar source-group block near the top.)
test_qec_decoder: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_qec_decoder \
	    tests/test_qec_decoder.c $(QEC_SRCS) src/toric_code.c src/kitaev_model.c \
	    $(LDFLAGS)

# v0.5 pillar P1.4 — Neural operator (FNO) + flow-matching scaffolds.
# (FNO_SRCS / FLOW_SRCS are defined in the pillar source-group block near the top.)
test_neural_operator: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_neural_operator \
	    tests/test_neural_operator.c $(FNO_SRCS) $(LDFLAGS)

test_fft: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_fft \
	    tests/test_fft.c $(FNO_SRCS) $(LDFLAGS)
test_flow_matching: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/test_flow_matching \
	    tests/test_flow_matching.c $(FLOW_SRCS) $(LDFLAGS)

# Aggregate: build and run every v0.4 test.  First failure propagates exit code.
test: test_majorana test_toric_code test_ising test_ising_sw test_kitaev test_topological_entropy \
      test_engine_adapter test_nn_backend test_spin_models test_energy_utils \
      test_disordered_model test_eshkol_bridge test_physics_loss test_berry_phase \
      test_reinforcement_learning test_quantum_mechanics test_ising_chain_qubits \
      test_matrix_neon test_nqs test_libirrep_bridge \
      test_nqs_convergence test_nqs_rbm test_nqs_complex_rbm test_nqs_holomorphic_sr \
      test_nqs_kitaev test_nqs_lanczos test_nqs_marshall test_nqs_translation \
      test_nqs_tvmc test_nqs_xxz test_nqs_kagome test_nqs_chi_F test_nqs_excited test_nqs_minsr \
      test_hopfield test_rbm_cd \
      test_torque_net test_torque_net_llg test_torque_net_golden \
      test_torque_net_heisenberg_fit \
      test_siren \
      test_thqcp_coupling test_noesis_bridge test_qgtl_bridge test_qllm_bridge \
      test_fibonacci_anyons test_mps \
      test_llg test_llg_spin_wave test_llg_demag test_llg_2d test_qec_decoder test_toric_code_mwpm \
      test_neural_operator test_fft test_flow_matching \
      test_mps_svd test_mps_dmrg test_mps_tebd test_pbit test_pillar_integration
	@echo "=== Running v0.4 foundation test suite ==="
	@$(BIN_DIR)/test_majorana
	@$(BIN_DIR)/test_toric_code
	@$(BIN_DIR)/test_ising
	@$(BIN_DIR)/test_ising_sw
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
	@$(BIN_DIR)/test_nqs
	@$(BIN_DIR)/test_libirrep_bridge
	@$(BIN_DIR)/test_nqs_convergence
	@$(BIN_DIR)/test_nqs_rbm
	@$(BIN_DIR)/test_nqs_complex_rbm
	@$(BIN_DIR)/test_nqs_holomorphic_sr
	@$(BIN_DIR)/test_nqs_kitaev
	@$(BIN_DIR)/test_nqs_kagome
	@$(BIN_DIR)/test_nqs_chi_F
	@$(BIN_DIR)/test_nqs_excited
	@$(BIN_DIR)/test_nqs_minsr
	@$(BIN_DIR)/test_nqs_lanczos
	@$(BIN_DIR)/test_nqs_marshall
	@$(BIN_DIR)/test_nqs_translation
	@$(BIN_DIR)/test_nqs_tvmc
	@$(BIN_DIR)/test_nqs_xxz
	@$(BIN_DIR)/test_hopfield
	@$(BIN_DIR)/test_rbm_cd
	@$(BIN_DIR)/test_torque_net
	@$(BIN_DIR)/test_torque_net_llg
	@$(BIN_DIR)/test_torque_net_golden
	@$(BIN_DIR)/test_torque_net_heisenberg_fit
	@$(BIN_DIR)/test_siren
	@$(BIN_DIR)/test_thqcp_coupling
	@$(BIN_DIR)/test_noesis_bridge
	@$(BIN_DIR)/test_qgtl_bridge
	@$(BIN_DIR)/test_qllm_bridge
	@$(BIN_DIR)/test_fibonacci_anyons
	@$(BIN_DIR)/test_mps
	@$(BIN_DIR)/test_llg
	@$(BIN_DIR)/test_llg_spin_wave
	@$(BIN_DIR)/test_llg_demag
	@$(BIN_DIR)/test_llg_2d
	@$(BIN_DIR)/test_qec_decoder
	@$(BIN_DIR)/test_toric_code_mwpm
	@$(BIN_DIR)/test_neural_operator
	@$(BIN_DIR)/test_fft
	@$(BIN_DIR)/test_flow_matching
	@$(BIN_DIR)/test_mps_svd
	@$(BIN_DIR)/test_mps_dmrg
	@$(BIN_DIR)/test_mps_tebd
	@$(BIN_DIR)/test_pbit
	@$(BIN_DIR)/test_pillar_integration

# ---- Benchmark harness (v0.4 foundation) ---------------------------------
BENCH_CFLAGS = $(CFLAGS_COMMON) -Ibenchmarks -O2 -DUSE_NEON_IF_AVAILABLE

# Benchmark targets: target name matches binary name under build/.
# (Historical `_bin` suffixes removed 2026-04-18 so scripts/run_benchmarks.sh
# and Makefile share the same vocabulary.)
bench_ising: $(BIN_DIR)
	$(CC) $(BENCH_CFLAGS) -o $(BIN_DIR)/bench_ising \
	    benchmarks/bench_ising.c src/ising_model.c $(LDFLAGS)

bench_kitaev: $(BIN_DIR)
	$(CC) $(BENCH_CFLAGS) -o $(BIN_DIR)/bench_kitaev \
	    benchmarks/bench_kitaev.c src/kitaev_model.c $(LDFLAGS)

bench_majorana_braid: $(BIN_DIR)
	$(CC) $(BENCH_CFLAGS) -o $(BIN_DIR)/bench_majorana_braid \
	    benchmarks/bench_majorana_braid.c src/majorana_modes.c $(LDFLAGS)

bench_toric_decoder: $(BIN_DIR)
	$(CC) $(BENCH_CFLAGS) -o $(BIN_DIR)/bench_toric_decoder \
	    benchmarks/bench_toric_decoder.c src/toric_code.c $(LDFLAGS)

bench_nqs: $(BIN_DIR)
	$(CC) $(BENCH_CFLAGS) -o $(BIN_DIR)/bench_nqs \
	    benchmarks/bench_nqs.c $(NQS_SRCS) $(LDFLAGS)

bench_thqcp: $(BIN_DIR)
	$(CC) $(BENCH_CFLAGS) -o $(BIN_DIR)/bench_thqcp \
	    benchmarks/bench_thqcp.c src/thqcp/coupling.c $(LDFLAGS)

bench: bench_ising bench_kitaev bench_majorana_braid bench_toric_decoder bench_nqs bench_thqcp
	@echo "=== benchmarks built. Run: ./scripts/run_benchmarks.sh ==="

# ---- Downstream-compatibility golden-vector generators ------------------
# Utilities that regenerate the golden JSON vendored into libirrep 1.2.
# Not run by `make test`; invoke explicitly when updating the snapshots.

generate_lattice_connectivity: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/generate_lattice_connectivity \
	    tests/test_downstream_compat/generate_lattice_connectivity.c \
	    src/equivariant_gnn/torque_net.c $(LDFLAGS)

generate_torque_net_golden: $(BIN_DIR)
	$(CC) $(TEST_CFLAGS) -o $(BIN_DIR)/generate_torque_net_golden \
	    tests/test_downstream_compat/torque_net_tp_paths/generate_golden.c \
	    src/equivariant_gnn/torque_net.c $(LDFLAGS)

# ---- Stack probe --------------------------------------------------------
check_stack:
	@sh scripts/check_stack.sh

clean:
	rm -rf $(BIN_DIR)
