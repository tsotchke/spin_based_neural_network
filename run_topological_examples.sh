#!/bin/bash
# run_topological_examples.sh — demonstrate the framework across several
# topological phases by forcing specific Chern / winding values and
# Majorana-chain parameters. Uses the `arm_testing` build, which enables
# the SPIN_NN_TESTING compile-time flag required to honor the
# CHERN_NUMBER / WINDING_NUMBER environment-variable overrides. These
# overrides are gated out of the default release build.

set -e

# Build the test-mode binary (includes -DSPIN_NN_TESTING)
make arm_testing

BIN=./build/spin_based_neural_computation_arm_testing

echo "==== Example 1: Z2 Topological Insulator (Chern number = 1, Winding number = 1) ===="
export CHERN_NUMBER=1
export WINDING_NUMBER=1
export MAJORANA_MU=0.5    # |μ| < 2|t| for topological phase
export MAJORANA_T=1.0     # Hopping parameter
export MAJORANA_DELTA=1.0 # Pairing strength
"$BIN" --iterations 10 --verbose --calculate-entropy \
    --calculate-invariants --use-error-correction --majorana-chain-length 3 \
    --toric-code-size "2 2"

echo -e "\n\n==== Example 2: Quantum Spin Hall Effect (Chern number = 2, Winding number = 0) ===="
# Set winding number to 0 (trivial phase) for this example
export CHERN_NUMBER=2
export WINDING_NUMBER=0
export MAJORANA_MU=2.5    # |μ| > 2|t| for trivial phase
export MAJORANA_T=1.0     # Hopping parameter
export MAJORANA_DELTA=1.0 # Pairing strength
"$BIN" --iterations 10 --verbose --calculate-entropy \
    --calculate-invariants --use-error-correction --majorana-chain-length 5 \
    --toric-code-size "3 3"

echo -e "\n\n==== Example 3: Fractional Quantum Hall Effect (Chern number = 1/3, Winding number = -1) ===="
export CHERN_NUMBER=0.333
export WINDING_NUMBER=-1
export MAJORANA_MU=0.2    # Small chemical potential
export MAJORANA_T=1.0     # Hopping parameter
export MAJORANA_DELTA=-1.0 # Negative pairing strength
"$BIN" --iterations 10 --verbose --calculate-entropy \
    --calculate-invariants --use-error-correction --majorana-chain-length 7 \
    --toric-code-size "4 4"
