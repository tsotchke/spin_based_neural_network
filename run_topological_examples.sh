#!/bin/bash

# Compile the program
make

echo "==== Example 1: Z2 Topological Insulator (Chern number = 1, Winding number = 1) ===="
export CHERN_NUMBER=1
export WINDING_NUMBER=1
export MAJORANA_MU=0.5    # |μ| < 2|t| for topological phase
export MAJORANA_T=1.0     # Hopping parameter
export MAJORANA_DELTA=1.0 # Pairing strength
./bin/spin_based_neural_computation --iterations 10 --verbose --calculate-entropy \
    --calculate-invariants --use-error-correction --majorana-chain-length 3 \
    --toric-code-size 2 2

echo -e "\n\n==== Example 2: Quantum Spin Hall Effect (Chern number = 2, Winding number = 0) ===="
# Set winding number to 0 (trivial phase) for this example
export CHERN_NUMBER=2
export WINDING_NUMBER=0
export MAJORANA_MU=2.5    # |μ| > 2|t| for trivial phase
export MAJORANA_T=1.0     # Hopping parameter
export MAJORANA_DELTA=1.0 # Pairing strength
./bin/spin_based_neural_computation --iterations 10 --verbose --calculate-entropy \
    --calculate-invariants --use-error-correction --majorana-chain-length 5 \
    --toric-code-size 3 3

echo -e "\n\n==== Example 3: Fractional Quantum Hall Effect (Chern number = 1/3, Winding number = -1) ===="
export CHERN_NUMBER=0.333
export WINDING_NUMBER=-1
export MAJORANA_MU=0.2    # Small chemical potential
export MAJORANA_T=1.0     # Hopping parameter (reverse sign to get negative winding)
export MAJORANA_DELTA=-1.0 # Negative pairing strength
./bin/spin_based_neural_computation --iterations 10 --verbose --calculate-entropy \
    --calculate-invariants --use-error-correction --majorana-chain-length 7 \
    --toric-code-size 4 4
