#!/usr/bin/env bash
# Launch the FOLLOW-UP MES experiment on the 4 lowest *distinct* sector
# states of the L=3 PBC kagome AFM Heisenberg manifold:
#
#   E_2 (Sz=1/2, doublet partner 1)   E_0 = -11.7795 J  (global GS)
#   A_1                                E_0 = -11.6099 J
#   E_1 (Sz=1/2, doublet partner 1)   E_0 = -11.5930 J
#   A_2                                E_0 = -11.5576 J
#
# Compares to the 4-of-1D-irrep MES at L3_4sector_6x6x6.json which uses
# A_1/A_2/B_1/B_2.  Both runs probe the same Z_2 TC vs U(1) Dirac
# question; the lowest-4 variant is more physically motivated since it
# tracks the actual lowest-energy manifold.
#
# Runtime: ~80 min on a 14-thread M-series Mac.
set -e
cd "$(dirname "$0")/.."
mkdir -p research_data/mes
env OMP_NUM_THREADS=14 ./build/research_kagome_mes 3 \
  research_data/eigvecs/kagome_3x3_E2_sz1_eigvec.bin \
  research_data/eigvecs/kagome_3x3_A1_eigvec.bin \
  research_data/eigvecs/kagome_3x3_E1_sz1_eigvec.bin \
  research_data/eigvecs/kagome_3x3_A2_eigvec.bin \
  6 6 6 > research_data/mes/L3_lowest4_6x6x6.json 2> research_data/mes/L3_lowest4_6x6x6.log
