#!/usr/bin/env bash
# Launch the FULL 5-state MES on the kagome 3×3 PBC AFM lowest-energy
# manifold including BOTH partners of the (Γ, E_2, Sz=1/2) doublet:
#
#   E_2 partner 1 (loaded from canonical p1 file)   E_0 = -11.7795 J
#   E_2 partner 2 (must exist before this is run)   E_0 = -11.7795 J
#   A_1                                              E_0 = -11.6099 J
#   E_1 partner 1                                    E_0 = -11.5930 J
#   A_2                                              E_0 = -11.5576 J
#
# This is the methodologically clean follow-up to L3_mes_lowest4_6x6x6
# (Frobenius distance from (1/2)·H_4: 1.07) — by including both E_2
# doublet partners we span the full lowest-energy quasi-degenerate
# manifold rather than asymmetrically using only one partner.
#
# Grid: 5×5×5×5 = 625 α-vectors per cycle  (4 spherical-coord angles)
# Estimated wall: 1250 evals × 11 s/eval ≈ 4 h on 14-thread M-series.
#
# Pre-condition: research_data/eigvecs/kagome_3x3_E2_p2_sz1_eigvec.bin
# must exist (produced by ./build/research_kagome_e2_p2).
set -e
cd "$(dirname "$0")/.."
P2=research_data/eigvecs/kagome_3x3_E2_p2_sz1_eigvec.bin
if [[ ! -s "$P2" ]]; then
    echo "ERROR: $P2 missing or empty — run research_kagome_e2_p2 first." >&2
    exit 1
fi
if [[ $(stat -f %z "$P2") -lt 1000000000 ]]; then
    echo "ERROR: $P2 is smaller than 1 GB — likely truncated." >&2
    exit 1
fi
mkdir -p research_data/mes
env OMP_NUM_THREADS=14 ./build/research_kagome_mes 3 5 \
  research_data/eigvecs/kagome_3x3_E2_sz1_eigvec.bin \
  "$P2" \
  research_data/eigvecs/kagome_3x3_A1_eigvec.bin \
  research_data/eigvecs/kagome_3x3_E1_sz1_eigvec.bin \
  research_data/eigvecs/kagome_3x3_A2_eigvec.bin \
  5 5 5 5 > research_data/mes/L3_5state_doublet_5x5x5x5.json \
         2> research_data/mes/L3_5state_doublet_5x5x5x5.log
