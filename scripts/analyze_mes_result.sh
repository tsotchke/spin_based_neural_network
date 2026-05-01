#!/usr/bin/env bash
# Post-processing for research_kagome_mes output.
# Usage: scripts/analyze_mes_result.sh research_data/mes/L3_4sector_6x6x6.json
#
# Reports:
#   - The empirical 4x4 modular S matrix |U_x→y|.
#   - The symbolic Z_2 TC prediction (1/2)·Hadamard_4.
#   - Element-wise |empirical| - 1/2 (deviation from Hadamard structure).
#   - Frobenius distance ||empirical| - (1/2)·H_4||_F (lower = better Z_2 fit,
#     up to permutation + sign gauge — see caveat in the MES output).
set -euo pipefail
file="${1:-research_data/mes/L3_4sector_6x6x6.json}"
if [[ ! -s "$file" ]]; then
    echo "FAIL: $file missing or empty" >&2
    exit 1
fi
python3 - <<PY
import json, math
d = json.load(open("$file"))
S = d["empirical_modular_S_in_MES_basis"]
H = d["symbolic_Z2_TC_modular_S"]
print("Empirical |U_x→y|:")
for r in S:
    print("  [" + ", ".join(f"{abs(x):+.4f}" for x in r) + "]")
print()
print("Symbolic (1/2)·H_4:")
for r in H:
    print("  [" + ", ".join(f"{x:+.4f}" for x in r) + "]")
print()
# Best gauge fit: try all 4! permutations and all 2^4 sign assignments,
# minimise Frobenius distance.
import itertools
absS = [[abs(x) for x in row] for row in S]
absH = [[abs(x) for x in row] for row in H]
best = None
for perm in itertools.permutations(range(4)):
    for signs in itertools.product([1,-1], repeat=4):
        # apply column-permutation + column-sign to S
        cand = [[signs[c] * S[r][perm[c]] for c in range(4)] for r in range(4)]
        d2 = sum((abs(cand[r][c]) - abs(H[r][c]))**2 for r in range(4) for c in range(4))
        if best is None or d2 < best[0]:
            best = (d2, perm, signs, cand)
d2, perm, signs, cand = best
print(f"Best column-gauge fit:")
print(f"  permutation = {perm}")
print(f"  signs       = {signs}")
print(f"  Frobenius distance ||·| - (1/2)H_4||_F = {math.sqrt(d2):.6f}")
print(f"  (perfect Z_2 TC: 0.0;  random 4x4 unitary: ~1)")
PY
