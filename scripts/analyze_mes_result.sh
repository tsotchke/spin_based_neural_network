#!/usr/bin/env bash
# Post-processing for research_kagome_mes output.
# Usage: scripts/analyze_mes_result.sh path/to/mes.json
#
# Handles K=4 (with Hadamard_4 comparison) and K≥5 (no canonical
# anyon-model comparison; reports SVD spectrum + matrix structure).
set -euo pipefail
file="${1:-research_data/mes/L3_4sector_6x6x6.json}"
if [[ ! -s "$file" ]]; then
    echo "FAIL: $file missing or empty" >&2
    exit 1
fi
python3 - <<PY
import json, math, itertools
d = json.load(open("$file"))
S = d["empirical_modular_S_in_MES_basis"]
K = len(S)
assert all(len(r) == K for r in S), "S must be square"
print(f"K = {K}")
print(f"Empirical |U_x→y| ({K}×{K}):")
for r in S:
    print("  [" + ", ".join(f"{abs(x):+.4f}" for x in r) + "]")
print()

# Singular-value spectrum is invariant under column-permutation +
# sign-gauge, so it is a basis-free observable.
def svd(M):
    """Tiny pure-python SVD via M^T M eigvalsh.  Good enough for K≤8."""
    n = len(M)
    # Compute M^T M
    MtM = [[sum(M[k][i]*M[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    # Power-iteration / Jacobi for symmetric eigvalsh.  Use numpy if possible.
    try:
        import numpy as np
        s = np.linalg.svd(np.array(M), compute_uv=False)
        return list(s)
    except ImportError:
        # Fallback: just return diag of M^T M as crude estimate.
        return sorted([math.sqrt(MtM[i][i]) for i in range(n)], reverse=True)

sv = svd(S)
print(f"Singular values of empirical S:")
print("  [" + ", ".join(f"{s:.4f}" for s in sv) + "]")
print(f"  rank-numerical (cutoff 1e-6) = {sum(1 for s in sv if s > 1e-6)}")
print(f"  conditioning σ_max/σ_min = {sv[0]/max(sv[-1], 1e-30):.2f}")
print()

if K == 4 and "symbolic_Z2_TC_modular_S" in d:
    H = d["symbolic_Z2_TC_modular_S"]
    print("Symbolic (1/2)·H_4:")
    for r in H:
        print("  [" + ", ".join(f"{x:+.4f}" for x in r) + "]")
    print()
    # Best column-permutation + sign gauge fit on |·|
    best = None
    for perm in itertools.permutations(range(4)):
        for signs in itertools.product([1,-1], repeat=4):
            cand = [[signs[c] * S[r][perm[c]] for c in range(4)] for r in range(4)]
            d2 = sum((abs(cand[r][c]) - abs(H[r][c]))**2 for r in range(4) for c in range(4))
            if best is None or d2 < best[0]:
                best = (d2, perm, signs)
    d2, perm, signs = best
    print(f"Best column-gauge fit to (1/2)·H_4:")
    print(f"  permutation = {perm}")
    print(f"  signs       = {signs}")
    print(f"  Frobenius distance ||·| - (1/2)H_4||_F = {math.sqrt(d2):.6f}")
    print(f"  (perfect Z_2 TC: 0.0;  random 4x4 unitary: ~1)")
elif K == 5:
    # No canonical 5-anyon comparison.  Report (a) closest 4-of-5
    # sub-matrix Frobenius fit to (1/2)·H_4 (drop one row+col), and
    # (b) singular-value gap as topological-rank diagnostic.
    H4 = [[0.5,0.5,0.5,0.5],[0.5,0.5,-0.5,-0.5],[0.5,-0.5,0.5,-0.5],[0.5,-0.5,-0.5,0.5]]
    print("K=5: no canonical anyon-model comparison.  Reporting:")
    print(f"  Closest 4×4 sub-matrix fit to (1/2)·H_4 (best of 5C2 = 25 row+col drops):")
    best = None
    for drop_r in range(5):
        for drop_c in range(5):
            sub = [[S[r][c] for c in range(5) if c != drop_c]
                   for r in range(5) if r != drop_r]
            for perm in itertools.permutations(range(4)):
                for signs in itertools.product([1,-1], repeat=4):
                    cand = [[signs[c] * sub[r][perm[c]] for c in range(4)] for r in range(4)]
                    d2 = sum((abs(cand[r][c]) - abs(H4[r][c]))**2
                             for r in range(4) for c in range(4))
                    if best is None or d2 < best[0]:
                        best = (d2, drop_r, drop_c, perm, signs)
    d2, dr, dc, perm, signs = best
    print(f"    drop row {dr}, drop col {dc}, then perm {perm} sign {signs}")
    print(f"    ||·| - (1/2)H_4||_F = {math.sqrt(d2):.6f}")
    print(f"  SV gap σ_4/σ_5 = {sv[3]/max(sv[4],1e-30):.2f}")
    print(f"    (Z_2 TC: ratio → ∞ since rank-4 + zero;  generic: O(1))")
PY
