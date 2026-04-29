#!/usr/bin/env python3
"""Synthesis analysis of kagome AFM finite-size flow JSON outputs.

Reads benchmarks/results/nqs/full_analysis/L{L}_ir{ir}.json files
(produced by research_kagome_full_analysis) and produces:

  1. Sector-resolved γ_TEE finite-size flow + linear-in-1/N extrapolation.
  2. Lowest cross-sector gap finite-size flow + extrapolation.
  3. Entanglement spectrum comparison (gap, multiplet structure).
  4. Distance-resolved correlation function, log|C| vs d to test
     power-law (gapless) vs exponential (gapped) decay.
  5. Hamiltonian-consistency residuals across all runs.

Output: stdout summary table + benchmarks/results/nqs/full_analysis/synthesis.json
"""
import json
import glob
import math
import os
import sys


def load_runs(directory):
    runs = {}
    for path in sorted(glob.glob(os.path.join(directory, 'L*_ir*.json'))):
        try:
            with open(path) as f:
                d = json.load(f)
            if 'system' not in d:
                continue
            L = d['system']['L']
            ir = d['system']['irrep']
            runs[(L, ir)] = d
        except (json.JSONDecodeError, ValueError):
            pass
    return runs


def linear_1_over_N(N1, y1, N2, y2):
    """Linear fit y = a + b/N → return (a, b, y_at_inf=a)."""
    inv1, inv2 = 1.0/N1, 1.0/N2
    b = (y1 - y2) / (inv1 - inv2)
    a = y1 - b * inv1
    return a, b


def main():
    directory = 'benchmarks/results/nqs/full_analysis'
    runs = load_runs(directory)
    if not runs:
        sys.exit("no JSON outputs found in " + directory)

    print("=" * 78)
    print("kagome AFM Heisenberg PBC — finite-size flow synthesis")
    print("=" * 78)

    # 1. Per-sector E_0 + γ_TEE table
    print()
    print("Per-sector ground state and γ_TEE:")
    print(f"{'L':<3} {'irrep':<5} {'E_0':<14} {'E_0/N':<10} {'S_total':<8} {'γ/log2':<8}")
    for (L, ir), d in sorted(runs.items()):
        N = d['system']['N']
        E0 = d['lanczos']['E_0']
        S = d['total_spin']['S_total']
        gam = d['tee']['gamma_log2'] if d.get('tee') else None
        gam_s = f"{gam:.3f}" if gam is not None else "-"
        print(f"{L:<3} {ir:<5} {E0:<14.6f} {E0/N:<10.4f} {S:<8.3f} {gam_s:<8}")

    # 2. γ_TEE finite-size flow per sector
    print()
    print("γ_TEE / log 2 finite-size flow (1/N extrapolation):")
    print(f"{'sector':<5} {'γ(L=2)':<10} {'γ(L=3)':<10} {'γ_inf (1/N fit)':<18} {'Δ from L=2 to L=3 (%)'}")
    for ir in ('A_1', 'A_2', 'B_1', 'B_2'):
        L2 = runs.get((2, ir))
        L3 = runs.get((3, ir))
        if L2 is None or L3 is None:
            continue
        g2 = L2['tee']['gamma_log2']
        g3 = L3['tee']['gamma_log2']
        a, _ = linear_1_over_N(12, g2, 27, g3)
        delta = (g3 - g2) / g2 * 100
        print(f"{ir:<5} {g2:<10.3f} {g3:<10.3f} {a:<18.3f} {delta:+.1f}")

    # 3. Cross-sector spin gap finite-size flow
    print()
    print("Lowest cross-sector spin gap finite-size flow:")
    for L_target in (2, 3):
        e0_per_irrep = {ir: r['lanczos']['E_0']
                         for (LL, ir), r in runs.items() if LL == L_target}
        if not e0_per_irrep:
            continue
        e_sorted = sorted(e0_per_irrep.values())
        if len(e_sorted) >= 2:
            gap = e_sorted[1] - e_sorted[0]
            print(f"  L={L_target}: E_0={e_sorted[0]:.6f},  E_1={e_sorted[1]:.6f},  gap = {gap:.6f} J")
    # Linear extrapolation
    e0_L2 = sorted(r['lanczos']['E_0']
                    for (LL, _), r in runs.items() if LL == 2)
    e0_L3 = sorted(r['lanczos']['E_0']
                    for (LL, _), r in runs.items() if LL == 3)
    if len(e0_L2) >= 2 and len(e0_L3) >= 2:
        gap2 = e0_L2[1] - e0_L2[0]
        gap3 = e0_L3[1] - e0_L3[0]
        a, b = linear_1_over_N(12, gap2, 27, gap3)
        print(f"  Linear-in-1/N extrapolation: Δ_∞ ≈ {a:.6f} J (intercept), b/N coefficient = {b:.4f}")
        print(f"  Z₂ spin liquid: Δ_∞ > 0 (gapped), expected ~0.05 J [Yan-Huse-White 2011]")
        print(f"  U(1) Dirac:     Δ_∞ = 0 (gapless), Δ ~ 1/N or 1/L scaling")

    # 4. Entanglement spectrum: gap and lowest level pattern
    print()
    print("Entanglement spectrum at the largest compact subsystem (nA=6):")
    print(f"{'L':<3} {'irrep':<5} {'lowest -ln λ':<14} {'gap':<10} {'pattern (top 8)'}")
    for (L, ir), d in sorted(runs.items()):
        spec = d.get('entanglement_spectrum_nA_6')
        if not spec:
            continue
        levels = spec['top_32_minus_log_lambda']
        gap = spec['largest_gap_in_lowest_8']
        # Detect degeneracies in lowest 8 levels
        eps = 0.05
        groups = []
        for v in levels[:8]:
            if not isinstance(v, (int, float)):
                groups.append('inf')
                continue
            if groups and abs(v - groups[-1][0]) < eps:
                groups[-1] = (groups[-1][0], groups[-1][1] + 1)
            else:
                groups.append((v, 1))
        pat = ' + '.join(f"{c}({m:.2f})" if isinstance(m, float)
                          else f"{c}({m})"
                          for v, c in groups for m in [v])
        print(f"{L:<3} {ir:<5} {levels[0]:<14.4f} {gap:<10.4f} {pat}")

    # 4b. Renyi entropy α=2,∞ from entanglement spectrum
    print()
    print("Renyi entropies from entanglement spectrum (S_α = (1/(1-α)) ln Σ λ^α):")
    print(f"{'L':<3} {'irrep':<5} {'S_1 (vN)':<10} {'S_2':<10} {'S_∞ (-ln λ_max)':<16}")
    for (L, ir), d in sorted(runs.items()):
        spec = d.get('entanglement_spectrum_nA_6')
        if not spec:
            continue
        levels = [v for v in spec['top_32_minus_log_lambda']
                   if isinstance(v, (int, float))]
        if not levels:
            continue
        # λ_i = exp(-level_i). Renyi α: S_α = ln(Σ λ^α) / (1-α).
        lams = [math.exp(-v) for v in levels]
        # Renormalise (top 32 may be a partial trace; rescale to sum=1).
        Z = sum(lams)
        lams = [x/Z for x in lams]
        S1 = -sum(x * math.log(x) for x in lams if x > 1e-15)
        S2 = -math.log(sum(x*x for x in lams))
        S_inf = -math.log(max(lams))
        print(f"{L:<3} {ir:<5} {S1:<10.4f} {S2:<10.4f} {S_inf:<16.4f}")
    print(f"{'':5} (nominal max S = nA·log 2 = 6·{math.log(2):.4f} = {6*math.log(2):.4f})")

    # 4c. CFT fit on entanglement spectrum: ξ_n − ξ_0 ≈ (2π v_F / L_∂A) · n
    # If the data fit a linear-in-n curve cleanly, that's a Dirac/CFT signature.
    # If non-linear (gapped + multiplet), that's Z₂.
    print()
    print("Linear-in-n CFT fit on lowest 8 entanglement levels:")
    print(f"{'L':<3} {'irrep':<5} {'slope (v_F · 2π / L_∂A)':<24} {'R² (linearity)':<14}")
    for (L, ir), d in sorted(runs.items()):
        spec = d.get('entanglement_spectrum_nA_6')
        if not spec:
            continue
        levels = [v for v in spec['top_32_minus_log_lambda'][:8]
                   if isinstance(v, (int, float))]
        if len(levels) < 4:
            continue
        # Subtract the lowest level
        ys = [v - levels[0] for v in levels]
        xs = list(range(len(ys)))
        # Linear regression
        n = len(xs)
        sx = sum(xs); sy = sum(ys)
        sxx = sum(x*x for x in xs); sxy = sum(x*y for x, y in zip(xs, ys))
        denom = n*sxx - sx*sx
        if denom == 0:
            continue
        slope = (n*sxy - sx*sy) / denom
        intercept = (sy - slope*sx) / n
        # R²
        ss_tot = sum((y - sy/n)**2 for y in ys)
        ss_res = sum((y - (slope*x + intercept))**2 for x, y in zip(xs, ys))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-15 else 1.0
        print(f"{L:<3} {ir:<5} {slope:<24.4f} {r2:<14.4f}")
    print("  Z₂ topological:   gapped + multiplets → R² < 0.95 (non-linear)")
    print("  U(1) Dirac:       linear CFT spectrum → R² > 0.95")

    # 5. C(d) decay: extract |C(d)| vs d, fit log decay rate
    print()
    print("Distance-resolved correlations |C(d)| at GS sectors:")
    for L_target in (2, 3):
        # Find global GS irrep at this L
        gs_ir = min((r['lanczos']['E_0'], ir)
                     for (LL, ir), r in runs.items() if LL == L_target)[1] if any(LL == L_target for (LL, _) in runs) else None
        if not gs_ir:
            continue
        d = runs[(L_target, gs_ir)]
        print(f"  L={L_target}, GS sector {gs_ir}:")
        for s in d['correlation_shells']:
            print(f"    d={s['d']:.4f}  |C|={abs(s['C_avg']):.6e}  log|C|={math.log10(abs(s['C_avg'])) if s['C_avg'] != 0 else float('-inf'):+.4f}")

    # 6. Sum-rule consistency
    print()
    print("Hamiltonian-consistency residuals (J·Σ_NN ⟨S·S⟩ vs E_0):")
    for (L, ir), d in sorted(runs.items()):
        res = d['hamiltonian_sum_rule']['residual']
        print(f"  L={L}  {ir}: {res:.3e}")

    print()
    print("=" * 78)


if __name__ == '__main__':
    main()
