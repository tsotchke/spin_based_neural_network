# Toric Code Implementation for Error Correction

## Abstract

This document presents the implementation of Kitaev's toric code for topological quantum error correction in the Spin-Based Neural Computation Framework. We describe the mathematical formalism, computational methods, and practical applications of the toric code as a stabilizer quantum error-correcting code. The implementation follows the theoretical framework detailed in the "Majorana Zero Modes in Topological Quantum Computing" paper [1], with emphasis on the topological protection against local errors and the ground state degeneracy that enables robust quantum information storage.

## 1. Introduction

Quantum error correction is essential for fault-tolerant quantum computation. Kitaev's toric code represents a milestone in the development of topological quantum error correction, offering protection that stems from the global, topological properties of the system rather than local redundancy [2]. This implementation realizes the toric code on a two-dimensional lattice, enabling the detection and correction of errors through stabilizer measurements.

The toric code offers several advantages for quantum error correction:

1. **Topological Protection**: Errors must form extended strings across the system to cause logical errors
2. **Stabilizer Formalism**: Error detection through local stabilizer measurements
3. **Geometrically Local Interactions**: All operations involve only nearest-neighbor interactions
4. **Non-Trivial Ground State Degeneracy**: Logical qubits encoded in the degenerate ground state manifold

## 2. Theoretical Framework

### 2.1 Toric Code Hamiltonian

The toric code Hamiltonian, introduced by Kitaev [2], is defined on a two-dimensional square lattice with spin-1/2 particles (qubits) placed on the edges. The Hamiltonian involves two types of stabilizer operators:

H<sub>tc</sub> = -J ∑<sub>v</sub> A<sub>v</sub> - K ∑<sub>p</sub> B<sub>p</sub>

where:

- A<sub>v</sub> = ∏<sub>i∈v</sub> σ<sub>i</sub><sup>x</sup> are star (vertex) operators
- B<sub>p</sub> = ∏<sub>i∈p</sub> σ<sub>i</sub><sup>z</sup> are plaquette operators
- J, K > 0 are coupling constants

These stabilizer operators commute with each other and with the Hamiltonian. The ground state |ψ⟩ satisfies:

A<sub>v</sub>|ψ⟩ = |ψ⟩ ∀v
B<sub>p</sub>|ψ⟩ = |ψ⟩ ∀p

### 2.2 Ground State Degeneracy and Logical Operators

On a torus (i.e., a square lattice with periodic boundary conditions), the toric code exhibits a ground state degeneracy of 4 = 2<sup>2g</sup> where g = 1 is the genus of the torus. This degeneracy allows for the encoding of two logical qubits.

The logical operators are defined as string operators along non-contractible loops:

- Z<sub>1</sub>, Z<sub>2</sub>: Products of σ<sup>z</sup> operators along non-contractible loops in the horizontal and vertical directions
- X<sub>1</sub>, X<sub>2</sub>: Products of σ<sup>x</sup> operators along non-contractible loops in the dual lattice

### 2.3 Error Models and Correction

In the toric code, errors are detected through stabilizer measurements:

- Z errors (phase-flip) anticommute with some A<sub>v</sub> operators
- X errors (bit-flip) anticommute with some B<sub>p</sub> operators

When an error occurs, the affected stabilizers yield -1 instead of +1 eigenvalues. The pattern of these violations (syndrome) is used to identify and correct errors.

## 3. Implementation Details

### 3.1 Data Structures

The toric code data model in v0.4 is defined in `include/toric_code.h` as
follows. The struct carries both the physical data-qubit accumulators
introduced in v0.4 and legacy fields kept around so pre-v0.4 callers
continue to compile unchanged:

```c
typedef struct {
    int size_x;                 /* L_x */
    int size_y;                 /* L_y */
    int num_links;              /* 2 * L_x * L_y data qubits */

    /* Physical data-qubit error accumulators (GF(2)) */
    int *x_errors;              /* size num_links, 0 or 1 */
    int *z_errors;              /* size num_links, 0 or 1 */

    /* Syndromes derived from current error state */
    int *vertex_syndrome;       /* size L_x*L_y, 0 (ok) / 1 (flagged) */
    int *plaquette_syndrome;    /* size L_x*L_y, 0 (ok) / 1 (flagged) */

    /* Legacy fields — preserved so pre-v0.4 code keeps compiling.
     * star_operators[i][0]  = vertex_syndrome[i]    mapped to ±1.
     * plaquette_operators[i][0] = plaquette_syndrome[i] mapped to ±1.
     * The other three legacy slots are held at +1. */
    int **star_operators;
    int **plaquette_operators;
    int *logical_operators_x;
    int *logical_operators_z;
} ToricCode;

typedef struct {
    int error_type;             /* 0 = X-error syndrome (plaquettes),
                                 * 1 = Z-error syndrome (vertices) */
    int *error_positions;       /* flagged stabilizer indices (0..Lx*Ly) */
    int num_errors;
} ErrorSyndrome;
```

Data qubits are indexed by `link_index(x, y, dir) = 2 * (x * L_y + y) + dir`,
where `dir = 0` is the horizontal link from `(x,y)` to `(x+1, y)` and
`dir = 1` is the vertical link from `(x,y)` to `(x, y+1)`. Helpers
`toric_code_link_index`, `toric_code_vertex_links`, and
`toric_code_plaquette_links` encapsulate the indexing arithmetic.

### 3.2 Toric Code Initialization

`initialize_toric_code` allocates the struct plus flat `x_errors`,
`z_errors`, `vertex_syndrome`, and `plaquette_syndrome` arrays, and
zero-initialises them (no errors, all syndromes +1). The legacy
`star_operators`/`plaquette_operators` arrays are still allocated (4
`int` slots per stabilizer) and initialised to +1, and the legacy 1-D
logical-operator arrays likewise, so pre-v0.4 readers see a consistent
view.

```c
/* v0.4 public signature. */
ToricCode *initialize_toric_code(int size_x, int size_y);
void       free_toric_code(ToricCode *code);
```

Typical v0.4 caller pattern:

```c
ToricCode *code = initialize_toric_code(5, 5);
apply_random_errors(code, 0.03);
toric_code_decode_mwpm(code);
int logical = toric_code_has_logical_error(code);
free_toric_code(code);
```

### 3.3 Stabilizer Calculation and Measurement

Stabilizers in v0.4 are derived on demand from the data-qubit error
arrays by `toric_code_refresh_syndromes`. A vertex stabilizer flags
when the XOR (GF(2) sum) of `z_errors` on its four incident links is
odd; a plaquette stabilizer flags when the XOR of `x_errors` on its
four bounding links is odd.

```c
void toric_code_refresh_syndromes(ToricCode *code) {
    int Lx = code->size_x, Ly = code->size_y;

    for (int vx = 0; vx < Lx; vx++) for (int vy = 0; vy < Ly; vy++) {
        int links[4];
        toric_code_vertex_links(code, vx, vy, links);
        int parity = 0;
        for (int j = 0; j < 4; j++) parity ^= code->z_errors[links[j]];
        code->vertex_syndrome[vertex_index(code, vx, vy)] = parity;
    }
    for (int px = 0; px < Lx; px++) for (int py = 0; py < Ly; py++) {
        int links[4];
        toric_code_plaquette_links(code, px, py, links);
        int parity = 0;
        for (int j = 0; j < 4; j++) parity ^= code->x_errors[links[j]];
        code->plaquette_syndrome[plaquette_index(code, px, py)] = parity;
    }
    /* Mirror to legacy ±1 arrays in slot 0 for back-compat. */
}
```

Every mutator (`toric_code_apply_x_error`, `toric_code_apply_z_error`,
`apply_random_errors`, `toric_code_decode_greedy`,
`toric_code_decode_mwpm`) calls `toric_code_refresh_syndromes` so the
syndrome arrays stay in sync with the underlying qubit state. Periodic
boundary conditions are handled by `wrap()` inside the link-indexing
helpers.

The pre-v0.4 `calculate_stabilizers(ToricCode*, KitaevLattice*)` entry
point is retained: it seeds `x_errors` from the Kitaev lattice's spin
pattern on horizontal links (`spin -1 → X-error bit 1`) and then calls
`toric_code_refresh_syndromes`. Pre-v0.4 demos that visually inspect
star/plaquette sign changes therefore still produce their expected
output.

### 3.4 Error Detection and Syndrome Measurement

v0.4 exposes three syndrome-extraction entry points, all of which
return heap-allocated `ErrorSyndrome` structs (free with
`free_error_syndrome`):

```c
/* v0.4: split by error channel. */
ErrorSyndrome *sx = toric_code_measure_x_syndrome(code);  /* plaquettes */
ErrorSyndrome *sz = toric_code_measure_z_syndrome(code);  /* vertices   */

/* Legacy: combined, returns whichever channel has more flagged
 * stabilizers (ties → plaquette/X). error_type is 0 for X, 1 for Z. */
ErrorSyndrome *s  = measure_error_syndrome(code);
```

All three call `toric_code_refresh_syndromes` first, so the returned
lists always reflect the current data-qubit state. `error_positions`
is a flat row-major index into the `L_x × L_y` grid
(`index = x * L_y + y`).

The `measure_error_syndrome` path is retained so pre-v0.4 demos that
assume one syndrome per call keep running; new code should prefer the
split APIs because decoding requires both channels.

### 3.5 Error Correction (v0.4: data-qubit model + greedy and MWPM decoders)

v0.4 introduces an explicit data-qubit model for the toric code. Each
link in the L_x × L_y torus carries independent GF(2) X and Z error
accumulators; syndromes are re-derived from the qubit state after every
correction step, so iterated error-and-correction cycles remain
self-consistent.

Two decoders are shipped:

- `toric_code_decode_greedy` — repeatedly pairs the two closest flagged
  stabilizers of a given type by toroidal taxicab distance, then applies
  corrections along the shortest link path (primal lattice for vertex
  syndromes / Z corrections, dual lattice for plaquette syndromes / X
  corrections). Correct at low error rates, O(k²) per channel where k
  is the number of flagged sites.
- `toric_code_decode_mwpm` — optimal minimum-weight perfect matching
  baseline. For K ≤ `MWPM_ENUM_MAX` (= 14) defects per channel the
  decoder enumerates all (K−1)!! perfect matchings with partial-weight
  pruning — genuinely optimal. For K > 14 it seeds from the greedy
  matching and runs 2-opt edge swaps until no further weight decrease
  is found. 2-opt is not provably optimal but comes within a small
  constant factor of MWPM on surface-code defect distributions.

```c
/* Apply a random depolarizing channel at rate p to each data qubit. */
ToricCode *code = initialize_toric_code(5, 5);
apply_random_errors(code, 0.03);

/* Flip individual data qubits by link index. */
int link = toric_code_link_index(code, 2, 3, 0 /* horizontal */);
toric_code_apply_x_error(code, link);       /* stamps an X error  */
toric_code_apply_x_correction(code, link);  /* flips it back      */

/* Query syndromes (re-derived on demand). */
toric_code_refresh_syndromes(code);
ErrorSyndrome *sx = toric_code_measure_x_syndrome(code);
ErrorSyndrome *sz = toric_code_measure_z_syndrome(code);
free_error_syndrome(sx);
free_error_syndrome(sz);

/* Decode. Prefer MWPM for accuracy; greedy is kept as a low-cost baseline. */
int rc_mwpm   = toric_code_decode_mwpm(code);
int rc_greedy = toric_code_decode_greedy(code);

/* Detect a logical error via H₁(dual, Z₂) winding numbers. */
int logical = toric_code_has_logical_error(code);
```

The `perform_error_correction(ToricCode *, ErrorSyndrome *)` entry
point from v0.3 is retained and delegates to the greedy decoder; the
`ErrorSyndrome` argument is effectively ignored (syndromes are
re-derived from the data-qubit state internally).

**Baseline logical-error rates** (greedy decoder; see
`benchmarks/results/toric_decoder/`, measured on M-series Mac):

| distance | p=1% | p=3% | p=5% |
|---|---|---|---|
| 3 | 0.8% | 6.8% | 15.4% |
| 5 | 0.2% | 5.6% | 14.0% |
| 7 | 0.0% | 2.4% | 9.2% |

At p=1% the logical error rate decreases with distance (below threshold);
at p=5% all distances show high rates (above threshold). Learned
decoders based on transformer / Mamba architectures [9–11] are
scheduled for v0.5 pillar P1.3 and will be benchmarked head-to-head
against the v0.4 MWPM baseline.

### 3.6 Logical Operations and Ground-State Verification

v0.4 exposes three query entry points that reason about the
logical-qubit state rather than the stabilizer syndromes directly:

```c
/* +1 = clean state (no flagged syndromes AND no logical error). */
int  is_ground_state(ToricCode *code);

/* 1 iff the accumulated errors form a non-contractible loop on the
 * primal or dual lattice — i.e. a logical X or Z has been applied. */
int  toric_code_has_logical_error(const ToricCode *code);

/* Ground-state degeneracy on a torus = 4 (two logical qubits). */
int  calculate_ground_state_degeneracy(ToricCode *code);
```

`toric_code_has_logical_error` computes homology classes directly from
`x_errors` and `z_errors`: the parities of intersections with a basis
of primal 1-cycles detect the X-chain class in `H_1(dual, Z_2)`, and
intersections with dual 1-cycles detect the Z-chain class in
`H_1(primal, Z_2)`. Any non-zero winding signals that a logical
operator has been applied.

To *apply* a logical operator explicitly (e.g. to test decoder
robustness against pre-applied logicals), the caller flips data qubits
along a non-contractible loop using `toric_code_apply_x_error` or
`toric_code_apply_z_error` — there is no dedicated
`apply_logical_x/z(n)` helper in the v0.4 API. A common pattern:

```c
/* Apply logical X̄₁: string of X errors along the horizontal row y = 0. */
for (int x = 0; x < code->size_x; x++) {
    int link = toric_code_link_index(code, x, 0, 0);
    toric_code_apply_x_error(code, link);
}
assert(toric_code_has_logical_error(code));
```

## 4. Integration with Majorana Zero Modes

Following reference [1], Majorana chains and toric codes are
conceptually related: pairs of Majorana modes can be mapped to
qubits, with vertex operators `A_v = γ_{2i} γ_{2i+1}` and plaquette
operators `B_p = γ_{2i+1} γ_{2i+2}`. A direct `map_majorana_to_toric_code`
helper is *not shipped* in v0.4 — the two modules are linked indirectly
via the shared `KitaevLattice` substrate (see `calculate_stabilizers`
and `map_chain_to_lattice` in `majorana_modes.h`). Native Majorana →
toric-code mapping is scheduled alongside the learned decoder work
(v0.5 pillar P1.3).

## 5. Usage Examples

### 5.1 Basic Toric Code Simulation

```c
#include <stdio.h>
#include "toric_code.h"

int main(void) {
    /* Create a 3x3 toric code (2 · 3 · 3 = 18 data qubits). */
    ToricCode *code = initialize_toric_code(3, 3);

    /* Stamp X errors on two specific links. */
    int e1 = toric_code_link_index(code, 1, 1, 0);  /* horizontal link (1,1)->(2,1) */
    int e2 = toric_code_link_index(code, 1, 2, 0);  /* horizontal link (1,2)->(2,2) */
    toric_code_apply_x_error(code, e1);
    toric_code_apply_x_error(code, e2);

    /* Inspect current syndrome state. */
    ErrorSyndrome *sx = toric_code_measure_x_syndrome(code);
    ErrorSyndrome *sz = toric_code_measure_z_syndrome(code);
    printf("Flagged plaquettes (X-channel): %d\n", sx->num_errors);
    printf("Flagged vertices  (Z-channel): %d\n", sz->num_errors);
    free_error_syndrome(sx);
    free_error_syndrome(sz);

    /* Run MWPM decoder and check the result. */
    toric_code_decode_mwpm(code);
    printf("Logical error after decoding: %d\n",
           toric_code_has_logical_error(code));
    printf("In ground state:              %d\n", is_ground_state(code));
    printf("Ground-state degeneracy:      %d\n",
           calculate_ground_state_degeneracy(code));

    free_toric_code(code);
    return 0;
}
```

### 5.2 Command Line Interface

```bash
# Run toric code error correction with a 2x2 lattice
./build/spin_based_neural_computation --use-error-correction --toric-code-size 2 2 --verbose
```

### 5.3 Error Correction with a Pre-Applied Logical

```c
/* Create a 4x4 toric code. */
ToricCode *code = initialize_toric_code(4, 4);

/* Apply a logical X̄₁: string of X-errors along the horizontal row y = 0. */
for (int x = 0; x < code->size_x; x++) {
    int link = toric_code_link_index(code, x, 0, 0);
    toric_code_apply_x_error(code, link);
}
assert(toric_code_has_logical_error(code));  /* non-contractible loop present */

/* Sprinkle depolarizing noise on top. */
apply_random_errors(code, 0.03);

/* Decode with MWPM. A competent decoder cannot tell a full logical from
 * a stabilizer correction — the logical will persist after decoding. */
toric_code_decode_mwpm(code);
printf("Logical error after decoding: %d\n",
       toric_code_has_logical_error(code));

free_toric_code(code);
```

## 6. Performance Considerations

The computational complexity of toric code operations depends on several factors:

- **Initialization**: O(L²) where L is the linear size of the lattice
- **Stabilizer Measurement**: O(L²)
- **Error Detection**: O(L²)
- **Error Correction**: O(L² log L) using the minimum-weight perfect matching algorithm
- **Logical Operations**: O(L)

For practical quantum error correction, the following performance optimizations are implemented:

1. Efficient edge indexing for fast stabilizer measurements
2. Sparse representation of error syndromes
3. Optimized matching algorithms for error correction
4. Look-up tables for common lattice operations

## 7. Advanced Topics

### 7.1 Code Distance and Error Threshold

The code distance d of the toric code is equal to the linear size L of the lattice. The probability of a logical error scales as:

P<sub>logical</sub> ~ (p/p<sub>th</sub>)<sup>d/2</sup>

where p is the physical error rate and p<sub>th</sub> is the threshold error rate (approximately 11% for the toric code with perfect measurements).

### 7.2 Surface Code Variant (planned, not in v0.4)

A planar surface-code variant (toric code with open boundary conditions
and ground-state degeneracy 2) is not shipped in v0.4. Callers who want
a planar code today can initialize a toric code and skip link updates
across one row and one column to emulate boundaries, but boundary
stabilizers are not correctly handled without code changes. Native
support is scheduled alongside the learned QEC decoder (v0.5 pillar
P1.3).

### 7.3 Non-Abelian Toric Codes (planned, not in v0.4)

Non-Abelian generalizations (e.g. S₃, D₄ stabilizer groups with
`|G|²`-dim ground-state degeneracy) are not shipped in v0.4. Infrastructure
for non-Abelian stabilizer measurement will land with the Fibonacci-anyon
module (v0.5 pillar P1.3); until then, the toric code in this framework
is strictly Z₂.

## 8. Future Directions

Ongoing development for the toric code implementation includes:

1. Support for measurement errors in the syndrome extraction
2. Implementation of fault-tolerant logical gates
3. Integration with physical qubit models for realistic noise simulations
4. Support for color codes and other topological codes
5. Exploration of 3D topological codes with improved thresholds

## 9. References

[1] tsotchke, "Majorana Zero Modes in Topological Quantum Computing: Error-Resistant Codes Through Dynamical Symmetries," 2022.

[2] A. Y. Kitaev, "Fault-tolerant quantum computation by anyons," Annals of Physics, vol. 303, no. 1, pp. 2-30, 2003.

[3] E. Dennis, A. Kitaev, A. Landahl, and J. Preskill, "Topological quantum memory," Journal of Mathematical Physics, vol. 43, no. 9, pp. 4452-4505, 2002.

[4] A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. N. Cleland, "Surface codes: Towards practical large-scale quantum computation," Physical Review A, vol. 86, no. 3, p. 032324, 2012.

[5] S. B. Bravyi and A. Y. Kitaev, "Quantum codes on a lattice with boundary," arXiv:quant-ph/9811052, 1998.

[6] H. Bombin and M. A. Martin-Delgado, "Topological Quantum Distillation," Physical Review Letters, vol. 97, no. 18, p. 180501, 2006.

[7] M. H. Freedman, D. A. Meyer, and F. Luo, "Z₂-systolic freedom and quantum codes," Mathematics of Quantum Computation, Chapman & Hall/CRC, pp. 287-320, 2002.

[8] J. Edmonds, "Paths, trees, and flowers," Canadian Journal of Mathematics, vol. 17, pp. 449-467, 1965.

[9] J. Bausch, A. Senior, F. Heras, T. Edlich, A. Davies, M. Newman, C. Jones, K. Satzinger, M. Y. Niu, S. Blackwell, G. Holland, D. Kafri, J. Atalaya, C. Gidney, D. Hassabis, S. Boixo, H. Neven, and P. Kohli, "Learning high-accuracy error decoding for quantum processors," Nature, vol. 635, pp. 834-840, 2024. DOI: 10.1038/s41586-024-08148-8.

[10] V. Ninkovic, O. Kundacina, D. Vukobratovic, and C. Häger, "Scalable Neural Decoders for Practical Real-Time Quantum Error Correction," arXiv:2510.22724, 2025.

[11] E. Dennis, A. Kitaev, A. Landahl, and J. Preskill, "Topological quantum memory," Journal of Mathematical Physics, vol. 43, no. 9, pp. 4452-4505, 2002 (see [3]; cited again for the original threshold result).
