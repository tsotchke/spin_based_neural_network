# Draft: libirrep PR #6 — `irrep/point_group.h`

Content for the PR we owe libirrep 1.2 on the point-group projection
machinery. Draft lives in this repo; actual PR lands in libirrep.

## Scope (per maintainer)

- Operator type + reduced labels + direct-sum decomposition.
- Minimum viable: **C₄ᵥ** and **D₆** (used for square and kagome
  lattice NQS projections respectively).
- Follow-up (same module, next cycle): **C₃ᵥ** and **D₃** (for
  triangular lattice variants).
- Character tables verified against Bradley–Cracknell.

## Header skeleton

```c
/* include/irrep/point_group.h */

#ifndef IRREP_POINT_GROUP_H
#define IRREP_POINT_GROUP_H

#include <stddef.h>
#include <irrep/export.h>
#include <irrep/multiset.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    IRREP_PG_C4V = 0,     /* square lattice */
    IRREP_PG_D6  = 1,     /* hexagonal / kagome */
    IRREP_PG_C3V = 2,     /* follow-up */
    IRREP_PG_D3  = 3      /* follow-up */
} irrep_point_group_t;

/* Opaque character-table handle. Build once, project many. */
typedef struct irrep_pg_table irrep_pg_table_t;

/* ---- lifecycle ---------------------------------------------------- */

IRREP_API irrep_pg_table_t *irrep_pg_table_build(irrep_point_group_t g);
IRREP_API void              irrep_pg_table_free (irrep_pg_table_t *t);

/* ---- metadata ---------------------------------------------------- */

/* Number of irreducible representations of the group (for C4v: 5; D6: 6). */
IRREP_API int irrep_pg_num_irreps(const irrep_pg_table_t *t);

/* Order of the group (for C4v: 8; D6: 12; C3v: 6; D3: 6). */
IRREP_API int irrep_pg_order(const irrep_pg_table_t *t);

/* Human-readable label for irrep mu (e.g. "A1", "A2", "B1", "B2", "E"
 * for C4v). Pointer to static storage; do not free. */
IRREP_API const char *irrep_pg_irrep_label(const irrep_pg_table_t *t,
                                             int mu);

/* ---- projection -------------------------------------------------- */

/* Apply the projector
 *     P_mu = (d_mu / |G|) * sum_{g in G} chi*_mu(g) * D(g)
 * to a feature vector of shape `spec` (an irrep_multiset). Output
 * length equals input length; only components in the mu-th
 * irreducible subspace survive. */
IRREP_API void irrep_pg_project(const irrep_pg_table_t *t,
                                 int mu,
                                 const irrep_multiset_t *spec,
                                 const double *in,
                                 double *out);

/* Reduce: decompose the feature spec into irreps under G.
 * `out_mult` receives num_irreps integers giving the multiplicity
 * of each irrep in the direct-sum decomposition. */
IRREP_API void irrep_pg_reduce(const irrep_pg_table_t *t,
                                const irrep_multiset_t *spec,
                                int *out_mult);

#ifdef __cplusplus
}
#endif

#endif /* IRREP_POINT_GROUP_H */
```

## C4v character table (reference, Bradley-Cracknell)

|        | E | 2C₄ | C₂ | 2σᵥ | 2σ_d |
|--------|---|-----|----|----|------|
| A₁     | 1 |  1  | 1  |  1 |  1   |
| A₂     | 1 |  1  | 1  | -1 | -1   |
| B₁     | 1 | -1  | 1  |  1 | -1   |
| B₂     | 1 | -1  | 1  | -1 |  1   |
| E      | 2 |  0  | -2 |  0 |  0   |

## D6 character table (reference, Bradley-Cracknell)

|        | E | 2C₆ | 2C₃ | C₂ | 3C₂′ | 3C₂″ |
|--------|---|-----|-----|----|----|------|
| A₁     | 1 |  1  |  1  |  1 |  1 |  1   |
| A₂     | 1 |  1  |  1  |  1 | -1 | -1   |
| B₁     | 1 | -1  |  1  | -1 |  1 | -1   |
| B₂     | 1 | -1  |  1  | -1 | -1 |  1   |
| E₁     | 2 |  1  | -1  | -2 |  0 |  0   |
| E₂     | 2 | -1  | -1  |  2 |  0 |  0   |

## Acceptance tests

1. **Table metadata.**
   - `irrep_pg_num_irreps(C4V) == 5`, `irrep_pg_order(C4V) == 8`.
   - `irrep_pg_num_irreps(D6) == 6`,  `irrep_pg_order(D6) == 12`.
2. **Character orthogonality.**
   For each group, assert `(1/|G|) Σ_g χ*_μ(g) χ_ν(g) = δ_{μν}` to 1e-12.
3. **Projector idempotence.**
   For each μ, `P_μ (P_μ v) == P_μ v` to bit-exactness.
4. **Projector sum.**
   `Σ_μ P_μ = I` on any input vector.
5. **Decomposition.**
   Applied to a known multiset (e.g. `1x0e + 1x1o + 1x2e` under C4v),
   the returned multiplicities match hand-computed Bradley-Cracknell
   reductions.

## Use case here (`spin_based_neural_network`)

The projector enables symmetry-projected NQS ansätze on square (C4v)
and kagome (D6) lattices — pillar P1.1.b follow-up work on the
frustrated J1-J2 benchmark. The projection is applied in the
equivariant-NQS forward pass after the NequIP tower's readout; only
the A₁-symmetric component is kept for ground-state search, reducing
the variational manifold's dimensionality by ~|G|× for a comparable
ansatz size.

## PR checklist

- [ ] Header (as above).
- [ ] Implementation (~400 LOC: four character tables + projector +
      reduce).
- [ ] Five acceptance tests.
- [ ] CHANGELOG entry under libirrep 1.2.
- [ ] API docs (Doxygen blocks + example).
