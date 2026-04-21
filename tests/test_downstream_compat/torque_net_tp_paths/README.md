# Golden vectors for torque-net ↔ libirrep NequIP convergence

This directory holds pinned input/output vectors for the five reference
configurations used to verify that the hand-rolled SO(3)-equivariant
torque-net primitives in `src/equivariant_gnn/torque_net.c` agree bit-
exactly with an `irrep_nequip_layer_apply`-driven implementation once
libirrep ≥ 1.2 ships.

Both trees (`spin_based_neural_network` and `libirrep`) vendor the same
files here, and both run `make test_downstream_compat` as part of their
regression suites. Any convention drift on either side — Condon-Shortley
sign, SH normalisation, tensor-product path ordering — fails both test
suites simultaneously, which is the early-warning signal we need.

## Files

- `README.md` — this document.
- `configs.json` — the five fixed input configurations (graph geometry,
  edge vectors, node features, TP weights).
- `expected_outputs.json` — pinned TP outputs with `{−1/√2, 1/√3}`
  prefactor book-keeping documented inline.

## Pinning status

- `configs.json` — frozen at libirrep 1.2 tag.
- `expected_outputs.json` — frozen alongside.
- Any future convention change in either repo requires re-generating
  both files via `scripts/regenerate_golden.py` (coming with the
  migration PR) and bumping the file header version tag.

## Prefactor accounting

The real spherical harmonics used throughout carry the standard
orthonormal Condon-Shortley normalisation

    Y_l^m(r̂) with ∫ Y_l^m · Y_{l'}^{m'} dΩ = δ_{ll'} δ_{mm'}.

For the torque-net's five basis terms (see `include/equivariant_gnn/
torque_net.h`), the mapping to libirrep TP paths uses:

    w0 · (m_j · r̂) m_i         ↔   1x1o → 1x1o via (1o ⊗ 1o)_0 · m_i
                                   prefactor −1/√3 from the CG scalar.
    w1 · (m_j × r̂)              ↔   1x1o → 1x1o via (1o ⊗ 1o)_1
                                   prefactor 1/√2 from the antisymmetric
                                   projector.
    w2 · (m_i × m_j)            ↔   1x1o → 1x1o via (1o ⊗ 1o)_1 on the
                                   node-centred product.
    w3 · (m_i · m_j) m_i        ↔   1x0e → 1x1o via (0e ⊗ 1o)_1
                                   prefactor unity.
    w4 · m_j                    ↔   1x1o → 1x1o identity path.

The `{−1/√2, 1/√3}` book-keeping is a shorthand for these CG-scalar
constants. The golden-output JSON freezes the full expected numeric
vector so the prefactor scheme is unambiguous.

## Status as of 2026-04-18

- [x] Directory + README committed.
- [ ] `configs.json` + `expected_outputs.json` to be generated during
      the libirrep-1.2 torque-net migration (blocks on libirrep 1.2 tag).
- [ ] `scripts/regenerate_golden.py` ships alongside the migration PR.
- [ ] libirrep CI vendors this directory into `tests/test_downstream_compat/`
      on their side.
