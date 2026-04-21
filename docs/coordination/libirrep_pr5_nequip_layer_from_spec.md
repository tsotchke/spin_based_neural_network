# Draft: libirrep PR #5 — `irrep_nequip_layer_from_spec`

Content for the PR we owe libirrep 1.2. Lives in this repo as the
working draft; the actual PR will be filed against
[github.com/tsotchke/libirrep](https://github.com/tsotchke/libirrep)
with the same text moved into that tree's headers and tests.

## BNF grammar

```
layer_spec   ::= hidden_in_multiset  "->"  hidden_out_multiset  [ option_list ]

hidden_in_multiset   ::= multiset
hidden_out_multiset  ::= multiset

multiset     ::= term  ( "+"  term )*
term         ::= multiplicity  "x"  l_value  parity_sigil
multiplicity ::= positive_integer                   /* e.g. "1", "4", "12" */
l_value      ::= non_negative_integer               /* e.g. "0", "1", "2", "3" */
parity_sigil ::= "e"  |  "o"                         /* "e" for even, "o" for odd */

option_list  ::= "["  option_entry  ( ","  option_entry )*  "]"
option_entry ::= option_name  "="  option_value
option_name  ::= "sh" | "radial" | "r_cut" | "cutoff"
option_value ::= value_literal | cutoff_specifier

cutoff_specifier ::= "cosine"  |  "polynomial(" positive_integer ")"
value_literal    ::= positive_real | non_negative_integer

positive_integer      ::= ( "1" | ... | "9" ) digit*
non_negative_integer  ::= "0" | positive_integer
positive_real         ::= digit+ [ "." digit+ ] [ exponent ]
digit                 ::= "0" | "1" | ... | "9"
exponent              ::= ( "e" | "E" ) [ "+" | "-" ] digit+
```

### Whitespace

- **Lenient** (0 or more spaces allowed) around operators `->`, `=`,
  `,`, `+`, `[`, `]`.
- **Strict** (no spaces) inside tokens: `"1x0e"`, not `"1 x 0 e"`.
- Matches the existing `irrep_multiset_parse` convention, as the
  maintainer specified.

### Defaults

When an option is omitted in the spec string:

| Option | Default |
|---|---|
| `sh` | 2 |
| `radial` | 8 |
| `r_cut` | 1.0 |
| `cutoff` | `polynomial(6)` |

Omitted options take the default silently. Caller overriding any of
these is additive — documented in `irrep/nequip.h` per maintainer's
direction.

### Error surface

Parser failures return `NULL` and set `irrep_last_error()` to a
string of the form:

    "nequip spec: unexpected token '...' at column N (expected ...)"

Keeps the existing "builders return NULL" convention rather than
introducing an `irrep_status_t` variant.

## Examples

Valid:

```
"1x0e + 1x1o -> 1x1o"                                      ; all defaults
"2x0e + 1x1o -> 1x1o [sh=2]"                                ; small shape
"4x0e + 2x1o + 1x2e -> 2x0e + 1x1o [sh=3, radial=8]"        ; medium shape
"8x0e+4x1o+2x2e+1x3o->4x0e+2x1o+1x2e[sh=4,radial=16]"        ; no whitespace
"1x0e + 1x1o -> 1x1o [cutoff=cosine]"                       ; cosine cutoff
"1x0e + 1x1o -> 1x1o [cutoff=polynomial(6), r_cut=1.5]"     ; polynomial p=6
```

Malformed:

```
"1x0e + 1x1o 1x1o"                    ; missing "->"
"1x0e + 1x1o -> 1x1o +"               ; trailing "+"
"1q0e + 1x1o -> 1x1o"                 ; invalid irrep (q is not e/o)
"1x0e + 1x1o -> 1x1o [cutoff=linear]" ; invalid cutoff specifier
"1x0e + 1x1o -> 1x1o [zoom=2]"        ; unknown option
```

## 10 test cases

Each case is `(spec_string, expected_behaviour)` with numerical
round-trip acceptance defined below.

### Valid cases (5)

1. **Default everything.**
   `"1x0e + 1x1o -> 1x1o"` →
   layer equivalent to `irrep_nequip_layer_build(hidden_in=parse("1x0e+1x1o"), l_sh_max=2, n_radial=8, r_cut=1.0, cutoff_kind=IRREP_NEQUIP_CUTOFF_POLYNOMIAL, cutoff_poly_p=6, hidden_out=parse("1x1o"))`.

2. **Override `sh`.**
   `"1x0e + 1x1o -> 1x1o [sh=3]"` →
   same as (1) but `l_sh_max=3`.

3. **Override `radial`.**
   `"1x0e + 1x1o -> 1x1o [radial=16]"` →
   same as (1) but `n_radial=16`.

4. **Override `r_cut`.**
   `"1x0e + 1x1o -> 1x1o [r_cut=1.5]"` →
   same as (1) but `r_cut=1.5`.

5. **Override everything.**
   `"4x0e + 2x1o + 1x2e -> 2x0e + 1x1o [sh=3, radial=16, r_cut=1.5, cutoff=cosine]"`.

### Malformed cases (5)

6. **Missing arrow.**
   `"1x0e + 1x1o 1x1o"` →
   returns `NULL`, `irrep_last_error()` reports missing arrow.

7. **Trailing operator.**
   `"1x0e + 1x1o -> 1x1o +"` →
   returns `NULL`, reports unexpected end-of-input after `+`.

8. **Invalid parity sigil.**
   `"1q0e + 1x1o -> 1x1o"` →
   returns `NULL`, reports "expected 'x' after multiplicity".

9. **Unknown cutoff specifier.**
   `"1x0e + 1x1o -> 1x1o [cutoff=linear]"` →
   returns `NULL`, reports invalid cutoff.

10. **Unknown option.**
    `"1x0e + 1x1o -> 1x1o [zoom=2]"` →
    returns `NULL`, reports unknown option.

## Acceptance

Every valid case (1–5) must pass a **round-trip bit-exact test**:

```c
/* Build layer A from spec. Build layer B from verbose API with
 * the equivalent parameters. Apply both to a fixed (h_in, edge_vec)
 * test input. Require output-bit-exact equality. */
irrep_nequip_layer_t *A = irrep_nequip_layer_from_spec(spec);
irrep_nequip_layer_t *B = irrep_nequip_layer_build(hidden_in, l_sh_max,
    n_radial, r_cut, cutoff_kind, cutoff_poly_p, hidden_out);
assert_equal_outputs(A, B, h_in_fixed, edge_vec_fixed);
```

The `(h_in, edge_vec)` fixture is deterministic and identical across
test cases — a 4-node, 8-edge micro-graph with unit-magnitude features
seeded from splitmix at a fixed seed. Committed alongside the test
as `tests/fixtures/nequip_spec_roundtrip_input.h`.

Every malformed case (6–10) must:
- Return `NULL`.
- Leave `irrep_last_error()` non-empty.
- Leave no allocations (valgrind / sanitizer-clean).

## File placement in libirrep

- **Header addition**: `include/irrep/nequip.h` gains the
  `irrep_nequip_layer_from_spec` declaration.
- **Implementation**: `src/nequip_spec.c` (new file).
- **Tests**: `tests/test_nequip_from_spec.c`.
- **Fixtures**: `tests/fixtures/nequip_spec_roundtrip_input.h`.

## PR checklist

- [ ] Header declaration + doxygen.
- [ ] Parser implementation (~200 LOC).
- [ ] 5 valid-case tests + 5 malformed-case tests.
- [ ] Fixture file for bit-exact round-trip.
- [ ] ABI hash file updated (ABI surface changes).
- [ ] CHANGELOG entry under 1.2 section.

## Draft — ready for PR submission to libirrep

Everything above is ready to copy-paste into a libirrep PR once the
1.2 milestone is open. This doc tracks the commitment from our
(spin_based_neural_network) side.
