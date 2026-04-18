# Engine Integration — `engine_adapter` and `eshkol_bridge`

v0.4 ships two engine-neutral bridge modules as dormant scaffolding.
They compile into every build but return "disabled" status codes unless
the appropriate `SPIN_NN_HAS_*` macro is defined. This document
describes how they are structured and how to plug in a concrete
external engine in v0.5+.

For the higher-level polymorphic NN handle built on top of these
bridges, see `training.md`.

## 1. `engine_adapter` — external NN / tensor / reasoning backend

### 1.1 Purpose

The `engine_adapter` is a **single translation unit** (`src/engine_adapter.c`)
through which the spin framework will call into any external engine that
implements three weak hooks:

```c
int         engine_backend_init(void);
int         engine_backend_shutdown(void);
const char *engine_backend_version(void);
```

Every other source file in the project that wants to use the engine
calls the public `engine_adapter_*` API defined in
`include/engine_adapter.h`. The link boundary is auditable in one
file — no other translation unit includes the concrete engine's
headers.

### 1.2 Public API

```c
#include "engine_adapter.h"

typedef enum {
    ENGINE_ADAPTER_OK         =  0,
    ENGINE_ADAPTER_EDISABLED  = -1,   /* built without SPIN_NN_HAS_ENGINE  */
    ENGINE_ADAPTER_ENOT_READY = -2,   /* engine_adapter_init() not called  */
    ENGINE_ADAPTER_ELIB       = -3,   /* underlying engine returned error  */
    ENGINE_ADAPTER_EARG       = -4,
} engine_adapter_status_t;

int   engine_adapter_init(void);         /* refcounted */
int   engine_adapter_shutdown(void);     /* refcounted */
int   engine_adapter_is_available(void); /* 1 if built with SPIN_NN_HAS_ENGINE */
const char *engine_adapter_engine_version(void);  /* engine-reported, NULL if disabled */
const char *engine_adapter_build_version(void);   /* -DSPIN_NN_ENGINE_VERSION, never NULL */

long engine_adapter_flatten_ising (const IsingLattice  *l, float *out, size_t cap);
long engine_adapter_flatten_kitaev(const KitaevLattice *l, float *out, size_t cap);
long engine_adapter_flatten_spin  (const SpinLattice   *l, float *out, size_t cap);
```

### 1.3 Flattener conventions

Each `engine_adapter_flatten_*` writes its lattice into a contiguous
`float` buffer, **row-major** with `x` outer, `y` middle, `z` inner:

```
index(x, y, z) = (x * size_y + z_stride_per_site) * ... = x*(size_y*size_z) + y*size_z + z
```

- `IsingLattice` and `KitaevLattice`: one `float` per site (+1.0 or -1.0).
- `SpinLattice`: three `float`s per site (`sx, sy, sz` in order), total
  length `size_x * size_y * size_z * 3`.

Usage — size probe, allocate, write:

```c
long need = engine_adapter_flatten_ising(ising, NULL, 0);
float *buf = malloc(need * sizeof(float));
long wrote = engine_adapter_flatten_ising(ising, buf, need);
assert(wrote == need);
```

Passing too small a `cap` returns `ENGINE_ADAPTER_EARG`.

### 1.4 Refcount semantics

`engine_adapter_init` and `engine_adapter_shutdown` maintain an
internal counter. The underlying engine's real `init`/`finalize` only
fires at the 0→1 and 1→0 transitions. This makes it safe for multiple
callers — e.g. an NQS trainer and a QEC-decoder trainer in the same
process — to own the engine lifetime without coordination.

### 1.5 Build-time enablement

Default build leaves the adapter dormant:

```sh
$ make arm
$ ./bin/spin_based_neural_computation_arm --nn-backend engine
nn_backend: built without SPIN_NN_HAS_ENGINE; falling back to legacy MLP.
```

To light up the adapter path, a downstream build recipe supplies both
the macro and an object file that defines the three backend hooks:

```sh
make ENGINE_ENABLE=1 ENGINE_ROOT=/path/to/my-engine
```

The Makefile's `ENGINE_ENABLE=1` branch then adds
`-DSPIN_NN_HAS_ENGINE=1 -DSPIN_NN_ENGINE_VERSION='"..."'` to the
compiler flags and links against the engine's shared library.

### 1.6 Plugging in a concrete engine

An engine provider ships a small translation unit:

```c
/* my_engine_backend.c — compile and link alongside the spin framework */
#include <my_engine/public_api.h>

int engine_backend_init(void) {
    return my_engine_global_init();   /* must return 0 on success */
}

int engine_backend_shutdown(void) {
    return my_engine_global_shutdown();
}

const char *engine_backend_version(void) {
    return my_engine_version_string();
}
```

Link it into the binary with `-lmy_engine` (or equivalent) and the
engine-adapter path will activate at runtime.

### 1.7 Planned concrete backends (v0.5+)

| Engine | Role | Status |
|---|---|---|
| Eshkol-native NN engine (working title `eshkol-transformers`) | Transformer / KAN / MoE with multi-dtype tensors, Flash Attention, SafeTensors loading, Riemannian optimisers on hyperbolic / spherical manifolds | v0.6+ target, built on https://github.com/tsotchke/eshkol |
| Noesis | Reasoning engine (symbolic / program-synthesis) | In development; not yet publicly released |

The engine-neutral API leaves the choice of concrete backend open
until v0.5 scope stabilises.

## 2. `eshkol_bridge` — Scheme-orchestrated training tapes

### 2.1 Purpose

The `eshkol_bridge` wraps the Eshkol FFI so that training drivers
implemented in Scheme (files under `eshkol/*.esk`) can record gradient
tapes and call back into spin-framework C kernels. It is independent
of the `engine_adapter` — the former exists so any external engine can
plug in; this one is specifically for Eshkol-driven autodiff over
hand-written kernels.

### 2.2 Public API

```c
#include "eshkol_bridge.h"

typedef enum {
    ESHKOL_BRIDGE_OK         =  0,
    ESHKOL_BRIDGE_EDISABLED  = -1,   /* built without SPIN_NN_HAS_ESHKOL */
    ESHKOL_BRIDGE_ENOT_READY = -2,
    ESHKOL_BRIDGE_ELIB       = -3,
    ESHKOL_BRIDGE_EARG       = -4,
} eshkol_bridge_status_t;

int eshkol_bridge_init(void);              /* refcounted */
int eshkol_bridge_shutdown(void);          /* refcounted */
int eshkol_bridge_is_available(void);

int eshkol_bridge_load_script(const char *path);       /* evaluate an .esk file */
int eshkol_bridge_eval_double(const char *source,
                              double *result);         /* eval a small expression */

const char *eshkol_bridge_last_error(void);            /* NULL if no error pending */
```

### 2.3 Enabling

```sh
make ENGINE_ENABLE=1 CFLAGS_EXTRA=-DSPIN_NN_HAS_ESHKOL=1 \
     ENGINE_ROOT=/path/to/eshkol-install
```

Without `SPIN_NN_HAS_ESHKOL=1`, every entry point returns
`ESHKOL_BRIDGE_EDISABLED`. Call-site code should handle this
gracefully:

```c
int rc = eshkol_bridge_init();
if (rc == ESHKOL_BRIDGE_EDISABLED) {
    /* fallback or skip — legacy training path continues */
} else if (rc != ESHKOL_BRIDGE_OK) {
    fprintf(stderr, "eshkol: %s\n", eshkol_bridge_last_error());
    return -1;
}
```

### 2.4 Script search path

`eshkol_bridge_load_script(path)` accepts an absolute path or a path
relative to the current working directory. Training scripts for v0.5
pillars live under `eshkol/` in the repo:

- `eshkol/train_nqs.esk` (pillar P1.1)
- `eshkol/train_equivariant_llg.esk` (pillar P1.2)
- `eshkol/train_qec_decoder.esk` (pillar P1.3)
- `eshkol/train_flow_matching.esk` (pillar P1.4)

v0.4 ships `eshkol/` as a placeholder directory; real scripts land
alongside each pillar.

### 2.5 Thread safety

The Eshkol runtime is single-threaded. All bridge calls must come from
the thread that invoked `eshkol_bridge_init`. The refcount semantics
above make nesting safe within that thread.

## 3. Tests

- `tests/test_engine_adapter.c` (8 tests) — disabled-mode status codes,
  flatteners for all three lattice types, row-major ordering, buffer
  size validation.
- `tests/test_eshkol_bridge.c` (5 tests) — disabled-mode lifecycle and
  argument validation. Engine-path tests land when a concrete engine
  is wired in.

Run `make test_engine_adapter && ./bin/test_engine_adapter` for the
adapter only.

## 4. See also

- `architecture_v0.4.md` §3 — the full engine-integration roadmap.
- `training.md` — the polymorphic `nn_backend` that consumes this
  adapter.
- `eshkol/README.md` — placeholder Scheme-script directory.
