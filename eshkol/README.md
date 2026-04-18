# Eshkol training scripts

This directory holds Scheme (`.esk`) training drivers that orchestrate
autodiff tapes and optimizer updates against the C kernels in `src/`.

v0.4 foundation ships an empty placeholder; real scripts land alongside
the v0.5 research pillars:

- `train_nqs.esk`             — pillar P1.1 (ViT wavefunctions for NQS)
- `train_equivariant_llg.esk` — pillar P1.2 (Equivariant LLG dynamics)
- `train_qec_decoder.esk`     — pillar P1.3 (learned surface-code decoder)
- `train_flow_matching.esk`   — pillar P1.4 (discrete flow-matching sampler)

The C-side bridge is `src/eshkol_bridge.c` (header
`include/eshkol_bridge.h`). It compiles without Eshkol present —
`ESHKOL_BRIDGE_EDISABLED` is returned from every entry point. Enable by
setting `-DSPIN_NN_HAS_ESHKOL=1` and pointing the build at the Eshkol
FFI header (see https://github.com/tsotchke/eshkol for the runtime).

Each script is expected to:

1. Declare `extern` bindings to the C kernels it needs (forward pass,
   local-energy estimator, etc.).
2. Build a gradient tape by invoking the C forward functions through
   the FFI.
3. Call an Eshkol optimizer (Adam / SGD / natural-gradient) to update
   parameters, writing updated weights back through the bridge.

Heavier lifting — transformer ops, Flash Attention, quantization — will
arrive as an Eshkol-native NN engine (working title
`eshkol-transformers`), a planned future sibling library. When that
ships, the bridge widens to include tensor / model / optimizer handles.
