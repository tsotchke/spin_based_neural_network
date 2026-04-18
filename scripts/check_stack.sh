#!/bin/sh
# scripts/check_stack.sh — advisory probe for optional stack components.
#
# v0.4 foundation: everything reported here is advisory. The main binary
# and test/benchmark suites do not require any of these to build or run.
# They light up as needed:
#   - external NN / reasoning engine — required for --nn-backend=engine
#       (planned backends: an Eshkol-native NN engine and Noesis;
#        v0.5 P1.1+)
#   - libirrep                    — required for equivariant-LLG pillar
#                                   (v0.5 P1.2)
#   - eshkol runtime              — required for Eshkol-driven training
#                                   scripts (v0.5 pillar drivers)
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HOST_OS="$(uname -s)"
HOST_ARCH="$(uname -m)"
case "$HOST_OS/$HOST_ARCH" in
    Darwin/arm64)  TRIPLE="macos-arm64" ;;
    Darwin/x86_64) TRIPLE="macos-x86_64" ;;
    Linux/x86_64)  TRIPLE="linux-x86_64" ;;
    Linux/aarch64) TRIPLE="linux-aarch64" ;;
    *)             TRIPLE="$HOST_OS/$HOST_ARCH" ;;
esac

echo "== Host triple: $TRIPLE"

# External engine — optional for v0.4. When ENGINE_ROOT is set in the
# environment, verify it exists; otherwise just note absence.
if [ -n "${ENGINE_ROOT:-}" ]; then
    if [ -d "$ENGINE_ROOT/include" ] && [ -d "$ENGINE_ROOT/lib" ]; then
        echo "== engine: $ENGINE_ROOT (available)"
    else
        echo "== engine: ENGINE_ROOT=$ENGINE_ROOT set but missing include/ or lib/"
    fi
else
    echo "== engine: not configured (set ENGINE_ROOT + ENGINE_ENABLE=1 to"
    echo "   enable; required from v0.5 for --nn-backend=engine)"
fi

# libirrep — arrives with v0.5
if [ -n "${IRREP_ROOT:-}" ] && [ -f "$IRREP_ROOT/include/irrep/irrep.h" ]; then
    echo "== libirrep: $IRREP_ROOT"
else
    echo "== libirrep: not configured (required from v0.5 equivariant-LLG pillar)"
fi

# Eshkol — optional for v0.4
if command -v eshkol >/dev/null 2>&1; then
    echo "== eshkol: $(eshkol --version 2>&1 | head -1)"
else
    echo "== eshkol: not on PATH (required from v0.5 for Eshkol-driven training)"
fi

echo "== OK (all probes advisory in v0.4)"
