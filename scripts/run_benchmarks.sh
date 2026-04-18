#!/bin/sh
# scripts/run_benchmarks.sh — run all v0.4 foundation benchmarks.
# Each benchmark emits a JSON file under benchmarks/results/<suite>/<name>.json.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Building benchmarks ==="
make bench

echo ""
echo "=== Running benchmarks ==="
./build/bench_ising
./build/bench_kitaev
./build/bench_majorana_braid
./build/bench_toric_decoder

echo ""
echo "=== Results ==="
find benchmarks/results -name "*.json" -type f | sort
