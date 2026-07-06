#!/bin/bash
# Run this ONCE (e.g. on the login node, or as a quick 'salloc' step) before
# submitting job_tiling_test.sh. It builds one binary per tile size so the
# benchmark job just runs them, instead of recompiling 60+ times.
#
# Usage: srun --partition=EPYC --account=dssc --nodes=1 --ntasks=1 --cpus-per-task=1 --time=00:10:00 bash compile_variants.sh stencil_parallel.c

set -e

SRC=${1:-stencil_parallel.c}
OUT_DIR=./bin
mkdir -p "$OUT_DIR"

module load openMPI/4.1.6

echo "Building baseline (no tiling, TILE_SIZE undefined -> original loop)..."
mpicc -fopenmp -O3 -march=native "$SRC" -o "$OUT_DIR/stencil_notile"

# Candidate tile sizes: below/around/above a typical 32KiB L1d estimate,
# plus a couple sized to fit L2 (512KiB/core) instead.
TILE_SIZES="8 16 32 64 128 256"

for T in $TILE_SIZES; do
    echo "Building TILE_SIZE=$T ..."
    mpicc -fopenmp -O3 -march=native -DTILE_SIZE=$T "$SRC" -o "$OUT_DIR/stencil_tile$T"
done

echo
echo "Built binaries:"
ls -la "$OUT_DIR"
