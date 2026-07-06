#!/bin/bash
#SBATCH --job-name=tiling_test
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48gb
#SBATCH --time=00:45:00
#SBATCH -A dssc
#SBATCH --output=tiling_test3.log

# ------------------------------------------------------------------------
# Tests whether cache-blocking (tiling) the interior stencil update helps,
# on a grid large enough that it cannot possibly fit in any level of cache
# (forcing the memory-bound regime tiling is meant to help with).
#
# Fixed: grid size, iteration count, MPI = 1 rank (isolates the OpenMP
# kernel from any MPI/network noise).
# Varies: TILE_SIZE (untiled + 6 tile candidates) x OMP_NUM_THREADS
#         (1 thread isolates the pure cache-blocking effect; 32 threads
#         gives a realistic multi-threaded picture).
# Repeated 3x per configuration to average out timing noise.
#
# Binaries must already exist in ./bin (run compile_variants.sh first).
# ------------------------------------------------------------------------

echo "Running on $(hostname)"
module load openMPI/4.1.6

export OMP_PLACES=cores
export OMP_PROC_BIND=close

GX=40000
GY=40000
NITER=50
BIN_DIR=./bin
REPS=1

VARIANTS="notile tile8 tile16 tile32 tile64 tile128 tile256"
THREAD_COUNTS="1 32"

for T in $THREAD_COUNTS; do
    export OMP_NUM_THREADS=$T
    for V in $VARIANTS; do
        for R in $(seq 1 $REPS); do
            echo "#PARAMS suite=tiling variant=$V np=1 omp=$T grid=${GX}x${GY} niter=$NITER rep=$R"
            mpirun -np 1 --bind-to none \
                "$BIN_DIR/stencil_$V" -x $GX -y $GY -n $NITER -o 0 -p 0
            echo "#END"
        done
    done
done