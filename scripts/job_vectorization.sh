#!/bin/bash
#SBATCH --job-name=vec_check
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8gb
#SBATCH --time=00:20:00
#SBATCH -A dssc
#SBATCH --output=vec_check.log

set -e

module load openMPI/4.1.6

echo "Running on $(hostname)"

# -------------------------
# SMALL PROBLEM SIZE (IMPORTANT)
# -------------------------
GX=1024
GY=1024
NITER=10

BIN_DIR=./bin
SRC=stencil_parallel.c

mkdir -p vec_logs "$BIN_DIR"

# -------------------------
# FUNCTION: compile with vec report
# -------------------------
compile_variant () {
    local name=$1
    local flags=$2
    local outbin="$BIN_DIR/$name"
    local vecfile_pass="vec_logs/${name}_pass.vec"
    local vecfile_miss="vec_logs/${name}_miss.vec"

    echo "Compiling $name ..."

    mpicc -O3 -march=native -fopenmp \
        $flags \
        -fopt-info-vec-optimized=$vecfile_pass \
        -fopt-info-vec-missed=$vecfile_miss \
        $SRC -o $outbin
}

# -------------------------
# BUILD BOTH VERSIONS
# -------------------------
compile_variant "notile" ""
compile_variant "tile64" "-DTILE_SIZE=64"

echo ""
echo "Compilation done. Running benchmarks..."
echo ""

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

run_variant () {
    local name=$1
    echo "# RUN $name"

    mpirun -np 1 --bind-to none \
        $BIN_DIR/$name \
        -x $GX -y $GY -n $NITER -o 0 -p 0

    echo "# END $name"
    echo ""
}

run_variant notile
run_variant tile64

echo "Done. Vectorization logs in vec_logs/"