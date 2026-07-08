#!/bin/bash
#SBATCH --job-name=cachegrind
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -A dssc
#SBATCH --output=cachegrind.out

# ---------------------------------------------------------------------------
# Cachegrind-based L1/L2 cache-miss comparison: no-tile vs TILE_SIZE=48 vs 64
#
# Design notes:
# - Cachegrind is a serial *emulator*: it single-steps the binary and models
#   the cache, so it's 20-100x slower than native and, unlike real hardware,
#   Valgrind schedules threads cooperatively (one at a time) rather than
#   running them concurrently. Since the vectorization-vs-cache-capacity
#   question is about the per-core access pattern inside one thread's tile,
#   OMP_NUM_THREADS=1 gives the cleanest and cheapest measurement: the effect
#   shows up identically whether one thread runs it or many run copies of it.
# - Cachegrind's counts are a deterministic simulation (not a hardware
#   timer), so unlike your wall-clock scaling runs, no repetitions are
#   needed - one run per config is enough.
# - Grid size is cut hard vs your production runs (25000x25000, NITER=100)
#   because of the emulation slowdown. XSIZE is kept large enough (8000,
#   i.e. a 62.5KB row of doubles) to still comfortably exceed L1d (32KB) per
#   row, so the untiled version still thrashes L1 the way it does at full
#   scale. NITER is cut hard since more iterations just repeat the same
#   access pattern - they add emulated instructions/runtime without
#   changing the steady-state miss RATE.
# - If the first run is too slow (or finishes suspiciously fast), adjust
#   NITER/XSIZE/YSIZE below and resubmit - actual Cachegrind slowdown is
#   workload-dependent and hard to predict exactly without a first data point.
# ---------------------------------------------------------------------------

module load openMPI/4.1.6

mpicc -fopenmp -O3 -march=native -o stencil_notile stencil_parallel.c
mpicc -fopenmp -O3 -march=native -DTILE_SIZE=64 -o stencil_tile64 stencil_parallel.c
mpicc -fopenmp -O3 -march=native -DTILE_SIZE=128 -o stencil_tile128 stencil_parallel.c
mpicc -fopenmp -O3 -march=native -DTILE_SIZE=256 -o stencil_tile256 stencil_parallel.c
mpicc -fopenmp -O3 -march=native -DTILE_SIZE=512 -o stencil_tile512 stencil_parallel.c
mpicc -fopenmp -O3 -march=native -DTILE_SIZE=1024 -o stencil_tile1024 stencil_parallel.c # 3 rows exceed L1

XSIZE=8192
YSIZE=8192
NITER=25
NSOURCES=4
PERIODIC=0

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# EPYC (Zen) cache geometry: 32KB 8-way 64B L1d/L1i, 512KB 8-way 64B L2.
# Passed explicitly so the simulated cache matches the real hardware
# regardless of Valgrind's auto-detection. Verify associativity/line size
# against `lscpu -C` on an Orfeo EPYC node if you want to be certain.
D1_CFG="32768,8,64"
I1_CFG="32768,8,64"
LL_CFG="524288,8,64"   # modelling L2 as Cachegrind's single "LL" level -
                       # Cachegrind only simulates two levels, so the shared
                       # L3 isn't represented; worth a caveat in the report.

RESULTS=cachegrind_results.csv
echo "config,I_refs,I1_misses,I1_miss_rate,D_refs,D1_misses,D1_miss_rate,LL_misses,LL_miss_rate" > ${RESULTS}

declare -A EXECS=( [notile]=stencil_notile [tile64]=stencil_tile64 [tile128]=stencil_tile128 [tile256]=stencil_tile256 [tile512]=stencil_tile512 [tile1024]=stencil_tile1024 )

for LABEL in notile tile64 tile128 tile256 tile512 tile1024; do
    EXEC=${EXECS[${LABEL}]}
    LOG=cachegrind_${LABEL}.log
    OUTFILE=cachegrind_${LABEL}.out

    echo "Running cachegrind on ${LABEL}..."
    mpirun -np 1 valgrind --tool=cachegrind \
        --cache-sim=yes \
        --D1=${D1_CFG} --I1=${I1_CFG} --LL=${LL_CFG} \
        --cachegrind-out-file=${OUTFILE} \
        ./${EXEC} -x ${XSIZE} -y ${YSIZE} -n ${NITER} -e ${NSOURCES} -p ${PERIODIC} -o 0 \
        > ${LOG} 2>&1

    I_REFS=$(grep -E "^==[0-9]+== I +refs:"          ${LOG} | awk '{print $4}' | tr -d ',')
    I1_MISS=$(grep -E "^==[0-9]+== I1 +misses:"       ${LOG} | awk '{print $4}' | tr -d ',')
    I1_RATE=$(grep -E "^==[0-9]+== I1 +miss rate:"    ${LOG} | awk '{print $5}' | tr -d '%')
    D_REFS=$(grep -E "^==[0-9]+== D +refs:"           ${LOG} | awk '{print $4}' | tr -d ',')
    D1_MISS=$(grep -E "^==[0-9]+== D1 +misses:"       ${LOG} | awk '{print $4}' | tr -d ',')
    D1_RATE=$(grep -E "^==[0-9]+== D1 +miss rate:"    ${LOG} | awk '{print $5}' | tr -d '%')
    LL_MISS=$(grep -E "^==[0-9]+== LL +misses:"       ${LOG} | awk '{print $4}' | tr -d ',')
    LL_RATE=$(grep -E "^==[0-9]+== LL +miss rate:"    ${LOG} | awk '{print $5}' | tr -d '%')

    echo "  I1 miss rate: ${I1_RATE}%  D1 miss rate: ${D1_RATE}%  LL miss rate: ${LL_RATE}%"
    echo "${LABEL},${I_REFS},${I1_MISS},${I1_RATE},${D_REFS},${D1_MISS},${D1_RATE},${LL_MISS},${LL_RATE}" >> ${RESULTS}
done

echo "Done. Summary in ${RESULTS}, full Cachegrind logs in cachegrind_<config>.log"
echo "For line-by-line hotspot detail (which lines in the tiled vs untiled"
echo "loop generate the misses), run e.g.:"
echo "  cg_annotate cachegrind_tile48.out stencil_parallel.c"