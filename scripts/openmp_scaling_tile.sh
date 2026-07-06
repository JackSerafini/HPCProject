#!/bin/bash
#SBATCH --job-name=omp_scaling
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -A dssc
#SBATCH --output=omp_scaling_tile48.out

# ------------------------------------------------------------------------
# Requirement (D), OpenMP leg: 1 MPI task, scale threads from 1 up to
# filling the whole node.
#
# Also worth knowing for interpreting the curve: Orfeo's 2 sockets are
# further split into 8 NUMA domains of 16 cores each (confirmed via the
# hardware probe, not the "2 domains" you'd naively assume from 2
# sockets) - so don't be surprised by small steps in the curve right
# around 16, 32, 48, 64, 96 threads, where this single rank's threads
# start spilling into the next NUMA domain.
# ------------------------------------------------------------------------

module load openMPI/4.1.6

mpicc -fopenmp -O3 -march=native -DTILE_SIZE=64 -o stencil_tile48 stencil_parallel.c

EXEC=./stencil_tile48
XSIZE=25000
YSIZE=25000
NITER=100
NSOURCES=4
PERIODIC=0
REPS=2

export OMP_PLACES=cores
export OMP_PROC_BIND=close

THREAD_COUNTS="1 2 4 8 16 32 64 96 128"

RAW=omp_scaling_tile48_raw.csv
OUTFILE=omp_scaling_tile48_results.csv
echo "threads,rep,elapsed_s,comp_max_s,comp_avg_s,comm_max_s,comm_avg_s" > ${RAW}

for T in ${THREAD_COUNTS}; do
    export OMP_NUM_THREADS=${T}
    for REP in $(seq 1 ${REPS}); do
        echo "Running with ${T} thread(s), rep ${REP}"
        OUTPUT=$(mpirun -np 1 --bind-to none --map-by node:PE=${T} ${EXEC} \
            -x ${XSIZE} -y ${YSIZE} -n ${NITER} -e ${NSOURCES} -p ${PERIODIC} -o 0)

        ELAPSED=$(echo "${OUTPUT}"   | grep "Elapsed time:"     | awk '{print $3}')
        COMP_MAX=$(echo "${OUTPUT}"  | grep "Computation time"  | awk '{print $6}')
        COMP_AVG=$(echo "${OUTPUT}"  | grep "Computation time"  | awk -F'avg ' '{print $2}' | tr -d ')')
        COMM_MAX=$(echo "${OUTPUT}"  | grep "Communication time"| awk '{print $6}')
        COMM_AVG=$(echo "${OUTPUT}"  | grep "Communication time"| awk -F'avg ' '{print $2}' | tr -d ')')

        echo "  Elapsed: ${ELAPSED}s  Comp(max/avg): ${COMP_MAX}/${COMP_AVG}s"
        echo "${T},${REP},${ELAPSED},${COMP_MAX},${COMP_AVG},${COMM_MAX},${COMM_AVG}" >> ${RAW}
    done
done

# Average the reps per thread count, then compute speedup/efficiency
# against the 1-thread baseline (using computation time, not elapsed -
# comp time isolates the OpenMP kernel from fixed I/O/setup overhead).
python3 - "${RAW}" "${OUTFILE}" <<'EOF'
import csv, sys
from collections import defaultdict

raw_path, out_path = sys.argv[1], sys.argv[2]

sums = defaultdict(lambda: [0.0, 0])  # threads -> [sum_comp_max, count]
with open(raw_path) as f:
    for row in csv.DictReader(f):
        t = int(row["threads"])
        sums[t][0] += float(row["comp_max_s"])
        sums[t][1] += 1

threads_sorted = sorted(sums)
baseline = sums[threads_sorted[0]][0] / sums[threads_sorted[0]][1]

with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["threads", "comp_max_s_avg", "speedup", "efficiency"])
    for t in threads_sorted:
        avg_comp = sums[t][0] / sums[t][1]
        speedup = baseline / avg_comp
        efficiency = speedup / t
        w.writerow([t, f"{avg_comp:.6f}", f"{speedup:.4f}", f"{efficiency:.4f}"])

print(f"Wrote {out_path}")
EOF

echo ""
echo "Per-rep raw data: ${RAW}"
echo "Averaged speedup/efficiency: ${OUTFILE}"