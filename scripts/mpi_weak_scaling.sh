#!/bin/bash
#SBATCH --job-name=mpi_weak
#SBATCH --partition=EPYC
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH -A dssc
#SBATCH --exclusive
#SBATCH --output=mpi_weak_scaling.out

# ------------------------------------------------------------------------
# MPI weak scaling - identical structure to mpi_strong_scaling.sh,
# except the grid side scales as sqrt(NODES) * BASE_SIDE so that total
# work / total cores stays roughly constant as NODES grows (that's what
# makes it "weak" scaling instead of strong).
# ------------------------------------------------------------------------

module load openMPI/4.1.6

mpicc -fopenmp -O3 -march=native -DTILE_SIZE=64 -o stencil_weak stencil_parallel.c

EXEC=./stencil_weak
BASE_SIDE=40000
NITER=100
NSOURCES=4
PERIODIC=0
REPS=3
NODES_LIST="1 2 4"

TASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}
THREADS_PER_TASK=${SLURM_CPUS_PER_TASK}

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=${THREADS_PER_TASK}

HOSTNAMES=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))

RAW=mpi_weak_scaling_raw.csv
OUTFILE=mpi_weak_scaling_results.csv
echo "nodes,tasks,rep,grid_side,elapsed_s,comp_max_s,comp_avg_s,comm_max_s,comm_avg_s" > ${RAW}

# Compute the rounded-to-nearest-64 grid side for a given node count,
# so the sqrt(NODES) scaling law is respected as closely as integer
# grids divisible by 64 allow.
grid_side_for () {
    local N=$1
    python3 -c "
import math
base = ${BASE_SIDE}
n = ${N}
ideal = base * math.sqrt(n)
tile = 64
rounded = round(ideal / tile) * tile
print(int(rounded))
"
}

for N in ${NODES_LIST}; do
    SIDE=$(grid_side_for ${N})
    NTASKS=$(( N * TASKS_PER_NODE ))

    # Build a rankfile: NTASKS ranks total, cycling through the first N
    # allocated hosts, TASKS_PER_NODE ranks per host, each rank pinned to
    # its own contiguous block of THREADS_PER_TASK cores.
    RANKFILE=rank_file_weak_${N}
    rm -f ${RANKFILE}
    rank=0
    for ((n = 0; n < N; n++)); do
        node=${HOSTNAMES[$n]}
        for ((r = 0; r < TASKS_PER_NODE; r++)); do
            start_core=$(( r * THREADS_PER_TASK ))
            end_core=$(( start_core + THREADS_PER_TASK - 1 ))
            echo "rank $rank=$node slot=$start_core-$end_core" >> ${RANKFILE}
            ((rank++))
        done
    done

    for REP in $(seq 1 ${REPS}); do
        echo "Running with ${N} node(s), ${NTASKS} MPI tasks, grid ${SIDE}x${SIDE}, rep ${REP}"
        OUTPUT=$(mpirun -np ${NTASKS} --rankfile ${RANKFILE} ${EXEC} \
            -x ${SIDE} -y ${SIDE} -n ${NITER} -e ${NSOURCES} -p ${PERIODIC} -o 0)
        ELAPSED=$(echo "${OUTPUT}"   | grep "Elapsed time:"     | awk '{print $3}')
        COMP_MAX=$(echo "${OUTPUT}"  | grep "Computation time"  | awk '{print $6}')
        COMP_AVG=$(echo "${OUTPUT}"  | grep "Computation time"  | awk -F'avg ' '{print $2}' | tr -d ')')
        COMM_MAX=$(echo "${OUTPUT}"  | grep "Communication time"| awk '{print $6}')
        COMM_AVG=$(echo "${OUTPUT}"  | grep "Communication time"| awk -F'avg ' '{print $2}' | tr -d ')')
        echo "  Elapsed: ${ELAPSED}s  Comp(max/avg): ${COMP_MAX}/${COMP_AVG}s  Comm(max/avg): ${COMM_MAX}/${COMM_AVG}s"
        echo "${N},${NTASKS},${REP},${SIDE},${ELAPSED},${COMP_MAX},${COMP_AVG},${COMM_MAX},${COMM_AVG}" >> ${RAW}
    done

    rm -f ${RANKFILE}
done

# Average the reps per node count, then compute weak-scaling efficiency
# against the 1-node baseline using computation time (isolates the
# kernel from fixed I/O/setup overhead, same rationale as the OMP script).
python3 - "${RAW}" "${OUTFILE}" <<'EOF'
import csv, sys
from collections import defaultdict
raw_path, out_path = sys.argv[1], sys.argv[2]
sums = defaultdict(lambda: [0.0, 0, None, None])  # nodes -> [sum_comp_max, count, tasks, grid_side]
with open(raw_path) as f:
    for row in csv.DictReader(f):
        n = int(row["nodes"])
        sums[n][0] += float(row["comp_max_s"])
        sums[n][1] += 1
        sums[n][2] = row["tasks"]
        sums[n][3] = row["grid_side"]
nodes_sorted = sorted(sums)
baseline = sums[nodes_sorted[0]][0] / sums[nodes_sorted[0]][1]
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["nodes", "tasks", "grid_side", "comp_max_s_avg", "efficiency"])
    for n in nodes_sorted:
        avg_comp = sums[n][0] / sums[n][1]
        efficiency = baseline / avg_comp  # weak scaling: ideal is comp time staying flat
        w.writerow([n, sums[n][2], sums[n][3], f"{avg_comp:.6f}", f"{efficiency:.4f}"])
print(f"Wrote {out_path}")
EOF
echo ""
echo "Per-rep raw data: ${RAW}"
echo "Averaged efficiency: ${OUTFILE}"