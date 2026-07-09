#!/bin/bash
#SBATCH --job-name=mpi_strong
#SBATCH --partition=EPYC
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH -A dssc
#SBATCH --exclusive
#SBATCH --output=mpi_strong_scaling.out

# ------------------------------------------------------------------------
# MPI strong scaling.
#
# Requests the FULL Orfeo EPYC partition (8 nodes - that's all it has;
# Leonardo-style 16 would over-ask here) up front, then loops NODES over
# 1,2,4,8, each time using only the first NODES hostnames from that one
# allocation to build a subset rankfile. Total grid size is held FIXED
# across all node counts.
#
# 8 tasks/node x 16 threads/task = 128 cores/node (full node), and 16
# lands each rank on exactly one Orfeo NUMA domain.
# ------------------------------------------------------------------------

module load openMPI/4.1.6

mpicc -fopenmp -O3 -march=native -DTILE_SIZE=64 -o stencil_strong stencil_parallel.c

EXEC=./stencil_strong
XSIZE=64000
YSIZE=64000
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

RAW=mpi_strong_scaling_raw.csv
OUTFILE=mpi_strong_scaling_results.csv
echo "nodes,tasks,rep,elapsed_s,comp_max_s,comp_avg_s,comm_max_s,comm_avg_s" > ${RAW}

for NODES in ${NODES_LIST}; do
    TOTAL_TASKS=$(( NODES * TASKS_PER_NODE ))

    RANKFILE=rank_file_strong_${NODES}
    rm -f ${RANKFILE}
    rank=0
    for ((n = 0; n < NODES; n++)); do
        node=${HOSTNAMES[$n]}
        for ((r = 0; r < TASKS_PER_NODE; r++)); do
            start_core=$(( r * THREADS_PER_TASK ))
            end_core=$(( start_core + THREADS_PER_TASK - 1 ))
            echo "rank $rank=$node slot=$start_core-$end_core" >> ${RANKFILE}
            ((rank++))
        done
    done

    for REP in $(seq 1 ${REPS}); do
        echo "Running with ${NODES} node(s) (${TOTAL_TASKS} MPI tasks), rep ${REP}"
        OUTPUT=$(mpirun -np ${TOTAL_TASKS} --rankfile ${RANKFILE} ${EXEC} \
            -x ${XSIZE} -y ${YSIZE} -n ${NITER} -e ${NSOURCES} -p ${PERIODIC} -o 0)
        # These field positions/patterns match this binary's ACTUAL printf
        # strings (checked against stencil_parallel.c) - not the same text
        # your friend's job greps for, since that's a different template.
        ELAPSED=$(echo "${OUTPUT}" | grep "Elapsed time:" | awk '{print $3}')
        COMP_MAX=$(echo "${OUTPUT}" | grep "Computation time" | awk '{print $6}')
        COMP_AVG=$(echo "${OUTPUT}" | grep "Computation time" | awk -F'avg ' '{print $2}' | tr -d ')')
        COMM_MAX=$(echo "${OUTPUT}" | grep "Communication time" | awk '{print $6}')
        COMM_AVG=$(echo "${OUTPUT}" | grep "Communication time" | awk -F'avg ' '{print $2}' | tr -d ')')
        echo "  Elapsed: ${ELAPSED}s  Comp(max/avg): ${COMP_MAX}/${COMP_AVG}s  Comm(max/avg): ${COMM_MAX}/${COMM_AVG}s"
        echo "${NODES},${TOTAL_TASKS},${REP},${ELAPSED},${COMP_MAX},${COMP_AVG},${COMM_MAX},${COMM_AVG}" >> ${RAW}
    done

    rm -f ${RANKFILE}
done

echo ""
echo "Results saved to ${OUTFILE}"

# Average the reps per node count, then compute speedup/efficiency against
# the 1-node baseline using computation time (isolates the kernel from
# fixed I/O/setup overhead, same rationale as the OMP/weak-scaling scripts).
python3 - "${RAW}" "${OUTFILE}" <<'EOF'
import csv, sys
from collections import defaultdict
raw_path, out_path = sys.argv[1], sys.argv[2]
sums = defaultdict(lambda: [0.0, 0, None])  # nodes -> [sum_comp_max, count, tasks]
with open(raw_path) as f:
    for row in csv.DictReader(f):
        n = int(row["nodes"])
        sums[n][0] += float(row["comp_max_s"])
        sums[n][1] += 1
        sums[n][2] = row["tasks"]
nodes_sorted = sorted(sums)
baseline = sums[nodes_sorted[0]][0] / sums[nodes_sorted[0]][1]
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["nodes", "tasks", "comp_max_s_avg", "speedup", "efficiency"])
    for n in nodes_sorted:
        avg_comp = sums[n][0] / sums[n][1]
        speedup = baseline / avg_comp
        efficiency = speedup / n
        w.writerow([n, sums[n][2], f"{avg_comp:.6f}", f"{speedup:.4f}", f"{efficiency:.4f}"])
print(f"Wrote {out_path}")
EOF
echo ""
echo "Per-rep raw data: ${RAW}"
echo "Averaged speedup/efficiency: ${OUTFILE}"