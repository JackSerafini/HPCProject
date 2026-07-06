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
# MPI strong scaling, single self-contained script (no other files needed
# besides the rankfile it builds itself, one per iteration).
#
# Requests the FULL Orfeo EPYC partition (8 nodes - that's all it has;
# Leonardo-style 16 would over-ask here) up front, then loops NODES over
# 1,2,4,8, each time using only the first NODES hostnames from that one
# allocation to build a subset rankfile. Total grid size is held FIXED
# across all node counts (that's what makes it "strong" scaling).
#
# 8 tasks/node x 16 threads/task = 128 cores/node (full node), and 16
# lands each rank on exactly one Orfeo NUMA domain (see hardware/NOTES.md
# from the probe job - each socket here is split into 4 NUMA nodes of 16
# cores, not 2 nodes of 64).
# ------------------------------------------------------------------------

module load openMPI/4.1.6

mpicc -fopenmp -O3 -march=native -o stencil stencil_parallel.c

EXEC=./stencil
XSIZE=60000
YSIZE=60000
NITER=100
NSOURCES=4
PERIODIC=0

TASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}
THREADS_PER_TASK=${SLURM_CPUS_PER_TASK}

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=${THREADS_PER_TASK}

HOSTNAMES=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))

OUTFILE="mpi_strong_scaling_results.csv"
echo "nodes,tasks,elapsed_s,comp_max_s,comp_avg_s,comm_max_s,comm_avg_s" > ${OUTFILE}

for NODES in 1 2 4; do
    TOTAL_TASKS=$(( NODES * TASKS_PER_NODE ))

    RANK_FILE=rank_file_strong_${NODES}
    rm -f ${RANK_FILE}
    rank=0
    for ((n = 0; n < NODES; n++)); do
        node=${HOSTNAMES[$n]}
        for ((r = 0; r < TASKS_PER_NODE; r++)); do
            start_core=$(( r * THREADS_PER_TASK ))
            end_core=$(( start_core + THREADS_PER_TASK - 1 ))
            echo "rank $rank=$node slot=$start_core-$end_core" >> ${RANK_FILE}
            ((rank++))
        done
    done

    echo "Running with ${NODES} node(s) (${TOTAL_TASKS} MPI tasks)"
    OUTPUT=$(mpirun --rankfile ${RANK_FILE} --report-bindings ${EXEC} \
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
    echo "${NODES},${TOTAL_TASKS},${ELAPSED},${COMP_MAX},${COMP_AVG},${COMM_MAX},${COMM_AVG}" >> ${OUTFILE}

    rm -f ${RANK_FILE}
done

echo ""
echo "Results saved to ${OUTFILE}"