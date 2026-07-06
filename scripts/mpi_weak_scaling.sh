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
# MPI weak scaling - identical structure to mpi_strong_scaling_standalone.sh,
# except the grid side scales as sqrt(NODES) * BASE_SIDE so that total
# work / total cores stays roughly constant as NODES grows (that's what
# makes it "weak" scaling instead of strong).
# ------------------------------------------------------------------------

module load openMPI/4.1.6

mpicc -fopenmp -O3 -march=native -o stencil stencil_parallel.c

EXEC=./stencil
BASE_SIDE=10000     # grid side at NODES=1
NITER=100
NSOURCES=4
PERIODIC=0

TASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}
THREADS_PER_TASK=${SLURM_CPUS_PER_TASK}

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=${THREADS_PER_TASK}

HOSTNAMES=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))

OUTFILE="mpi_weak_scaling_results.csv"
echo "nodes,tasks,grid_side,elapsed_s,comp_max_s,comp_avg_s,comm_max_s,comm_avg_s" > ${OUTFILE}

for NODES in 1 2 4; do
    TOTAL_TASKS=$(( NODES * TASKS_PER_NODE ))
    SIDE=$(python3 -c "import math; print(round(${BASE_SIDE} * math.sqrt(${NODES})))")

    RANK_FILE=rank_file_weak_${NODES}
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

    echo "Running with ${NODES} node(s) (${TOTAL_TASKS} MPI tasks), grid ${SIDE}x${SIDE}"
    OUTPUT=$(mpirun --rankfile ${RANK_FILE} --report-bindings ${EXEC} \
        -x ${SIDE} -y ${SIDE} -n ${NITER} -e ${NSOURCES} -p ${PERIODIC} -o 0)

    ELAPSED=$(echo "${OUTPUT}" | grep "Elapsed time:" | awk '{print $3}')
    COMP_MAX=$(echo "${OUTPUT}" | grep "Computation time" | awk '{print $6}')
    COMP_AVG=$(echo "${OUTPUT}" | grep "Computation time" | awk -F'avg ' '{print $2}' | tr -d ')')
    COMM_MAX=$(echo "${OUTPUT}" | grep "Communication time" | awk '{print $6}')
    COMM_AVG=$(echo "${OUTPUT}" | grep "Communication time" | awk -F'avg ' '{print $2}' | tr -d ')')

    echo "  Elapsed: ${ELAPSED}s  Comp(max/avg): ${COMP_MAX}/${COMP_AVG}s  Comm(max/avg): ${COMM_MAX}/${COMM_AVG}s"
    echo "${NODES},${TOTAL_TASKS},${SIDE},${ELAPSED},${COMP_MAX},${COMP_AVG},${COMM_MAX},${COMM_AVG}" >> ${OUTFILE}

    rm -f ${RANK_FILE}
done

echo ""
echo "Results saved to ${OUTFILE}"