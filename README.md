# HPCProject — Hybrid MPI + OpenMP Stencil Solver

A hybrid **MPI + OpenMP** solver for the 2D five-point stencil heat-diffusion equation, implemented and benchmarked for the *Foundations of HPC* course (University of Trieste). The repository contains the solver itself, a serial reference implementation, the full SLURM benchmarking suite used to characterize it on the [Orfeo](https://orfeo-doc.areasciencepark.it) cluster, the raw/processed results, and the written report and slide deck.

## The problem
 
The solver simulates heat diffusing across a 2D plate of size `Sx × Sy`. At each iteration:
 
1. A fixed set of point sources injects energy into the grid.
2. Every interior cell is updated with a five-point stencil:
```
   new[i,j] = 0.5 * old[i,j] + 0.125 * (old[i-1,j] + old[i+1,j] + old[i,j-1] + old[i,j+1])
```
 
## Repository layout
 
```
HPCProject/
├── src/
│   ├── stencil_parallel.c          # hybrid MPI+OpenMP solver (main deliverable)
│   └── stencil_template_serial.c   # serial reference implementation
├── include/
│   ├── stencil_parallel.h
│   └── stencil_template_serial.h
├── scripts/
│   ├── probe_hardware.sh           # SLURM job: dumps lscpu/cache/NUMA topology, suggests TILE_SIZE candidates
│   ├── openmp_scaling_tile.sh      # SLURM job: OpenMP thread scaling, 1 MPI rank, 1→128 threads
│   ├── mpi_strong_scaling.sh       # SLURM job: MPI strong scaling, fixed problem size, 1/2/4 nodes
│   ├── mpi_weak_scaling.sh         # SLURM job: MPI weak scaling, grid side ∝ √(nodes)
│   ├── cachegrind.sh               # SLURM job: Valgrind Cachegrind sweep across TILE_SIZE variants
│   └── plot.ipynb                  # notebook that produces the scaling plots used in the report
├── runs/                           # collected benchmark data
├── logs/
│   └── hardware_report.txt         # output of probe_hardware.sh on the Orfeo EPYC node used for benchmarking
├── Dockerfile                      # Ubuntu 24.04 + OpenMPI + libomp image, for local dev
├── go_dcgp                         # original SLURM template for CINECA Leonardo's DCGP partition
├── assignment.pdf                  # course assignment specification
├── Report.pdf                      # written technical report
├── Presentation.pdf                # slide deck
└── LICENSE                         # MIT
```
 
## Implementation highlights
 
- **Decomposition** — 2D Cartesian block decomposition of the plate across MPI ranks. The process grid's aspect ratio is chosen to mirror the plate's aspect ratio (falling back to a 1D split when the rank count is small relative to the plate's elongation), via a simple prime-factorization of `Ntasks`.
- **Halo exchange** — non-blocking `MPI_Isend`/`MPI_Irecv` on all four directions, issued unconditionally every iteration. Boundary ranks get `MPI_PROC_NULL` neighbours, which makes their corresponding sends/receives automatic no-ops — no per-direction branching needed.
- **Compute/communication overlap** — `update_plane_inside()` (the interior, which doesn't depend on halo data) runs while the halo transfer is in flight; only `update_plane_border()` blocks on `MPI_Waitall`.
- **Zero-copy N/S halos** — north/south rows are already contiguous in memory, so their send/recv buffers are pointers directly into the plane data. Only east/west columns (strided in row-major layout) require explicit pack/unpack loops.
- **OpenMP** — `#pragma omp parallel for collapse(2)` over the interior update, with an optional compile-time cache-blocked variant (`-DTILE_SIZE=N`) using `#pragma omp simd` on the innermost loop.
- **NUMA-aware first touch** — the initial grid-zeroing pass follows the same tiling pattern as the interior update, so pages land on the NUMA node that will actually touch them under a first-touch allocation policy.
- **Memory** — 64-byte aligned allocation (`posix_memalign`) and `restrict`-qualified pointers throughout, to enable vectorization and rule out aliasing between the old/new grid planes.
- **Timing** — `MPI_Wtime` brackets the compute and communication phases separately each iteration; totals are reduced with both `MPI_MAX` (critical path) and `MPI_SUM`/`Ntasks` (average) to expose load imbalance.

## Building
 
### Local development (Docker)
 
The image targets local correctness testing (e.g. on Apple Silicon) — it does **not** reproduce the cluster's `-march=native` instruction set or NUMA layout, so it isn't used for benchmarking.
 
```bash
docker build -t hpc-dev .
docker run -it --rm -v "$(pwd)":/workspace hpc-dev
 
# inside the container
mpicc -fopenmp src/stencil_parallel.c -o stencil
mpirun --allow-run-as-root -np 4 ./stencil -x 2000 -y 2000 -n 200
```
 
### Cluster (Orfeo, EPYC partition)
 
```bash
module load openMPI/4.1.6
mpicc -fopenmp -O3 -march=native -DTILE_SIZE=64 -o stencil src/stencil_parallel.c
```
 
- `-DTILE_SIZE=<N>` *(optional)* — enables the cache-blocked interior loop with an `N×N` tile; omit it (or set `0`) for the plain, untiled loop. All benchmarks in `runs/` use `TILE_SIZE=64`.

## Running
 
```bash
mpirun -np <N> ./stencil [flags]
```
 
| Flag | Meaning | Default |
|------|---------|---------|
| `-x` | x size of the plate | `10000` |
| `-y` | y size of the plate | `10000` |
| `-e` | number of energy sources | `4` |
| `-E` | energy per source | `1.0` |
| `-n` | number of iterations | `1000` |
| `-p` | periodic boundaries (`0`/`1`) | `0` (off) |
| `-o` | print energy stats every step (`0`/`1`) | *unset — pass explicitly* |
| `-v` | verbosity: print rank/grid/neighbour layout if `> 0` | `0` |
| `-h` | print usage and exit | — |
 
`-o` has no default value in the code, so all benchmarking scripts pass `-o 0` explicitly to disable per-step output (it's needed for correctness checks, not for timing runs).
 
## Benchmarking
 
Each `scripts/*.sh` file is a self-contained SLURM batch script: it compiles the required binary, sweeps a parameter, greps timing out of the solver's stdout, writes a raw per-repetition CSV, then reduces it to an averaged speedup/efficiency CSV via an inline Python block.
 
| Script | Sweep | Fixed problem size |
|--------|-------|---------------------|
| `probe_hardware.sh` | — | reports cache/NUMA topology + suggests `TILE_SIZE` candidates |
| `openmp_scaling_tile.sh` | 1 MPI rank, `OMP_NUM_THREADS` ∈ {1,2,4,8,16,32,64,96,128} | 25600 × 25600 |
| `mpi_strong_scaling.sh` | 1/2/4 nodes × 8 ranks/node × 16 threads/rank (1 rank per NUMA domain) | 64000 × 64000 (fixed total) |
| `mpi_weak_scaling.sh` | 1/2/4 nodes, same rank/thread layout | grid side ∝ √(nodes), base 40000 |
| `cachegrind.sh` | `TILE_SIZE` ∈ {none,64,128,256,512,1024}, 1 thread | 8192 × 8192 (reduced for emulation cost) |
 
Submit on Orfeo with e.g. `sbatch scripts/mpi_strong_scaling.sh`; outputs land alongside the script (`*_raw.csv`, `*_results.csv`, `*.out`). `scripts/plot.ipynb` reads the `runs/*_results.csv` files and produces the speedup/efficiency plots (log2 axes, ideal-scaling reference line, annotated data points).
 
## Results
 
Measured on Orfeo's EPYC partition: speedup/efficiency are computed from the total time, averaged over 3 repetitions.
 
**OpenMP scaling** (1 MPI rank, `TILE_SIZE=64`, 25600×25600 grid):
 
| Threads | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| Speedup | 1.00 | 1.95 | 3.45 | 4.39 | 6.10 | 12.19 | 24.22 | 35.93 | 47.53 |
| Efficiency | 1.00 | 0.98 | 0.86 | 0.55 | 0.38 | 0.38 | 0.38 | 0.37 | 0.37 |
 
**MPI strong scaling** (8 ranks/node × 16 threads/rank, 64000×64000 grid fixed):
 
| Nodes | 1 | 2 | 4 |
|---|---|---|---|
| Tasks | 8 | 16 | 32 |
| Speedup | 1.00 | 1.992 | 3.952 |
| Efficiency | 1.00 | 0.996 | 0.988 |
 
**MPI weak scaling** (grid side ∝ √(nodes), base 40000):
 
| Nodes | 1 | 2 | 4 |
|---|---|---|---|
| Tasks | 8 | 16 | 32 |
| Grid side | 40000 | 56576 | 80000 |
| Efficiency | 1.00 | 0.996 | 0.979 |
