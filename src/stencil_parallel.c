/*
mysizex   :   local x-extendion of your patch
mysizey   :   local y-extension of your patch
*/

#include "../include/stencil_parallel.h"

// ------------------------------------------------------------------

int main(int argc, char **argv) {
    MPI_Comm myCOMM_WORLD;
    int Rank, Ntasks; // this process's ID (0 to Ntasks-1); total number of MPI processes
    uint neighbours[4]; // ranks of this process's North/South/East/West neighbors

    int Niterations;
    int periodic;
    vec2_t S, N; // global plate size (Sx, Sy); process grid dimensions (Nx, Ny): how the Ntasks processes are arranged in a 2D grid
    
    int Nsources; // total heat sources across the whole simulation
    int Nsources_local; // how many of those landed on this process's patch
    vec2_t *Sources_local; // their local coordinates
    double energy_per_source;

    plane_t planes[2]; // the OLD/NEW double-buffer
    buffers_t buffers[2]; // SEND/RECV halo-communication buffers (one set per direction, per SEND/RECV)
    
    int output_energy_stat_perstep;
    
    /* initialize MPI envrionment */
    {
        int level_obtained;
        
        // NOTE: change MPI_FUNNELED (= only the main thread will ever call MPI functions) if appropriate
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &level_obtained); // declare what level of thread-safety this program needs
        if ( level_obtained < MPI_THREAD_FUNNELED )
        {
            printf("MPI_thread level obtained is %d instead of %d\n", level_obtained, MPI_THREAD_FUNNELED);
            MPI_Finalize();
            exit(1);
        }
        
        // this process's ID and the total count
        MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
        MPI_Comm_size(MPI_COMM_WORLD, &Ntasks);
        MPI_Comm_dup (MPI_COMM_WORLD, &myCOMM_WORLD); // make a private copy of the communicator
    }

    // argument checking and setting
    int ret = initialize(&myCOMM_WORLD, Rank, Ntasks, argc, argv, &S, &N, &periodic, &output_energy_stat_perstep,
                neighbours, &Niterations,
                &Nsources, &Nsources_local, &Sources_local, &energy_per_source,
                &planes[0], &buffers[0]);

    if (ret)
    {
        printf("task %d is opting out with termination code %d\n", Rank, ret);
        
        MPI_Finalize();
        return 0;
    }
    
    int current = OLD;
    // accumulators for this rank's total time spent in compute vs. comm
    // sections across the whole run; t0 is a scratch variable reused to
    // bracket each region with MPI_Wtime()
    double t_comp = 0.0, t_comm = 0.0, t0;
    double t1 = MPI_Wtime(); /* take wall-clock time */
    
    for (int iter = 0; iter < Niterations; ++iter)
    {
        MPI_Request reqs[8]; // handle for up to 8 non-blocking MPI operations (4 directions x send+receive)
        
        /* new energy from sources */
        t0 = MPI_Wtime();
        // each iteration first injects fresh heat at the source points into the current plane
        inject_energy(periodic, Nsources_local, Sources_local, energy_per_source, &planes[current], N);
        t_comp += MPI_Wtime() - t0;

        // before updating the grid, each process needs its neighbors' edge data copied into its own halo cells
        /* -------------------------------------- */
        const uint fxsize = planes[current].size[_x_] + 2; // full (haloed) row width
        const uint xsize = planes[current].size[_x_];
        const uint ysize = planes[current].size[_y_];
 
        #define IDX(i, j) ( (j)*fxsize + (i) )
 
        double *restrict data = planes[current].data;

        // [A] fill the buffers, and/or make the buffers' pointers pointing to the correct position
        // Pack the data you need to send into a buffer, or just point directly at it if it's already contiguous

        // NORTH/SOUTH: a row of the grid is already contiguous in memory (row-major layout), so there is nothing to copy
        // -> repoint the SEND buffer pointer at the first/last interior row of `data` -> "zero-copy" trick
        buffers[SEND][NORTH] = &data[IDX(1, 1)]; // first interior row (j=1)
        buffers[SEND][SOUTH] = &data[IDX(1, ysize)]; // last interior row  (j=ysize)
        // for (uint i = 1; i <= xsize; i++)
        // {
        //     buffers[SEND][NORTH][i-1] = data[IDX(i, 1)];
        //     buffers[SEND][SOUTH][i-1] = data[IDX(i, ysize)];
        // }

        // EAST/WEST: a column is strided in memory, so MPI cannot send it directly as a contiguous buffer
        // -> manual packing into the pre-allocated buffers[SEND][EAST/WEST] (sized `ysize` in memory_allocate) 
        // -> copy is O(ysize) and is the unavoidable price of row-major storage with column halos
        t0 = MPI_Wtime();
        for (uint j = 1; j <= ysize; j++)
        {
            buffers[SEND][EAST][j-1] = data[IDX(xsize, j)]; // last interior column
            buffers[SEND][WEST][j-1] = data[IDX(1, j)]; // first interior column
        }
        t_comm += MPI_Wtime() - t0;
 
        // RECV buffers: same zero-copy trick for north/south —> point RECV[NORTH/SOUTH]
        // directly at the halo rows (j=0 and j=ysize+1), so MPI writes the incoming
        // data straight into its final place and step [C] needs no extra copy for these two directions
        buffers[RECV][NORTH] = &data[IDX(1, 0)]; // north halo row
        buffers[RECV][SOUTH] = &data[IDX(1, ysize + 1)]; // south halo row
        // buffers[RECV][EAST] / buffers[RECV][WEST] keep the malloc'd packed buffers
        // from memory_allocate (sized `ysize`) -> unpack those by hand in step [C]
        // since the halo columns are strided and can't be filled in place

        // [B] perform the halo communications

        // With Isend/Irecv, the call returns immediately after MPI has been told
        // about the transfer; the actual data movement happens in the background
        // (network/progress engine) -> start computing the interior of the grid (the cells that do NOT depend on any neighbour's halo)
        // *while* the halo data is still in flight, and only block (MPI_Waitall) right before we need the halo values, 
        // i.e. when updating the border ring of the patch.
        // MPI_PROC_NULL neighbours (no neighbour in that direction, e.g. on a
        // non-periodic boundary) make the corresponding Isend/Irecv an
        // automatic no-op with a request that completes immediately —> issue all 8 calls unconditionally with no per-direction branching
 
        t0 = MPI_Wtime();
        MPI_Irecv(buffers[RECV][NORTH], xsize, MPI_DOUBLE, neighbours[NORTH], NORTH, myCOMM_WORLD, &reqs[0]);
        MPI_Irecv(buffers[RECV][SOUTH], xsize, MPI_DOUBLE, neighbours[SOUTH], SOUTH, myCOMM_WORLD, &reqs[1]);
        MPI_Irecv(buffers[RECV][EAST], ysize, MPI_DOUBLE, neighbours[EAST], EAST, myCOMM_WORLD, &reqs[2]);
        MPI_Irecv(buffers[RECV][WEST], ysize, MPI_DOUBLE, neighbours[WEST], WEST, myCOMM_WORLD, &reqs[3]);
 
        MPI_Isend(buffers[SEND][NORTH], xsize, MPI_DOUBLE, neighbours[NORTH], SOUTH, myCOMM_WORLD, &reqs[4]);
        MPI_Isend(buffers[SEND][SOUTH], xsize, MPI_DOUBLE, neighbours[SOUTH], NORTH, myCOMM_WORLD, &reqs[5]);
        MPI_Isend(buffers[SEND][EAST], ysize, MPI_DOUBLE, neighbours[EAST], WEST, myCOMM_WORLD, &reqs[6]);
        MPI_Isend(buffers[SEND][WEST], ysize, MPI_DOUBLE, neighbours[WEST], EAST, myCOMM_WORLD, &reqs[7]);
        t_comm += MPI_Wtime() - t0;

        // Tag convention: each Isend uses the tag of the direction its receiver is
        // looking for it under. E.g. my NORTH neighbour's "SOUTH" recv is filled by
        // my "NORTH"-directed send arriving at them from their south side, so I tag
        // it SOUTH; symmetrically I post my own MPI_Irecv[NORTH] expecting a message
        // tagged NORTH from that same neighbour (sent by them as their SOUTH send).
        // This NORTH<->SOUTH, EAST<->WEST tag pairing is what lets every process
        // post matching sends/recvs with no special-casing, including when the
        // "neighbour" is MPI_PROC_NULL (tag is then irrelevant, call is a no-op)

        t0 = MPI_Wtime();
        update_plane_inside(periodic, N, &planes[current], &planes[!current]);
        t_comp += MPI_Wtime() - t0;
 
        // this is the *exposed* communication cost: whatever wasn't already
        // hidden behind update_plane_inside above. If overlap is working well,
        // this should be small (ideally close to zero once the interior work
        // is large enough to fully hide the halo transfer time)
        t0 = MPI_Wtime();
        MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
        t_comm += MPI_Wtime() - t0;
        // We wait for completion of all 8 requests right here
        // MPI_STATUSES_IGNORE: don't need the MPI_Status info (source rank, tag, error code) for any of these
        // requests, since we already know exactly what we asked for
        
        // [C] copy the haloes data
        // once received, copy/place the incoming halo data into your plane's border cells

        // NORTH/SOUTH: already written directly into the halo rows by MPI_Irecv,
        // thanks to pointing buffers[RECV][NORTH/SOUTH] at IDX(1,0) / IDX(1,ysize+1) above
        // for (uint i = 1; i <= xsize; i++)
        // {
        //     data[IDX(i, 0)] = buffers[RECV][NORTH][i-1];
        //     data[IDX(i, ysize + 1)] = buffers[RECV][SOUTH][i-1];
        // }

        // EAST/WEST: the received data is sitting in the packed RECV buffers (contiguous, ysize doubles)
        // we must scatter it into the strided halo column of `data` by hand, mirroring the packing loop in [A]
        t0 = MPI_Wtime();
        for (uint j = 1; j <= ysize; j++)
        {
            data[IDX(xsize + 1, j)] = buffers[RECV][EAST][j-1]; // east halo column
            data[IDX(0, j)] = buffers[RECV][WEST][j-1]; // west halo column
        }
        t_comm += MPI_Wtime() - t0;
 
        #undef IDX

        /* -------------------------------------- */

        /* update grid points */
        t0 = MPI_Wtime();
        update_plane_border(periodic, N, &planes[current], &planes[!current]);
        t_comp += MPI_Wtime() - t0;

        /* output if needed */
        if ( output_energy_stat_perstep ) // optionally prints diagnostics every step
            output_energy_stat(iter, &planes[!current], (iter+1) * Nsources*energy_per_source, Rank, &myCOMM_WORLD, S);
        
        /* swap plane indexes for the new iteration */
        // swaps which buffer is "current" for the next iteration
        current = !current;
    }
    
    t1 = MPI_Wtime() - t1;

    {
        // report both the max (the critical path / what limits speedup) and
        // the average (what a naive sum/Ntasks would suggest) across ranks;
        // a wide gap between them signals load imbalance, not just overhead
        double t_comp_max, t_comp_sum, t_comm_max, t_comm_sum;
        // MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
        MPI_Reduce(&t_comp, &t_comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);
        MPI_Reduce(&t_comp, &t_comp_sum, 1, MPI_DOUBLE, MPI_SUM, 0, myCOMM_WORLD);
        MPI_Reduce(&t_comm, &t_comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);
        MPI_Reduce(&t_comm, &t_comm_sum, 1, MPI_DOUBLE, MPI_SUM, 0, myCOMM_WORLD);

        // t_comp_max and t_comm_max can come from *different* ranks, so their
        // sum can legitimately exceed any single rank's real elapsed time.
        // Reduce the per-rank TOTAL instead to get a residual that can't go
        // negative: this is guaranteed to come from one consistent rank.
        double t_total_local = t_comp + t_comm;
        double t_total_max;
        MPI_Reduce(&t_total_local, &t_total_max, 1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);

        if (Rank == 0)
        {
            printf("Elapsed time:    %f seconds\n", t1);
            printf("Computation time (max across ranks): %f seconds (avg %f)\n",
                t_comp_max, t_comp_sum / Ntasks);
            printf("Communication time (max across ranks): %f seconds (avg %f)\n",
                t_comm_max, t_comm_sum / Ntasks);
            printf("Unaccounted (I/O, RNG, imbalance, etc.): %f seconds\n",
                t1 - t_total_max);
        }
    }

    // final energy report
    output_energy_stat(-1, &planes[!current], Niterations * Nsources*energy_per_source, Rank, &myCOMM_WORLD, S);
    
    memory_release(buffers, planes);
    
    MPI_Finalize();
    return 0;
}

/* ==========================================================================
   =                                                                        =
   =   routines called within the integration loop                          =
   ========================================================================== */

/* ==========================================================================
   =                                                                        =
   =   initialization                                                       =
   ========================================================================== */

uint simple_factorization(uint, int *, uint **);

int initialize_sources(
    int,
    int,
    MPI_Comm *,
    uint [2], // just another way of writing vec2_t
    int,
    int *,
    vec2_t **
);

int memory_allocate(
    const int *,
    const vec2_t,
    buffers_t *,
    plane_t *
);

int initialize(
    MPI_Comm *Comm,
    int Me, // the rank of the calling process
    int Ntasks, // the total number of MPI ranks
    int argc, // the argc from command line
    char **argv, // the argv from command line
    vec2_t *S, // the size of the plane
    vec2_t *N, // two-uint array defining the MPI tasks' grid
    int *periodic, // periodic-boundary tag
    int *output_energy_stat,
    int *neighbours, // four-int array that gives back the neighbours of the calling task
    int *Niterations, // how many iterations
    int *Nsources, // how many heat sources
    int *Nsources_local,
    vec2_t **Sources_local,
    double *energy_per_source, // how much heat per source
    plane_t *planes,
    buffers_t *buffers
)
{
    int halt = 0;
    int ret;
    int verbose = 0;
    
    // ··································································
    // set default values

    (*S)[_x_] = 10000;
    (*S)[_y_] = 10000;
    *periodic = 0;
    *Nsources = 4;
    *Nsources_local = 0;
    *Sources_local = NULL;
    *Niterations = 1000;
    *energy_per_source = 1.0;

    if ( planes == NULL )
    {
        perror("Error: NULL pointer passed. Failed to access the plane.");
		exit(1);
    }

    planes[OLD].size[0] = planes[OLD].size[1] = 0;
    planes[NEW].size[0] = planes[NEW].size[1] = 0;
    
    // initializes all 4 neighbor slots to MPI_PROC_NULL
    for ( int i = 0; i < 4; i++ )
        neighbours[i] = MPI_PROC_NULL;

    // initializes both SEND/RECV buffer sets, all 4 directions, to NULL (not yet allocated)
    for ( int b = 0; b < 2; b++ )
        for ( int d = 0; d < 4; d++ )
            buffers[b][d] = NULL;
    
    // ··································································
    // process the command line
    // 
    while ( 1 )
    {
        int opt;
        while((opt = getopt(argc, argv, ":hx:y:e:E:n:o:p:v:")) != -1)
        {
            switch( opt )
            {
                case 'x': (*S)[_x_] = (uint)atoi(optarg);
                    break;

                case 'y': (*S)[_y_] = (uint)atoi(optarg);
                    break;

                case 'e': *Nsources = atoi(optarg);
                    break;

                case 'E': *energy_per_source = atof(optarg);
                    break;

                case 'n': *Niterations = atoi(optarg);
                    break;

                case 'o': *output_energy_stat = (atoi(optarg) > 0);
                    break;

                case 'p': *periodic = (atoi(optarg) > 0);
                    break;

                case 'v': verbose = atoi(optarg);
                    break;

                case 'h': {
                    if ( Me == 0 )
                        printf( "\nvalid options are ( values between [] are the default values ):\n"
                            "-x    x size of the plate [10000]\n"
                            "-y    y size of the plate [10000]\n"
                            "-e    how many energy sources on the plate [4]\n"
                            "-E    how much energy per source [1.0]\n"
                            "-n    how many iterations [1000]\n"
                            "-p    whether periodic boundaries applies [0 = false]\n\n"
                        );
                    halt = 1; }
                    break;
                    
                case ':': printf( "option -%c requires an argument\n", optopt);
                    break;
                    
                case '?': printf(" -------- help unavailable ----------\n");
                    break;
            }
        }

        if ( opt == -1 )
            break;
    }

    if ( halt )
        return 1;
    
    // ··································································
    /*
    * here we should check for all the parms being meaningful
    */

    // ...

    // ··································································
    /*
    * find a suitable domain decomposition
    * very simple algorithm, you may want to
    * substitute it with a better one
    *
    * the plane Sx x Sy will be solved with a grid
    * of Nx x Ny MPI tasks
    */

    vec2_t Grid;
    // the plate's aspect ratio, always ≥ 1 (longer side over shorter side)
    double formfactor = ((*S)[_x_] >= (*S)[_y_] ? (double)(*S)[_x_]/(*S)[_y_] : (double)(*S)[_y_]/(*S)[_x_] );
    // whether to use a 1D decomposition (split along one axis only) or 2D decomposition (split along both)
    // if the number of tasks is small relative to the plate's elongation (Ntasks <= formfactor+1), a 1D split is good enough 
    // (since the plate is already very elongated, splitting along the short axis wouldn't help much); otherwise use 2D
    int dimensions = 2 - (Ntasks <= ((int)formfactor+1) );

    // put all tasks along whichever axis is longer
    if ( dimensions == 1 )
    {
        if ( (*S)[_x_] >= (*S)[_y_] )
            Grid[_x_] = Ntasks, Grid[_y_] = 1;
        else
            Grid[_x_] = 1, Grid[_y_] = Ntasks;
    }
    // factorize Ntasks into its prime factors (e.g., 12 → 2,2,3), then greedily multiply factors together into first until 
    // (Ntasks/first)/first (the ratio between the two grid dimensions if split as first × Ntasks/first) drops <= the plate's aspect ratio
    // -> tries to make the process grid's aspect ratio mirror the plate's aspect ratio, so each process gets a roughly square-ish chunk relative to overall proportions
    // —> better for surface-to-volume communication efficiency -> assigns the larger grid dimension to whichever plate axis is longer
    else
    {
        int Nf;
        uint *factors;
        uint first = 1;
        ret = simple_factorization( Ntasks, &Nf, &factors );
        
        for ( int i = 0; (i < Nf) && ((Ntasks/first)/first > formfactor); i++ )
            first *= factors[i];

        if ( (*S)[_x_] > (*S)[_y_] )
            Grid[_x_] = Ntasks/first, Grid[_y_] = first;
        else
            Grid[_x_] = first, Grid[_y_] = Ntasks/first;
    }

    // stores the chosen process-grid shape into the output parameter N
    (*N)[_x_] = Grid[_x_];
    (*N)[_y_] = Grid[_y_];
    
    // ··································································
    // my coordinates in the grid of processors
    // finding this process's grid coordinates
    // treats the Ntasks ranks as laid out row-major in a Grid[_x_] × Grid[_y_] grid: rank Me's column is Me % Grid[_x_], 
    // row is Me / Grid[_x_] (integer division) -> with Grid[_x_]=4: rank 6 → X=2, Y=1
    int X = Me % Grid[_x_];
    int Y = Me / Grid[_x_];

    // ··································································
    // find my neighbours
    //

    if ( Grid[_x_] > 1 ) // more than one column
    // if there's just 1 column, there's no East/West neighbor at all
    {  
        if ( *periodic )
        // the grid wraps around — the rightmost column's East neighbor is the leftmost column of the same row, and vice versa
        {
            // Y*Grid[_x_] + (Me+1)%Grid[_x_] computes "same row, next column wrapping via modulo"
            neighbours[EAST]  = Y*Grid[_x_] + (Me + 1 ) % Grid[_x_];
            // (Y+1)*Grid[_x_]-1 is "same row, last column" —> used when X==0 (leftmost), wrapping to the rightmost column of that row
            neighbours[WEST]  = (X%Grid[_x_] > 0 ? Me-1 : (Y+1)*Grid[_x_]-1);
        }
        else
        {
            // East neighbor is Me+1 if not at the rightmost column, otherwise no neighbor
            neighbours[EAST]  = ( X < Grid[_x_]-1 ? Me+1 : MPI_PROC_NULL );
            // West is Me-1 if not at the leftmost column, otherwise no neighbor
            neighbours[WEST]  = ( X > 0 ? (Me-1)%Ntasks : MPI_PROC_NULL );
        }  
    }

    if ( Grid[_y_] > 1 )
    {
        if ( *periodic )
        {      
            neighbours[NORTH] = (Ntasks + Me - Grid[_x_]) % Ntasks;
            neighbours[SOUTH] = (Ntasks + Me + Grid[_x_]) % Ntasks;
        }
        else
        {    
            neighbours[NORTH] = ( Y > 0 ? Me - Grid[_x_]: MPI_PROC_NULL );
            neighbours[SOUTH] = ( Y < Grid[_y_]-1 ? Me + Grid[_x_] : MPI_PROC_NULL );
        }
    }

    // ··································································
    // the size of my patch
    //

    /*
    * every MPI task determines the size sx x sy of its own domain
    * REMIND: the computational domain will be embedded into a frame (sx+2) x (sy+2)
    *         the outern frame will be used for halo communication or
    */
    
    // this distributes the plate size as evenly as possible when it doesn't divide evenly by the grid dimension
    // -> "distribute the remainder to the first few chunks" pattern
    vec2_t mysize;
    uint s = (*S)[_x_] / Grid[_x_];
    uint r = (*S)[_x_] % Grid[_x_];
    mysize[_x_] = s + (X < r);
    s = (*S)[_y_] / Grid[_y_];
    r = (*S)[_y_] % Grid[_y_];
    mysize[_y_] = s + (Y < r);

    // both buffers get this process's local patch dimensions 
    planes[OLD].size[0] = mysize[0];
    planes[OLD].size[1] = mysize[1];
    planes[NEW].size[0] = mysize[0];
    planes[NEW].size[1] = mysize[1];
    
    if ( verbose > 0 )
    {
        if ( Me == 0 )
        {
            printf("Tasks are decomposed in a grid %d x %d\n\n", Grid[_x_], Grid[_y_] );
            fflush(stdout);
        }

        MPI_Barrier(*Comm); // -> serialize output
        // without this, all processes printing simultaneously would produce garbled terminal output
        
        for ( int t = 0; t < Ntasks; t++ )
        {
            if ( t == Me )
            {
                printf("Task %4d :: "
                    "\tgrid coordinates : %3d, %3d\n"
                    "\tneighbours: N %4d    E %4d    S %4d    W %4d\n",
                    Me, X, Y,
                    neighbours[NORTH], neighbours[EAST],
                    neighbours[SOUTH], neighbours[WEST] );
                fflush(stdout);
            }

        MPI_Barrier(*Comm);
        }   
    }

    // ··································································
    // allocate the needed memory
    //
    ret = memory_allocate( neighbours, *N, buffers, planes ); // *N could also be Grid -> look better
    // ··································································
    // allocate the heat sources
    //
    ret = initialize_sources( Me, Ntasks, Comm, mysize, *Nsources, Nsources_local, Sources_local );
    
    return 0;  
}

uint simple_factorization(uint A, int *Nfactors, uint **factors) {
    // first pass: count how many prime factors A has (with repetition — e.g., 12 = 2×2×3 has 3 factors)
    // for each candidate divisor f starting at 2, divides it out of _A_ repeatedly while possible, counting each division.
    int N = 0;
    int f = 2;
    uint _A_ = A;

    while ( f < A )
    {
        while( _A_ % f == 0 )
        {
            N++;
            _A_ /= f;
        }
        f++;
    }

    *Nfactors = N;
    // second pass: now that we know how many factors there are, allocate exactly that much memory and redo the same loop, this time storing each factor
    // doing it in two passes avoids needing to guess an array size or use dynamic resizing —> simple but inefficient (factorizes twice)
    uint *_factors_ = (uint*)malloc( N * sizeof(uint) );

    N = 0;
    f = 2;
    _A_ = A;

    while ( f < A )
    {
        while( _A_ % f == 0 )
        {
            _factors_[N++] = f;
            _A_ /= f;
        }
        f++;
    }

    *factors = _factors_;
    return 0;
}

// randomly placing heat sources
int initialize_sources(
    int Me,
    int Ntasks,
    MPI_Comm *Comm,
    vec2_t mysize,
    int Nsources,
    int *Nsources_local,
    vec2_t **Sources
)
{
    // seeds the random number generator: time(NULL) ^ Me XORs the current time with this process's rank
    // -> every process gets a different seed even though they all call this at roughly the same wall-clock time — otherwise all processes would generate the identical "random" sequence
    srand48(time(NULL) ^ Me);
    int *tasks_with_sources = (int*)malloc( Nsources * sizeof(int) );
    
    // only rank 0 randomly decides, for each of the Nsources heat sources, which process it will live on
    if ( Me == 0 )
    {
        for ( int i = 0; i < Nsources; i++ )
            tasks_with_sources[i] = (int)lrand48() % Ntasks;
    }
    
    // MPI_Bcast broadcasts this array from rank 0 to every other process —> a collective communication ensuring everyone agrees on the same assignment
    MPI_Bcast( tasks_with_sources, Nsources, MPI_INT, 0, *Comm );

    // each process counts how many of the Nsources were assigned to itself
    int nlocal = 0;
    for ( int i = 0; i < Nsources; i++ )
        nlocal += (tasks_with_sources[i] == Me); // tasks... == Me evaluates to 0 or 1, summed as an int —> counting matches
    *Nsources_local = nlocal;
    
    // if this process owns at least one source, allocate an array of (x,y) coordinates and randomly pick a position within this process's local patch 
    // (the 1 + ensures coordinates start at 1, not 0, matching the interior region — recall index 0 is the halo border)
    if ( nlocal > 0 )
    {
        vec2_t * restrict helper = (vec2_t*)malloc( nlocal * sizeof(vec2_t) );      
        for ( int s = 0; s < nlocal; s++ )
        {
            // each process independently calls lrand48() here without any further synchronization needed, since these positions only need to be locally meaningful, not globally agreed upon
            helper[s][_x_] = 1 + lrand48() % mysize[_x_];
            helper[s][_y_] = 1 + lrand48() % mysize[_y_];
        }

        *Sources = helper;
    }
    // frees the temporary broadcast array since it's no longer needed.
    free( tasks_with_sources );
    return 0;
}

int memory_allocate(const int *neighbours, const vec2_t N, buffers_t *buffers_ptr, plane_t *planes_ptr) {
    /*
    here you allocate the memory buffers that you need to
    (i)  hold the results of your computation
    (ii) communicate with your neighbours

    The memory layout that I propose to you is as follows:

    (i) --- calculations
    you need 2 memory regions: the "OLD" one that contains the
    results for the step (i-1)th, and the "NEW" one that will contain
    the updated results from the step ith.

    Then, the "NEW" will be treated as "OLD" and viceversa.

    These two memory regions are indexed by *plate_ptr:

    planew_ptr[0] ==> the "OLD" region
    plamew_ptr[1] ==> the "NEW" region

    (ii) --- communications

    you may need two buffers (one for sending and one for receiving)
    for each one of your neighnours, that are at most 4: north, south, east amd west
    -> up to 4 pairs of SEND/RECV buffers, one pair per neighbor direction, each large enough to hold one edge's worth of data (mysizex or mysizey doubles)

    To them you need to communicate at most mysizex or mysizey double data.

    These buffers are indexed by the buffer_ptr pointer so that
    (*buffers_ptr)[SEND][ {NORTH,...,WEST} ] = .. some memory regions
    (*buffers_ptr)[RECV][ {NORTH,...,WEST} ] = .. some memory regions
    
    north/south buffers might not even be needed as separate allocations, since a row of the grid is already stored contiguously in memory (because of the row-major flattening)
    —> you could just point directly at the right offset in the existing array rather than copying. East/west columns, by contrast, are not contiguous (they're strided), so those genuinely need a packed buffer
    */

    if (planes_ptr == NULL )
    {
        perror("Error: NULL pointer passed. Failed to access the plane.");
        exit(1);
    }

    if (buffers_ptr == NULL )
    {
        perror("Error: NULL pointer passed. Failed to access the buffer.");
        exit(1);
    }
        
    // ··················································
    // allocate memory for data
    // we allocate the space needed for the plane plus a contour frame
    // that will contains data form neighbouring MPI tasks
    // frame_size = (x+2)*(y+2) — the local patch plus its 1-cell halo border on every side: both OLD and NEW buffers get allocated at this size and zeroed via memset 
    // -> initial temperature is 0 everywhere, including halos, before any heat is injected
    unsigned int frame_size = (planes_ptr[OLD].size[_x_]+2) * (planes_ptr[OLD].size[_y_]+2);

    // 64-byte aligned allocation for cache-line-aligned SIMD loads/stores
    int ret_old = posix_memalign((void**)&planes_ptr[OLD].data, 64, frame_size * sizeof(double));
    if ( ret_old != 0 )
    {
        fprintf(stderr, "Error: posix_memalign failed for OLD plane: %s\n", strerror(ret_old));
        exit(1);
    }

    int ret_new = posix_memalign((void**)&planes_ptr[NEW].data, 64, frame_size * sizeof(double));
    if ( ret_new != 0 )
    {
        fprintf(stderr, "Error: posix_memalign failed for NEW plane: %s\n", strerror(ret_old));
        exit(1);
    }

    // NUMA-aware first-touch with tiled pattern matching update_plane_interior's collapse(2) tile distribution
    // Each thread touches the same tiles it will compute, causing Linux to allocate pages on that thread's local NUMA node.
	// {
    //     uint fxsize = planes_ptr[OLD].size[_x_] + 2;
    //     uint fysize  = planes_ptr[OLD].size[_y_] + 2;

    //     #define TILE 64
    //     // #pragma omp parallel for collapse(2) schedule(static)
    //     for (uint jj = 0; jj < ysize; jj += TILE)
    //         for (uint ii = 0; ii < fxsize; ii += TILE)
    //         {
    //             uint j_end = (jj + TILE < ysize) ? jj + TILE : ysize;
    //             uint i_end = (ii + TILE < fxsize) ? ii + TILE : fxsize;
    //             for (uint j = jj; j < j_end; j++)
    //                 for (uint i = ii; i < i_end; i++)
    //                 {
    //                     planes_ptr[OLD].data[j * fxsize + i] = 0.0;
    //                     planes_ptr[NEW].data[j * fxsize + i] = 0.0;
    //                 }
    //         }
    //     #undef TILE
	// }
    {
        uint fxsize = planes_ptr[OLD].size[_x_] + 2;
        uint fysize  = planes_ptr[OLD].size[_y_] + 2;
        #pragma omp parallel for collapse(2) schedule(static)
        for (uint j = 0; j < fysize; j++)
            for (uint i = 0; i < fxsize; i++)
            {
                planes_ptr[OLD].data[j * fxsize + i] = 0.0;
                planes_ptr[NEW].data[j * fxsize + i] = 0.0;
            }
    }

    // ··················································
    // buffers for north and south communication are not really needed
    // in fact, they are already contiguous, just the first and last line of every rank's plane
    //
    // you may just make some pointers pointing to the correct positions
    //
    // or, if you prefer, just go on and allocate buffers also for north and south communications
    // ··················································
    // allocate buffers
    //
    // ··················································
    // we allocate a packed SEND and RECV buffer for every direction
    // that actually has a neighbour. NORTH/SOUTH buffers hold one
    // row's worth of data (xsize doubles), EAST/WEST buffers hold one
    // column's worth of data (ysize doubles), since with a 5-points
    // stencil only the edge (not the corners) needs to be exchanged.
    //
    // NOTE: north/south rows are already contiguous in `data`, so in
    // principle you could skip allocating/copying for those two
    // directions and just point the buffer pointers at the right
    // offset inside planes_ptr[...].data instead (see the comment
    // above)
    // ··················································
 
    const uint xsize = planes_ptr[OLD].size[_x_];
    const uint ysize = planes_ptr[OLD].size[_y_];
 
    uint buffer_size[4];
    buffer_size[NORTH] = buffer_size[SOUTH] = xsize;
    buffer_size[EAST] = buffer_size[WEST] = ysize;
 
    for (int b = 0; b < 2; b++) // SEND, RECV
    {
        for (int d = 0; d < 4; d++) // NORTH, SOUTH, EAST, WEST
        {
            if (d == NORTH || d == SOUTH)
            {
                buffers_ptr[b][d] = NULL; // aliased into plane data by main(), never malloc'd
                continue;
            }

            buffers_ptr[b][d] = (double*)malloc(buffer_size[d] * sizeof(double));
            if (buffers_ptr[b][d] == NULL)
            {
                // manage the malloc fail
                perror("Failed to allocate the receiving buffer.");
                exit(1);
            }
            memset(buffers_ptr[b][d], 0, buffer_size[d] * sizeof(double));
        }
    }

    return 0;
}

int memory_release(buffers_t *buffers, plane_t *planes) {
    if ( buffers != NULL )
    {
        for ( int b = 0; b < 2; b++ ) // SEND, RECV
        {
            for ( int d = 0; d < 4; d++ ) // NORTH, SOUTH, EAST, WEST
            {
                if ( d == NORTH || d == SOUTH )
                    continue;

                if ( buffers[b][d] != NULL )
                    free (buffers[b][d]);
            }
        }
    }

    if ( planes != NULL ) // free(NULL) is actually safe/a no-op in C anyway
    {
        if ( planes[OLD].data != NULL )
            free (planes[OLD].data);
        
        if ( planes[NEW].data != NULL )
            free (planes[NEW].data);
    }

    return 0;
}

int output_energy_stat (int step, plane_t *plane, double budget, int Me, MPI_Comm *Comm, vec2_t S) {
    double system_energy = 0;
    double tot_system_energy = 0;
    // each process computes its own local total energy via get_total_energy
    get_total_energy ( plane, &system_energy );
    
    // MPI_Reduce then combines values from all processes using MPI_SUM, with the result landing only on rank 0 (the 0 argument is the destination rank)
    MPI_Reduce ( &system_energy, &tot_system_energy, 1, MPI_DOUBLE, MPI_SUM, 0, *Comm );
    
    // only rank 0 prints
    if ( Me == 0 )
    {
        if ( step >= 0 )
            printf(" [ step %4d ] ", step ); 
            
        fflush(stdout);

        printf( "total injected energy is %g, "
            "system energy is %g "
            "( in avg %g per grid point)\n",
            budget,
            tot_system_energy,
            // tot_system_energy / (plane->size[_x_]*plane->size[_y_]) );
            tot_system_energy / (S[_x_]*S[_y_]) );
    }
    
    return 0;
}
