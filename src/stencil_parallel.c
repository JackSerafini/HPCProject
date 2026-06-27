/*
mysizex   :   local x-extendion of your patch
mysizey   :   local y-extension of your patch
*/

#include "../include/stencil_parallel.h"

// ------------------------------------------------------------------
// ------------------------------------------------------------------

int main(int argc, char **argv)
{
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
    buffers_t buffers[2]; // SEND/RECV halo-communication buffers (one set per direction, per OLD/NEW... actually per SEND/RECV)
    
    int output_energy_stat_perstep;
    
    // initialize MPI envrionment
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
        MPI_Comm_dup (MPI_COMM_WORLD, &myCOMM_WORLD);
    }
    
    // argument checking and setting
    int ret = initialize(&myCOMM_WORLD, Rank, Ntasks, argc, argv, &S, &N, &periodic, &output_energy_stat_perstep,
                neighbours, &Niterations,
                &Nsources, &Nsources_local, &Sources_local, &energy_per_source,
                &planes[0], &buffers[0]);

    if ( ret )
    {
        printf("task %d is opting out with termination code %d\n", Rank, ret);
        
        MPI_Finalize();
        return 0;
    }
    
    int current = OLD;
    double t1 = MPI_Wtime(); /* take wall-clock time */
    
    for (int iter = 0; iter < Niterations; ++iter)
    {
        // declared but not yet used (another unfinished part):
        MPI_Request reqs[8]; // handle for up to 8 non-blocking MPI operations (e.g., 4 directions × send+receive)
        
        /* new energy from sources */
        // each iteration first injects fresh heat at the source points into the current plane
        inject_energy(periodic, Nsources_local, Sources_local, energy_per_source, &planes[current], N);

        // before updating the grid, each process needs its neighbors' edge data copied into its own halo cells
        /* -------------------------------------- */

        // [A] fill the buffers, and/or make the buffers' pointers pointing to the correct position
        // Pack the data you need to send into a buffer, or just point directly at it if it's already contiguous

        // [B] perfoem the halo communications
        //     (1) use Send / Recv
        //     (2) use Isend / Irecv
        //         --> can you overlap communication and compution in this way?
        // exchange this data over MPI either with blocking MPI_Send/MPI_Recv (simple but slow, can deadlock if not ordered carefully) 
        // or non-blocking MPI_Isend/MPI_Irecv (lets you overlap communication with other work, e.g., compute interior points while waiting for boundary data)
        
        // [C] copy the haloes data
        // once received, copy/place the incoming halo data into your plane's border cells

        /* --------------------------------------  */

        /* update grid points */
        update_plane(periodic, N, &planes[current], &planes[!current]);

        /* output if needed */
        if ( output_energy_stat_perstep ) // optionally prints diagnostics every step
            output_energy_stat(iter, &planes[!current], (iter+1) * Nsources*energy_per_source, Rank, &myCOMM_WORLD);
        
        /* swap plane indexes for the new iteration */
        // swaps which buffer is "current" for the next iteration
        current = !current;
    }
    
    t1 = MPI_Wtime() - t1;

    // final energy report
    output_energy_stat(-1, &planes[!current], Niterations * Nsources*energy_per_source, Rank, &myCOMM_WORLD);
    
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
    // set deffault values

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
        // manage the situation
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
    * REMIND: the computational domain will be embedded into a frame
    *         that is (sx+2) x (sy+2)
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
    // allocae the needed memory
    //
    ret = memory_allocate( neighbours, *N, buffers, planes ); // *N could also be Grid -> look better
    // ··································································
    // allocate the heat sources
    //
    ret = initialize_sources( Me, Ntasks, Comm, mysize, *Nsources, Nsources_local, Sources_local );
    
    return 0;  
}

uint simple_factorization( uint A, int *Nfactors, uint **factors )
/*
 * rought factorization;
 * assumes that A is small, of the order of <~ 10^5 max,
 * since it represents the number of tasks
*/
{
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

int memory_allocate(
    const int *neighbours,
    const vec2_t N,
    buffers_t *buffers_ptr,
    plane_t *planes_ptr
)
{
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
    for each one of your neighnours, that are at most 4:
    north, south, east amd west. -> up to 4 pairs of SEND/RECV buffers, one pair per neighbor direction, each large enough to hold one edge's worth of data (mysizex or mysizey doubles)

    To them you need to communicate at most mysizex or mysizey
    daouble data.

    These buffers are indexed by the buffer_ptr pointer so that

    (*buffers_ptr)[SEND][ {NORTH,...,WEST} ] = .. some memory regions
    (*buffers_ptr)[RECV][ {NORTH,...,WEST} ] = .. some memory regions
    
    --->> Of course you can change this layout as you prefer
    north/south buffers might not even be needed as separate allocations, since a row of the grid is already stored contiguously in memory (because of the row-major flattening)
    —> you could just point directly at the right offset in the existing array rather than copying. East/west columns, by contrast, are not contiguous (they're strided), so those genuinely need a packed buffer
    */

    if (planes_ptr == NULL )
        // an invalid pointer has been passed
        // manage the situation
        ;

    if (buffers_ptr == NULL )
        // an invalid pointer has been passed
        // manage the situation
        ;
        
    // ··················································
    // allocate memory for data
    // we allocate the space needed for the plane plus a contour frame
    // that will contains data form neighbouring MPI tasks
    // frame_size = (x+2)*(y+2) — the local patch plus its 1-cell halo border on every side: both OLD and NEW buffers get allocated at this size and zeroed via memset 
    // (so initial temperature is 0 everywhere, including halos, before any heat is injected)
    unsigned int frame_size = (planes_ptr[OLD].size[_x_]+2) * (planes_ptr[OLD].size[_y_]+2);

    planes_ptr[OLD].data = (double*)malloc( frame_size * sizeof(double) );
    if ( planes_ptr[OLD].data == NULL )
        // manage the malloc fail
        ;
    memset ( planes_ptr[OLD].data, 0, frame_size * sizeof(double) );

    planes_ptr[NEW].data = (double*)malloc( frame_size * sizeof(double) );
    if ( planes_ptr[NEW].data == NULL )
        // manage the malloc fail
        ;
    memset ( planes_ptr[NEW].data, 0, frame_size * sizeof(double) );

    // the actual halo-communication buffer allocation is not implemented 
    // ··················································
    // buffers for north and south communication 
    // are not really needed
    //
    // in fact, they are already contiguous, just the
    // first and last line of every rank's plane
    //
    // you may just make some pointers pointing to the
    // correct positions
    //
    // or, if you prefer, just go on and allocate buffers
    // also for north and south communications

    // ··················································
    // allocate buffers
    //
    // ··················································

    return 0;
}

int memory_release( // TODO: free buffer
    buffers_t *buffers, // need to add a buffers parameter here and free those too
    plane_t *planes
)
{
    if ( planes != NULL ) // free(NULL) is actually safe/a no-op in C anyway
    {
        if ( planes[OLD].data != NULL )
            free (planes[OLD].data);
        
        if ( planes[NEW].data != NULL )
            free (planes[NEW].data);
    }

    return 0;
}

int output_energy_stat ( int step, plane_t *plane, double budget, int Me, MPI_Comm *Comm )
{
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
            tot_system_energy / (plane->size[_x_]*plane->size[_y_]) ); // plane->size[_x_]*plane->size[_y_] is each process's local patch size
            // since plane passed in is one specific process's plane, and only rank 0 prints, this divides the global summed energy by rank 0's local grid-point count, not the global count,
            // unless Ntasks == 1 -> the correct denominator should probably be S[_x_]*S[_y_] (the global plate size), or this average is computed incorrectly whenever running with more than one process
    }
    
    return 0;
}
