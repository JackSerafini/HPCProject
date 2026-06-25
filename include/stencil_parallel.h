/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
See COPYRIGHT in top-level directory.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <math.h>

#include <omp.h>
#include <mpi.h>

#define NORTH 0
#define SOUTH 1
#define EAST  2
#define WEST  3

#define SEND 0
#define RECV 1

#define OLD 0
#define NEW 1

#define _x_ 0
#define _y_ 1

typedef unsigned int uint;

typedef uint vec2_t[2]; // array of 2 unsigned ints -> used for anything 2D
typedef double *restrict buffers_t[4]; // array of 4 pointers to double, one per direction (N/S/E/W) -> used for communication buffers
// restrict is a compiler hint meaning "the compiler may assume this pointer doesn't alias/overlap other pointers"

typedef struct {
    double *restrict data; // the flattened 2D array of doubles holding the actual temperature/energy values for this process's local patch
    vec2_t size; // its (x, y) dimensions (size[_x_], size[_y_])
} plane_t;

extern int inject_energy (
    const int,
    const int,
    const vec2_t *,
    const double,
    plane_t *,
    const vec2_t
);

extern int update_plane (
    const int,
    const vec2_t,
    const plane_t *,
    plane_t *
);

extern int get_total_energy(
    plane_t *,
    double *
);

// The following functions are declared here, but defined in the .c file

int initialize (
    MPI_Comm *,
    int,
    int,
    int,
    char **,
    vec2_t *,
    vec2_t *,
    int *,
    int *,
    int *,
    int *,
    int *,
    int *,
    vec2_t **,
    double *,
    plane_t *,
    buffers_t *
);

int memory_release (
    buffers_t *,
    plane_t *
);

int output_energy_stat ( 
    int,
    plane_t *,
    double,
    int,
    MPI_Comm *
);

inline int inject_energy (
    const int periodic,
    const int Nsources,
    const vec2_t *Sources, // pointer to an array of vec2_t —> list of (x,y) coordinates of heat sources
    const double energy,
    plane_t *plane,
    const vec2_t N // process grid dimensions (Nx, Ny)
)
{
    const uint register sizex = plane->size[_x_]+2; // the patch is stored with a halo/ghost border of 1 extra cell on each side (for neighbor data)
    // -> the actual allocated width is size_x + 2
    double *restrict data = plane->data;
    
   #define IDX( i, j ) ( (j)*sizex + (i) ) // grid stored as a 1D array simulating 2D, defined locally
    for (int s = 0; s < Nsources; s++) // for each heat source, add energy to that grid cell
    {
        int x = Sources[s][_x_];
        int y = Sources[s][_y_];
        
        data[ IDX(x,y) ] += energy;
        
        if ( periodic )
        {
            if ( (N[_x_] == 1)  )
            {
                // propagate the boundaries if needed
                // check the serial version
            }
            
            if ( (N[_y_] == 1) )
            {
                // propagate the boundaries if needed
                // check the serial version
            }
        }                
    }
   #undef IDX
    
    return 0;
}

inline int update_plane (
    const int periodic, 
    const vec2_t N, // the grid of MPI tasks
    const plane_t *oldplane,
    plane_t *newplane
)
{
    // "full" size including the +2 halo border
    uint register fxsize = oldplane->size[_x_]+2;
    uint register fysize = oldplane->size[_y_]+2;
    // the actual interior size (no border) —> the loop bounds
    uint register xsize = oldplane->size[_x_];
    uint register ysize = oldplane->size[_y_];
    
   #define IDX( i, j ) ( (j)*fxsize + (i) ) // fxsize, not xsize, because the underlying array's row width is the full (haloed) width, even though we only loop over interior cells
    
    // HINT: you may attempt to
    //       (i)  manually unroll the loop
    //       (ii) ask the compiler to do it
    // for instance
    // #pragma GCC unroll 4
    //
    // HINT: in any case, this loop is a good candidate
    //       for openmp parallelization

    double *restrict old = oldplane->data;
    double *restrict new = newplane->data;
    
    // loop indices run from 1 to ysize/xsize inclusive, skipping index 0 and index size+1, which are the halo cells 
    // (border, filled with neighbor data or treated as zero/"heat sink" if there's no neighbor)
    for (uint j = 1; j <= ysize; j++)
        for ( uint i = 1; i <= xsize; i++)
        {
            // NOTE: (i-1,j), (i+1,j), (i,j-1) and (i,j+1) always exist even
            //       if this patch is at some border without periodic conditions;
            //       in that case it is assumed that the +-1 points are outside the
            //       plate and always have a value of 0, i.e. they are an
            //       "infinite sink" of heat
            
            // five-points stencil formula
            //
            // HINT : check the serial version for some optimization
            //
            new[ IDX(i,j) ] =
                old[ IDX(i,j) ] / 2.0 + ( old[IDX(i-1, j)] + old[IDX(i+1, j)] +
                                            old[IDX(i, j-1)] + old[IDX(i, j+1)] ) / 4.0 / 2.0;
            
        }

    if ( periodic )
    {
        if ( N[_x_] == 1 )
        {
            // propagate the boundaries as needed
            // check the serial version
        }

        if ( N[_y_] == 1 ) 
        {
            // propagate the boundaries as needed
            // check the serial version
        }
    }
   #undef IDX
 
    return 0;
}

inline int get_total_energy(
    plane_t *plane,
    double *energy
)
/*
NOTE: this routine a good candiadate for openmp parallelization
*/
{
    const int register xsize = plane->size[_x_];
    const int register ysize = plane->size[_y_];
    const int register fsize = xsize+2;

    double *restrict data = plane->data;
    
   #define IDX( i, j ) ( (j)*fsize + (i) )

   // compile-time conditional -> if compile with -DLONG_ACCURACY, it uses long double (extended precision) for the running sum
   // otherwise plain double -> summing millions of small values can accumulate floating-point rounding error
   #if defined(LONG_ACCURACY)    
    long double totenergy = 0;
   #else
    double totenergy = 0;    
   #endif

    // HINT: you may attempt to
    //       (i)  manually unroll the loop
    //       (ii) ask the compiler to do it
    // for instance
    // #pragma GCC unroll 4
    for ( int j = 1; j <= ysize; j++ )
        for ( int i = 1; i <= xsize; i++ )
            totenergy += data[ IDX(i, j) ];

   #undef IDX

    *energy = (double)totenergy;
    return 0;
}
