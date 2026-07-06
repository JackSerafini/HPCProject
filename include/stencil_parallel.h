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

extern int update_plane_inside (
    const int,
    const vec2_t,
    const plane_t *,
    plane_t *
);

extern int update_plane_border (
    const int,
    const vec2_t,
    const plane_t *,
    plane_t *
);

extern int get_total_energy(
    plane_t *,
    double *
);

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
    MPI_Comm *,
    vec2_t
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
    double *restrict data = plane->data;
    
    #define IDX( i, j ) ( (j)*sizex + (i) ) // grid stored as a 1D array simulating 2D, defined locally

    // Hoist loop-invariant periodic checks outside the source loop
    const int periodic_x = periodic && (N[_x_] == 1);
    const int periodic_y = periodic && (N[_y_] == 1);

    for (int s = 0; s < Nsources; s++) // for each heat source, add energy to that grid cell
    {
        int x = Sources[s][_x_];
        int y = Sources[s][_y_];
        
        data[ IDX(x,y) ] += energy;

        if ( periodic_x )
        {
            if ( x == 1 )
                data[IDX(plane->size[_x_] + 1, y)] += energy;
            if ( x == plane->size[_x_] ) 
                data[IDX(0, y)] += energy;
        }
        if ( periodic_y )
        {
            if ( y == 1 )
                data[IDX(x, plane->size[_y_] + 1)] += energy;
            if ( y == plane->size[_y_] )
                data[IDX(x, 0)] += energy;
        }        
    }
    #undef IDX
    
    return 0;
}

#ifndef TILE_SIZE
#define TILE_SIZE 0
#endif

inline int update_plane_inside (
    const int periodic, 
    const vec2_t N, // the grid of MPI tasks
    const plane_t *oldplane,
    plane_t *newplane
)
{
    // "full" size including the +2 halo border
    uint register fxsize = oldplane->size[_x_]+2;
    // the actual interior size (no border) —> the loop bounds
    uint register xsize = oldplane->size[_x_];
    uint register ysize = oldplane->size[_y_];
    
    #define IDX( i, j ) ( (j)*fxsize + (i) ) // fxsize, not xsize, because the underlying array's row width is the full (haloed) width, even though we only loop over interior cells

    double *restrict old = oldplane->data;
    double *restrict new = newplane->data;

    const double alpha = 0.5;
    const double alpha_neighbour = 0.125; // (1/4 * 1/2)

#if TILE_SIZE > 0
    #pragma omp parallel for collapse(2) schedule(static)
    for (uint jj = 2; jj < ysize; jj += TILE_SIZE)
        for (uint ii = 2; ii < xsize; ii += TILE_SIZE)
        {
            uint j_end = (jj + TILE_SIZE < ysize) ? jj + TILE_SIZE : ysize;
            uint i_end = (ii + TILE_SIZE < xsize) ? ii + TILE_SIZE : xsize;
            for (uint j = jj; j < j_end; j++)
            {
                #pragma omp simd
                for (uint i = ii; i < i_end; i++)
                {
                    new[ IDX(i,j) ] = old[ IDX(i,j) ] * alpha + 
                        ( old[IDX(i-1, j)] + old[IDX(i+1, j)] + old[IDX(i, j-1)] + old[IDX(i, j+1)] ) * alpha_neighbour;
                }
            }
        }
#else
    #pragma omp parallel for collapse(2) schedule(static)
    for (uint j = 2; j < ysize; j++)
        for (uint i = 2; i < xsize; i++)
        {
            new[ IDX(i,j) ] = old[ IDX(i,j) ] * alpha + 
                ( old[IDX(i-1, j)] + old[IDX(i+1, j)] + old[IDX(i, j-1)] + old[IDX(i, j+1)] ) * alpha_neighbour;
        }
#endif
    #undef IDX
 
    return 0;
}

inline int update_plane_border(
    const int periodic, 
    const vec2_t N, // the grid of MPI tasks
    const plane_t *oldplane,
    plane_t *newplane
) {
    uint register fxsize = oldplane->size[_x_]+2;
    uint register xsize = oldplane->size[_x_];
    uint register ysize = oldplane->size[_y_];
    
    #define IDX( i, j ) ( (j)*fxsize + (i) )

    double *restrict old = oldplane->data;
    double *restrict new = newplane->data;

    const double alpha = 0.5;
    const double alpha_neighbour = 0.125; // (1/4 * 1/2)

    #pragma omp parallel
    {
        // Top row (j=1, all columns)
        #pragma omp for schedule(static)
        for (uint i = 1; i <= xsize; i++)
            new[ IDX(i,1) ] = old[ IDX(i,1) ] * alpha + 
                ( old[IDX(i-1, 1)] + old[IDX(i+1, 1)] + old[IDX(i, 0)] + old[IDX(i, 2)] ) * alpha_neighbour;

        // Bottom row (j=ysize, all columns)
        if (ysize > 1)
            #pragma omp for schedule(static)
            for (uint i = 1; i <= xsize; i++)
                new[ IDX(i,ysize) ] = old[ IDX(i,ysize) ] * alpha + 
                    ( old[IDX(i-1, ysize)] + old[IDX(i+1, ysize)] + old[IDX(i, ysize-1)] + old[IDX(i, ysize+1)] ) * alpha_neighbour;

        // Left column (i=1, skip corners already done by top/bottom rows)
        #pragma omp for schedule(static)
        for (uint j = 2; j < ysize; j++)
            new[ IDX(1,j) ] = old[ IDX(1,j) ] * alpha + 
                ( old[IDX(0, j)] + old[IDX(2, j)] + old[IDX(1, j-1)] + old[IDX(1, j+1)] ) * alpha_neighbour;

        // Right column (i=xsize, skip corners already done by top/bottom rows)
        if (xsize > 1)
            #pragma omp for schedule(static)
            for (uint j = 2; j < ysize; j++)
                new[ IDX(xsize,j) ] = old[ IDX(xsize,j) ] * alpha + 
                    ( old[IDX(xsize-1, j)] + old[IDX(xsize+1, j)] + old[IDX(xsize, j-1)] + old[IDX(xsize, j+1)] ) * alpha_neighbour;
    }

    if ( periodic )
    {
        if ( N[_x_] == 1 )
        {
            for ( int j = 1; j <= ysize; j++ )
            {
                new[ IDX(0, j) ] = new[ IDX(xsize, j) ];
                new[ IDX(xsize+1, j) ] = new[ IDX(1, j) ];
            }
        }

        if ( N[_y_] == 1 ) 
        {
            for ( int i = 1; i <= xsize; i++ )
            {
                new[ i ] = new[ IDX(i, ysize) ];
                new[ IDX(i, ysize + 1) ] = new[ IDX(i, 1) ];
            }
        }
    }
    #undef IDX
 
    return 0;
}

inline int get_total_energy(plane_t *plane, double *energy) {
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

    #pragma omp parallel for collapse(2) schedule(static) reduction(+:totenergy)
    for ( int j = 1; j <= ysize; j++ )
        for ( int i = 1; i <= xsize; i++ )
            totenergy += data[ IDX(i, j) ];

    #undef IDX

    *energy = (double)totenergy;
    return 0;
}
