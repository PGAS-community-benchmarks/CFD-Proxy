#ifndef SOLVER_DATA_H
#define SOLVER_DATA_H

#ifdef USE_GASPI
#include <GASPI.h>
#endif
#include <omp.h>

#define NGRAD 7
#define NFLUX 3

typedef struct 
{
  int global  __attribute__((aligned(64)));
} counter_t;

typedef struct 
{
  omp_lock_t lock  __attribute__((aligned(64)));
} lock_t;

typedef struct
{
  int  (*fpoint)[2];
  double (*fnormal)[3];
} solver_data_local;

typedef struct RangeList_t
{
  // next slice - linked list
  struct RangeList_t *succ;

  // meta data
  int  start;
  int  stop;
  int  ftype; //  face type   
  
  // points of color 
  int  nall_points_of_color; // incl. addpoints
  int  *all_points_of_color;
  int  nfirst_points_of_color; // first touch  
  int  *first_points_of_color;
  int  nlast_points_of_color; // last touch
  int  *last_points_of_color;
  
   // comm vars, color local, send 
  int nsendcount;
  int *sendpartner;
  int *sendcount;
  int **sendindex;
  int **sendoffset;

  // comm vars, color local, recv
  int nrecvcount;
  int *recvpartner;
  int *recvcount;
  int **recvindex;
  int **recvoffset;

  // thread id
  int tid;

} RangeList;


typedef struct 
{
  int     nfaces;
  int     nallfaces;
  int     nownpoints;
  int     nallpoints;
  int     ncolors;
  int     (*fpoint)[2];
  double  (*fnormal)[3];
  double  *pvolume;
  double  (*var)[NGRAD];
  double  (*grad)[NGRAD][3];
  double  (*psd_flux)[NFLUX];
  RangeList *fcolor;
  int     niter;
} solver_data ;


void init_solver_data(solver_data *sd, int NITER);
void read_solver_data(int ncid, solver_data *sd);

#endif
