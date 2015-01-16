/*
 * This file is part of a small exa2ct benchmark kernel
 * The kernel aims at a dataflow implementation for 
 * hybrid solvers which make use of unstructured meshes.
 *
 * Contact point for exa2ct: 
 *                 https://projects.imec.be/exa2ct
 *
 * Contact point for this kernel: 
 *                 christian.simmendinger@t-systems.com
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <stdbool.h>

#include "read_netcdf.h"
#include "solver_data.h"
#include "error_handling.h"
#include "util.h"
#include "rangelist.h"
#include "points_of_color.h"
#include "threads.h"

/* global stage counters for comp */
static volatile counter_t *comp_stage_global = NULL; 

// threadprivate rangelist data
static int ncolors_local;
#pragma omp threadprivate(ncolors_local)
static RangeList *color_local;
#pragma omp threadprivate(color_local)

// threadprivate neighbour threads
static int nngb_threads_local;
#pragma omp threadprivate(nngb_threads_local)
static int *ngb_threads_local;
#pragma omp threadprivate(ngb_threads_local)

// threadprivate solver data
static solver_data_local solver_local;
#pragma omp threadprivate(solver_local)


solver_data_local* get_solver_local(void)
{
  return &solver_local;
}

int get_ncolors_local(void)
{
  return ncolors_local;
}

void inc_stage_counter_global(int tid)
{
  my_fetch_and_add(&comp_stage_global[tid].global,1);      
}

int get_stage_counter_global(int tid)
{
  return comp_stage_global[tid].global;      
}

void init_rangelist(RangeList *fcolor)
{  
  // next slice - linked list
  fcolor->succ = NULL;

  // meta data
  fcolor->start = -1;
  fcolor->stop = -1;
  fcolor->ftype = -1; //  face type   

  // points of color
  fcolor->nall_points_of_color = 0;
  fcolor->all_points_of_color = NULL;
  fcolor->nfirst_points_of_color = 0;
  fcolor->first_points_of_color = NULL;
  fcolor->nlast_points_of_color = 0;
  fcolor->last_points_of_color = NULL;

  // comm vars - send
  fcolor->nsendcount = 0;
  fcolor->sendpartner = NULL;
  fcolor->sendcount = NULL;
  fcolor->sendindex = NULL;
  fcolor->sendoffset = NULL;

  // comm vars - recv
  fcolor->nrecvcount = 0;
  fcolor->recvpartner = NULL;
  fcolor->recvcount = NULL;
  fcolor->recvindex = NULL;
  fcolor->recvoffset = NULL;

  // thread id
  fcolor->tid = -1; 

}

void init_halo_type(int *htype
		    , comm_data *cd
		    , solver_data *sd
		    )
{
  int i,j;
  /* all own points */
  for(i = 0; i < sd->nownpoints; i++) 
    {
      htype[i] = 1;
    }
  /* set halo type for sendindex data (inner halo) */
  for(i = 0; i < cd->ncommdomains; i++)
    {
      int k = cd->commpartner[i];
      int count = cd->sendcount[k];      
      if(count > 0)
	{
	  for(j = 0; j < count; j++)
	    {
	      int pnt = cd->sendindex[k][j];
	      htype[pnt] = 2;
	    }
	}
    }
  /* addpoints (outer halo) */
  for(i = sd->nownpoints; i < sd->nallpoints; i++) 
    {
      htype[i] = 3;
    }
}


void init_meta_data(int *pid
		    , int NTHREADS
		    , solver_data *sd
		    )
{ 
  /* set thread range */
  int min_size = sd->nallpoints/NTHREADS;
  int i, j, k;

  for(i = 0; i < sd->nallpoints; i++) 
    {
      pid[i] = -1;
    }

  int count = 0;
  int ncolors = 0;
  int rl_start = 0;
  int rl_stop = 0;
  for(k = 0; k < NTHREADS; k++)
    {
      int npnt  = 0;
      for(i = rl_start; i < sd->ncolors; i++)
	{
	  RangeList *color = &(sd->fcolor[i]); 
	  npnt += color->nall_points_of_color;
	  if (npnt >= min_size && k < NTHREADS-1)
	    {
	      break;
	    }
	}
      rl_stop = MIN(i+1, sd->ncolors);      

      /* all halo colors */
      for(i = rl_start; i < rl_stop; i++)
	{
	  RangeList *color = &(sd->fcolor[i]); 
	  int  *all_points_of_color = color->all_points_of_color;
	  int  nall_points_of_color = color->nall_points_of_color;
	  for(j = 0; j < nall_points_of_color; j++) 
	    {
	      int pnt = all_points_of_color[j];
	      ASSERT(pid[pnt] == -1);
	      pid[pnt] = k;
	    }
	  count += nall_points_of_color;
	  ncolors++;
	}
      rl_start =  rl_stop;
    }

  /* re-assign cross edges ending in addpoints (outer halo) */
  int face;
  for (face = 0; face < sd->nfaces; face++)
    {
      int p0 = sd->fpoint[face][0];
      int p1 = sd->fpoint[face][1];
      if (p1 >= sd->nownpoints && pid[p0] != pid[p1])
	{
	  pid[p1] = pid[p0];
	}
      if (p0 >= sd->nownpoints && pid[p1] != pid[p0])
	{
	  pid[p0] = pid[p1];
	}
    }

  /* sanity check */
  ASSERT(count == sd->nallpoints);
  ASSERT(ncolors == sd->ncolors);
  for(i = 0; i < sd->nallpoints; i++) 
    {
      ASSERT(pid[i] != -1);
    }
}

void init_thread_neighbours(comm_data *cd
			    , solver_data *sd
			    , int tid
			    , int *pid
			    )
{
  const int nthreads = omp_get_num_threads();
  int ncolors = ncolors_local;
  int (*fpoint)[2] = solver_local.fpoint;
  int *tmp1 = check_malloc(nthreads * sizeof(int));
  int j, face;

  ASSERT(cd != NULL);
  ASSERT(sd != NULL);

  nngb_threads_local = 0;
  ngb_threads_local = NULL;

  if (ncolors == 0)
    {
      return;
    }

  for(j = 0; j < nthreads; j++)   
    {
      tmp1[j] = 0;
    }
  
  RangeList *color;
  for (color = get_color(); color != NULL
	 ; color = get_next_color(color)) 
    {      
      ASSERT(color->tid != -1);
      for(face = color->start; face < color->stop; face++)
        {
          const int p0 = fpoint[face][0];
          const int p1 = fpoint[face][1];
          const int t0 = pid[p0];
          const int t1 = pid[p1];
          if (t0 != color->tid)
            {
              if (tmp1[t0] == 0)
                {
                  tmp1[t0] = 1;
                }
              ASSERT(t1 == tid);
              
            }
          if (t1 != color->tid)
            {
              if (tmp1[t1] == 0)
                {
                  tmp1[t1] = 1;
                }
              ASSERT(t0 == tid);
            }
        }
    }
  int nngb = 0;
  for(j = 0; j < nthreads; j++)   
    {
      if (tmp1[j] == 1)
        {
          nngb++;
        }
    }

  ASSERT(nngb != 0);

  /* set thread local neighbours */
  nngb_threads_local = nngb;
  ngb_threads_local = check_malloc(nngb * sizeof(int));  

  nngb = 0;;
  for(j = 0; j < nthreads; j++)   
    {
      if (tmp1[j] == 1)
        {
          ngb_threads_local[nngb++] = tmp1[j];
        }
    }

  check_free(tmp1);

#ifdef DEBUG
  printf("iProc: %6d tid: %4d nngb: %8d\n"
         ,cd->iProc,tid,nngb);
  fflush(stdout);
#endif


}


void init_thread_rangelist(comm_data *cd
			   , solver_data *sd
			   , int tid
			   , int *pid
			   , int *htype
			   )
{
  ASSERT(cd != NULL);
  ASSERT(sd != NULL);

  int face;
  int nfaces = 0;

  for (face = 0; face < sd->nfaces; face++)
    {
      int p0 = sd->fpoint[face][0];
      int p1 = sd->fpoint[face][1];

      if ((pid[p0] == tid && htype[p0] < 3) || 
	  (pid[p1] == tid && htype[p1] < 3))
	{
	  nfaces++;
	}
    }

  ncolors_local = 0;
  color_local = NULL;  

  solver_local.fpoint = NULL;
  solver_local.fnormal = NULL;

  if (nfaces == 0)
    {
      return;
    }

  /* init thread private face data */
  int     *ttype       = check_malloc(nfaces * sizeof(int));
  int    (*fpoint)[2]  = check_malloc(2 * nfaces * sizeof(int));
  double (*fnormal)[3] = check_malloc(3 * nfaces * sizeof(double));

  int nf = 0;
  for (face = 0; face < sd->nfaces; face++)
    {
      int p0 = sd->fpoint[face][0];
      int p1 = sd->fpoint[face][1];
      if ((pid[p0] == tid && htype[p0] < 3) || 
	  (pid[p1] == tid && htype[p1] < 3))
	{
	  memcpy(&(fpoint[nf][0])
		 , &(sd->fpoint[face][0])
		 , 2 * sizeof(int)
		 );
	  memcpy(&(fnormal[nf][0])
		 , &(sd->fnormal[face][0])
		 , 3 * sizeof(double)
		 );

	  /* init type */
	  ttype[nf] = -1;

	  nf++;
	}
    }
  ASSERT(nf == nfaces);

  /* set tmp face type for sorting */
  for (face = 0; face < nfaces; face++)
    {
      int p0 = fpoint[face][0];
      int p1 = fpoint[face][1];

      if (htype[p1] == 2 || htype[p0] == 2)
	{
	  /* all inner halo faces, writing only p1 */      
	  if (pid[p0] != tid || htype[p0] == 3)
	    {
	      ttype[face] = 0;
	    }
	  /* all inner halo faces, writing only p0 */      
	  else if (pid[p1] != tid || htype[p1] == 3)
	    {
	      ttype[face] = 1;
	    }
	  /* all inner halo faces, writing p1/p0 */      
	  else 
	    {
	      ttype[face] = 2;
	    }
	}
      else
	{
	  /* all other faces, writing only p1 */      
	  if (pid[p0] != tid || htype[p0] == 3)
	    {
	      ttype[face] = 3;
	    }
	  /* all other faces, writing only p0 */      
	  else if (pid[p1] != tid || htype[p1] == 3)
	    {
	      ttype[face] = 4;
	    }
	  /* all other faces, writing p1/p0 */      
	  else 
	    {
	      ttype[face] = 5;
	    }
	}
    }
  for (face = 0; face < nfaces; face++)
    {
      ASSERT(ttype[face] != -1);
    }

  /* init permutation vector for sorting */
  int *pm = check_malloc(nfaces * sizeof(int));
  for (face = 0; face < nfaces; face++)
    {
      pm[face] = face;
    }

  /* sort faces for type and p1/p0 */
  sort_faces(pm, fpoint, ttype, nfaces);

  /* fix face permutation */
  int     *tt     = check_malloc(nfaces * sizeof(int));
  int    (*fp)[2] = check_malloc(2 * nfaces * sizeof(int));
  double (*fn)[3] = check_malloc(3 * nfaces * sizeof(double));
  for (face = 0; face < nfaces; face++)
    {
      int tf = pm[face];
      memcpy(&(tt[face])
             , &(ttype[tf])
             , sizeof(int)
             );
      memcpy(&(fp[face][0])
             , &(fpoint[tf][0])
             , 2 * sizeof(int)
             );
      memcpy(&(fn[face][0])
             , &(fnormal[tf][0])
             , 3 * sizeof(double)
             );
    }
  check_free(pm);
  check_free(ttype);
  check_free(fpoint);
  check_free(fnormal);

  // thread local solver (face) data
  solver_local.fpoint = fp;
  solver_local.fnormal = fn;
  ttype = tt;

#define MAX_FACES_IN_COLOR 96

  int count = 0;
  int ncolors = 0;
  for(face = 1 ; face < nfaces; face++)
    {
      if((++count) == MAX_FACES_IN_COLOR || 
	 ttype[face] != ttype[face-1])	
	{
	  count = 0;
	  ncolors++;
	} 
    }
  /* last color */
  ncolors++;
        
  /* alloc threadprivate rangelist */
  ncolors_local = ncolors;
  color_local = check_malloc(ncolors * sizeof(RangeList));  

  RangeList *tl;  
  int i0;
  for (i0 = 0; i0 < ncolors; i0++)
    {
      tl = &(color_local[i0]); 
      init_rangelist(tl);
    }

  /* set color range */
  count = 0;
  i0 = 0;
  int start = 0;  
  for(face = 1 ; face < nfaces; face++)
    {
      if((++count) == MAX_FACES_IN_COLOR || 
	 ttype[face] != ttype[face-1])	
	{
	  tl = &(color_local[i0]); 
	  tl->start  = start;
	  tl->stop   = face;

	  start = face;
	  count = 0;
	  i0++;
	} 
    }
  /* last color */
  tl = &(color_local[i0]); 
  tl->start  = start;
  tl->stop   = nfaces;  
  i0++;

  ASSERT(i0 == ncolors);

  /* set face type value, succ */
  int i;
  for(i = 0; i < ncolors; i++)
    {
      tl = &(color_local[i]); 

      /* tid */
      tl->tid = tid;

      int fstart = tl->start;

      /* ftype, writing only p1 */
      if(ttype[fstart] == 0 || 
	 ttype[fstart] == 3)
	{
	  tl->ftype = 1;
	}
      /* ftype, writing only p0 */
      else if (ttype[fstart] == 1 || 
	       ttype[fstart] == 4)
	{
	  tl->ftype = 2;
	}
      /* ftype, writing p0/p1 */
      else if (ttype[fstart] == 2 || 
	       ttype[fstart] == 5)
	{
	  tl->ftype = 3;
	}
      /* external, writing outer halo */
      else
	{
	  ASSERT(0);
	}

      /* succ */
      if (i < ncolors - 1 )
	{
	  tl->succ = &(color_local[i+1]);
	}
      else
	{
	  tl->succ = NULL;
	}
    } 
  check_free(ttype);

  /* points of color */
  set_all_points_of_color(sd, tid, pid);
  
  /* first points of color */
  set_first_points_of_color(sd);

  /* last points of color */
  set_last_points_of_color(sd, tid, pid);

}


void init_comp_stage_global(int nthreads)
{
  int i;
  comp_stage_global = check_malloc(nthreads * sizeof(counter_t));
  for(i = 0; i < nthreads; i++)
    {
      comp_stage_global[i].global = 0;
    }

}


void wait_for_local_neighbours(void)
{
  int i;
  int const tid = omp_get_thread_num();   
  
  /* test for neighbour thread stage */
  for (i = 0; i < nngb_threads_local; ++i)
    {
      volatile int global;
      int id = ngb_threads_local[i];
      while ((global = comp_stage_global[id].global) 
	     < comp_stage_global[tid].global)
        {
          _mm_pause();
        }
    }
}


void wait_for_all_neighbours(void)
{
  int i;
  int const tid = omp_get_thread_num();   
  int const nthreads = omp_get_thread_num();

  /* test for neighbour thread stage */
  for (i = 0; i < nthreads; ++i)
    {
      if (i != tid)
        {
	  volatile int global;
	  while ((global = comp_stage_global[i].global) 
		 < comp_stage_global[tid].global)
            {
              _mm_pause();
            }
        }
    }
}



RangeList* private_get_color(RangeList *const prev)
{
  RangeList *color = NULL;

  if(prev != NULL)
    {
      color = prev->succ;      
    }
  else
    {
      color = color_local;
    }

  return color;
}


RangeList* private_get_color_and_exchange(RangeList *const prev
					  , send_fn send
					  , exch_fn exch
					  , comm_data *cd
					  , double *data
					  , int dim2
					  , int final
					  )
{
  RangeList *color = NULL;
  if(prev != NULL)
    {
      color = prev->succ;      

      /* initate thread comm, pack/send */
      if (send != NULL)
	{
	  send(prev
	       , cd
	       , data
	       , dim2
	       );
      }
    }
  else
    {
      color = color_local;
    }

  if (color == NULL)
    {
      int const tid = omp_get_thread_num();         
      inc_stage_counter_global(tid);
      
      /* exchange_dbl recv/unpack */      
      if (exch != NULL)
	{
	  exch(cd
	       , data
	       , dim2
	       , final
	       );
	}

      /* wait for neighbour threads */      
      wait_for_local_neighbours();
      
    }

  return color;

}




