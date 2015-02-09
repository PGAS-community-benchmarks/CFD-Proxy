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
#include "eval.h"
#include "threads.h"

void eval_thread_comm(comm_data *cd)
{
  ASSERT(cd != NULL);

#ifdef DEBUG
#pragma omp parallel default (none) shared(cd\
            , stdout, stderr)
    {
      int const tid = omp_get_thread_num();
      RangeList *color;

      int ncolors = 0;
      int scolors = 0;
      for (color = get_color(); color != NULL
	     ; color = get_next_color(color)) 
	{
	  ncolors++;
	  if (color->nsendcount > 0)
	    {
	      scolors++;
	    }
	}
      printf("iProc: %6d tid: %4d ncolors: %d scolors: %d\n"
	     ,cd->iProc,tid,ncolors,scolors);

      int nsend = 0;
      for (color = get_color(); color != NULL
	     ; color = get_next_color(color)) 
	{      
	  int i;
	  for(i = 0; i < color->nsendcount; i++)
	    {
	      int i1 = color->sendpartner[i];
	      int sendcount_color = color->sendcount[i];
	      int sendcount_local = get_sendcount_local(i1);
	      if (sendcount_color > 0 && 
		  sendcount_local > 0)
		{		  
		  int inc_local = inc_send_counter_local(i1, sendcount_color);
		  if(inc_local % sendcount_local == 0)
		    {
		      int inc_global = inc_send_counter_global(i1, sendcount_local);
		      int k = cd->commpartner[i1];
		      if (inc_global % cd->sendcount[k] == 0)
			{
			  printf("iProc: %6d tid: %4d send num: %8d to: %6d -- complete. final send color: %6d\n"
				 ,cd->iProc,tid,cd->sendcount[k],k,nsend);
			}
		    }
		}
	    }
	  nsend++;
	}

    }
#endif
}


void eval_thread_rangelist(solver_data *sd)
{
  int i0;

  /* sanity check */
  int *tmp1 = check_malloc(sd->nallpoints * sizeof(int));
  int *tmp2 = check_malloc(sd->nallpoints * sizeof(int));

  /* validate all_points_of_color */
  for(i0 = 0; i0 < sd->nallpoints; i0++)   
    {
      tmp1[i0] = -1;
    }
#pragma omp parallel default (none) shared(sd\
            , tmp1, stdout, stderr)
    {
      RangeList *color;
      for (color = get_color(); color != NULL
	     ; color = get_next_color(color)) 
	{      
	  int  nall_points_of_color = color->nall_points_of_color;
	  int  *all_points_of_color = color->all_points_of_color;
	  int i1;

	  for(i1 = 0; i1 < nall_points_of_color; i1++) 
	    {
	      int pnt = all_points_of_color[i1];
	      ASSERT(tmp1[pnt] == -1);
	      tmp1[pnt] += 1;
	    }
	}
    }
  for(i0 = 0; i0 < sd->nallpoints; i0++) 
    {      
      ASSERT(tmp1[i0] == 0);
    }


  /* validate first_points_of_color, ftype and tid*/
  for(i0 = 0; i0 < sd->nallpoints; i0++) 
    {      
      tmp1[i0] = -1;
      tmp2[i0] = -1;
    }
#pragma omp parallel default (none) shared(sd\
            , tmp1, tmp2, stdout, stderr)
    {
      RangeList *color;
      int const tid = omp_get_thread_num();
      solver_data_local* solver_local = get_solver_local();
      int    (*fpoint)[2]        = solver_local->fpoint;
      for (color = get_color(); color != NULL
	     ; color = get_next_color(color)) 
	{      
	  int  nfirst_points_of_color = color->nfirst_points_of_color;
	  int  *first_points_of_color = color->first_points_of_color;
	  int i1, face;

	  for(i1 = 0; i1 < nfirst_points_of_color; i1++) 
	    {
	      int pnt = first_points_of_color[i1];
	      ASSERT(tmp1[pnt] == -1);
	      tmp1[pnt] = tid;
	    }
	  
	  /* validate tid */
	  ASSERT(color->tid == tid);

	  /* validate ftype */
	  ASSERT(color->ftype != 0);
	  for(face = color->start; face < color->stop; face++)
	    {
	      const int  p0    = fpoint[face][0];
	      const int  p1    = fpoint[face][1];

	      if (color->ftype == 2 || 
		  color->ftype == 3)
		{
		  if (tmp2[p0] == -1)
		    {
		      tmp2[p0] = tid;
		    }
		  else 
		    {
		      ASSERT(tmp2[p0] == tid);
		    }
		}
	      if (color->ftype == 1 || 
		  color->ftype == 3)
		{
		  if (tmp2[p1] == -1)
		    {
		      tmp2[p1] = tid;
		    }
		  else
		    {
		      ASSERT(tmp2[p1] == tid);
		    }
		}
	    }
	}
    }
  for(i0 = 0; i0 < sd->nownpoints; i0++)   
    {
      ASSERT(tmp1[i0] != -1);
      ASSERT(tmp2[i0] != -1);
    }
  for(i0 = sd->nownpoints; i0 < sd->nallpoints; i0++)   
    {
      ASSERT(tmp1[i0] == -1);
      ASSERT(tmp2[i0] == -1);
    }

  /* validate last_points_of_color */
  for(i0 = 0; i0 < sd->nallpoints; i0++)   
    {
      tmp1[i0] = -1;
    }
#pragma omp parallel default (none) shared(sd\
            , tmp1, stdout, stderr)
    {
      RangeList *color;
      for (color = get_color(); color != NULL
	     ; color = get_next_color(color)) 
	{      
	  int  nlast_points_of_color = color->nlast_points_of_color;
	  int  *last_points_of_color = color->last_points_of_color;
	  int i1;
	  for(i1 = 0; i1 < nlast_points_of_color; i1++) 
	    {
	      int pnt = last_points_of_color[i1];
	      ASSERT(tmp1[pnt] == -1);
	      tmp1[pnt] += 1;
	    }
	}
    }
  for(i0 = 0; i0 < sd->nownpoints; i0++) 
    {      
      ASSERT(tmp1[i0] == 0);
    }
  for(i0 = sd->nownpoints; i0 < sd->nallpoints; i0++)   
    {
      ASSERT(tmp1[i0] == -1);
    }

  check_free(tmp1);
  check_free(tmp2);
}   
