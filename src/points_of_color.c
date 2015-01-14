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
#include "threads.h"

void set_all_points_of_color(solver_data *sd
			     , int tid
			     , int *pid
			     )
{
  int i, face;
  solver_data_local* solver_local = get_solver_local();
  int    (*fpoint)[2]  = solver_local->fpoint;
  RangeList *color;

  ASSERT(sd->nallpoints > 0);
  ASSERT(fpoint != NULL);

  int *tmp1   = check_malloc(sd->nallpoints * sizeof(int));
  for(i = 0; i < sd->nallpoints; i++) 
    {
      tmp1[i] = -1;
    }
  /* all first points of color including outer halo */
  int nallpoints = 0;
  for (color = get_color(); color != NULL
	 ; color = get_next_color(color)) 
    {      
      int npoints = 0;
      for(face = color->start ; face < color->stop; face++)
	{
	  int p0 = fpoint[face][0];
	  int p1 = fpoint[face][1];
	  if (pid[p0] == tid && 
	      tmp1[p0] == -1)
	    {
	      tmp1[p0] = i;
	      npoints++;
	    }
	  if (pid[p1] == tid && 
	      tmp1[p1] == -1)
	    {
	      tmp1[p1] = i;
	      npoints++;
	    }
	}
      color->nall_points_of_color = npoints;
      nallpoints += npoints;
    }
  check_free(tmp1);

  if (nallpoints == 0)
    {
      return;
    }
  
  int *tmp2   = check_malloc(sd->nallpoints * sizeof(int));  
  for(i = 0; i < sd->nallpoints; i++) 
    {
      tmp2[i] = -1;
    }

  int *points = check_malloc(nallpoints * sizeof(int));
  for (color = get_color(); color != NULL
	 ; color = get_next_color(color)) 
    {      
      color->all_points_of_color = points;
      points += color->nall_points_of_color;
      int npoints = 0;
      for(face = color->start ; face < color->stop; face++)
	{
	  int p0 = fpoint[face][0];
	  int p1 = fpoint[face][1];
	  if (pid[p0] == tid && 
	      tmp2[p0] == -1)
	    {
	      tmp2[p0] = i;
	      color->all_points_of_color[npoints++] = p0;
	    }
	  if (pid[p1] == tid && 
	      tmp2[p1] == -1)
	    {
	      tmp2[p1] = i;
	      color->all_points_of_color[npoints++] = p1;
	    }
	}
    }
  check_free(tmp2);  

}


void set_last_points_of_color(solver_data *sd
			      , int tid
			      , int *pid
			      )
{
  int i, face;
  solver_data_local* solver_local = get_solver_local();
  int    (*fpoint)[2]        = solver_local->fpoint;
  RangeList *color;

  ASSERT(sd->nallpoints > 0);
  ASSERT(fpoint != NULL);

  int *tmp1   = check_malloc(sd->nallpoints * sizeof(int));
  for(i = 0; i < sd->nallpoints; i++) 
    {
      tmp1[i] = 0;
    }

  /* all finalized/last points of color */
  for (color = get_color(); color != NULL
	 ; color = get_next_color(color)) 
    {      
      for(face = color->start ; face < color->stop; face++)
        {
          int p0 = fpoint[face][0];
          int p1 = fpoint[face][1];
          if (pid[p0] == tid && 
	      p0 < sd->nownpoints)
            {
              tmp1[p0]++;
            }
          if (pid[p1] == tid && 
	      p1 < sd->nownpoints)
            {
              tmp1[p1]++;
            }
        }
    }

  int *tmp2   = check_malloc(sd->nallpoints * sizeof(int));
  for(i = 0; i < sd->nallpoints; i++) 
    {
      tmp2[i] = 0;
    }

  int nlastpoints = 0;
  int nfirstpoints = 0;
  for (color = get_color(); color != NULL
	 ; color = get_next_color(color)) 
    {      
      int npoints = 0;
      for(face = color->start ; face < color->stop; face++)
        {
          int p0 = fpoint[face][0];
          int p1 = fpoint[face][1];
          if (pid[p0] == tid && 
	      p0 < sd->nownpoints)
            {
              if (++tmp2[p0] == tmp1[p0])
                {
                  npoints++;
                }           
            }
          if (pid[p1] == tid && 
	      p1 < sd->nownpoints)
            {
              if (++tmp2[p1] == tmp1[p1])
                {
                  npoints++;
                }           
            }
        }
      color->nlast_points_of_color = npoints;
      nlastpoints += npoints;
      nfirstpoints += color->nfirst_points_of_color;
    }
  if (nlastpoints == 0)
    {
      return;
    }

  /* sanity check */
  ASSERT(nlastpoints == nfirstpoints);
  for(i = 0; i < sd->nallpoints; i++) 
    {
      tmp2[i] = 0;
    }

  int *points = check_malloc(nlastpoints * sizeof(int));
  for (color = get_color(); color != NULL
	 ; color = get_next_color(color)) 
    {      
      color->last_points_of_color = points;
      points += color->nlast_points_of_color;
      int npoints = 0;
      for(face = color->start ; face < color->stop; face++)
        {
          int p0 = fpoint[face][0];
          int p1 = fpoint[face][1];
          if (pid[p0] == tid && 
	      p0 < sd->nownpoints)
            {
              if (++tmp2[p0] == tmp1[p0])
                {
		  ASSERT(color->nlast_points_of_color != 0);
                  color->last_points_of_color[npoints++] = p0;
                }           
            }
          if (pid[p1] == tid && 
	      p1 < sd->nownpoints)
            {
              if (++tmp2[p1] == tmp1[p1])
                {
		  ASSERT(color->nlast_points_of_color != 0);
                  color->last_points_of_color[npoints++] = p1;
                }           
            }
        }
    }
  check_free(tmp2);  
  check_free(tmp1);  

}

void set_first_points_of_color(solver_data *sd)
{
  int k;
  RangeList *color;

  /* all first points of color excluding outer halo */
  int nfirstpoints = 0;
  for (color = get_color(); color != NULL
	 ; color = get_next_color(color)) 
    {      
      int  nall_points_of_color = color->nall_points_of_color;
      int  *all_points_of_color = color->all_points_of_color;
      int npoints = 0;
      for(k = 0; k < nall_points_of_color; k++) 
	{
	  int pnt = all_points_of_color[k];
	  if (pnt < sd->nownpoints)
	    {
	      npoints++;
	    }
	}
      color->nfirst_points_of_color = npoints;
      nfirstpoints += npoints;
    }

  if (nfirstpoints == 0)
    {
      return;
    }

  int *points = check_malloc(nfirstpoints * sizeof(int));
  for (color = get_color(); color != NULL
	 ; color = get_next_color(color)) 
    {      
      int  nall_points_of_color = color->nall_points_of_color;
      int  *all_points_of_color = color->all_points_of_color;
      color->first_points_of_color = points;
      points += color->nfirst_points_of_color;
      int npoints = 0;
      for(k = 0; k < nall_points_of_color; k++) 
	{
	  int pnt = all_points_of_color[k];
	  if (pnt < sd->nownpoints)
	    {
	      color->first_points_of_color[npoints++] = pnt;
	    }
	}
    }

}

