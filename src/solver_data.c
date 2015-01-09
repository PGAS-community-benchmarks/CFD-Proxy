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
#include <mpi.h>
#include <string.h>

#include "read_netcdf.h"
#include "solver_data.h"
#include "error_handling.h"
#include "rangelist.h"
#include "util.h"


static void init_var(double (*var)[NGRAD], int nallpoints)
{
  int i,j;
  for (i = 0; i< nallpoints; ++i)
    {
      for (j = 0; j < NGRAD; ++j)
	{
	  var[i][j] = 1.0;
	}
    }
}

static void init_grad(double (*grad)[NGRAD][3], int nallpoints)
{
  int i,j,k;
  for (i = 0; i< nallpoints; ++i)
    {
      for (j = 0; j < NGRAD; ++j)
	{
	  for (k = 0; k < 3; ++k)
	    {
	      grad[i][j][k] = 1.0;
	    }
	}
    }
}

static void init_flux(double (*psd_flux)[NFLUX], int nallpoints)
{
  int i,j;
  for (i = 0; i< nallpoints; ++i)
    {
      for (j = 0; j < NFLUX; ++j)
        {         
          psd_flux[i][j] = 1.0;
        }
    }
}

void init_solver_data(solver_data *sd, int NITER)
{
  ASSERT(sd != NULL);
  ASSERT(sd->nallpoints != 0);

  /* initialize var/grad */
  init_var(sd->var, sd->nallpoints);
  init_grad(sd->grad, sd->nallpoints);
  init_flux(sd->psd_flux, sd->nallpoints);

  /* set num iterations */
  sd->niter = NITER;
}


void read_solver_data(int ncid, solver_data *sd)
{
  ASSERT(sd != NULL);

  sd->nfaces = 0;
  sd->nallfaces = 0;
  sd->nownpoints = 0;
  sd->nallpoints = 0;
  sd->ncolors = 0;
  sd->fpoint = NULL;
  sd->fnormal = NULL;
  sd->pvolume = NULL;
  sd->var = NULL;
  sd->grad = NULL;
  sd->fcolor = NULL;
  sd->niter = 0;

  /* read val */
  sd->ncolors = get_nc_val(ncid,"ncolors");
  sd->nfaces = get_nc_val(ncid,"nfaces");
  sd->nownpoints = get_nc_val(ncid,"nownpoints");
  sd->nallpoints = get_nc_val(ncid,"nallpoints");

  /* sanity check*/
  ASSERT(sd->ncolors > 0);
  ASSERT(sd->nfaces > 0);
  ASSERT(sd->nownpoints > 0);
  ASSERT(sd->nallpoints > 0);
  ASSERT(NGRAD > 0);

  /* alloc */
  sd->fpoint = check_malloc(sd->nfaces * 2 * sizeof(int));
  sd->fnormal = check_malloc(sd->nfaces * 3 * sizeof(double));
  sd->pvolume = check_malloc(sd->nallpoints * sizeof(double));
  sd->var = check_malloc(sd->nallpoints * NGRAD * sizeof(double));
  sd->grad = check_malloc(sd->nallpoints * NGRAD * 3 * sizeof(double));
  sd->psd_flux = check_malloc(sd->nallpoints * NFLUX * sizeof(double));
  sd->fcolor = check_malloc(sd->ncolors * sizeof(RangeList));

  /* read data */
  get_nc_int(ncid,"fpoint",&(sd->fpoint[0][0]));
  get_nc_double(ncid,"fnormal",&(sd->fnormal[0][0]));
  get_nc_double(ncid,"pvolume",sd->pvolume);

  /* rangelist */
  int i;
  for(i = 0; i < sd->ncolors; i++)
    {
      RangeList *rl = &(sd->fcolor[i]); 
      init_rangelist(rl);
    }

  /* npoints of color */
  int *npoints = check_malloc(sd->ncolors * sizeof(int));
  get_nc_int(ncid,"fcolor_npoints",npoints);
  for(i = 0; i < sd->ncolors; i++)
    {
      RangeList *rl = &(sd->fcolor[i]); 
      rl->nall_points_of_color = npoints[i];
    }
  check_free(npoints);

  /* points of color */
  int *points = check_malloc(sd->nallpoints * sizeof(int));
  get_nc_int(ncid,"fcolor_points",points);
  for(i = 0; i < sd->ncolors; i++)
    {
      RangeList *rl = &(sd->fcolor[i]); 
      int nall_points_of_color = rl->nall_points_of_color;      
      if (nall_points_of_color > 0)
	{	  
	  rl->all_points_of_color = points;      
	  points += nall_points_of_color;
	}
      else
	{
	  rl->all_points_of_color = NULL;
	}      
    }

}







