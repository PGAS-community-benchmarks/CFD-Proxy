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

#include <stdio.h>
#include <stdlib.h>
#include "flux.h"
#include "comm_data.h"
#include "solver_data.h"
#include "rangelist.h"
#include "threads.h"
#include "exchange_data_mpi.h"
#include "exchange_data_mpidma.h"
#ifdef USE_GASPI
#include "exchange_data_gaspi.h"
#endif

#if 0

static void private_compute_psd_flux(RangeList *color, solver_data *sd)
{

  solver_data_local* solver_local = get_solver_local();
  int    (*fpoint)[2]         = solver_local->fpoint;
  double  (*fnormal)[3]       = solver_local->fnormal; 
  double (*psd_flux)[NFLUX]   = sd->psd_flux;
  double (*grad)[NGRAD][3]    = sd->grad;
  int  nfirst_points_of_color = color->nfirst_points_of_color;
  int  *first_points_of_color = color->first_points_of_color;

  const int       start = color->start;
  const int       stop  = color->stop;      
  const int       ftype = color->ftype;

  const double mue_eff = 1.0;
  int i, face;

  for(i = 0; i < nfirst_points_of_color; i++) 
    {
      int pnt = first_points_of_color[i];
      psd_flux[pnt][IVX] = 0.0;
      psd_flux[pnt][IVY] = 0.0;
      psd_flux[pnt][IVZ] = 0.0;      
    }

  for(face = start; face < stop; face++)
    {
      //      double flux[NFLUX];

      const int    p0 = fpoint[face][0];
      const int    p1 = fpoint[face][1];
      const double nx = fnormal[face][0];
      const double ny = fnormal[face][1];
      const double nz = fnormal[face][2];
      
      /*----------------------------------------------------------------------
      | Averaged spatial derivatives for pseudo fluxes
      ----------------------------------------------------------------------*/

      const double dvx_dx  = 0.5 * (grad[p0][IVX][0] + grad[p1][IVX][0]);
      const double dvx_dy  = 0.5 * (grad[p0][IVX][1] + grad[p1][IVX][1]);
      const double dvx_dz  = 0.5 * (grad[p0][IVX][2] + grad[p1][IVX][2]);

      const double dvy_dx  = 0.5 * (grad[p0][IVY][0] + grad[p1][IVY][0]);
      const double dvy_dy  = 0.5 * (grad[p0][IVY][1] + grad[p1][IVY][1]);
      const double dvy_dz  = 0.5 * (grad[p0][IVY][2] + grad[p1][IVY][2]);

      const double dvz_dx  = 0.5 * (grad[p0][IVZ][0] + grad[p1][IVZ][0]);
      const double dvz_dy  = 0.5 * (grad[p0][IVZ][1] + grad[p1][IVZ][1]);
      const double dvz_dz  = 0.5 * (grad[p0][IVZ][2] + grad[p1][IVZ][2]);

      const double lambda   = - 2.0/3.0 * mue_eff;

      const double sts_xx = lambda * (dvy_dy + dvz_dz - 2.0 * dvx_dx);
      const double sts_yy = lambda * (dvx_dx + dvz_dz - 2.0 * dvy_dy);
      const double sts_zz = lambda * (dvx_dx + dvy_dy - 2.0 * dvz_dz);

      const double sts_xy = mue_eff * (dvx_dy + dvy_dx);
      const double sts_xz = mue_eff * (dvx_dz + dvz_dx);
      const double sts_yz = mue_eff * (dvy_dz + dvz_dy);

      const double flux_IVX = -(sts_xx * nx + sts_xy * ny + sts_xz * nz);
      const double flux_IVY = -(sts_xy * nx + sts_yy * ny + sts_yz * nz);
      const double flux_IVZ = -(sts_xz * nx + sts_yz * ny + sts_zz * nz);

      if (ftype != 3)
	{         
	  psd_flux[p0][IVX] += flux_IVX;         
	  psd_flux[p0][IVY] += flux_IVY;
	  psd_flux[p0][IVZ] += flux_IVZ;
	}
      if (ftype != 2)
	{                 
	  psd_flux[p1][IVX] -= flux_IVX;
	  psd_flux[p1][IVZ] -= flux_IVZ;
	  psd_flux[p1][IVY] -= flux_IVY;
	}
    }
}

#else

static void private_compute_psd_flux(RangeList *color, solver_data *sd)
{

  solver_data_local* solver_local = get_solver_local();
  int    (*fpoint)[2]         = solver_local->fpoint;
  double  (*fnormal)[3]       = solver_local->fnormal; 
  double (*psd_flux)[NFLUX]   = sd->psd_flux;
  double (*grad)[NGRAD][3]    = sd->grad;
  int  nfirst_points_of_color = color->nfirst_points_of_color;
  int  *first_points_of_color = color->first_points_of_color;

  const int       start = color->start;
  const int       stop  = color->stop;      
  const int       ftype = color->ftype;

  const double mue_eff = 1.0;
  int i, face;

  for(i = 0; i < nfirst_points_of_color; i++) 
    {
      int pnt = first_points_of_color[i];
      psd_flux[pnt][IVX] = 0.0;
      psd_flux[pnt][IVY] = 0.0;
      psd_flux[pnt][IVZ] = 0.0;      
    }

  for(face = start; face < stop; face++)
    {
      //      double flux[NFLUX];

      const int    p0 = fpoint[face][0];
      const int    p1 = fpoint[face][1];
      const double nx = fnormal[face][0];
      const double ny = fnormal[face][1];
      const double nz = fnormal[face][2];
      
      /*----------------------------------------------------------------------
      | Averaged spatial derivatives for pseudo fluxes
      ----------------------------------------------------------------------*/

      const double dvx_dx  = 0.5 * (grad[p0][IVX][0] + grad[p1][IVX][0]);
      const double dvx_dy  = 0.5 * (grad[p0][IVX][1] + grad[p1][IVX][1]);
      const double dvx_dz  = 0.5 * (grad[p0][IVX][2] + grad[p1][IVX][2]);

      const double dvy_dx  = 0.5 * (grad[p0][IVY][0] + grad[p1][IVY][0]);
      const double dvy_dy  = 0.5 * (grad[p0][IVY][1] + grad[p1][IVY][1]);
      const double dvy_dz  = 0.5 * (grad[p0][IVY][2] + grad[p1][IVY][2]);

      const double dvz_dx  = 0.5 * (grad[p0][IVZ][0] + grad[p1][IVZ][0]);
      const double dvz_dy  = 0.5 * (grad[p0][IVZ][1] + grad[p1][IVZ][1]);
      const double dvz_dz  = 0.5 * (grad[p0][IVZ][2] + grad[p1][IVZ][2]);

      const double lambda   = - 2.0/3.0 * mue_eff;

      const double sts_xx = lambda * (dvy_dy + dvz_dz - 2.0 * dvx_dx);
      const double sts_yy = lambda * (dvx_dx + dvz_dz - 2.0 * dvy_dy);
      const double sts_zz = lambda * (dvx_dx + dvy_dy - 2.0 * dvz_dz);

      const double sts_xy = mue_eff * (dvx_dy + dvy_dx);
      const double sts_xz = mue_eff * (dvx_dz + dvz_dx);
      const double sts_yz = mue_eff * (dvy_dz + dvz_dy);

      const double flux_IVX = -(sts_xx * nx + sts_xy * ny + sts_xz * nz);
      const double flux_IVY = -(sts_xy * nx + sts_yy * ny + sts_yz * nz);
      const double flux_IVZ = -(sts_xz * nx + sts_yz * ny + sts_zz * nz);

      if (ftype != 3)
	{         
	  psd_flux[p0][IVX] += flux_IVX;         
	  psd_flux[p0][IVY] += flux_IVY;
	  psd_flux[p0][IVZ] += flux_IVZ;
	}
      if (ftype != 2)
	{                 
	  psd_flux[p1][IVX] -= flux_IVX;
	  psd_flux[p1][IVZ] -= flux_IVZ;
	  psd_flux[p1][IVY] -= flux_IVY;
	}
    }
}

#endif

void compute_psd_flux(solver_data *sd)
{
  RangeList *color;  
  for (color = get_color(); color != NULL; color = get_next_color(color)) 
    {
      private_compute_psd_flux(color, sd);
    }
}

