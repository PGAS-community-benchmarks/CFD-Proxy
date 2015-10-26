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
#include "comm_data.h"
#include "solver_data.h"
#include "rangelist.h"
#include "threads.h"
#include "exchange_data_mpi.h"
#include "exchange_data_mpidma.h"
#ifdef USE_GASPI
#include "exchange_data_gaspi.h"
#endif

static void private_compute_gradients_gg(RangeList *color, solver_data *sd)
{
  solver_data_local* solver_local = get_solver_local();
  int    (*fpoint)[2]        = solver_local->fpoint;
  double  (*fnormal)[3]      = solver_local->fnormal; 

  double (*var)[NGRAD]       = sd->var;
  double (*grad)[NGRAD][3]   = sd->grad;
  const double *pvolume      = sd->pvolume;
  int i, eq, pnt;

  /*----------------------------------------------------------------------------
  | loop over all faces in the current grid domain and calculating the gradients
  | for the primitive variables except for the turbulence variables 
  | these gradients are calculated with the Gauss divergence theorem
  | first loop over all colors - second loop over colored inner faces
  | strip mining is applied to first and last points of color
  ----------------------------------------------------------------------------*/

  int  nfirst_points_of_color  = color->nfirst_points_of_color;
  int  *first_points_of_color  = color->first_points_of_color;
  int  nlast_points_of_color = color->nlast_points_of_color;
  int  *last_points_of_color = color->last_points_of_color;

  const int       start = color->start;
  const int       stop  = color->stop;      
  const int       ftype = color->ftype;
  int face;

  for(i = 0; i < nfirst_points_of_color; i++) 
    {
      pnt = first_points_of_color[i];
      for(eq = 0; eq < NGRAD; eq++)
	{
	  grad[pnt][eq][0] = 0.0;
	  grad[pnt][eq][1] = 0.0;
	  grad[pnt][eq][2] = 0.0;
	}
    }

  if (ftype == 2)
    {
      for(face = start; face < stop; face++)
	{
	  const int  p0    = fpoint[face][0];
	  const int  p1    = fpoint[face][1];
	  const double anx = fnormal[face][0];
	  const double any = fnormal[face][1];
	  const double anz = fnormal[face][2];
	  
	  for(eq = 0; eq < NGRAD; eq++)
	    {
	      const double val = 0.5 * (var[p0][eq] + var[p1][eq]);
	      const double vx = anx * val, vy = any * val, vz = anz * val;	      

	      grad[p0][eq][0] += vx; 
	      grad[p0][eq][1] += vy;
	      grad[p0][eq][2] += vz;
	    }
	}
    }

  if (ftype == 1)
    {
      for(face = start; face < stop; face++)
	{
	  const int  p0    = fpoint[face][0];
	  const int  p1    = fpoint[face][1];
	  const double anx = fnormal[face][0];
	  const double any = fnormal[face][1];
	  const double anz = fnormal[face][2];
	  
	  for(eq = 0; eq < NGRAD; eq++)
	    {
	      const double val = 0.5 * (var[p0][eq] + var[p1][eq]);
	      const double vx = anx * val, vy = any * val, vz = anz * val;	      

	      grad[p1][eq][0] -= vx;
	      grad[p1][eq][1] -= vy;
	      grad[p1][eq][2] -= vz; 
	    }
	}
    }

  if (ftype == 3)
    {
      for(face = start; face < stop; face++)
	{
	  const int  p0    = fpoint[face][0];
	  const int  p1    = fpoint[face][1];
	  const double anx = fnormal[face][0];
	  const double any = fnormal[face][1];
	  const double anz = fnormal[face][2];
	  
	  for(eq = 0; eq < NGRAD; eq++)
	    {
	      const double val = 0.5 * (var[p0][eq] + var[p1][eq]);
	      const double vx = anx * val, vy = any * val, vz = anz * val;	      

	      grad[p0][eq][0] += vx; 
	      grad[p0][eq][1] += vy;
	      grad[p0][eq][2] += vz;

	      grad[p1][eq][0] -= vx;
	      grad[p1][eq][1] -= vy;
	      grad[p1][eq][2] -= vz; 
	    }
	}
    }

  for(i = 0; i < nlast_points_of_color; i++) 
    {
      pnt = last_points_of_color[i];
      const double tmp = 1 / pvolume[pnt];
      for(eq = 0; eq < NGRAD; eq++)
	{  
	  grad[pnt][eq][0] *= tmp;
	  grad[pnt][eq][1] *= tmp;
	  grad[pnt][eq][2] *= tmp;
	}
    }

}


void compute_gradients_gg_comm_free(comm_data *cd, solver_data *sd, int final)
{
  RangeList *color;  
  send_fn send = NULL;
  exch_fn exch = NULL;
  double *data = NULL;
  int     dim2 = 0;
  for (color = get_color_and_exchange(send, exch, cd, data, dim2, final)
	 ; color != NULL
	 ; color = get_next_color_and_exchange(color, send, exch
					       , cd, data, dim2, final)) 
    {
      private_compute_gradients_gg(color, sd);
    }
}


void compute_gradients_gg_mpi_bulk_sync(comm_data *cd, solver_data *sd, int final)
{
  RangeList *color;  
#if defined(USE_PACK_IN_BULK_SYNC) || defined(USE_PARALLEL_GATHER)
  send_fn send = initiate_thread_comm_mpi_pack;
#else
  send_fn send = NULL;
#endif
  exch_fn exch = exchange_dbl_mpi_bulk_sync;
  double *data = &(sd->grad[0][0][0]);
  int     dim2 = NGRAD * 3;
  for (color = get_color_and_exchange(send, exch, cd, data, dim2, final)
	 ; color != NULL
	 ; color = get_next_color_and_exchange(color, send, exch
					       , cd, data, dim2, final)) 
    {
      private_compute_gradients_gg(color, sd);
    }
}


void compute_gradients_gg_mpi_early_recv(comm_data *cd, solver_data *sd, int final)
{
  RangeList *color;  
#if defined(USE_PACK_IN_BULK_SYNC) || defined(USE_PARALLEL_GATHER)
  send_fn send = initiate_thread_comm_mpi_pack;
#else
  send_fn send = NULL;
#endif
  exch_fn exch = exchange_dbl_mpi_early_recv;
  double *data = &(sd->grad[0][0][0]);
  int     dim2 = NGRAD * 3;
  for (color = get_color_and_exchange(send, exch, cd, data, dim2, final)
	 ; color != NULL
	 ; color = get_next_color_and_exchange(color, send, exch
					       , cd, data, dim2, final)) 
    {
      private_compute_gradients_gg(color, sd);
    }
}

void compute_gradients_gg_mpi_async(comm_data *cd, solver_data *sd, int final)
{
  RangeList *color;
  send_fn send = initiate_thread_comm_mpi_send;
  exch_fn exch = exchange_dbl_mpi_async;
  double *data = &(sd->grad[0][0][0]);
  int     dim2 = NGRAD * 3;
  for (color = get_color_and_exchange(send, exch, cd, data, dim2, final)
	 ; color != NULL
	 ; color = get_next_color_and_exchange(color, send, exch
					       , cd, data, dim2, final)) 
    {
      private_compute_gradients_gg(color, sd);
    }
}


#ifdef USE_GASPI
void compute_gradients_gg_gaspi_bulk_sync(comm_data *cd, solver_data *sd, int final)
{
  RangeList *color;  
#if defined(USE_PACK_IN_BULK_SYNC) || defined(USE_PARALLEL_GATHER)
  send_fn send = initiate_thread_comm_mpi_pack;
#else
  send_fn send = NULL;
#endif
  exch_fn exch = exchange_dbl_gaspi_bulk_sync;
  double *data = &(sd->grad[0][0][0]);
  int     dim2 = NGRAD * 3;
  for (color = get_color_and_exchange(send, exch, cd, data, dim2, final)
	 ; color != NULL
	 ; color = get_next_color_and_exchange(color, send, exch
					       , cd, data, dim2, final)) 
    {
      private_compute_gradients_gg(color, sd);
    }
}

void compute_gradients_gg_gaspi_async(comm_data *cd, solver_data *sd, int final)
{
  RangeList *color;  
  send_fn send = initiate_thread_comm_gaspi;
  exch_fn exch = exchange_dbl_gaspi_async;
  double *data = &(sd->grad[0][0][0]);
  int     dim2 = NGRAD * 3;
  for (color = get_color_and_exchange(send, exch, cd, data, dim2, final)
	 ; color != NULL
	 ; color = get_next_color_and_exchange(color, send, exch
					       , cd, data, dim2, final)) 
    {
      private_compute_gradients_gg(color, sd);
    }
}
#endif


void compute_gradients_gg_mpifence_bulk_sync(comm_data *cd, solver_data *sd, int final)
{
  RangeList *color;
#if defined(USE_PACK_IN_BULK_SYNC) || defined(USE_PARALLEL_GATHER)
  send_fn send = initiate_thread_comm_mpi_pack;
#else
  send_fn send = NULL;
#endif
  exch_fn exch = exchange_dbl_mpifence_bulk_sync;
  double *data = &(sd->grad[0][0][0]);
  int     dim2 = NGRAD * 3;
  for (color = get_color_and_exchange(send, exch, cd, data, dim2, final)
	 ; color != NULL
	 ; color = get_next_color_and_exchange(color, send, exch
					       , cd, data, dim2, final)) 
    {
      private_compute_gradients_gg(color, sd);
    }
}

void compute_gradients_gg_mpifence_async(comm_data *cd, solver_data *sd, int final)
{
  RangeList *color;  
  send_fn send = initiate_thread_comm_mpi_fence;
  exch_fn exch = exchange_dbl_mpifence_async;
  double *data = &(sd->grad[0][0][0]);
  int     dim2 = NGRAD * 3;
  for (color = get_color_and_exchange(send, exch, cd, data, dim2, final)
	 ; color != NULL
	 ; color = get_next_color_and_exchange(color, send, exch
					       , cd, data, dim2, final)) 
    {
      private_compute_gradients_gg(color, sd);
    }
}


void compute_gradients_gg_mpipscw_bulk_sync(comm_data *cd, solver_data *sd, int final)
{
  RangeList *color;
#if defined(USE_PACK_IN_BULK_SYNC) || defined(USE_PARALLEL_GATHER)
  send_fn send = initiate_thread_comm_mpi_pack;
#else
  send_fn send = NULL;
#endif
  exch_fn exch = exchange_dbl_mpipscw_bulk_sync;
  double *data = &(sd->grad[0][0][0]);
  int     dim2 = NGRAD * 3;
  for (color = get_color_and_exchange(send, exch, cd, data, dim2, final)
	 ; color != NULL
	 ; color = get_next_color_and_exchange(color, send, exch
					       , cd, data, dim2, final)) 
    {
      private_compute_gradients_gg(color, sd);
    }
}

void compute_gradients_gg_mpipscw_async(comm_data *cd, solver_data *sd, int final)
{
  RangeList *color;  
  send_fn send = initiate_thread_comm_mpi_pscw;
  exch_fn exch = exchange_dbl_mpipscw_async;
  double *data = &(sd->grad[0][0][0]);
  int     dim2 = NGRAD * 3;
  for (color = get_color_and_exchange(send, exch, cd, data, dim2, final)
	 ; color != NULL
	 ; color = get_next_color_and_exchange(color, send, exch
					       , cd, data, dim2, final)) 
    {
      private_compute_gradients_gg(color, sd);
    }
}




