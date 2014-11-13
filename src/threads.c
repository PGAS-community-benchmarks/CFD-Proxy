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
#ifdef USE_GASPI
#include <GASPI.h>
#endif
#include <mpi.h>
#include <omp.h>
#include <stdbool.h>

#include "read_netcdf.h"
#include "solver_data.h"
#include "exchange_data_mpi.h"
#include "exchange_data_mpidma.h"
#ifdef USE_GASPI
#include "exchange_data_gaspi.h"
#endif
#include "error_handling.h"
#include "util.h"
#include "rangelist.h"
#include "threads.h"


/* comm var for threadprivate comm */
static volatile counter_t *inc_send = NULL; 
static int *inc_send_local = NULL;
#pragma omp threadprivate(inc_send_local)
static int *sendcount_local = NULL;
#pragma omp threadprivate(sendcount_local)

/* getter/setter functions for global increments */
int get_inc_send(int i)
{
  return inc_send[i].global;
}
int set_inc_send(int i, int val)
{
  return my_add_and_fetch(&inc_send[i].global, val);
}

/* getter/setter functions for threadlocal sendcount */
int get_sendcount_local(int i)
{
  return sendcount_local[i];
}
int set_inc_send_local(int i, int val)
{
  inc_send_local[i] += val;
  return inc_send_local[i];
}

int my_add_and_fetch(volatile int *ptr, int val)
{
#ifdef GCC_EXTENSION
  int t = __sync_add_and_fetch(ptr, val);
  ASSERT(t >= 0);
  return t;
#else
  int t;
  //#pragma omp atomic capture 
#pragma omp critical
  { 
    t = *ptr; 
    *ptr += val; 
  }
  t += val;
  ASSERT(t >= 0);
  return t;
#endif
}


int my_fetch_and_add(volatile int *ptr, int val)
{
#ifdef GCC_EXTENSION
  int t = __sync_fetch_and_add(ptr, val);
  ASSERT(t >= 0);
  return t;
#else
  int t;
  //#pragma omp atomic capture 
#pragma omp critical
  { 
    t = *ptr; 
    *ptr += val; 
  }
  ASSERT(t >= 0);
  return t;
#endif
}

int this_is_the_first_thread(void)
{
  static volatile int shared_counter = 0;
  static int local_next     = 0;
#pragma omp threadprivate(local_next)

  const int nthreads = omp_get_num_threads(),
            first    = local_next;

  if(nthreads == 1)
    return 1;
  
  while(shared_counter < local_next)
    _mm_pause();

  local_next += nthreads;

  return(my_fetch_and_add(&shared_counter,1) == first);
}

int this_is_the_last_thread(void)
{
  static volatile int shared_counter = 0;
  static int local_next     = 0;
#pragma omp threadprivate(local_next)

  int const nthreads = omp_get_num_threads();

  if(nthreads == 1)
    return 1;

  while(shared_counter < local_next)
    _mm_pause();

  local_next += nthreads;

  return (my_add_and_fetch(&shared_counter,1) == local_next);
}


void initiate_thread_comm_mpi(RangeList *color
			     , comm_data *cd
			     , double *data
			     , int dim2
			     )
{
  int i;
  for(i = 0; i < color->nsendcount; i++)
    {
      int i1 = color->sendpartner[i];
      int sendcount_color = color->sendcount[i];
      if (sendcount_color > 0 && sendcount_local[i1] > 0)
	{
	  inc_send_local[i1] += sendcount_color;
	  if(inc_send_local[i1] % sendcount_local[i1] == 0)
	    {
	      int inc_global = set_inc_send(i1, sendcount_local[i1]);
	      int k = cd->commpartner[i1];
	      if (inc_global % cd->sendcount[k] == 0)
		{
#ifndef USE_MPI_MULTI_THREADED
#pragma omp critical
#endif
		  {
		    exchange_dbl_mpi_send(cd
					  , data
					  , dim2
					  , i1
					  );
		  }
		}
	    }
	}
    }

}

void initiate_thread_comm_mpifence(RangeList *color
				   , comm_data *cd
				   , double *data
				   , int dim2
				   )
{
  int i;
  for(i = 0; i < color->nsendcount; i++)
    {
      int i1 = color->sendpartner[i];
      int sendcount_color = color->sendcount[i];
      if (sendcount_color > 0 && sendcount_local[i1] > 0)
	{
	  inc_send_local[i1] += sendcount_color;
	  if(inc_send_local[i1] % sendcount_local[i1] == 0)
	    {
	      int inc_global = set_inc_send(i1, sendcount_local[i1]);
	      int k = cd->commpartner[i1];
	      if (inc_global % cd->sendcount[k] == 0)
		{
#ifndef USE_MPI_MULTI_THREADED
#pragma omp critical
#endif
		  {
		    exchange_dbl_mpidma_write(cd, data, dim2, i1);
		  }
		}
	    }
	}
    }
}


void initiate_thread_comm_mpipscw(RangeList *color
				  , comm_data *cd
				  , double *data
				  , int dim2
				  )
{
  int i;
  static volatile int shared_nput = 0;
  for(i = 0; i < color->nsendcount; i++)
    {
      int i1 = color->sendpartner[i];
      int sendcount_color = color->sendcount[i];
      if (sendcount_color > 0 && sendcount_local[i1] > 0)
	{
	  inc_send_local[i1] += sendcount_color;
	  if(inc_send_local[i1] % sendcount_local[i1] == 0)
	    {
	      int inc_global = set_inc_send(i1, sendcount_local[i1]);
	      int k = cd->commpartner[i1];
	      if (inc_global % cd->sendcount[k] == 0)
		{
#ifndef USE_MPI_MULTI_THREADED
#pragma omp critical
#endif
		  {
		    exchange_dbl_mpidma_write(cd, data, dim2, i1);
		    if (my_add_and_fetch(&shared_nput, 1) % cd->ncommdomains == 0)
		      {
			mpidma_async_complete();
		      }
		  }
		}
	    }
	}
    }
}


#ifdef USE_GASPI
void initiate_thread_comm_gaspi(RangeList *color
			       , comm_data *cd
			       , double *data
			       , int dim2
			       )
{
  int i;
  for(i = 0; i < color->nsendcount; i++)
    {
      int i1 = color->sendpartner[i];
      int sendcount_color = color->sendcount[i];
      if (sendcount_color > 0 && sendcount_local[i1] > 0)
	{
	  int buffer_id = cd->send_stage % 2;
	  inc_send_local[i1] += sendcount_color;
	  if(inc_send_local[i1] % sendcount_local[i1] == 0)
	    {
	      int inc_global = set_inc_send(i1, sendcount_local[i1]);
	      int k = cd->commpartner[i1];
	      if (inc_global % cd->sendcount[k] == 0)
		{
		  exchange_dbl_gaspi_write(cd
					   , data
					   , dim2
					   , buffer_id
					   , i1
					   );
		}
	    }
	}
    }
}
#endif


static void allocate_thread_private_comm_data(comm_data *cd)
{
  int j ;
  /* thread private increments for send/recv counter */
#pragma omp parallel default (none) shared(cd)
  {
    int j1;
    inc_send_local = check_malloc(cd->ncommdomains * sizeof(int));
    for(j1 = 0; j1 < cd->ncommdomains; j1++)
      {	
	inc_send_local[j1] = 0;
      }
  }

  /* global increments for send/recv counter */
  inc_send = check_malloc(cd->ncommdomains * sizeof(counter_t));
  for(j = 0; j < cd->ncommdomains; j++)
    {
      inc_send[j].global = 0;
    }


    /* set num of thread private sendcounts */
#pragma omp parallel default (none) shared(cd, stdout, stderr)
    {
      int i1;
      sendcount_local = check_malloc(cd->ncommdomains * sizeof(int));
      for(i1 = 0; i1 < cd->ncommdomains; i1++)
	{
	  sendcount_local[i1] = 0;
	}

      RangeList *color;
      for (color = get_color(); color != NULL
	     ; color = get_next_color(color)) 
	{
	  int i2;
	  for(i2 = 0; i2 < color->nsendcount; i2++)
	    {
	      int i3 = color->sendpartner[i2];
	      sendcount_local[i3] += color->sendcount[i2];
	    }
	}
    }
}


void init_threads(comm_data *cd
		  , solver_data *sd
		  , int NTHREADS
		  )
{
  int *pid = check_malloc(sd->nallpoints * sizeof(int));
  int *htype = check_malloc(sd->nallpoints * sizeof(int));

  /* meta data for threadprivate rangelist, reorder face data */
  init_thread_meta_data(pid, htype, cd, sd, NTHREADS);

  /* free old rangelist */
  check_free(sd->fcolor->all_points_of_color);
  check_free(sd->fcolor);

  /* assign cross edge type, first/last points of color etc.*/
#pragma omp parallel default (none) shared(pid, htype, cd, sd, stderr)
  {
    int const tid = omp_get_thread_num();
    init_thread_rangelist(cd, sd, tid, pid, htype);
  }

  /* init thread communication */
  init_thread_comm(cd, sd);
  allocate_thread_private_comm_data(cd);

  /* sanity check */
  test_thread_rangelist(sd);
  eval_thread_comm(cd);

  check_free(pid);
  check_free(htype);


}

