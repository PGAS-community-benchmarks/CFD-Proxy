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
#include "thread_comm.h"
#include "eval.h"
#include "threads.h"


/* global comm var for threadprivate comm */
static volatile counter_t *send_counter_global = NULL; 

/* private comm var for threadprivate comm */
static int *send_counter_local = NULL;
#pragma omp threadprivate(send_counter_local)

static int nsendcount_local = 0;
#pragma omp threadprivate(nsendcount_local)
static int *sendcount_local = NULL;
#pragma omp threadprivate(sendcount_local)
static int **sendindex_local = NULL;
#pragma omp threadprivate(sendindex_local)
static int **sendoffset_local = NULL;
#pragma omp threadprivate(sendoffset_local)

static int nrecvcount_local = 0;
#pragma omp threadprivate(nrecvcount_local)
static int *recvcount_local = NULL;
#pragma omp threadprivate(recvcount_local)
static int **recvindex_local = NULL;
#pragma omp threadprivate(recvindex_local)
static int **recvoffset_local = NULL;
#pragma omp threadprivate(recvoffset_local)

/* getter/setter functions for global/local counters */
int get_send_counter_global(int i)
{
#pragma omp flush
  return send_counter_global[i].global;
}
int inc_send_counter_global(int i, int val)
{
#pragma omp flush
  const int global = my_add_and_fetch(&send_counter_global[i].global, val);
#pragma omp flush
  return global;
}

/* getter/setter functions for thread local send/recv count */
int inc_send_counter_local(int i, int val)
{
  send_counter_local[i] += val;
  return send_counter_local[i];
}
int get_nsendcount_local(void)
{
  return nsendcount_local;
}
int get_sendcount_local(int i)
{
  return sendcount_local[i];
}
int get_nrecvcount_local(void)
{
  return nrecvcount_local;
}
int get_recvcount_local(int i)
{
  return recvcount_local[i];
}


/* fetch/add wrapper */
int my_add_and_fetch(volatile int *ptr, int val)
{
#ifdef GCC_EXTENSION
  int t = __sync_add_and_fetch(ptr, val);
  ASSERT(t >= 0);
  return t;
#else
  int t;
  //#pragma omp atomic capture 
#pragma omp critical (add_and_fetch)
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
#pragma omp critical (fetch_and_add)
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


/*
 * just pack, no send, used in bulk_sync 
 * exchange in order to send pre-packed
 * comm buffer. 
 */
void initiate_thread_comm_mpi_pack(RangeList *color
				   , comm_data *cd
				   , double *data
				   , int dim2
				   )
{
  int i;
  static volatile int shared = 0;
  for(i = 0; i < color->nsendcount; i++)
    {
      int i1 = color->sendpartner[i];
      double *sbuf = cd->sendbuf[i1];
      if (color->sendcount[i] > 0 
	  && sendcount_local[i1] > 0)
	{
	  send_counter_local[i1] += color->sendcount[i];
	  if(send_counter_local[i1] 
	     % sendcount_local[i1] == 0)
	    {
#ifdef USE_PARALLEL_GATHER
	      /*
	       * multithreaded gather, all threads
	       * pack per commbuffer (i.e. per target rank)
	       */
	      exchange_dbl_copy_in_local(sbuf
					 , data
					 , dim2
					 , i1);
#endif
	      int inc_global 
		= inc_send_counter_global(i1
					  , sendcount_local[i1]
					  );
	      int k = cd->commpartner[i1];
	      if (inc_global 
		  % cd->sendcount[k] == 0)
		{
#ifndef USE_PARALLEL_GATHER
		  /*
		   * multithreaded pack, last thread
		   * packs per commbuffer (i.e. per target rank)
		   */
		  exchange_dbl_copy_in(cd
				       , sbuf
				       , data
				       , dim2
				       , i1
				       );
#endif
		  if (my_add_and_fetch(&shared, 1) 
		      % cd->ncommdomains == 0)
		    {
		      /* 
		       * increase comm_stage, 
		       * (flag last pack) 
		       */
		      cd->comm_stage++;
		    }
		}
	    }
	}
    }
}



void initiate_thread_comm_mpi_send(RangeList *color
				   , comm_data *cd
				   , double *data
				   , int dim2
				   )
{
  int i;
  static volatile int shared = 0;
  for(i = 0; i < color->nsendcount; i++)
    {
      int i1 = color->sendpartner[i];
      double *sbuf = cd->sendbuf[i1];
      if (color->sendcount[i] > 0 
	  && sendcount_local[i1] > 0)
	{
	  send_counter_local[i1] += color->sendcount[i];
	  if(send_counter_local[i1] 
	     % sendcount_local[i1] == 0)
	    {
#ifdef USE_PARALLEL_GATHER
	      /*
	       * multithreaded gather, all threads
	       * pack per commbuffer (i.e. per target rank)
	       */
	      exchange_dbl_copy_in_local(sbuf
					 , data
					 , dim2
					 , i1);
#endif
	      int inc_global 
		= inc_send_counter_global(i1
					  , sendcount_local[i1]
					  );
	      int k = cd->commpartner[i1];
	      if (inc_global 
		  % cd->sendcount[k] == 0)
		{
#ifndef USE_PARALLEL_GATHER
		  /*
		   * multithreaded pack, last thread
		   * packs per commbuffer (i.e. per target rank)
		   */
		  exchange_dbl_copy_in(cd
				       , sbuf
				       , data
				       , dim2
				       , i1
				       );
#endif
#ifndef USE_MPI_MULTI_THREADED
#pragma omp critical (mpi)
#endif
		  {
		    exchange_dbl_mpi_send(cd
					  , data
					  , dim2
					  , i1
					  );
		  }
		  /* 
		   * increase comm_stage, flag last send 
		   * required for MPI_EARLY_WAIT
		   */
		  if (my_add_and_fetch(&shared, 1) 
		      % cd->ncommdomains == 0)
		    {
		      cd->comm_stage++;
		    }
		}
	    }
	}
    }

  /*
   * try to achieve some progress, only useful
   * with MPI_THREAD_SERIALIZED and late waiting. 
   * Better: MPI_EARLY_WAIT -- achieve progress in waitany. 
   */

#if defined(USE_MPI_TEST)
#if defined(USE_MPI_TEST_MASTER_ONLY)
#pragma omp master
#endif
  {
#ifndef USE_MPI_MULTI_THREADED
#pragma omp critical (mpi)
#endif
    {
      exchange_dbl_mpi_test(cd);
    }
  }
#endif

}

void initiate_thread_comm_mpi_fence(RangeList *color
				    , comm_data *cd
				    , double *data
				    , int dim2
				    )
{
  int i;
  static volatile int shared = 0;
  for(i = 0; i < color->nsendcount; i++)
    {
      int i1 = color->sendpartner[i];
      void *sndbuf = get_sndbuf();
      int k = cd->commpartner[i1];
      double *const sbuf = (double *) ((char *) sndbuf + cd->local_send_offset[k]);
      if (color->sendcount[i] > 0 
	  && sendcount_local[i1] > 0)
	{
	  send_counter_local[i1] += color->sendcount[i];
	  if(send_counter_local[i1] 
	     % sendcount_local[i1] == 0)
	    {
#ifdef USE_PARALLEL_GATHER
	      /*
	       * multithreaded gather, all threads
	       * pack per commbuffer (i.e. per target rank)
	       */
	      exchange_dbl_copy_in_local(sbuf
					 , data
					 , dim2
					 , i1);
#endif
	      int inc_global 
		= inc_send_counter_global(i1
					  , sendcount_local[i1]
					  );
	      if (inc_global 
		  % cd->sendcount[k] == 0)
		{
#ifndef USE_PARALLEL_GATHER
		  /*
		   * multithreaded pack, last thread
		   * packs per commbuffer (i.e. per target rank)
		   */
		  exchange_dbl_copy_in(cd
				       , sbuf
				       , data
				       , dim2
				       , i1
				       );
#endif
#ifndef USE_MPI_MULTI_THREADED
#pragma omp critical (mpi)
#endif
		  {
		    exchange_dbl_mpidma_write(cd, data, dim2, i1);

		  }
		  /* 
		   * increase comm_stage, flag last send 
		   * required for MPI_EARLY_WAIT
		   */
		  if (my_add_and_fetch(&shared, 1) 
		      % cd->ncommdomains == 0)
		    {
		      cd->comm_stage++;
		    }
		}
	    }
	}
    }
}


void initiate_thread_comm_mpi_pscw(RangeList *color
				   , comm_data *cd
				   , double *data
				   , int dim2
				   )
{
  int i;
  static volatile int shared = 0;
  for(i = 0; i < color->nsendcount; i++)
    {
      int i1 = color->sendpartner[i];
      void *sndbuf = get_sndbuf();
      int k = cd->commpartner[i1];
      double *const sbuf = (double *) ((char *) sndbuf + cd->local_send_offset[k]);
      if (color->sendcount[i] > 0 
	  && sendcount_local[i1] > 0)
	{
	  send_counter_local[i1] += color->sendcount[i];
	  if(send_counter_local[i1] 
	     % sendcount_local[i1] == 0)
	    {
#ifdef USE_PARALLEL_GATHER
	      /*
	       * multithreaded gather, all threads
	       * pack per commbuffer (i.e. per target rank)
	       */
	      exchange_dbl_copy_in_local(sbuf
					 , data
					 , dim2
					 , i1);
#endif
	      int inc_global 
		= inc_send_counter_global(i1
					  , sendcount_local[i1]
					  );
	      if (inc_global 
		  % cd->sendcount[k] == 0)
		{
#ifndef USE_PARALLEL_GATHER
		  /*
		   * multithreaded pack, last thread
		   * packs per commbuffer (i.e. per target rank)
		   */
		  exchange_dbl_copy_in(cd
				       , sbuf
				       , data
				       , dim2
				       , i1
				       );
#endif
#ifndef USE_MPI_MULTI_THREADED
#pragma omp critical (mpi)
#endif
		  {
		    exchange_dbl_mpidma_write(cd, data, dim2, i1);
		    if (my_add_and_fetch(&shared, 1) 
			% cd->ncommdomains == 0)
		        {
			  mpidma_async_complete();
			  /* 
			   * increase comm_stage, flag last send 
			   * required for MPI_EARLY_WAIT
			   */
			  cd->comm_stage++;
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
  static volatile int shared = 0;
  for(i = 0; i < color->nsendcount; i++)
    {
      int i1 = color->sendpartner[i];
      gaspi_pointer_t ptr;
      int buffer_id = cd->send_stage % 2;
      SUCCESS_OR_DIE(gaspi_segment_ptr(buffer_id, &ptr));
      int k = cd->commpartner[i1];
      double *sbuf = (double *) ((char *) ptr + cd->local_send_offset[k]); 
      if (color->sendcount[i] > 0 
	  && sendcount_local[i1] > 0)
	{
	  send_counter_local[i1] += color->sendcount[i];
	  if(send_counter_local[i1] 
	     % sendcount_local[i1] == 0)
	    {
#ifdef USE_PARALLEL_GATHER
	      /*
	       * multithreaded gather, all threads
	       * pack per commbuffer (i.e. per target rank)
	       */
	      exchange_dbl_copy_in_local(sbuf
					 , data
					 , dim2
					 , i1
					 );
#endif
	      int inc_global 
		= inc_send_counter_global(i1
					  , sendcount_local[i1]
					  );
	      if (inc_global 
		  % cd->sendcount[k] == 0)
		{
#ifndef USE_PARALLEL_GATHER
		  /*
		   * multithreaded pack, last thread
		   * packs per commbuffer (i.e. per target rank)
		   */
		  exchange_dbl_copy_in(cd
				       , sbuf
				       , data
				       , dim2
				       , i1
				       );
#endif
		  exchange_dbl_gaspi_write(cd
					   , data
					   , dim2
					   , buffer_id
					   , i1
					   );
		  /* 
		   * increase comm_stage, flag last send 
		   * required for MPI_EARLY_WAIT
		   */
		  if (my_add_and_fetch(&shared, 1) 
		      % cd->ncommdomains == 0)
		    {
		      cd->comm_stage++;
		    }
		}
	    }
	}
    }
}
#endif


static void allocate_comm_data(comm_data *cd)
{
  int j ;

  /* global counters for send/recv counter */
  send_counter_global = check_malloc(cd->ncommdomains * sizeof(counter_t));
  for(j = 0; j < cd->ncommdomains; j++)
    {
      send_counter_global[j].global = 0;
    }

  /* thread private counters for send/recv counter */
#pragma omp parallel default (none) shared(cd)
  {
    int j1;
    send_counter_local = check_malloc(cd->ncommdomains * sizeof(int));
    for(j1 = 0; j1 < cd->ncommdomains; j1++)
      { 
        send_counter_local[j1] = 0;
      }
  }

#pragma omp parallel default (none) shared(cd, stdout, stderr)
  {
    int i1;
    RangeList *color;

    /* alloc num of thread private sendcounts */
    sendcount_local = check_malloc(cd->ncommdomains * sizeof(int));
    for(i1 = 0; i1 < cd->ncommdomains; i1++)
      {
	sendcount_local[i1] = 0;
      }
    for (color = get_color(); color != NULL
	   ; color = get_next_color(color)) 
      {
	int i2;
	for(i2 = 0; i2 < color->nsendcount; i2++)
	  {
	    int i3  = color->sendpartner[i2];
	    sendcount_local[i3] += color->sendcount[i2];
	  }
      }
    int nsendcount = 0;
    for(i1 = 0; i1 < cd->ncommdomains; i1++)
      {
	if (sendcount_local[i1] > 0)
	  {
	    nsendcount++;
	  }
      }
    nsendcount_local = nsendcount;

    /* alloc thread private sendindex/sendoffset */
    sendindex_local = check_malloc(cd->ncommdomains * sizeof(int*));
    sendoffset_local = check_malloc(cd->ncommdomains * sizeof(int*));
    for(i1 = 0; i1 < cd->ncommdomains; i1++)
      {
	int count = sendcount_local[i1];
	if(count > 0)
	  {
	    sendindex_local[i1] = check_malloc(count * sizeof(int));
	    sendoffset_local[i1] = check_malloc(count * sizeof(int));
	  }
      }
    int *tmp1 = check_malloc(cd->ncommdomains * sizeof(int));
    for(i1 = 0; i1 < cd->ncommdomains; i1++)
      {
	tmp1[i1] = 0;
      }
    for (color = get_color(); color != NULL
	   ; color = get_next_color(color)) 
      {
	int i2, i4;
	for(i2 = 0; i2 < color->nsendcount; i2++)
	  {
	    int i3  = color->sendpartner[i2];
	    for(i4 = 0; i4 < color->sendcount[i2]; i4++)
	      {                     
		sendindex_local[i3][tmp1[i3]+i4] = color->sendindex[i2][i4];
		sendoffset_local[i3][tmp1[i3]+i4] = color->sendoffset[i2][i4];
	      }
	    tmp1[i3] += color->sendcount[i2];
	  }
      }
    check_free(tmp1);    
  }

#pragma omp parallel default (none) shared(cd, stdout, stderr)
  {
    int i1;
    RangeList *color;

    /* alloc num of thread private recvcounts */
    recvcount_local = check_malloc(cd->ncommdomains * sizeof(int));
    for(i1 = 0; i1 < cd->ncommdomains; i1++)
      {
	recvcount_local[i1] = 0;
      }
    for (color = get_color(); color != NULL
	   ; color = get_next_color(color)) 
      {
	int i2;
	for(i2 = 0; i2 < color->nrecvcount; i2++)
	  {
	    int i3  = color->recvpartner[i2];
	    recvcount_local[i3] += color->recvcount[i2];
	  }
      }
    int nrecvcount = 0;
    for(i1 = 0; i1 < cd->ncommdomains; i1++)
      {
	if (recvcount_local[i1] > 0)
	  {
	    nrecvcount++;
	  }
      }
    nrecvcount_local = nrecvcount;

    /* alloc thread private recvindex/recvoffset */
    recvindex_local = check_malloc(cd->ncommdomains * sizeof(int*));
    recvoffset_local = check_malloc(cd->ncommdomains * sizeof(int*));
    for(i1 = 0; i1 < cd->ncommdomains; i1++)
      {
	int count = recvcount_local[i1];
	if(count > 0)
	  {
	    recvindex_local[i1] = check_malloc(count * sizeof(int));
	    recvoffset_local[i1] = check_malloc(count * sizeof(int));
	  }
      }

    int *tmp1 = check_malloc(cd->ncommdomains * sizeof(int));
    for(i1 = 0; i1 < cd->ncommdomains; i1++)
      {
	tmp1[i1] = 0;
      }
    for (color = get_color(); color != NULL
	   ; color = get_next_color(color)) 
      {
	int i2, i4;
	for(i2 = 0; i2 < color->nrecvcount; i2++)
	  {
	    int i3  = color->recvpartner[i2];
	    for(i4 = 0; i4 < color->recvcount[i2]; i4++)
	      {                     
		recvindex_local[i3][tmp1[i3]+i4] = color->recvindex[i2][i4];
		recvoffset_local[i3][tmp1[i3]+i4] = color->recvoffset[i2][i4];
	      }
	    tmp1[i3] += color->recvcount[i2];
	  }
      }
    check_free(tmp1);    
  }

}



void init_threads(comm_data *cd
		  , solver_data *sd
		  , int NTHREADS
		  )
{
  int *pid = check_malloc(sd->nallpoints * sizeof(int));
  int *htype = check_malloc(sd->nallpoints * sizeof(int));

  /* global comp stage counter */
  init_comp_stage_global(NTHREADS);

  /* set thread id, color id */
  init_thread_id(NTHREADS, pid, sd);

  /* init halo type */
  init_halo_type(htype, cd, sd);

  /* free old rangelist */
  check_free(sd->fcolor->all_points_of_color);
  check_free(sd->fcolor);

  /* assign cross edge type, first/last points of color etc.*/
#pragma omp parallel default (none) shared(pid, htype, cd, sd, stderr)
  {
    int const tid = omp_get_thread_num();
    init_thread_rangelist(cd, sd, tid, pid, htype);
  }

  /* sanity check */
  eval_thread_rangelist(sd);

  /* assign thread neighbours*/
#pragma omp parallel default (none) shared(pid, cd, sd, stderr)
  {
    int const tid = omp_get_thread_num();
    init_thread_neighbours(cd, sd, tid, pid);
  }

  /* free metadata */
  check_free(pid);
  check_free(htype);

  if (cd->ndomains == 1)
    {
      return;
    }

  /* init thread communication */
  init_thread_comm(cd, sd);

  /* thread private comm data allocate */
  allocate_comm_data(cd);

  /* sanity check */
  eval_thread_comm(cd);



}


void exchange_dbl_copy_in(comm_data *cd
			  , double *sbuf
			  , double *data
			  , int dim2
			  , int i
			  )
{
  int j;
  int *sendcount    = cd->sendcount;
  int **sendindex   = cd->sendindex;
  int k = cd->commpartner[i];
  int count = sendcount[k];

  if(count > 0)
    {
      for(j = 0; j < count; j++)
	{
	  int n1 = dim2 * j;
	  int n2 = dim2 * sendindex[k][j];
	  memcpy(&sbuf[n1], &data[n2], dim2 * sizeof(double));
	}
    }
}


void exchange_dbl_copy_out(comm_data *cd
			   , double *rbuf
			   , double *data
			   , int dim2
			   , int i
			   )
{
  int j;
  int *recvcount    = cd->recvcount;
  int **recvindex   = cd->recvindex;
  int k = cd->commpartner[i];
  int count = recvcount[k];

  if(count > 0)
    {
      for(j = 0; j < count; j++)
	{
	  int n1 = dim2 * j;
	  int n2 = dim2 * recvindex[k][j];
	  memcpy(&data[n2], &rbuf[n1], dim2 * sizeof(double));
	}
    }

}


void exchange_dbl_copy_in_local(double *sbuf
				, double *data
				, int dim2
				, int i)
{
  int j;
  for(j = 0; j < sendcount_local[i]; j++)
    {
      int n1 = dim2 * sendoffset_local[i][j];
      int n2 = dim2 * sendindex_local[i][j];
      memcpy(&sbuf[n1], &data[n2], dim2 * sizeof(double));
    }  
}


void exchange_dbl_copy_out_local(double *rbuf
			  , double *data
			  , int dim2
			  , int i)
{
  int j;
  for(j = 0; j < recvcount_local[i]; j++)
    {
      int n1 = dim2 * recvoffset_local[i][j];
      int n2 = dim2 * recvindex_local[i][j];
      memcpy(&data[n2], &rbuf[n1], dim2 * sizeof(double));
    }
}




