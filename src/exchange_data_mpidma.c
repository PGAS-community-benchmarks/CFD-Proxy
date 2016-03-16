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
#include <mpi.h>
#include <assert.h>

#include "exchange_data_mpidma.h"
#include "solver_data.h"
#include "comm_data.h"
#include "rangelist.h"
#include "threads.h"
#include "util.h"

#include "error_handling.h"

static void *sndbuf = 0;
static void *rcvbuf = 0;
static MPI_Win rcvwin;
static MPI_Group comm_group;

void init_mpidma_buffers(comm_data *cd
			 , int dim2
			 )
{
  int i;
  const int max_elem_sz = NGRAD * 3;
  const size_t szd = sizeof(double);
  ASSERT(dim2 == max_elem_sz);  

  int rsz = 0, ssz = 0;
  for(i = 0; i < cd->ncommdomains; i++)
    {
      int k = cd->commpartner[i];
      ssz +=  cd->sendcount[k] * max_elem_sz * szd;
      rsz +=  cd->recvcount[k] * max_elem_sz * szd;
    }  

  // buffer local MPI_Put() calls will read from
  MPI_Alloc_mem(ssz, MPI_INFO_NULL, &sndbuf);
  
  // buffer and window remote MPI_Put() calls will write into
  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Info_set(info, "no_locks", "true");

#if defined MPI_VERSION && MPI_VERSION == 3
  MPI_Win_allocate(rsz, 1, info, MPI_COMM_WORLD, &rcvbuf, &rcvwin);
  int *model, flag=0;
  MPI_Win_get_attr(rcvwin, MPI_WIN_MODEL, &model, &flag);
  assert(flag); // getting attributes succeeded
  assert(*model == MPI_WIN_UNIFIED);
#else
#warning MPI_Win_allocate() not available since no MPI-3
  MPI_Alloc_mem(rsz, MPI_INFO_NULL, &rcvbuf);
  MPI_Win_create(rcvbuf, rsz, 1, info, MPI_COMM_WORLD, &rcvwin);
#endif

  // set PSCW group
  static MPI_Group all_group;
  MPI_Comm_group( MPI_COMM_WORLD, &all_group );
  MPI_Group_incl( all_group, cd->ncommdomains, cd->commpartner, &comm_group );

}

void *get_sndbuf(void)
{
  return sndbuf;
}

void *get_rcvbuf(void)
{
  return rcvbuf;
}

void free_mpidma_win(void)
{
  MPI_Group_free( &comm_group );
  MPI_Win_free(&rcvwin);
}

void exchange_dbl_mpidma_write(comm_data *cd
			      , double *data
			      , int dim2
			      , int i)
{
  int *commpartner  = cd->commpartner;
  int *sendcount    = cd->sendcount;

  gaspi_offset_t *remote_recv_offset    = cd->remote_recv_offset;
  gaspi_offset_t *local_send_offset     = cd->local_send_offset;

  int count;
  size_t szd = sizeof(double);

  int k = commpartner[i];
  count = sendcount[k];

  ASSERT(data != NULL)
 
  if(count > 0)
    {
      double *const sbuf = (double *) ((char *) sndbuf + local_send_offset[k]);
      int size = count * dim2 * szd;
      MPI_Put(sbuf
	      , size // num items to copy
	      , MPI_CHAR // type pf items to copy
	      , k // target rank to copy to
	      , remote_recv_offset[k] // target_disp
	      , size  // target_count
	      , MPI_CHAR // type at target
	      , rcvwin // MPI DMA win to use
	      );
      cd->send_flag[i].global++;
    }
}


static void exchange_dbl_mpidma_scatter(comm_data *cd
					, double *data
					, int dim2
					)
{
  int ncommdomains  = cd->ncommdomains;
  int *commpartner  = cd->commpartner;
  gaspi_offset_t *local_recv_offset = cd->local_recv_offset;

  int i;
  for (i = 0; i < ncommdomains; ++i)
    {
      int nrecv = get_recvcount_local(i);
      if (nrecv > 0)
	{
	  volatile int flag;
	  while ((flag = cd->recv_flag[i].global) == cd->recv_stage)
	    {
	      _mm_pause();
	    }
	  /* sanity check */
	  ASSERT(flag == (cd->recv_stage + 1));

	  /* copy the data from the recvbuf into out data field */
	  int k = commpartner[i];
	  double *rbuf = (double *) ((char *) rcvbuf + local_recv_offset[k]);
	  exchange_dbl_copy_out_local(rbuf, data, dim2, i);	  
	} 
    }
}



void exchange_dbl_mpifence_bulk_sync(comm_data *cd
				     , double *data
				     , int dim2
				     , int final
				     )
{

  int ncommdomains  = cd->ncommdomains;
  int *commpartner  = cd->commpartner;
  int *recvcount    = cd->recvcount;
  int **recvindex   = cd->recvindex;
  
  gaspi_offset_t *remote_recv_offset    = cd->remote_recv_offset;
  gaspi_offset_t *local_recv_offset     = cd->local_recv_offset;
  
  int i;
  
  ASSERT(dim2 > 0);
  ASSERT(ncommdomains != 0);
  ASSERT(commpartner != NULL);
  ASSERT(recvcount != NULL);
  ASSERT(recvindex != NULL);
  ASSERT(remote_recv_offset != NULL);
  ASSERT(local_recv_offset != NULL);
  ASSERT((final == 0 || final == 1));
  
  /* wait for completed computation before send */
  if (this_is_the_last_thread())
  {

    MPI_Win_fence(MPI_MODE_NOPRECEDE, rcvwin);
    for(i = 0; i < ncommdomains; i++)
      {
#if !defined(USE_PACK_IN_BULK_SYNC) && !defined(USE_PARALLEL_GATHER)
	int k = cd->commpartner[i];
	double *const sbuf = (double *) ((char*) sndbuf + cd->local_send_offset[k]);
	exchange_dbl_copy_in(cd, sbuf, data, dim2, i);
#endif
        exchange_dbl_mpidma_write(cd, data, dim2, i);
      }
    MPI_Win_fence(MPI_MODE_NOSUCCEED | MPI_MODE_NOSTORE , rcvwin); // make sure data has arrived

    for(i = 0; i < ncommdomains; i++)
      {
	// flag received buffer 
	cd->recv_flag[i].global++;

#ifndef USE_PARALLEL_SCATTER
	/* copy the data from the recvbuffer into out data field */
        int k = commpartner[i];
	double *rbuf = (double *) ((char *) rcvbuf + local_recv_offset[k]);
	exchange_dbl_copy_out(cd, rbuf, data, dim2, i);
#endif

      }
  }
    
#ifdef USE_PARALLEL_SCATTER
  exchange_dbl_mpidma_scatter(cd, data, dim2);
#endif

  if (this_is_the_last_thread())
  {
    cd->send_stage++;
    cd->recv_stage++;
  }

#ifndef USE_PARALLEL_SCATTER
/* wait for recv/unpack */
#pragma omp barrier
#else
/* no barrier -- in parallel scatter all threads unpack 
   the specifically required parts of recv */   
#endif

}



void exchange_dbl_mpifence_async(comm_data *cd
				 , double *data
				 , int dim2
				 , int final
				 )
{
  int ncommdomains  = cd->ncommdomains;
  int *commpartner  = cd->commpartner;
  int *recvcount    = cd->recvcount;
  int **recvindex   = cd->recvindex;
  
  gaspi_offset_t *local_recv_offset = cd->local_recv_offset;
  
  ASSERT(dim2 > 0);
  ASSERT(ncommdomains != 0);
  ASSERT(commpartner != NULL);
  ASSERT(recvcount != NULL);
  ASSERT(recvindex != NULL);
  ASSERT(local_recv_offset != NULL);
  ASSERT((final == 0 || final == 1));
  
  int i;
  if (this_is_the_last_thread())
    {
      MPI_Win_fence(MPI_MODE_NOSTORE , rcvwin); // make sure data has arrived AND start next round

      for (i = 0; i < ncommdomains; ++i)
	{
	  // flag received buffer 
	  cd->recv_flag[i].global++;

#ifndef USE_PARALLEL_SCATTER
	  /* copy the data from the recvbuffer into out data field */
	  int k = commpartner[i];
	  double *rbuf = (double *) ((char *) rcvbuf + local_recv_offset[k]);
	  exchange_dbl_copy_out(cd, rbuf, data, dim2, i);
#endif
	}
    }

#ifdef USE_PARALLEL_SCATTER
  exchange_dbl_mpidma_scatter(cd, data, dim2);
#endif

  if (this_is_the_last_thread())
  {
    // inc stage counter
    cd->send_stage++;
    cd->recv_stage++;
  }

#ifndef USE_PARALLEL_SCATTER
/* wait for recv/unpack */
#pragma omp barrier
#else
/* no barrier -- in parallel scatter all threads unpack 
   the specifically required parts of recv */   
#endif

}

void exchange_dbl_mpipscw_bulk_sync(comm_data *cd
				    , double *data
				    , int dim2
				    , int final
				    )
{

  int ncommdomains  = cd->ncommdomains;
  int *commpartner  = cd->commpartner;
  int *recvcount    = cd->recvcount;
  int **recvindex   = cd->recvindex;
  
  gaspi_offset_t *remote_recv_offset    = cd->remote_recv_offset;
  gaspi_offset_t *local_recv_offset     = cd->local_recv_offset;
  
  int i;
  
  ASSERT(dim2 > 0);
  ASSERT(ncommdomains != 0);
  ASSERT(commpartner != NULL);
  ASSERT(recvcount != NULL);
  ASSERT(recvindex != NULL);
  ASSERT(remote_recv_offset != NULL);
  ASSERT(local_recv_offset != NULL);
  ASSERT((final == 0 || final == 1));
  
  /* wait for completed computation before send */
  if (this_is_the_last_thread())
    {
      mpidma_async_post_start();

      for(i = 0; i < ncommdomains; i++)
	{
#if !defined(USE_PACK_IN_BULK_SYNC) && !defined(USE_PARALLEL_GATHER)
	  int k = cd->commpartner[i];
	  double *const sbuf = (double *) ((char *) sndbuf + cd->local_send_offset[k]);
	  exchange_dbl_copy_in(cd, sbuf, data, dim2, i);
#endif
	  exchange_dbl_mpidma_write(cd, data, dim2, i);
	}

      mpidma_async_complete();
      mpidma_async_wait();

      for(i = 0; i < ncommdomains; i++)
	{
	  // flag received buffer 
	  cd->recv_flag[i].global++;

#ifndef USE_PARALLEL_SCATTER
	  /* copy the data from the recvbuffer into out data field */
	  int k = commpartner[i];
	  double *rbuf = (double *) ((char *) rcvbuf + local_recv_offset[k]);
	  exchange_dbl_copy_out(cd, rbuf, data, dim2, i);
#endif
	}
    }

#ifdef USE_PARALLEL_SCATTER
  exchange_dbl_mpidma_scatter(cd, data, dim2);
#endif

  if (this_is_the_last_thread())
    {
      // inc stage counter
      cd->send_stage++;
      cd->recv_stage++;
    }

#ifndef USE_PARALLEL_SCATTER
/* wait for recv/unpack */
#pragma omp barrier
#else
/* no barrier -- in parallel scatter all threads unpack 
   the specifically required parts of recv */   
#endif

}



void exchange_dbl_mpipscw_async(comm_data *cd
				, double *data
				, int dim2
				, int final
				)
{


  int ncommdomains  = cd->ncommdomains;
  int *commpartner  = cd->commpartner;
  int *recvcount    = cd->recvcount;
  int **recvindex   = cd->recvindex;
  gaspi_offset_t *local_recv_offset = cd->local_recv_offset;

  ASSERT(dim2 > 0);
  ASSERT(ncommdomains != 0);
  ASSERT(commpartner != NULL);
  ASSERT(recvcount != NULL);
  ASSERT(recvindex != NULL);
  ASSERT(local_recv_offset != NULL);
  ASSERT((final == 0 || final == 1));
  int i;

#if defined(USE_MPI_IMMEDIATE_WAIT)
#if !defined(USE_MPI_MULTI_THREADED)
#error MPI_IMMEDIATE_WAIT requires MPI_MULTI_THREADED
#endif 
  if (this_is_the_first_thread()) 
    {
    /* unconditional waiting with recv first thread, 
       requires MPI_MULTI_THREADED */ 

#elif defined(USE_MPI_EARLY_WAIT)
#if !defined(USE_MPI_MULTI_THREADED) && defined(USE_MPI_TEST)
#error MPI_EARLY_WAIT with MPI_THREAD_SERIALIZED must not use USE_MPI_TEST
#endif
  if (this_is_the_first_thread()) 
    {

    /* wait for completed comm_stage before entering recv */ 
    volatile int comm;
    while ((comm = cd->comm_stage) == cd->send_stage)
      {
	_mm_pause();
      }
    /* sanity check */
    ASSERT(comm == (cd->send_stage + 1));

#else /* late wait */
  if (this_is_the_last_thread())
    {
#endif

    mpidma_async_wait();

    for (i = 0; i < ncommdomains; ++i)
      {
	// flag received buffer 
	cd->recv_flag[i].global++;

#ifndef USE_PARALLEL_SCATTER
	/* copy the data from the recvbuf into out data field */
        int k = commpartner[i];
	double *rbuf = (double *) ((char *) rcvbuf + local_recv_offset[k]);
	exchange_dbl_copy_out(cd, rbuf, data, dim2, i);
#endif

      }
    }

#ifdef USE_PARALLEL_SCATTER
  exchange_dbl_mpidma_scatter(cd, data, dim2);
#endif

  if (this_is_the_last_thread())
  {
    // inc stage counter
    cd->send_stage++;
    cd->recv_stage++;

    if (! final)
      {
	/* start next round */
	mpidma_async_post_start();
      }
  }

#ifndef USE_PARALLEL_SCATTER
/* wait for recv/unpack */
#pragma omp barrier
#else
/* no barrier -- in parallel scatter all threads unpack 
   the specifically required parts of recv */   
#endif

}

void mpidma_async_post_start(void)
{
    MPI_Win_post(comm_group, 0, rcvwin );
    MPI_Win_start(comm_group, 0, rcvwin );
}

void mpidma_async_complete(void)
{
    MPI_Win_complete(rcvwin);
}

void mpidma_async_wait(void)
{
    MPI_Win_wait(rcvwin);
}

void mpidma_async_win_fence(int assertion)
{
  MPI_Win_fence(assertion, rcvwin);
}
