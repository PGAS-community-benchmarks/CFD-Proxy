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
#include "solver_data.h"
#include "comm_data.h"
#include "rangelist.h"
#include "threads.h"
#include "util.h"
#include "error_handling.h"

#define DATAKEY 4712


void init_mpi_requests(comm_data *cd, int dim2)
{
  int i;
  const int max_elem_sz = NGRAD * 3;
  ASSERT(dim2 == max_elem_sz);  
  size_t szd = sizeof(double);

  ASSERT(cd->nreq == 0);
  ASSERT(cd->ncommdomains > 0);
  
  size_t szr = 2 * cd->ncommdomains * sizeof(MPI_Request);
  size_t szs = 2 * cd->ncommdomains * sizeof(MPI_Status);   
  cd->req   = (MPI_Request *)check_malloc(szr);
  cd->stat  = (MPI_Status  *)check_malloc(szs);
  cd->nreq  = cd->ncommdomains;

  /* status flag */
  cd->send_flag = check_malloc(cd->ncommdomains*sizeof(counter_t));
  cd->recv_flag = check_malloc(cd->ncommdomains*sizeof(counter_t));
  for(i = 0; i < cd->ncommdomains; i++)
    {
      cd->send_flag[i].global = 0;
      cd->recv_flag[i].global = 0;
    }

  /* sendbuffer, recvbuffer  */
  cd->sendbuf = check_malloc(cd->ncommdomains * sizeof(double*));
  cd->recvbuf = check_malloc(cd->ncommdomains * sizeof(double*));
  for(i = 0; i < cd->ncommdomains; i++)
    {
      cd->sendbuf[i] = NULL;
      cd->recvbuf[i] = NULL;
    }
  for(i = 0; i < cd->ncommdomains; i++)
    {
      int k = cd->commpartner[i];
      if (cd->sendcount[k] > 0)
	{
	  cd->sendbuf[i] = 
	    check_malloc(dim2 * cd->sendcount[k] * max_elem_sz * szd);
	}
      if (cd->recvcount[k] > 0)
	{
	  cd->recvbuf[i] = 
	    check_malloc(dim2 * cd->recvcount[k] * max_elem_sz * szd);
	}
    }  


}

void exchange_dbl_mpi_test(comm_data *cd)
{
  int i, flag = 0;
  int ncommdomains  = cd->ncommdomains;
  
  for (i = 0; i < ncommdomains; i++)
    {
      volatile int send; 
      if ((send = cd->send_flag[i].global) > cd->send_stage)
	{
	  MPI_Test(&(cd->req[ncommdomains + i])
		   , &flag 
		   , &(cd->stat[ncommdomains + i])
		   );
	}
    }
}

void exchange_dbl_mpi_send(comm_data *cd
			   , double *data
			   , int dim2
			   , int i
			   )
{
  int ncommdomains  = cd->ncommdomains;
  int *commpartner  = cd->commpartner;
  int *sendcount    = cd->sendcount;

  size_t size, szd = sizeof(double);

  /* send */
  int k = commpartner[i];
  int count = sendcount[k];

  ASSERT(data != NULL);
 
  if(count > 0)
    {
      double *sbuf = cd->sendbuf[i];
      count *= dim2;
      size = count * szd;
      MPI_Isend(sbuf
		, size
		, MPI_BYTE
		, k
		, DATAKEY
		, MPI_COMM_WORLD
		, &(cd->req[ncommdomains + i])
		);
      /* inc send flag */
      cd->send_flag[i].global++;
    }
}



void exchange_dbl_mpi_post_recv(comm_data *cd
				, int dim2
				)
{
  int ncommdomains  = cd->ncommdomains;
  int *commpartner  = cd->commpartner;
  int *recvcount    = cd->recvcount;

  int i;
  size_t size, szd = sizeof(double);

  /* recv */
  for(i = 0; i < ncommdomains; i++)
    { 
      int k = commpartner[i];
      int count = recvcount[k] * dim2;

      if(count > 0)
	{
	  double *rbuf = cd->recvbuf[i]; 
	  size  = count * szd;
	  MPI_Irecv(rbuf
		   , size
		   , MPI_BYTE
		   , k
		   , DATAKEY
		   , MPI_COMM_WORLD
		    , &(cd->req[i])
		   );
	}
    }

}



static void exchange_dbl_mpi_scatter(comm_data *cd
				     , double *data
				     , int dim2
				     )
{
  int ncommdomains  = cd->ncommdomains;

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
	  double *rbuf = cd->recvbuf[i];
	  exchange_dbl_copy_out_local(rbuf, data, dim2, i);	  
	} 
    }
}


void exchange_dbl_mpi_bulk_sync(comm_data *cd
				, double *data
				, int dim2
				, int final
				)
{
  int ncommdomains  = cd->ncommdomains;
  int nreq          = cd->nreq;

  int *commpartner  = cd->commpartner;
  int *sendcount    = cd->sendcount;
  int *recvcount    = cd->recvcount;
  int **sendindex   = cd->sendindex;
  int **recvindex   = cd->recvindex;

  int i;

  ASSERT(dim2 > 0);
  ASSERT(ncommdomains != 0);
  ASSERT(nreq > 0);

  ASSERT(sendcount != NULL);
  ASSERT(recvcount != NULL);
  ASSERT(sendindex != NULL);
  ASSERT(recvindex != NULL);
  ASSERT((final == 0 || final == 1));

  /* wait for completed computation before send */
  if (this_is_the_last_thread())
    {
      exchange_dbl_mpi_post_recv(cd, dim2);      

      for(i = 0; i < ncommdomains; i++)
	{
	  int k = commpartner[i];
	  int count = sendcount[k];
	  if (count > 0)
	    {
#if !defined(USE_PACK_IN_BULK_SYNC) && !defined(USE_PARALLEL_GATHER)
	      double *sbuf = cd->sendbuf[i];	  
	      exchange_dbl_copy_in(cd, sbuf, data, dim2, i);
#endif
	      exchange_dbl_mpi_send(cd, data, dim2, i);
	    }
	}      

      MPI_Waitall(2 * ncommdomains
		  , cd->req
		  , cd->stat
		  );


      for(i = 0; i < ncommdomains; i++)
	{
	  // flag received buffer 
	  cd->recv_flag[i].global++;

#ifndef USE_PARALLEL_SCATTER
	  /* copy the data from the recvbuf into out data field */
	  double *rbuf = cd->recvbuf[i];
	  exchange_dbl_copy_out(cd, rbuf, data, dim2, i);
#endif
	}
    }

#ifdef USE_PARALLEL_SCATTER
  exchange_dbl_mpi_scatter(cd, data, dim2);
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


void exchange_dbl_mpi_early_recv(comm_data *cd
				 , double *data
				 , int dim2
				 , int final
				 )
{
  int ncommdomains  = cd->ncommdomains;
  int nreq          = cd->nreq;

  int *commpartner  = cd->commpartner;
  int *sendcount    = cd->sendcount;
  int *recvcount    = cd->recvcount;
  int **sendindex   = cd->sendindex;
  int **recvindex   = cd->recvindex;

  int i;

  ASSERT(dim2 > 0);
  ASSERT(ncommdomains != 0);
  ASSERT(nreq > 0);

  ASSERT(sendcount != NULL);
  ASSERT(recvcount != NULL);
  ASSERT(sendindex != NULL);
  ASSERT(recvindex != NULL);
  ASSERT((final == 0 || final == 1));

  /* wait for completed computation before send */
  if (this_is_the_last_thread())
    {
      for(i = 0; i < ncommdomains; i++)
	{
	  int k = commpartner[i];
	  int count = sendcount[k];
	  if (count > 0)
	    {
#if !defined(USE_PACK_IN_BULK_SYNC) && !defined(USE_PARALLEL_GATHER)
	      double *sbuf = cd->sendbuf[i];	  
	      exchange_dbl_copy_in(cd, sbuf, data, dim2, i);
#endif
	      exchange_dbl_mpi_send(cd, data, dim2, i);
	    }
	}      
      MPI_Waitall(2 * ncommdomains
		  , cd->req
		  , cd->stat
		  );      

      for(i = 0; i < ncommdomains; i++)
	{
	  // flag received buffer 
	  cd->recv_flag[i].global++;

#ifndef USE_PARALLEL_SCATTER
	  /* copy the data from the recvbuf into out data field */
	  double *rbuf = cd->recvbuf[i];
	  exchange_dbl_copy_out(cd, rbuf, data, dim2, i);
#endif
	}
    }

#ifdef USE_PARALLEL_SCATTER
  exchange_dbl_mpi_scatter(cd, data, dim2);
#endif

  if (this_is_the_last_thread())
    {
      // inc stage counter
      cd->send_stage++;
      cd->recv_stage++;

      if (! final)
	{
	  /* start next round */
	  exchange_dbl_mpi_post_recv(cd, NGRAD * 3);
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

void exchange_dbl_mpi_async(comm_data *cd
			    , double *data
			    , int dim2
			    , int final
			    )
{
  int ncommdomains  = cd->ncommdomains;
  int nreq          = cd->nreq;

  int *sendcount    = cd->sendcount;
  int *recvcount    = cd->recvcount;
  int **sendindex   = cd->sendindex;
  int **recvindex   = cd->recvindex;
  int i;

  ASSERT(dim2 > 0);
  ASSERT(ncommdomains != 0);
  ASSERT(nreq > 0);

  ASSERT(sendcount != NULL);
  ASSERT(recvcount != NULL);
  ASSERT(sendindex != NULL);
  ASSERT(recvindex != NULL);


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
#if defined(USE_MPI_WAIT_ANY )
      for (i = 0; i < ncommdomains; ++i)
	{
	  int id = -1;
	  MPI_Waitany(ncommdomains
		      , cd->req
		      , &id
		      , cd->stat
		      );

	  ASSERT(id >= 0 && id < ncommdomains);      

	  // flag received buffer 
	  cd->recv_flag[id].global++;

#ifndef USE_PARALLEL_SCATTER
	  /* copy the data from the recvbuf into out data field */
	  double *rbuf = cd->recvbuf[id];
	  exchange_dbl_copy_out(cd, rbuf, data, dim2, id);	  
#endif
	}

#elif defined(USE_MPI_TEST_ANY )
      int count = 0;
      while (count < ncommdomains)
	{
	  int flag = 0;
	  int id   = -1;
	  MPI_Testany(ncommdomains
		      , cd->req
		      , &id
		      , &flag
		      , cd->stat);
	  if (flag)
	    {
	      if (id == MPI_UNDEFINED)
		{
		  ASSERT(count == ncommdomains);
		  break;
		}
	      else
		{
		  ASSERT(id >= 0 && id < ncommdomains);      
		  
		  // flag received buffer 
		  cd->recv_flag[id].global++;
		  
#ifndef USE_PARALLEL_SCATTER
		  /* copy the data from the recvbuf into out data field */
		  double *rbuf = cd->recvbuf[id];
		  exchange_dbl_copy_out(cd, rbuf, data, dim2, id);	  
#endif
		  count++;
		}
	    }
	  else
	    {
	      _mm_pause();
	    }
	}
      
#else /* USE_MPI_WAIT_ANY */
      MPI_Waitall(ncommdomains
		  , cd->req
		  , cd->stat
		  );

      for (i = 0; i < ncommdomains; ++i)
	{
	  // flag received buffer 
	  cd->recv_flag[i].global++;

#ifndef USE_PARALLEL_SCATTER
	  /* copy the data from the recvbuf into out data field */
	  double *rbuf = cd->recvbuf[i];
	  exchange_dbl_copy_out(cd, rbuf, data, dim2, i);	  
#endif
	}

#endif /* !USE_MPI_WAIT_ANY */
    }

#ifdef USE_PARALLEL_SCATTER
   exchange_dbl_mpi_scatter(cd, data, dim2);
#endif

  if (this_is_the_last_thread())
    {
      // wait for all MPI_Isend we have issued
      MPI_Waitall(ncommdomains
		  , &(cd->req[ncommdomains])
		  , &(cd->stat[ncommdomains])
		  );

      // inc stage counter
      cd->send_stage++;
      cd->recv_stage++;

      if (! final)
	{
	/* start next round */
	  exchange_dbl_mpi_post_recv(cd, NGRAD * 3);
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





