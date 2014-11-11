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

  ASSERT(cd->nrecv == 0);
  ASSERT(cd->nsend == 0);
  ASSERT(cd->nreq == 0);
  
  size_t szr = 2 * cd->ncommdomains * sizeof(MPI_Request);
  size_t szs = 2 * cd->ncommdomains * sizeof(MPI_Status);   
  cd->req   = (MPI_Request *)check_malloc(szr);
  cd->stat  = (MPI_Status  *)check_malloc(szs);
  cd->nreq  = cd->ncommdomains;

  int rsz = 0, ssz = 0;
  for(i = 0; i < cd->ncommdomains; i++)
    {
      int k = cd->commpartner[i];
      ssz += dim2 * cd->sendcount[k] * max_elem_sz * szd;
      rsz += dim2 * cd->recvcount[k] * max_elem_sz * szd;
    }  

  cd->sendbuf = check_malloc(ssz);
  cd->nsend = ssz;

  cd->recvbuf = check_malloc(rsz);
  cd->nrecv = rsz;

}


static void exchange_dbl_mpi_copy_out(comm_data *cd
				      , double *data
				      , int dim2
				      , int k
				      )
{
  int *recvcount    = cd->recvcount;
  int **recvindex   = cd->recvindex;

  int j;
  double *rbuf = cd->recvbuf;
  int count = recvcount[k];

  if(count > 0)
    {
      for(j = 0; j < count; j++)
	{
	  int n1 = dim2 * j;
	  int n2 = dim2 * recvindex[k][j];
	  memcpy(&data[n2], &rbuf[n1], dim2 * sizeof(double));
	}
      rbuf += dim2 * count;
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
  int **sendindex   = cd->sendindex;

  int j;
  size_t size, szd = sizeof(double);
  double *sbuf = cd->sendbuf;

  /* send */
  int k = commpartner[i];
  int count = sendcount[k];
 
  if(count > 0)
    {
      for(j = 0; j < count; j++)
	{
	  int n1 = dim2 * j;
	  int n2 = dim2 * sendindex[k][j];
	  memcpy(&sbuf[n1], &data[n2], dim2 * sizeof(double));
	}

      count *= dim2;
      size = count * szd;
#ifndef USE_MPI_MULTI_THREADED
#pragma omp critical
#endif
      { 
	MPI_Isend(sbuf
		   , size
		   , MPI_BYTE
		   , k
		   , DATAKEY
		   , MPI_COMM_WORLD
		   , &(cd->req[ncommdomains + i])
		   );
      }

      sbuf += count;
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
  double *rbuf = cd->recvbuf; 

  /* recv */
  for(i = 0; i < ncommdomains; i++)
    { 
      int k = commpartner[i];
      int count = recvcount[k] * dim2;

      if(count > 0)
	{
	  size  = count * szd;
	  MPI_Irecv(rbuf
		   , size
		   , MPI_BYTE
		   , k
		   , DATAKEY
		   , MPI_COMM_WORLD
		    , &(cd->req[i])
		   );
	  rbuf += count;
	}
    }

}


void exchange_dbl_mpi_bulk_sync(comm_data *cd
				, double *data
				, int dim2
				)
{
  int ncommdomains  = cd->ncommdomains;
  int nsend         = cd->nsend;
  int nrecv         = cd->nrecv;
  int nreq          = cd->nreq;

  int *commpartner  = cd->commpartner;
  int *sendcount    = cd->sendcount;
  int *recvcount    = cd->recvcount;
  int **sendindex   = cd->sendindex;
  int **recvindex   = cd->recvindex;

  int i;

  ASSERT(dim2 > 0);
  ASSERT(ncommdomains != 0);
  ASSERT(nsend > 0);
  ASSERT(nrecv > 0);
  ASSERT(nreq > 0);

  ASSERT(sendcount != NULL);
  ASSERT(recvcount != NULL);
  ASSERT(sendindex != NULL);
  ASSERT(recvindex != NULL);


  /* wait for completed computation before send */
  if (this_is_the_last_thread())
    {
      exchange_dbl_mpi_post_recv(cd, dim2);      
      for(i = 0; i < ncommdomains; i++)
	{
	  exchange_dbl_mpi_send(cd, data, dim2, i);
	}      
      MPI_Waitall(2 * ncommdomains
		  , cd->req
		  , cd->stat
		  );      
      /* copy the data from the recvbuf into out data field */
      for(i = 0; i < ncommdomains; i++)
	{
	  int k = commpartner[i];
	  exchange_dbl_mpi_copy_out(cd, data, dim2, k);
	}

      // inc stage counter
      cd->send_stage++;
      cd->recv_stage++;

    }


}


void exchange_dbl_mpi_early_recv(comm_data *cd
				 , double *data
				 , int dim2
				 , int final
				 )
{
  int ncommdomains  = cd->ncommdomains;
  int nsend         = cd->nsend;
  int nrecv         = cd->nrecv;
  int nreq          = cd->nreq;

  int *commpartner  = cd->commpartner;
  int *sendcount    = cd->sendcount;
  int *recvcount    = cd->recvcount;
  int **sendindex   = cd->sendindex;
  int **recvindex   = cd->recvindex;

  int i;

  ASSERT(dim2 > 0);
  ASSERT(ncommdomains != 0);
  ASSERT(nsend > 0);
  ASSERT(nrecv > 0);
  ASSERT(nreq > 0);

  ASSERT(sendcount != NULL);
  ASSERT(recvcount != NULL);
  ASSERT(sendindex != NULL);
  ASSERT(recvindex != NULL);


  /* wait for completed computation before send */
  if (this_is_the_last_thread())
    {
      for(i = 0; i < ncommdomains; i++)
	{
	  exchange_dbl_mpi_send(cd, data, dim2, i);
	}      
      MPI_Waitall(2 * ncommdomains
		  , cd->req
		  , cd->stat
		  );      
      /* copy the data from the recvbuf into out data field */
      for(i = 0; i < ncommdomains; i++)
	{
	  int k = commpartner[i];
	  exchange_dbl_mpi_copy_out(cd, data, dim2, k);
	}

      // inc stage counter
      cd->send_stage++;
      cd->recv_stage++;

      if (! final)
	{
	  /* start next round */
	  exchange_dbl_mpi_post_recv(cd, NGRAD * 3);
	}

    }


}


void exchange_dbl_mpi_async(comm_data *cd
			    , double *data
			    , int dim2
			    , int final
			    )
{
  int ncommdomains  = cd->ncommdomains;
  int nsend         = cd->nsend;
  int nrecv         = cd->nrecv;
  int nreq          = cd->nreq;

  int *commpartner  = cd->commpartner;
  int *sendcount    = cd->sendcount;
  int *recvcount    = cd->recvcount;
  int **sendindex   = cd->sendindex;
  int **recvindex   = cd->recvindex;


  ASSERT(dim2 > 0);
  ASSERT(ncommdomains != 0);
  ASSERT(nsend > 0);
  ASSERT(nrecv > 0);
  ASSERT(nreq > 0);

  ASSERT(sendcount != NULL);
  ASSERT(recvcount != NULL);
  ASSERT(sendindex != NULL);
  ASSERT(recvindex != NULL);

#if defined(USE_MPI_MULTI_THREADED) && defined(USE_MPI_WAIT_ANY)
  
  if (this_is_the_first_thread())
    {
      int i;
      for (i = 0; i < ncommdomains; ++i)
	{
	  int id = -1;
	  MPI_Waitany(ncommdomains
		      , cd->req
		      , &id
		      , cd->stat
		      );

	  ASSERT(id >= 0 && id < ncommdomains);      
	  int k = commpartner[id];

	  /* copy the data from the recvbuf into out data field */
	  exchange_dbl_mpi_copy_out(cd, data, dim2, k);	  
	} 

      MPI_Waitall(ncommdomains
		  , &(cd->req[ncommdomains])
		  , &(cd->stat[ncommdomains])
		  );
    }


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

#else

  if (this_is_the_last_thread())
    {

      MPI_Waitall(2*ncommdomains
		  , cd->req
		  , cd->stat
		  );


      int i;
      for (i = 0; i < ncommdomains; ++i)
	{
	  int k = commpartner[i];

	  /* copy the data from the recvbuf into out data field */
	  exchange_dbl_mpi_copy_out(cd, data, dim2, k);
	  
	} 

      // inc stage counter
      cd->send_stage++;
      cd->recv_stage++;

      if (! final)
	{
	  /* start next round */
	  exchange_dbl_mpi_post_recv(cd, NGRAD * 3);
	}
    }




#endif




}





