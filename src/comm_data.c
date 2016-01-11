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
#ifdef USE_GASPI
#include <GASPI.h>
#endif
#include <string.h>

#include "exchange_data_mpi.h"
#include "exchange_data_mpidma.h"
#include "exchange_data_gaspi.h"
#include "read_netcdf.h"
#include "comm_data.h"
#include "solver_data.h"
#include "error_handling.h"
#include "util.h"
#include "rangelist.h"

#define DATAKEY 4711

static void init_communication_data(int iProc, int nProc, comm_data *cd)
{
  ASSERT(cd != NULL);

  /* init comm data */
  cd->nProc = nProc;
  cd->iProc = iProc;

  cd->ndomains = 0;
  cd->ncommdomains = 0;
  cd->nownpoints = 0;
  cd->naddpoints = 0;

  cd->commpartner = NULL;
  cd->sendcount = NULL;
  cd->recvcount = NULL;
  cd->addpoint_owner = NULL;
  cd->addpoint_id = NULL;
  cd->recvindex = NULL;
  cd->sendindex = NULL;

  cd->nreq = 0;
  cd->req = NULL;
  cd->stat = NULL;
  cd->recvbuf = NULL;
  cd->sendbuf = NULL;

  cd->remote_recv_offset = NULL;
  cd->local_recv_offset = NULL;
  cd->local_send_offset = NULL;
  cd->notification = NULL;

  cd->send_flag = NULL;
  cd->recv_flag = NULL;
  cd->send_stage = 0;
  cd->recv_stage = 0;
  cd->comm_stage = 0;

}

void read_communication_data(int ncid, comm_data *cd)
{
  ASSERT(cd != NULL);

  /* read val */
  cd->ndomains = get_nc_val(ncid,"ndomains");
  cd->nownpoints = get_nc_val(ncid,"nownpoints");
 
 /* threading model only */
  if (cd->ndomains == 1)
    {
      return;
    }

  /* read val */
  cd->naddpoints = get_nc_val(ncid,"naddpoints");
  cd->ncommdomains = get_nc_val(ncid,"ncommdomains");

  /* sanity check*/
  ASSERT(cd->ndomains >= 1);
  ASSERT(cd->ndomains == cd->nProc);

  /* sanity check*/
  ASSERT(cd->naddpoints > 0);
  ASSERT(cd->ncommdomains > 0);

  /* alloc */
  cd->commpartner = check_malloc(cd->ncommdomains * sizeof(int));
  cd->sendcount = check_malloc(cd->ndomains * sizeof(int));
  cd->recvcount = check_malloc(cd->ndomains * sizeof(int));
  cd->addpoint_owner = check_malloc(cd->naddpoints * sizeof(int));
  cd->addpoint_id = check_malloc(cd->naddpoints * sizeof(int));

  /* read data */
  get_nc_int(ncid,"commpartner",cd->commpartner);
  get_nc_int(ncid,"sendcount",cd->sendcount);
  get_nc_int(ncid,"recvcount",cd->recvcount);
  get_nc_int(ncid,"addpoint_owner",cd->addpoint_owner);
  get_nc_int(ncid,"addpoint_idx",cd->addpoint_id);

}

static void create_recvsend_index(comm_data *cd)
{
  ASSERT(cd != NULL);

  /* threading model only */
  if (cd->ndomains == 1)
    {
      return;
    }

  int i;
  const int nown   = cd->nownpoints;
  const int nadd   = cd->naddpoints;
  const int nProc  = cd->nProc;
  const int iProc  = cd->iProc;

  ASSERT(cd != NULL);
  ASSERT(cd->ndomains >= 1);
  ASSERT(cd->ncommdomains != 0);
  ASSERT(cd->sendcount != NULL);
  ASSERT(cd->recvcount != NULL);
  ASSERT(cd->addpoint_id != NULL);
  ASSERT(cd->addpoint_owner != NULL);

  /* alloc index tables */
  cd->sendindex = check_malloc(nProc * sizeof(int *));
  cd->recvindex = check_malloc(nProc * sizeof(int *));

  for(i = 0; i < nProc; i++)
    {
      cd->sendindex[i] = NULL;
      cd->recvindex[i] = NULL;
    }

  for(i = 0; i < cd->ncommdomains; i++)
    {
      int k = cd->commpartner[i];
      int j;
      if(cd->sendcount[k] > 0)
	{
	  cd->sendindex[k] = check_malloc(cd->sendcount[k] * sizeof(int));
	  for(j = 0; j < cd->sendcount[k]; j++)
	    {
	      cd->sendindex[k][j] = -1;
	    }
	}

      if(cd->recvcount[k] > 0)
	{
	  int count = 0;
	  cd->recvindex[k] = check_malloc(cd->recvcount[k] * sizeof(int));
	  for(j = 0; j < nadd; j++)
	    {
	      if(cd->addpoint_owner[j] == k)
		{
		  cd->recvindex[k][count++] = nown + j;
		}
	    }
	}
    }

  size_t sz = 0;
  for(i = 0; i < cd->ncommdomains; i++)
    {
      int k = cd->commpartner[i];
      sz = MAX(sz, (size_t)cd->sendcount[k]);
      sz = MAX(sz, (size_t)cd->recvcount[k]);
    }

  int *ibuf = check_malloc(sz * sizeof(int));
  for(i = 0; i < cd->ncommdomains; i++)
    {      
      int k          = cd->commpartner[i]; 
      int recvcount  = cd->recvcount[k];
      int sendcount  = cd->sendcount[k];
      int *recvindex = cd->recvindex[k];
      int *sendindex = cd->sendindex[k];
      int j;
          
      if(k > iProc) /* first send */
	{
	  for(j = 0; j < recvcount; j++)
	    {	      
	      int idx = recvindex[j] - nown;
	      ibuf[j] = cd->addpoint_id[idx]; 
	    }

	  MPI_Send(ibuf
		   , recvcount * sizeof(int)
		   , MPI_BYTE
		   , k
		   , DATAKEY
		   , MPI_COMM_WORLD
		   );
	  MPI_Recv(ibuf
		   , sendcount * sizeof(int)
		   , MPI_BYTE
		   , k
		   , DATAKEY
		   , MPI_COMM_WORLD
		   , MPI_STATUS_IGNORE
		   );

	  for(j = 0; j < sendcount; j++)
	    {
	      sendindex[j] = ibuf[j];
	    }
	}
      else  /* first receive */
	{
	  MPI_Recv(ibuf
		   , sendcount * sizeof(int)
		   , MPI_BYTE
		   , k
		   , DATAKEY
		   , MPI_COMM_WORLD
		   , MPI_STATUS_IGNORE
		   );
	  for(j = 0; j < sendcount; j++)
	    {
	      sendindex[j] = ibuf[j];
	    }
	  for(j = 0; j < recvcount; j++)
	    {
	      int idx = recvindex[j] - nown;
	      ibuf[j] = cd->addpoint_id[idx];
	    }
	  MPI_Send(ibuf
		   , recvcount * sizeof(int)
		   , MPI_BYTE
		   , k
		   , DATAKEY
		   , MPI_COMM_WORLD
		   );
	}
    }

  check_free(ibuf);

}

void init_communication(int argc, char *argv[], comm_data *cd)
{
  ASSERT(cd != NULL);

  /* MPI init */
  int nProc, iProc;

#ifdef USE_MPI_MULTI_THREADED
  int provided, required = MPI_THREAD_MULTIPLE;
  MPI_Init_thread(&argc, &argv, required, &provided);
  ASSERT(provided >= MPI_THREAD_MULTIPLE);
#else
  int provided, required = MPI_THREAD_SERIALIZED;
  MPI_Init_thread(&argc, &argv, required, &provided);
  ASSERT(provided >= MPI_THREAD_SERIALIZED);
#endif

  MPI_Comm_size(MPI_COMM_WORLD, &nProc);
  MPI_Comm_rank(MPI_COMM_WORLD, &iProc);

#ifdef DEBUG
  printf("Hello from rank: %8d of numranks: %8d (MPI)\n",iProc,nProc);
  fflush(stdout);
#endif

  init_communication_data(iProc, nProc, cd);

#ifdef USE_GASPI
  /* threading model only */
  if (nProc == 1)
    {      
      return;
    }

  /* GASPI init */
  gaspi_rank_t nProcGASPI, iProcGASPI;
  SUCCESS_OR_DIE(gaspi_proc_init(GASPI_BLOCK));
  SUCCESS_OR_DIE(gaspi_proc_rank(&iProcGASPI));
  SUCCESS_OR_DIE(gaspi_proc_num(&nProcGASPI));
  ASSERT(iProcGASPI == iProc);
  ASSERT(nProcGASPI == nProc);

#ifdef DEBUG
  printf("Hello from rank: %8d of numranks: %8d (GASPI)\n",iProcGASPI,nProcGASPI);
  fflush(stdout);
#endif
#endif



}

static void compute_offset_tables(comm_data *cd)
{
  ASSERT(cd != NULL);

  /* threading model only */
  if (cd->ndomains == 1)
    {
      return;
    }

  int i;
  ASSERT(cd != NULL);
  ASSERT(cd->ndomains >= 1);
  ASSERT(cd->ncommdomains != 0);
  ASSERT(cd->sendcount != NULL);
  ASSERT(cd->recvcount != NULL);

  const int nProc  = cd->nProc;
  const int iProc  = cd->iProc;

  /* alloc offset tables */
  cd->local_recv_offset  
    = check_malloc(nProc * sizeof(gaspi_offset_t));
  cd->local_send_offset  
    = check_malloc(nProc * sizeof(gaspi_offset_t));
  cd->remote_recv_offset 
    = check_malloc(nProc * sizeof(gaspi_offset_t));
  cd->notification       
    = check_malloc(nProc * sizeof(gaspi_notification_id_t));

  const int max_elem_sz = NGRAD * 3;
  const size_t szd = sizeof(double);

  /* determine offsets */
  gaspi_offset_t ssz = 0;
  gaspi_offset_t rsz = 0;
  for(i = 0; i < cd->ncommdomains; i++)
    {
      int k = cd->commpartner[i];      
      cd->local_send_offset[k] = ssz;
      cd->local_recv_offset[k] = rsz;
      ssz += cd->sendcount[k] * max_elem_sz * szd;
      rsz += cd->recvcount[k] * max_elem_sz * szd;	  
    }

  /* mutual exchange of recv offsets */
  for(i = 0; i < cd->ncommdomains; i++)
    {      
      int k = cd->commpartner[i]; 
	  
      if(k > iProc) /* first send */
	{
	  MPI_Send(&(cd->local_recv_offset[k]) 
		   , sizeof(gaspi_offset_t)
		   , MPI_BYTE
		   , k
		   , DATAKEY
		   , MPI_COMM_WORLD
		   );
	  MPI_Recv(&(cd->remote_recv_offset[k])
		   , sizeof(gaspi_offset_t)
		   , MPI_BYTE
		   , k
		   , DATAKEY
		   , MPI_COMM_WORLD
		   , MPI_STATUS_IGNORE
		   );
	}
      else  /* first receive */
	{
	  MPI_Recv(&(cd->remote_recv_offset[k])
		   , sizeof(gaspi_offset_t)
		   , MPI_BYTE
		   , k
		   , DATAKEY
		   , MPI_COMM_WORLD
		   , MPI_STATUS_IGNORE
		   );

	  MPI_Send(&(cd->local_recv_offset[k])
		   , sizeof(gaspi_offset_t)
		   , MPI_BYTE
		   , k
		   , DATAKEY
		   , MPI_COMM_WORLD
		   );
	}
    }

  /* mutual exchange of notification */
  for(i = 0; i < cd->ncommdomains; i++)
    {      
      int k = cd->commpartner[i]; 
      gaspi_notification_id_t nfct = i;
	  
      if(k > iProc) /* first send */
	{
	  MPI_Send(&nfct 
		   , sizeof(gaspi_notification_id_t)
		   , MPI_BYTE
		   , k
		   , DATAKEY
		   , MPI_COMM_WORLD
		   );
	  MPI_Recv(&(cd->notification[k])
		   , sizeof(gaspi_notification_id_t)
		   , MPI_BYTE
		   , k
		   , DATAKEY
		   , MPI_COMM_WORLD
		   , MPI_STATUS_IGNORE
		   );
	}
      else  /* first receive */
	{
	  MPI_Recv(&(cd->notification[k])
		   , sizeof(gaspi_notification_id_t)
		   , MPI_BYTE
		   , k
		   , DATAKEY
		   , MPI_COMM_WORLD
		   , MPI_STATUS_IGNORE
		   );

	  MPI_Send(&nfct
		   , sizeof(gaspi_notification_id_t)
		   , MPI_BYTE
		   , k
		   , DATAKEY
		   , MPI_COMM_WORLD
		   );
	}
    }

}


void compute_communication_tables(comm_data *cd)
{
  ASSERT(cd != NULL);

  /* threading model only */
  if (cd->ndomains == 1)
    {
      return;
    }

  ASSERT(cd != NULL);
  ASSERT(cd->naddpoints != 0);
  ASSERT(cd->addpoint_owner != NULL);
  ASSERT(cd->addpoint_id != NULL);
  ASSERT(cd->commpartner != NULL);
  ASSERT(cd->sendcount != NULL);
  ASSERT(cd->recvcount != NULL);

  create_recvsend_index(cd);

#ifdef DEBUG
  int i;
  for(i = 0; i < cd->ncommdomains; i++)
    {
      int k = cd->commpartner[i]; 
      printf(" rank %8d: send %8d to   %8d\n", 
	     cd->iProc, cd->sendcount[k], k);
      printf(" rank %8d: recv %8d from %8d\n", 
	     cd->iProc, cd->recvcount[k], k);
    }
#endif

  compute_offset_tables(cd);

#ifdef DEBUG
  for(i = 0; i < cd->ncommdomains; i++)
    {
      int k = cd->commpartner[i]; 
      printf(" rank %8d: to %8d sendcount: %8d remote_recv_offset: %lu notification: %d\n", 
	     cd->iProc, k, cd->sendcount[k], cd->remote_recv_offset[k],cd->notification[k]);
    }
#endif

  const int max_elem_sz = NGRAD * 3;

  /* allocate requests, statuses and buffer */
  init_mpi_requests(cd, max_elem_sz);

#ifdef USE_GASPI
  /* allocate gaspi segments and lock */
  init_gaspi_segments(cd, max_elem_sz);
#endif  

  /* allocate buffers and window for MPI DMA */
  init_mpidma_buffers(cd, max_elem_sz);

}


void free_communication_ressources(comm_data *cd)
{	
  ASSERT(cd != NULL);

  /* threading model only */
  if (cd->ndomains == 1)
    {
      return;
    }

  MPI_Barrier(MPI_COMM_WORLD);
  free_mpidma_win();

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

}
