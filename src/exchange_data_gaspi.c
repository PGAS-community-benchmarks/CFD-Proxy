#ifdef USE_GASPI
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
#include <GASPI.h>

#include "exchange_data_gaspi.h"
#include "solver_data.h"
#include "comm_data.h"
#include "rangelist.h"
#include "threads.h"
#include "waitsome.h"
#include "queue.h"
#include "util.h"

#include "error_handling.h"

#ifdef USE_GASPI_TEST
#ifndef USE_GASPI
#error GASPI_TEST requires GASPI
#endif
int *testsome_local = NULL;
#pragma omp threadprivate(testsome_local)
#endif

void init_gaspi_segments(comm_data *cd
			 , int dim2
			 )
{

  int i, j;  
  const int max_elem_sz = NGRAD * 3;
  const size_t szd = sizeof(double);
  ASSERT(dim2 == max_elem_sz);  

  gaspi_number_t snum = 0; 
  SUCCESS_OR_DIE(gaspi_segment_num(&snum));
  ASSERT(snum == 0);  

  gaspi_size_t rsz = 0, ssz = 0;
  for(i = 0; i < cd->ncommdomains; i++)
    {
      int k = cd->commpartner[i];
      ssz +=  cd->sendcount[k] * max_elem_sz * szd;
      rsz +=  cd->recvcount[k] * max_elem_sz * szd;
    }  

#ifdef DEBUG
  printf("GASPI segment size rsz: %lu ssz: %lu\n",rsz,ssz);
  fflush(stdout);
#endif
  
  for(j = 0; j < 2; j++)
    {      
      /* create gaspi send segments for gradients */ 
      SUCCESS_OR_DIE(gaspi_segment_create(j
			   , ssz
			   , GASPI_GROUP_ALL
			   , GASPI_BLOCK
			   , GASPI_ALLOC_DEFAULT
					  ));
      
      /* create gaspi recv segments for gradients */ 
      SUCCESS_OR_DIE(gaspi_segment_create(2+j
			   , rsz
			   , GASPI_GROUP_ALL
			   , GASPI_BLOCK
			   , GASPI_ALLOC_DEFAULT
					  ));
    }


#ifdef DEBUG
  printf("GASPI segment creation complete.\n");
  fflush(stdout);
#endif


#ifdef USE_GASPI_TEST
#pragma omp parallel default (none) shared(cd)
  {
    if (testsome_local == NULL)
      {
	testsome_local = check_malloc(cd->ncommdomains * sizeof(int));
      }
  }
#endif

}



void exchange_dbl_gaspi_write(comm_data *cd
			      , double *data
			      , int dim2
			      , int buffer_id
			      , int i)
{
  int *commpartner  = cd->commpartner;
  int *sendcount    = cd->sendcount;

  gaspi_queue_id_t queue_id = 0;
  gaspi_offset_t *local_send_offset     = cd->local_send_offset;
  gaspi_offset_t *remote_recv_offset    = cd->remote_recv_offset;
  gaspi_notification_id_t *notification = cd->notification;

  int count;
  size_t szd = sizeof(double);

  int k = commpartner[i];
  count = sendcount[k];

  ASSERT(data != NULL);
 
  if(count > 0)
    {

      gaspi_size_t size = count * dim2 * szd;

      // issue write
      wait_for_queue_max_half (&queue_id);
      SUCCESS_OR_DIE ( gaspi_write_notify
		       ( buffer_id
			 , local_send_offset[k]
			 , k 
			 , 2+buffer_id
			 , remote_recv_offset[k]
			 , size
			 , notification[k]
			 , 1
			 , queue_id
			 , GASPI_BLOCK
			 ));
      /* inc send flag */
      cd->send_flag[i].global++;

    }

}

static void exchange_dbl_gaspi_scatter(comm_data *cd
				       , double *data
				       , int dim2
				       )
{
  int ncommdomains  = cd->ncommdomains;
  int *commpartner  = cd->commpartner;
  int *recvcount    = cd->recvcount;
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

	  int k = commpartner[i];
	  ASSERT(recvcount[k] > 0);	 

	  /* copy the data from the recvbuf into out data field */
	  int buffer_id = cd->recv_stage % 2;	  
	  gaspi_pointer_t ptr;
	  SUCCESS_OR_DIE(gaspi_segment_ptr(2+buffer_id, &ptr));		  
	  double *rbuf = (double *) ((char *) ptr + local_recv_offset[k]);		  
	  exchange_dbl_copy_out_local(rbuf, data, dim2, i);	  
	} 
    }
}

void exchange_dbl_gaspi_bulk_sync(comm_data *cd
				  , double *data
				  , int dim2
				  , int final
				  )
{
  int ncommdomains  = cd->ncommdomains;
  int *commpartner  = cd->commpartner;
  int *recvcount    = cd->recvcount;
  int *sendcount    = cd->sendcount;

  gaspi_offset_t *remote_recv_offset    = cd->remote_recv_offset;
  gaspi_offset_t *local_recv_offset     = cd->local_recv_offset;
  gaspi_offset_t *local_send_offset     = cd->local_send_offset;

  int i;

  ASSERT(dim2 > 0);
  ASSERT(ncommdomains != 0);

  ASSERT(recvcount != NULL);
  ASSERT(sendcount != NULL);

  ASSERT(remote_recv_offset != NULL);
  ASSERT(local_recv_offset != NULL);
  ASSERT(local_send_offset != NULL);
  ASSERT((final == 0 || final == 1));

  gaspi_number_t snum = 0; 
  SUCCESS_OR_DIE(gaspi_segment_num(&snum));
  ASSERT(snum == 4);

  /* wait for completed computation before send */
  if (this_is_the_last_thread())
    {
      for(i = 0; i < ncommdomains; i++)
	{
	  int k = commpartner[i];
	  if (sendcount[k] > 0)
	    {
	      int buffer_id = cd->send_stage % 2;
#if !defined(USE_PACK_IN_BULK_SYNC) && !defined(USE_PARALLEL_GATHER)
	      gaspi_pointer_t ptr;
	      SUCCESS_OR_DIE(gaspi_segment_ptr(buffer_id, &ptr));		  
	      double *sbuf = (double *) ((char *) ptr + local_send_offset[k]);		  
	      exchange_dbl_copy_in(cd
				   , sbuf
				   , data
				   , dim2
				   , i
				   );
#endif
	      exchange_dbl_gaspi_write(cd, data, dim2, buffer_id, i);
	    }
	}

      /* wait for all notify */
      for(i = 0; i < ncommdomains; i++)
	{ 
	  gaspi_notification_id_t id;
	  gaspi_notification_t value = 0;	  	  
	  int buffer_id = cd->recv_stage % 2;	  
	  SUCCESS_OR_DIE(gaspi_notify_waitsome (2+buffer_id
						, 0
						, ncommdomains
						, &id
						, GASPI_BLOCK
						));
          SUCCESS_OR_DIE (gaspi_notify_reset (2+buffer_id
                                              , id
                                              , &value
                                              ));
          ASSERT(value != 0);

	  int k = commpartner[id];	  
	  ASSERT(recvcount[k] > 0);	 

	  // flag received buffer 
	  cd->recv_flag[id].global++;

#ifndef USE_PARALLEL_SCATTER
	  /* copy the data from the recvbuffer into out data field */
	  gaspi_pointer_t ptr;
	  SUCCESS_OR_DIE(gaspi_segment_ptr(2+buffer_id, &ptr));		  
	  double *rbuf = (double *) ((char *) ptr + local_recv_offset[k]);		  
	  exchange_dbl_copy_out(cd
				, rbuf
				, data
				, dim2
				, id
				);
#endif
	}
    }

#ifdef USE_PARALLEL_SCATTER
  exchange_dbl_gaspi_scatter(cd, data, dim2);
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


void exchange_dbl_gaspi_async(comm_data *cd
			      , double *data
			      , int dim2
			      , int final
			      )
{

  int ncommdomains  = cd->ncommdomains;
  int *commpartner  = cd->commpartner;
  int *recvcount    = cd->recvcount;

  gaspi_offset_t *local_recv_offset = cd->local_recv_offset;

  ASSERT(dim2 > 0);
  ASSERT(ncommdomains != 0);

  ASSERT(recvcount != NULL);
  ASSERT(local_recv_offset != NULL);
  ASSERT((final == 0 || final == 1));

  gaspi_number_t snum = 0; 
  SUCCESS_OR_DIE(gaspi_segment_num(&snum));
  ASSERT(snum == 4);

  int i;      

  /*----------------------------------------------------------------------
  |   Note: GASPI leverages parallel scatter by default. The GASPI API 
  |   uses a multithreaded evaluation of notifications. No flags for 
  |   the received data hence are required, instead notifications are 
  |   actually are flags for received data.
  ----------------------------------------------------------------------*/

#ifdef USE_PARALLEL_SCATTER
#ifdef USE_GASPI_TEST

  ASSERT(testsome_local != NULL);
  for (i = 0; i < ncommdomains; ++i)
    {
      testsome_local[i] = 0;
    }

  int count = 0;
  int nrecvcount = get_nrecvcount_local();
  while (count < nrecvcount)
    {
      for (i = 0; i < ncommdomains; ++i)
	{
	  gaspi_notification_id_t nid, id = i;
	  int nrecv = get_recvcount_local(i);
	  if (nrecv > 0 && testsome_local[i] == 0)
	    {
	      gaspi_return_t ret;
	      int buffer_id = cd->recv_stage % 2;	
	      if ((ret = gaspi_notify_waitsome (2+buffer_id
						, id
						, 1
						, &nid
						, GASPI_TEST
						)) == GASPI_SUCCESS
		  )
		{
		  ASSERT (id == nid);
		  int k = commpartner[i];
		  ASSERT(recvcount[k] > 0);	 
	      
		  /* copy the data from the recvbuffer into out data field */
		  gaspi_pointer_t ptr;
		  SUCCESS_OR_DIE(gaspi_segment_ptr(2+buffer_id, &ptr));		  
		  double *rbuf = (double *) ((char *) ptr + local_recv_offset[k]);		  
		  exchange_dbl_copy_out_local(rbuf
					      , data
					      , dim2
					      , i
					      );
		  testsome_local[i] = 1;
		  count++;
		}
	    }
	}
    }
#else /* USE_PARALLEL_SCATTER && USE_GASPI_TEST */
  for (i = 0; i < ncommdomains; ++i)
    {
      gaspi_notification_id_t nid, id = i;
      int nrecv = get_recvcount_local(i);
      if (nrecv > 0)
	{
	  int buffer_id = cd->recv_stage % 2;	
	  SUCCESS_OR_DIE(gaspi_notify_waitsome (2+buffer_id
						, id
						, 1
						, &nid
						, GASPI_BLOCK
						));

	  int k = commpartner[i];
	  ASSERT(recvcount[k] > 0);	 
	      
	  /* copy the data from the recvbuffer into out data field */
	  gaspi_pointer_t ptr;
	  SUCCESS_OR_DIE(gaspi_segment_ptr(2+buffer_id, &ptr));		  
	  double *rbuf = (double *) ((char *) ptr + local_recv_offset[k]);		  
	  exchange_dbl_copy_out_local(rbuf
				      , data
				      , dim2
				      , i
				      );
	}
    }
#endif /* USE_PARALLEL_SCATTER && !USE_GASPI_TEST */

  if (this_is_the_last_thread())
    {
      for (i = 0; i < ncommdomains; ++i)
	{
	  gaspi_notification_id_t id = i;
	  gaspi_notification_t value = 0;	  	  
	  int buffer_id = cd->recv_stage % 2;	  

	  // flag received buffer 
	  cd->recv_flag[id].global++;

	  /* .. and reset */
	  SUCCESS_OR_DIE (gaspi_notify_reset (2+buffer_id
					      , id
					      , &value
					      ));
	  ASSERT(value != 0);
	  
	}

      // inc stage counter
      cd->send_stage++;
      cd->recv_stage++;
    }

/* no barrier -- in parallel scatter all threads unpack 
   the specifically required parts of recv */   

#else /* USE_PARALLEL_SCATTER */

  if (this_is_the_first_thread())
    {
      /* buffer_id */
      int buffer_id = cd->recv_stage % 2;
      for (i = 0; i < ncommdomains; ++i)
	{
	  gaspi_notification_id_t id;
	  gaspi_notification_t value = 0;	  	  
	  /* test .. */
	  SUCCESS_OR_DIE(gaspi_notify_waitsome (2+buffer_id
						, 0
						, ncommdomains
						, &id
						, GASPI_BLOCK
						));

          SUCCESS_OR_DIE (gaspi_notify_reset (2+buffer_id
                                              , id
                                              , &value
                                              ));
          ASSERT(value != 0);

	  int k = commpartner[id];	  
	  ASSERT(recvcount[k] > 0);	 

	  // flag received buffer 
	  cd->recv_flag[id].global++;

	  /* copy the data from the recvbuffer into out data field */
	  gaspi_pointer_t ptr;
	  SUCCESS_OR_DIE(gaspi_segment_ptr(2+buffer_id, &ptr));		  
	  double *rbuf = (double *) ((char *) ptr + local_recv_offset[k]);		  
	  exchange_dbl_copy_out(cd
				, rbuf
				, data
				, dim2
				, id
				);
	}
    }
  
  if (this_is_the_last_thread())
    {
      // inc stage counter
      cd->send_stage++;
      cd->recv_stage++;
    }

#pragma omp barrier

#endif /* !USE_PARALLEL_SCATTER */


}

#endif











