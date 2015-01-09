#ifndef EXCHANGE_DATA_MPIDMA_H
#define EXCHANGE_DATA_MPIDMA_H

#include "comm_data.h"

void exchange_dbl_mpifence_bulk_sync(comm_data *cd
				     , double *data
				     , int dim2
				     , int final
				     );

void exchange_dbl_mpifence_async(comm_data *cd
				 , double *data
				 , int dim2
				 , int final
				 );

void exchange_dbl_mpipscw_bulk_sync(comm_data *cd
				    , double *data
				    , int dim2
				    , int final
				    );

void exchange_dbl_mpipscw_async(comm_data *cd
				, double *data
				, int dim2
				, int final
				);

void init_mpidma_buffers(comm_data *cd
			 , int dim2
			 );

void exchange_dbl_mpidma_write(comm_data *cd
			      , double *data
			      , int dim2
			      , int i
			       );

void mpidma_async_win_fence(int assertion);

void mpidma_async_post_start(void);

void mpidma_async_complete(void);

void mpidma_async_wait(void);

void free_mpidma_win(void);

void *get_sndbuf(void);

void *get_rcvbuf(void);

#endif


