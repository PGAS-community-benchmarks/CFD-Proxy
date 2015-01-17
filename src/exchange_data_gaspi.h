#ifndef EXCHANGE_DATA_GASPI_H
#define EXCHANGE_DATA_GASPI_H

#include "comm_data.h"

void init_gaspi_segments(comm_data *cd
			 , int dim2
			 );

void exchange_dbl_gaspi_bulk_sync(comm_data *cd
				  , double *data
				  , int dim2
				  , int final
				  );

void exchange_dbl_gaspi_async(comm_data *cd
			      , double *data
			      , int dim2
			      , int final
			      );

void exchange_dbl_gaspi_write(comm_data *cd
			      , double *data
			      , int dim2
			      , int buffer_id
			      , int i);

#endif


