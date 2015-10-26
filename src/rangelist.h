#ifndef RANGELIST_H
#define RANGELIST_H

#ifdef USE_GASPI
#include <GASPI.h>
#endif
#include <omp.h>
#include <stdbool.h>


#include "threads.h"
#include "comm_data.h"
#include "solver_data.h"
#include "error_handling.h"


void init_threads(comm_data *cd
		  , solver_data *sd
		  , int NTHREADS
		  );

void init_rangelist(RangeList *fcolor);


void init_thread_neighbours(comm_data *cd
			    , solver_data *sd
			    , int tid
			    , int *pid
			    );

void init_comp_stage_global(int nthreads);


 
void init_thread_rangelist(comm_data *cd
			   , solver_data *sd
			   , int tid
			   , int *pid
			   , int *htype
			   );

void init_halo_type(int *htype
		    , comm_data *cd
		    , solver_data *sd
		    );

void init_thread_id(int NTHREADS
		    , int *pid
		    , solver_data *sd
		    );

int get_ncolors_local(void);

solver_data_local* get_solver_local(void);

RangeList* private_get_color(RangeList *const prev);

static inline RangeList* get_color(void)
{
  return private_get_color(NULL);
}

static inline RangeList* get_next_color(RangeList *const prev)
{
  return private_get_color(prev);
}

RangeList* private_get_color_and_exchange(RangeList *const prev
					  , send_fn send
					  , exch_fn exch
					  , comm_data *cd
					  , double *data
					  , int dim2
					  , int final
					  );

static inline RangeList* get_color_and_exchange(send_fn send
						, exch_fn exch
						, comm_data *cd
						, double *data
						, int dim2
						, int final				   
						)
{
  return private_get_color_and_exchange(NULL
					, send
					, exch
					, cd
					, data
					, dim2
					, final
					);
}

static inline RangeList* get_next_color_and_exchange(RangeList *const prev
						     , send_fn send
						     , exch_fn exch
						     , comm_data *cd
						     , double *data
						     , int dim2
						     , int final
						     )
{
  return private_get_color_and_exchange(prev
					, send
					, exch
					, cd
					, data
					, dim2
					, final
					);
}


#endif
