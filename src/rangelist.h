#ifndef RANGELIST_H
#define RANGELIST_H

#ifdef USE_GASPI
#include <GASPI.h>
#endif
#include <omp.h>
#include <stdbool.h>

#include "comm_data.h"
#include "solver_data.h"
#include "error_handling.h"


void init_rangelist(RangeList *fcolor);

void init_threads(comm_data *cd
		  , solver_data *sd
		  , int NTHREADS
		  );

void initiate_thread_comm_mpi(RangeList *color
			      , comm_data *cd
			      , double *data
			      , int dim2
			      );

void initiate_thread_comm_gaspi(RangeList *color
				, comm_data *cd
				, double *data
				, int dim2
				);
 
void init_thread_comm(comm_data *cd
		      , solver_data *sd
		      );


void init_thread_rangelist(comm_data *cd
			   , solver_data *sd
			   , int tid
			   , int *pid
			   , int *htype
			   );

void init_thread_meta_data(int *pid
			   , int *htype
			   , comm_data *cd
			   , solver_data *sd
			   , int NTHREADS
			   );

void eval_thread_comm(comm_data *cd);

void eval_thread_rangelist(solver_data *sd);

solver_data_local* get_solver_data(void);

int get_ncolors(void);

RangeList* private_get_color(RangeList *const prev);

static inline RangeList* get_color(void)
{
  return private_get_color(NULL);
}

static inline RangeList* get_next_color(RangeList *const prev)
{
  return private_get_color(prev);
}


#endif
