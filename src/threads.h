#ifndef THREADS_H
#define THREADS_H

#include <omp.h>
#include <stdbool.h>

#include "comm_data.h"
#include "solver_data.h"
#include "error_handling.h"


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
 
void initiate_thread_comm_mpifence(RangeList *color
				   , comm_data *cd
				   , double *data
				   , int dim2
				   );


void initiate_thread_comm_mpipscw(RangeList *color
				  , comm_data *cd
				  , double *data
				  , int dim2
				  );

int my_add_and_fetch(volatile int *ptr, int val);
int my_fetch_and_add(volatile int *ptr, int val);

int this_is_the_first_thread(void);
int this_is_the_last_thread(void);

/* getter functions for global increments */
int get_inc_send(int i);
int set_inc_send(int i, int val);

/* getter functions for stage counters */
int get_thread_stage(int i);
int inc_thread_stage(int i, int val);

int  get_thread_stage_local(int i);
void inc_thread_stage_local(int i, int val);

int get_sendcount_local(int i);
int set_inc_send_local(int i, int val);

static __inline void _mm_pause (void)
{
  __asm__ __volatile__ ("rep; nop" : : );
}

#endif
