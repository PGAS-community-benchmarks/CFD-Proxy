#ifndef THREAD_COMM_H
#define THREAD_COMM_H

#ifdef USE_GASPI
#include <GASPI.h>
#endif
#include <omp.h>
#include <stdbool.h>

#include "comm_data.h"
#include "solver_data.h"
#include "error_handling.h"


void init_thread_comm(comm_data *cd
		      , solver_data *sd
		      );

#endif
