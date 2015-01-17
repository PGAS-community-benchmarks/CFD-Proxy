#ifndef EVAL_H
#define EVAL_H

#ifdef USE_GASPI
#include <GASPI.h>
#endif
#include <omp.h>
#include <stdbool.h>

#include "comm_data.h"
#include "solver_data.h"
#include "error_handling.h"


void eval_thread_comm(comm_data *cd);

void eval_thread_rangelist(solver_data *sd);

#endif
