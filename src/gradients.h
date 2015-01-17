#ifndef GRADIENTS_H
#define GRADIENTS_H

#include "comm_data.h"
#include "solver_data.h"

void compute_gradients_gg_comm_free(comm_data *cd, solver_data *sd, int final);

void compute_gradients_gg_mpi_bulk_sync(comm_data *cd, solver_data *sd, int final);

void compute_gradients_gg_mpi_early_recv(comm_data *cd, solver_data *sd, int final);

void compute_gradients_gg_gaspi_bulk_sync(comm_data *cd, solver_data *sd, int final);

void compute_gradients_gg_mpi_async(comm_data *cd, solver_data *sd, int final);

void compute_gradients_gg_gaspi_async(comm_data *cd, solver_data *sd, int final);

void compute_gradients_gg_mpifence_async(comm_data *cd, solver_data *sd, int final);

void compute_gradients_gg_mpifence_bulk_sync(comm_data *cd, solver_data *sd, int final);

void compute_gradients_gg_mpipscw_async(comm_data *cd, solver_data *sd, int final);

void compute_gradients_gg_mpipscw_bulk_sync(comm_data *cd, solver_data *sd, int final);

#endif
