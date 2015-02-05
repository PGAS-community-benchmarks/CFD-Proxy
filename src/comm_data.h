#ifndef COMM_DATA_H
#define COMM_DATA_H

#include <mpi.h>

#ifdef USE_GASPI
#include <GASPI.h>
#else
typedef unsigned long gaspi_offset_t;
typedef unsigned short gaspi_notification_id_t;
#endif

#include "solver_data.h"

typedef struct 
{

  int nProc;
  int iProc;

  int ndomains;
  int ncommdomains;
  int nownpoints;
  int naddpoints;

  /* general comm */
  int *addpoint_owner;
  int *addpoint_id;
  int *commpartner;
  int *sendcount;
  int *recvcount;
  int **recvindex;
  int **sendindex;

  /* comm vars mpi */
  int nreq;
  MPI_Request *req;
  MPI_Status *stat;
  double **recvbuf;
  double **sendbuf;

  /* offset vars gaspi */
  gaspi_offset_t *remote_recv_offset;
  gaspi_offset_t *local_recv_offset;
  gaspi_offset_t *local_send_offset;
  gaspi_notification_id_t *notification;

  /* global stage counter */
  volatile counter_t *recv_flag;
  volatile counter_t *send_flag;
  volatile int recv_stage;
  volatile int send_stage;
  volatile int comm_stage;

} comm_data ;


void init_communication(int argc, char *argv[], comm_data *cd);
void read_communication_data(int ncid, comm_data *cd);
void compute_communication_tables(comm_data *cd);
void free_communication_ressources(comm_data *cd);

#endif
