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
#include <netcdf.h>
#include <mpi.h>

#include "solver.h"
#include "comm_data.h"
#include "solver_data.h"
#include "read_netcdf.h"
#include "rangelist.h"
#include "error_handling.h"
#include "util.h"

int main(int argc, char *argv[])
{ 
  /* loop indexes, and error handling. */
  int retval, ncid;
  comm_data cd;
  solver_data sd;

  if (argc != 4 || strcmp(argv[1],"-lvl") != 0)
    {
      printf("Usage: %s -lvl [1-4] GRID_PREFIX\n",argv[0]);
      exit(EXIT_FAILURE);
    }


#ifndef USE_NTHREADS
#warning USE_NTHREADS undefined
  int NTHREADS;
  char* OMP_NUM_THREADS = getenv("OMP_NUM_THREADS");
  ASSERT(OMP_NUM_THREADS != NULL);
  NTHREADS = atoi(OMP_NUM_THREADS);
#else
  const int NTHREADS = USE_NTHREADS;
  omp_set_num_threads(NTHREADS);
#endif


  /* init communication */
  init_communication(argc, argv, &cd);

  /* open the file */
  char fname[80] = "";
  sprintf(fname, "%s_domain_%d_lvl_%d"
	  ,argv[3]
	  ,cd.iProc
	  ,atoi(argv[2])
	  );

  ASSERT (f_exist(fname));
  if ((retval = nc_open(fname, NC_NOWRITE, &ncid)))
    ERR(retval);

  /* read solver data */
  read_solver_data(ncid, &sd);

  /* init solver */
  const int NITER = 25;
  init_solver_data(&sd, NITER);

  /* read comm data */
  read_communication_data(ncid, &cd);

  /* compute comm tables */
  compute_communication_tables(&cd);

  /* init thread range, rangelist */
  init_threads(&cd, &sd, NTHREADS);

  /* run solver */
  test_solver(&cd, &sd, NTHREADS);

  /* free comm ressources */
  free_communication_ressources(&cd);

  /* close file. */
  if ((retval = nc_close(ncid)))
    ERR(retval);

  if (cd.iProc == 0)
    {
      printf("*** SUCCESS\n");
      fflush(stdout);
    }

  return 0;
}
