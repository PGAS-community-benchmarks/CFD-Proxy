#ifdef USE_GASPI
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

#include <stdio.h>
#include <stdlib.h>

#include "queue.h"
#include "error_handling.h"

void wait_for_queue_max_half (gaspi_queue_id_t* queue)
{
  gaspi_number_t queue_size_max;
  gaspi_number_t queue_size;

  SUCCESS_OR_DIE (gaspi_queue_size_max (&queue_size_max));
  SUCCESS_OR_DIE (gaspi_queue_size (*queue, &queue_size));

  if (queue_size >= queue_size_max/2)
    {
      SUCCESS_OR_DIE (gaspi_wait (*queue, GASPI_BLOCK));
    }

}
#endif
