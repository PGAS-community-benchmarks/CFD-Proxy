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

#include "waitsome.h"
#include "error_handling.h"

void wait_or_die( gaspi_segment_id_t segment_id
		  , gaspi_notification_id_t notification_id
		  , gaspi_notification_t expected
		  )
{
  gaspi_notification_id_t id;

  SUCCESS_OR_DIE
    (gaspi_notify_waitsome (segment_id, notification_id, 1, &id, GASPI_BLOCK));

  ASSERT (id == notification_id);

  gaspi_notification_t value;

  SUCCESS_OR_DIE (gaspi_notify_reset (segment_id, id, &value));

  ASSERT (value == expected);
}

#endif
