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

#include <sys/stat.h>
#include <unistd.h>

#include "error_handling.h"

int f_exist (char *fname)
{
  struct stat sb;   
  return (stat (fname, &sb) == 0);
}

