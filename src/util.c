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
#include <sys/time.h>
#include "error_handling.h"
#include "solver_data.h"
#include "util.h"

void check_free(void *ptr)
{
  if(ptr != NULL)
  {
    free(ptr);
  }

  return;

}

void *check_malloc(size_t bytes)
{
  void *tmp;
  ASSERT(bytes > 0);
  tmp = malloc(bytes);
  ASSERT(tmp != NULL);

  return tmp;
}



void *check_realloc(void *old, size_t bytes)
{
  void *tmp;
  ASSERT(bytes > 0);
  tmp = realloc(old, bytes);
  ASSERT(tmp != NULL);

  return tmp;
}


static void swap(double *a, double *b)
{
  double tmp = *a;
  *a = *b;
  *b = tmp;
}

void sort_median(double *begin, double *end)
{
  double *ptr;
  double *split;
  if (end - begin <= 1)
    return;
  ptr = begin;
  split = begin + 1;
  while (++ptr != end) {
    if (*ptr < *begin) {
      swap(ptr, split);
      ++split;
    }
  }
  swap(begin, split - 1);
  sort_median(begin, split - 1);
  sort_median(split, end);
}


double now()
{
  struct timeval tp;
  int i;
  i = gettimeofday(&tp,NULL);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}




