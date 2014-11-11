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

#include <netcdf.h>
#include "read_netcdf.h"
#include "error_handling.h"

void get_nc_int(int ncid, const char *name, int *array)
{
  int varid, retval;

  /* Get the varid of the data variable, based on its name. */
  if ((retval = nc_inq_varid(ncid, name, &varid)))
    ERR(retval);
  /* get array */
  if ((retval = nc_get_var_int(ncid, varid, array)))
    ERR(retval);

}

void get_nc_double(int ncid, const char *name, double *array)
{
  int varid, retval;

  /* Get the varid of the data variable, based on its name. */
  if ((retval = nc_inq_varid(ncid, name, &varid)))
    ERR(retval);
  /* get array */
  if ((retval = nc_get_var_double(ncid, varid, array)))
    ERR(retval);

}

int get_nc_val(int ncid, const char *name)
{

  int dimid, retval;
  size_t val;

  /* Get the dimid of the data variable, based on its name. */
  if ((retval = nc_inq_dimid(ncid, name, &dimid)))
    ERR(retval);
  /* get value */
  if ((retval = nc_inq_dimlen(ncid, dimid, &val)))
    ERR(retval);

  return val;

}

