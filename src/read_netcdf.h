#ifndef NETCDF_H
#define NETCDF_H

void get_nc_double(int ncid, const char *name, double *array);
void get_nc_int(int ncid, const char *name, int *array);
int  get_nc_val(int ncid, const char *name);

#endif
