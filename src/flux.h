#ifndef FLUX_H
#define FLUX_H

#include "comm_data.h"
#include "solver_data.h"

#define IVX 0
#define IVY 1
#define IVZ 2


void compute_psd_flux(solver_data *sd);

#endif
