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
#include <omp.h>
#include <stdbool.h>

#include "read_netcdf.h"
#include "solver_data.h"
#include "error_handling.h"
#include "util.h"
#include "rangelist.h"
#include "threads.h"

static void gather_sendcount(comm_data *cd
			     , solver_data *sd)
{
  int i, j, i0;

  int **offset = check_malloc(sd->nallpoints * sizeof(int*));
  for(i = 0; i < sd->nallpoints; i++)
    {
      offset[i] = NULL;
    }
  /* gather sendindex data per pnt, per target */
  for(i = 0; i < cd->ncommdomains; i++)
    {
      int k = cd->commpartner[i];
      int count = cd->sendcount[k];      
      if(count > 0)
	{
	  for(j = 0; j < count; j++)
	    {
	      int pnt = cd->sendindex[k][j];
	      if (offset[pnt] == NULL)
		{
		  offset[pnt] = check_malloc(cd->ncommdomains * sizeof(int));
		  for(i0 = 0; i0 < cd->ncommdomains; i0++)   
		    {
		      offset[pnt][i0] = -1;
		    }
 		}             
	      // assert initial state per pnt, per target
	      ASSERT(offset[pnt][i] == -1);
	      offset[pnt][i] = j;
	    }
	}
    }

  /* assemble partial sendcounts per color, per target */
#pragma omp parallel default (none) shared(cd, sd\
            , offset, stdout, stderr)
    {
      RangeList *color;
      int  *nsendcount = check_malloc(cd->ncommdomains * sizeof(int));

      for (color = get_color(); color != NULL
	     ; color = get_next_color(color)) 
	{      
	  int  nlast_points_of_color = color->nlast_points_of_color;
	  int  *last_points_of_color = color->last_points_of_color;
	  int i1, i3;	

	  /* gather color specific metadata */
	  for(i1 = 0; i1 < cd->ncommdomains; i1++) 
	    {
	      nsendcount[i1] = 0;
	    }
	  for(i1 = 0; i1 < nlast_points_of_color; i1++) 
	    {
	      int pnt = last_points_of_color[i1];
	      if (offset[pnt] != NULL)
		{
		  for(i3 = 0; i3 < cd->ncommdomains; i3++)
		    {
		      int j1 = offset[pnt][i3];
		      if (j1 != -1) 
			{
			  nsendcount[i3]++;
			}
		    }
		}
	    }
	  int nsend = 0;
	  for(i1 = 0; i1 < cd->ncommdomains; i1++) 
	    {
	      if (nsendcount[i1] > 0)
		{
		  nsend++;
		}
	    }
	  color->nsendcount = nsend;

          /* init sends, color local */
          if (color->nsendcount > 0)
            {
              color->sendpartner = check_malloc(color->nsendcount * sizeof(int));
              color->sendcount   = check_malloc(color->nsendcount * sizeof(int));
              color->sendindex   = check_malloc(color->nsendcount * sizeof(int*));
              color->sendoffset  = check_malloc(color->nsendcount * sizeof(int*));

              int i2 = 0;
              for(i1 = 0; i1 < cd->ncommdomains; i1++) 
                {
                  if (nsendcount[i1] > 0)
                    {
                      color->sendpartner[i2] = i1;
                      color->sendcount[i2]   = nsendcount[i1];
                      color->sendindex[i2]   = check_malloc(nsendcount[i1] * sizeof(int));
                      color->sendoffset[i2]  = check_malloc(nsendcount[i1] * sizeof(int));

                      int i4 = 0;
                      for(i3 = 0; i3 < nlast_points_of_color; i3++) 
                        {
                          int pnt = last_points_of_color[i3];
                          if (offset[pnt] != NULL)
                            {
                              int j1 = offset[pnt][i1];
                              if (j1 != -1) 
                                {
                                  color->sendindex[i2][i4]  = pnt;
				  color->sendoffset[i2][i4] = j1;
				  i4++;
                                }
                            }
                        }
                      ASSERT(i4 == color->sendcount[i2]);
                      i2++;
                    }
                }
              ASSERT(i2 == color->nsendcount);
            }
	}
      check_free(nsendcount);
    }

  for(i = 0; i < sd->nallpoints; i++)
    {
      if (offset[i] != NULL)
	{
	  check_free(offset[i]);
	}
    }
  check_free(offset);


  /* flag for testing color sendindex */
  int *flag = check_malloc(sd->nallpoints * sizeof(int));
  for(i = 0; i < sd->nallpoints; i++)
    {
      flag[i] = 0;
    }

  /* testing for multiple sends per pnt  */
  for(i = 0; i < cd->ncommdomains; i++)
    {
      int k = cd->commpartner[i];
      int count = cd->sendcount[k];      
      if(count > 0)
	{
	  for(j = 0; j < count; j++)
	    {
	      int pnt = cd->sendindex[k][j];
	      flag[pnt] -= 1;
	    }
	}
    }

  /* assemble partial sendcounts per color, per target */
#pragma omp parallel default (none) shared(cd, sd\
           , flag, stdout, stderr)
    {
      RangeList *color;
      for (color = get_color(); color != NULL
	     ; color = get_next_color(color)) 
	{      
	  int i2, i4;	
	  for(i2 = 0; i2 < color->nsendcount; i2++) 
	    {
	      for(i4 = 0; i4 < color->sendcount[i2]; i4++) 
		{
		  int pnt = color->sendindex[i2][i4];
		  flag[pnt] += 1;
		}
	    }
	}
    }

  for(i = 0; i < sd->nallpoints; i++)
    {
      ASSERT(flag[i] == 0);
    }
  check_free(flag);


}

static void gather_recvcount(comm_data *cd
			     , solver_data *sd)
{
  int i, j;

  int *offset = check_malloc(sd->nallpoints * sizeof(int));
  int *owner  = check_malloc(sd->nallpoints * sizeof(int));
  for(i = 0; i < sd->nallpoints; i++)
    {
      offset[i] = -1;
      owner[i] = -1;
    }

  /* gather recvindex data per pnt - similar to addpoint owner */
  for(i = 0; i < cd->ncommdomains; i++)
    {
      int k = cd->commpartner[i];
      if(cd->recvcount[k] > 0)
	{
	  for(j = 0; j < cd->recvcount[k]; j++)
	    {
	      int pnt = cd->recvindex[k][j];
	      ASSERT(pnt >= sd->nownpoints);
	      ASSERT(offset[pnt] == -1);
	      ASSERT(owner[pnt] == -1);
	      offset[pnt] = j;	      
	      owner[pnt] = i;
	    }
	}
    }


  /* assemble partial recvcounts per color, per target */
#pragma omp parallel default (none) shared(cd, sd			\
					   , offset, owner, stdout, stderr)
  {
    RangeList *color;
    int  *nrecvcount = check_malloc(cd->ncommdomains * sizeof(int));

    for (color = get_color(); color != NULL
	   ; color = get_next_color(color)) 
      {      
	int  nall_points_of_color = color->nall_points_of_color;
	int  *all_points_of_color = color->all_points_of_color;
	int i1, i3;	

	/* gather color specific metadata */
	for(i1 = 0; i1 < cd->ncommdomains; i1++) 
	  {
	    nrecvcount[i1] = 0;
	  }
	for(i1 = 0; i1 < nall_points_of_color; i1++) 
	  {
	    int pnt = all_points_of_color[i1];
	    if (pnt >= cd->nownpoints)
	      {
		ASSERT(offset[pnt] != -1);
		ASSERT(owner[pnt] != -1);
		i3 = owner[pnt];
		nrecvcount[i3]++;		  
	      }
	  }

	int nrecv = 0;
	for(i1 = 0; i1 < cd->ncommdomains; i1++) 
	  {
	    if (nrecvcount[i1] > 0)
	      {
		nrecv++;
	      }
	  }
	color->nrecvcount = nrecv;

	/* init recvs, color local */
	if (color->nrecvcount > 0)
	  {
	    color->recvpartner = check_malloc(color->nrecvcount * sizeof(int));
	    color->recvcount   = check_malloc(color->nrecvcount * sizeof(int));
	    color->recvindex   = check_malloc(color->nrecvcount * sizeof(int*));
	    color->recvoffset  = check_malloc(color->nrecvcount * sizeof(int*));

	    int i2 = 0;
	    for(i1 = 0; i1 < cd->ncommdomains; i1++) 
	      {
		if (nrecvcount[i1] > 0)
		  {
		    color->recvpartner[i2] = i1;
		    color->recvcount[i2]   = nrecvcount[i1];
		    color->recvindex[i2]   = check_malloc(nrecvcount[i1] * sizeof(int));
		    color->recvoffset[i2]  = check_malloc(nrecvcount[i1] * sizeof(int));

		    int i4 = 0;
		    for(i3 = 0; i3 < nall_points_of_color; i3++) 
		      {
			int pnt = all_points_of_color[i3];
			if (pnt >= cd->nownpoints && 
			    owner[pnt] == i1)
			  {
			    int j1 = offset[pnt];
			    ASSERT(j1 != -1);
			    color->recvindex[i2][i4]  = pnt;
			    color->recvoffset[i2][i4] = j1;
			    i4++;
			  }
		      }
		    ASSERT(i4 == color->recvcount[i2]);
		    i2++;
		  }
	      }
	    ASSERT(i2 == color->nrecvcount);
	  }
      }
    check_free(nrecvcount);
  }

  check_free(offset);
  check_free(owner);


  /* flag for testing color recvindex */
  int *flag = check_malloc(sd->nallpoints * sizeof(int));
  for(i = 0; i < sd->nallpoints; i++)
    {
      flag[i] =  0;
    }

  /* testing for multiple recvs per pnt  */
  for(i = 0; i < cd->ncommdomains; i++)
    {
      int k = cd->commpartner[i];
      int count = cd->recvcount[k];      
      if(count > 0)
	{
	  for(j = 0; j < count; j++)
	    {
	      int pnt = cd->recvindex[k][j];
	      ASSERT(flag[pnt] == 0);
	      flag[pnt] -= 1;
	    }
	}
    }


  for(i = sd->nownpoints; i < sd->nallpoints; i++)
    {
      ASSERT(flag[i] ==  -1);
    }

  /* assemble partial recvcounts per color, per target */
#pragma omp parallel default (none) shared(cd, sd			\
	   , flag, stdout, stderr)
  {
    RangeList *color;

    for (color = get_color(); color != NULL
	   ; color = get_next_color(color)) 
      {      
	int  nall_points_of_color = color->nall_points_of_color;
	int  *all_points_of_color = color->all_points_of_color;
	int i3;
	for(i3 = 0; i3 < nall_points_of_color; i3++) 
	  {
	    int pnt = all_points_of_color[i3];
	    if (pnt >= cd->nownpoints)
	      {
		ASSERT(flag[pnt] == -1);
		flag[pnt] += 1;
	      }
	  }
      }
  }	     

  for(i = 0; i < sd->nallpoints; i++)
    {
      ASSERT(flag[i] == 0);
    }

  /* testing for multiple recvs per pnt  */
  for(i = 0; i < cd->ncommdomains; i++)
    {
      int k = cd->commpartner[i];
      int count = cd->recvcount[k];      
      if(count > 0)
	{
	  for(j = 0; j < count; j++)
	    {
	      int pnt = cd->recvindex[k][j];
	      ASSERT(flag[pnt] == 0);
	      flag[pnt] -= 1;
	    }
	}
    }


  /* assemble partial recvcounts per color, per target */
#pragma omp parallel default (none) shared(cd, sd			\
					   , flag, stdout, stderr)
  {
    RangeList *color;
    for (color = get_color(); color != NULL
	   ; color = get_next_color(color)) 
      {      
	int i2, i4;	
	for(i2 = 0; i2 < color->nrecvcount; i2++) 
	  {
	    for(i4 = 0; i4 < color->recvcount[i2]; i4++) 
	      {
		int pnt = color->recvindex[i2][i4];
		ASSERT(flag[pnt] == -1);
		flag[pnt] += 1;
	      }
	  }
      }
  }

  for(i = 0; i < sd->nallpoints; i++)
    {
      ASSERT(flag[i] == 0);
    }
  check_free(flag);


}


void init_thread_comm(comm_data *cd
		      , solver_data *sd)
{

  /* determine required sends per color */
  gather_sendcount(cd, sd);

  /* determine required recvs per color */
  gather_recvcount(cd, sd);

}


