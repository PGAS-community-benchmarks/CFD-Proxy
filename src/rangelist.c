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

// threadprivate rangelist data
static int ncolors_local = 0;
#pragma omp threadprivate(ncolors_local)
static RangeList *color_local = NULL;
#pragma omp threadprivate(color_local)

// threadprivate solver data
static solver_data_local solver_local;
#pragma omp threadprivate(solver_local)

void init_rangelist(RangeList *fcolor)
{  
  // next slice - linked list
  fcolor->succ = NULL;

  // meta data
  fcolor->start = -1;
  fcolor->stop = -1;
  fcolor->ftype = -1; //  face type   

  // points of color
  fcolor->nall_points_of_color = 0;
  fcolor->all_points_of_color = NULL;
  fcolor->nfirst_points_of_color = 0;
  fcolor->first_points_of_color = NULL;
  fcolor->nlast_points_of_color = 0;
  fcolor->last_points_of_color = NULL;

  // comm vars - send
  fcolor->nsendcount = 0;
  fcolor->sendpartner = NULL;
  fcolor->sendcount = NULL;
  fcolor->sendindex = NULL;
  fcolor->sendoffset = NULL;

  // comm vars - recv
  fcolor->nrecvcount = 0;
  fcolor->recvpartner = NULL;
  fcolor->recvcount = NULL;
  fcolor->recvindex = NULL;
  fcolor->recvoffset = NULL;

  // thread id
  fcolor->tid = -1; 

}

static void init_halo_type(int *htype
			   , comm_data *cd
			   , solver_data *sd
			   )
{
  int i,j;
  /* all own points */
  for(i = 0; i < sd->nownpoints; i++) 
    {
      htype[i] = 1;
    }
  /* set halo type for sendindex data (inner halo) */
  for(i = 0; i < cd->ncommdomains; i++)
    {
      int k = cd->commpartner[i];
      int count = cd->sendcount[k];      
      if(count > 0)
	{
	  for(j = 0; j < count; j++)
	    {
	      int pnt = cd->sendindex[k][j];
	      htype[pnt] = 2;
	    }
	}
    }
  /* addpoints (outer halo) */
  for(i = sd->nownpoints; i < sd->nallpoints; i++) 
    {
      htype[i] = 3;
    }
}


static void init_meta_data(int *pid
			   , int NTHREADS
			   , solver_data *sd
			   )
{ 
  /* set thread range */
  int min_size = sd->nallpoints/NTHREADS;
  int i, j, k;

  for(i = 0; i < sd->nallpoints; i++) 
    {
      pid[i] = -1;
    }

  int count = 0;
  int ncolors = 0;
  int rl_start = 0;
  int rl_stop = 0;
  for(k = 0; k < NTHREADS; k++)
    {
      int npnt  = 0;
      for(i = rl_start; i < sd->ncolors; i++)
	{
	  RangeList *color = &(sd->fcolor[i]); 
	  npnt += color->nall_points_of_color;
	  if (npnt >= min_size && k < NTHREADS-1)
	    {
	      break;
	    }
	}
      rl_stop = MIN(i+1, sd->ncolors);      

      /* all halo colors */
      for(i = rl_start; i < rl_stop; i++)
	{
	  RangeList *color = &(sd->fcolor[i]); 
	  int  *all_points_of_color = color->all_points_of_color;
	  int  nall_points_of_color = color->nall_points_of_color;
	  for(j = 0; j < nall_points_of_color; j++) 
	    {
	      int pnt = all_points_of_color[j];
	      ASSERT(pid[pnt] == -1);
	      pid[pnt] = k;
	    }
	  count += nall_points_of_color;
	  ncolors++;
	}
      rl_start =  rl_stop;
    }

  /* re-assign cross edges ending in addpoints (outer halo) */
  int face;
  for (face = 0; face < sd->nfaces; face++)
    {
      int p0 = sd->fpoint[face][0];
      int p1 = sd->fpoint[face][1];
      if (p1 >= sd->nownpoints && pid[p0] != pid[p1])
	{
	  pid[p1] = pid[p0];
	}
      if (p0 >= sd->nownpoints && pid[p1] != pid[p0])
	{
	  pid[p0] = pid[p1];
	}
    }

  /* sanity check */
  ASSERT(count == sd->nallpoints);
  ASSERT(ncolors == sd->ncolors);
  for(i = 0; i < sd->nallpoints; i++) 
    {
      ASSERT(pid[i] != -1);
    }
}



static void set_all_points_of_color(solver_data *sd
				, int tid
				, int *pid
				, int ncolors
				)
{
  int i, face;
  RangeList *color = color_local;
  int    (*fpoint)[2] = solver_local.fpoint;
  ASSERT (fpoint != NULL);
  ASSERT(sd->nallpoints > 0);
  ASSERT(color != NULL);

  int *tmp1   = check_malloc(sd->nallpoints * sizeof(int));
  for(i = 0; i < sd->nallpoints; i++) 
    {
      tmp1[i] = -1;
    }
  /* all first points of color including outer halo */
  int nallpoints = 0;
  for(i = 0; i < ncolors; i++)
    {
      RangeList *rl = &(color[i]); 
      int npoints = 0;
      for(face = rl->start ; face < rl->stop; face++)
	{
	  int p0 = fpoint[face][0];
	  int p1 = fpoint[face][1];
	  if (pid[p0] == tid && 
	      tmp1[p0] == -1)
	    {
	      tmp1[p0] = i;
	      npoints++;
	    }
	  if (pid[p1] == tid && 
	      tmp1[p1] == -1)
	    {
	      tmp1[p1] = i;
	      npoints++;
	    }
	}
      rl->nall_points_of_color = npoints;
      nallpoints += npoints;
    }
  check_free(tmp1);

  if (nallpoints == 0)
    {
      return;
    }
  
  int *tmp2   = check_malloc(sd->nallpoints * sizeof(int));  
  for(i = 0; i < sd->nallpoints; i++) 
    {
      tmp2[i] = -1;
    }
  int *points = check_malloc(nallpoints * sizeof(int));
  for(i = 0; i < ncolors; i++)
    {
      RangeList *rl = &(color[i]); 
      rl->all_points_of_color = points;
      points += rl->nall_points_of_color;
      int npoints = 0;
      for(face = rl->start ; face < rl->stop; face++)
	{
	  int p0 = fpoint[face][0];
	  int p1 = fpoint[face][1];
	  if (pid[p0] == tid && 
	      tmp2[p0] == -1)
	    {
	      tmp2[p0] = i;
	      rl->all_points_of_color[npoints++] = p0;
	    }
	  if (pid[p1] == tid && 
	      tmp2[p1] == -1)
	    {
	      tmp2[p1] = i;
	      rl->all_points_of_color[npoints++] = p1;
	    }
	}
    }
  check_free(tmp2);  

}


static void set_last_points_of_color(solver_data *sd
                                     , int tid
                                     , int *pid
                                     , int ncolors
                                     )
{
  int i, face;
  RangeList *color = color_local;
  int    (*fpoint)[2] = solver_local.fpoint;
  ASSERT (fpoint != NULL);
  ASSERT(sd->nallpoints > 0);
  ASSERT(color != NULL);

  int *tmp1   = check_malloc(sd->nallpoints * sizeof(int));
  for(i = 0; i < sd->nallpoints; i++) 
    {
      tmp1[i] = 0;
    }
  /* all finalized/last points of color */
  for(i = 0; i < ncolors; i++)
    {
      RangeList *rl = &(color[i]); 
      for(face = rl->start ; face < rl->stop; face++)
        {
          int p0 = fpoint[face][0];
          int p1 = fpoint[face][1];
          if (pid[p0] == tid && 
	      p0 < sd->nownpoints)
            {
              tmp1[p0]++;
            }
          if (pid[p1] == tid && 
	      p1 < sd->nownpoints)
            {
              tmp1[p1]++;
            }
        }
    }

  int *tmp2   = check_malloc(sd->nallpoints * sizeof(int));
  for(i = 0; i < sd->nallpoints; i++) 
    {
      tmp2[i] = 0;
    }
  int nlastpoints = 0;
  int nfirstpoints = 0;
  for(i = 0; i < ncolors; i++)
    {
      RangeList *rl = &(color[i]); 
      int npoints = 0;
      for(face = rl->start ; face < rl->stop; face++)
        {
          int p0 = fpoint[face][0];
          int p1 = fpoint[face][1];
          if (pid[p0] == tid && 
	      p0 < sd->nownpoints)
            {
              if (++tmp2[p0] == tmp1[p0])
                {
                  npoints++;
                }           
            }
          if (pid[p1] == tid && 
	      p1 < sd->nownpoints)
            {
              if (++tmp2[p1] == tmp1[p1])
                {
                  npoints++;
                }           
            }
        }
      rl->nlast_points_of_color = npoints;
      nlastpoints += npoints;
      nfirstpoints += rl->nfirst_points_of_color;
    }
  if (nlastpoints == 0)
    {
      return;
    }

  /* sanity check */
  ASSERT(nlastpoints == nfirstpoints);
  for(i = 0; i < sd->nallpoints; i++) 
    {
      tmp2[i] = 0;
    }

  int *points = check_malloc(nlastpoints * sizeof(int));
  for(i = 0; i < ncolors; i++)
    {
      RangeList *rl = &(color[i]); 
      rl->last_points_of_color = points;
      points += rl->nlast_points_of_color;
      int npoints = 0;
      for(face = rl->start ; face < rl->stop; face++)
        {
          int p0 = fpoint[face][0];
          int p1 = fpoint[face][1];
          if (pid[p0] == tid && 
	      p0 < sd->nownpoints)
            {
              if (++tmp2[p0] == tmp1[p0])
                {
		  ASSERT(rl->nlast_points_of_color != 0);
                  rl->last_points_of_color[npoints++] = p0;
                }           
            }
          if (pid[p1] == tid && 
	      p1 < sd->nownpoints)
            {
              if (++tmp2[p1] == tmp1[p1])
                {
		  ASSERT(rl->nlast_points_of_color != 0);
                  rl->last_points_of_color[npoints++] = p1;
                }           
            }
        }
    }
  check_free(tmp2);  
  check_free(tmp1);  

}

static void set_first_points_of_color(solver_data *sd
				    , int ncolors
				    )
{
  int i, k;
  RangeList *color = color_local;
  int    (*fpoint)[2] = solver_local.fpoint;
  ASSERT (fpoint != NULL);

  ASSERT(color != NULL);

  /* all first points of color excluding outer halo */
  int nfirstpoints = 0;
  for(i = 0; i < ncolors; i++)
    {
      RangeList *rl = &(color[i]); 
      int  nall_points_of_color = rl->nall_points_of_color;
      int  *all_points_of_color = rl->all_points_of_color;
      int npoints = 0;
      for(k = 0; k < nall_points_of_color; k++) 
	{
	  int pnt = all_points_of_color[k];
	  if (pnt < sd->nownpoints)
	    {
	      npoints++;
	    }
	}
      rl->nfirst_points_of_color = npoints;
      nfirstpoints += npoints;
    }

  if (nfirstpoints == 0)
    {
      return;
    }

  int *points = check_malloc(nfirstpoints * sizeof(int));
  for(i = 0; i < ncolors; i++)
    {
      RangeList *rl = &(color[i]); 
      int  nall_points_of_color = rl->nall_points_of_color;
      int  *all_points_of_color = rl->all_points_of_color;
      rl->first_points_of_color = points;
      points += rl->nfirst_points_of_color;
      int npoints = 0;
      for(k = 0; k < nall_points_of_color; k++) 
	{
	  int pnt = all_points_of_color[k];
	  if (pnt < sd->nownpoints)
	    {
	      rl->first_points_of_color[npoints++] = pnt;
	    }
	}
    }

}

void init_thread_rangelist(comm_data *cd
			   , solver_data *sd
			   , int tid
			   , int *pid
			   , int *htype
			   )
{
  int face;
  int nfaces = 0;
  for (face = 0; face < sd->nfaces; face++)
    {
      int p0 = sd->fpoint[face][0];
      int p1 = sd->fpoint[face][1];

      if ((pid[p0] == tid && htype[p0] < 3) || 
	  (pid[p1] == tid && htype[p1] < 3))
	{
	  nfaces++;
	}
    }

  solver_local.fpoint = NULL;
  solver_local.fnormal = NULL;

  if (nfaces == 0)
    {
      return;
    }

  /* init thread private face data */
  int     *ttype       = check_malloc(nfaces * sizeof(int));
  int    (*fpoint)[2]  = check_malloc(2 * nfaces * sizeof(int));
  double (*fnormal)[3] = check_malloc(3 * nfaces * sizeof(double));

  int nf = 0;
  for (face = 0; face < sd->nfaces; face++)
    {
      int p0 = sd->fpoint[face][0];
      int p1 = sd->fpoint[face][1];
      if ((pid[p0] == tid && htype[p0] < 3) || 
	  (pid[p1] == tid && htype[p1] < 3))
	{
	  memcpy(&(fpoint[nf][0])
		 , &(sd->fpoint[face][0])
		 , 2 * sizeof(int)
		 );
	  memcpy(&(fnormal[nf][0])
		 , &(sd->fnormal[face][0])
		 , 3 * sizeof(double)
		 );

	  /* init type */
	  ttype[nf] = -1;

	  nf++;
	}
    }
  ASSERT(nf == nfaces);

  /* set tmp face type for sorting */
  for (face = 0; face < nfaces; face++)
    {
      int p0 = fpoint[face][0];
      int p1 = fpoint[face][1];

      if (htype[p1] == 2 || htype[p0] == 2)
	{
	  /* all inner halo faces, writing only p1 */      
	  if (pid[p0] != tid || htype[p0] == 3)
	    {
	      ttype[face] = 0;
	    }
	  /* all inner halo faces, writing only p0 */      
	  else if (pid[p1] != tid || htype[p1] == 3)
	    {
	      ttype[face] = 1;
	    }
	  /* all inner halo faces, writing p1/p0 */      
	  else 
	    {
	      ttype[face] = 2;
	    }
	}
      else
	{
	  /* all other faces, writing only p1 */      
	  if (pid[p0] != tid || htype[p0] == 3)
	    {
	      ttype[face] = 3;
	    }
	  /* all other faces, writing only p0 */      
	  else if (pid[p1] != tid || htype[p1] == 3)
	    {
	      ttype[face] = 4;
	    }
	  /* all other faces, writing p1/p0 */      
	  else 
	    {
	      ttype[face] = 5;
	    }
	}
    }
  for (face = 0; face < nfaces; face++)
    {
      ASSERT(ttype[face] != -1);
    }

  /* init permutation vector for sorting */
  int *pm = check_malloc(nfaces * sizeof(int));
  for (face = 0; face < nfaces; face++)
    {
      pm[face] = face;
    }

  /* sort faces for type and p1/p0 */
  sort_faces(pm, fpoint, ttype, sd, nfaces);

  /* fix face permutation */
  int     *tt     = check_malloc(nfaces * sizeof(int));
  int    (*fp)[2] = check_malloc(2 * nfaces * sizeof(int));
  double (*fn)[3] = check_malloc(3 * nfaces * sizeof(double));
  for (face = 0; face < nfaces; face++)
    {
      int tf = pm[face];
      memcpy(&(tt[face])
             , &(ttype[tf])
             , sizeof(int)
             );
      memcpy(&(fp[face][0])
             , &(fpoint[tf][0])
             , 2 * sizeof(int)
             );
      memcpy(&(fn[face][0])
             , &(fnormal[tf][0])
             , 3 * sizeof(double)
             );
    }
  check_free(pm);
  check_free(ttype);
  check_free(fpoint);
  check_free(fnormal);

  // thread local solver (face) data
  solver_local.fpoint = fp;
  solver_local.fnormal = fn;
  ttype = tt;

#define MAX_FACES_IN_COLOR 96

  int count = 0;
  int ncolors = 0;
  for(face = 1 ; face < nfaces; face++)
    {
      if((++count) == MAX_FACES_IN_COLOR || 
	 ttype[face] != ttype[face-1])	
	{
	  count = 0;
	  ncolors++;
	} 
    }
  /* last color */
  ncolors++;
        
  /* alloc threadprivate rangelist */
  ncolors_local = ncolors;
  color_local = check_malloc(ncolors * sizeof(RangeList));  

  RangeList *tl;  
  int i0;
  for (i0 = 0; i0 < ncolors; i0++)
    {
      tl = &(color_local[i0]); 
      init_rangelist(tl);
    }

  /* set color range */
  count = 0;
  i0 = 0;
  int start = 0;  
  for(face = 1 ; face < nfaces; face++)
    {
      if((++count) == MAX_FACES_IN_COLOR || 
	 ttype[face] != ttype[face-1])	
	{
	  tl = &(color_local[i0]); 
	  tl->start  = start;
	  tl->stop   = face;

	  start = face;
	  count = 0;
	  i0++;
	} 
    }
  /* last color */
  tl = &(color_local[i0]); 
  tl->start  = start;
  tl->stop   = nfaces;  
  i0++;

  ASSERT(i0 == ncolors);

  /* set face type value, succ */
  int i;
  for(i = 0; i < ncolors; i++)
    {
      tl = &(color_local[i]); 

      /* tid */
      tl->tid = tid;

      int fstart = tl->start;

      /* ftype, writing only p1 */
      if(ttype[fstart] == 0 || 
	 ttype[fstart] == 3)
	{
	  tl->ftype = 1;
	}
      /* ftype, writing only p0 */
      else if (ttype[fstart] == 1 || 
	       ttype[fstart] == 4)
	{
	  tl->ftype = 2;
	}
      /* ftype, writing p0/p1 */
      else if (ttype[fstart] == 2 || 
	       ttype[fstart] == 5)
	{
	  tl->ftype = 3;
	}
      /* external, writing outer halo */
      else
	{
	  ASSERT(0);
	}

      /* succ */
      if (i < ncolors - 1 )
	{
	  tl->succ = &(color_local[i+1]);
	}
      else
	{
	  tl->succ = NULL;
	}
    } 
  check_free(ttype);

  /* points of color */
  set_all_points_of_color(sd, tid, pid, ncolors);
  
  /* first points of color */
  set_first_points_of_color(sd, ncolors);

  /* last points of color */
  set_last_points_of_color(sd, tid, pid, ncolors);

}


void init_thread_meta_data(int *pid
			   , int *htype
			   , comm_data *cd
			   , solver_data *sd
			   , int NTHREADS
			   )
{

  /* set thread id, color id */
  init_meta_data(pid, NTHREADS, sd);

  /* init halo type */
  init_halo_type(htype, cd, sd);


#ifdef DDEBUG
  if (cd->iProc == 0)
    {
      int face;
      for (face = 0; face < sd->nfaces; face++)
	{
	  int p0 = sd->fpoint[face][0];
	  int p1 = sd->fpoint[face][1];
	  int pid0 = pid[p0];
	  int pid1 = pid[p1];
	  int htype0 = htype[p0];
	  int htype1 = htype[p1];
	  printf ("face: %d nfaces: %d p0: %d p1: %d pid0: %d pid1: %d htype0: %d htype1: %d\n"
		  ,face,sd->nfaces,p0,p1,pid0,pid1,htype0,htype1);
	}
    }
#endif

}

void eval_thread_comm(comm_data *cd)
{
#pragma omp parallel default (none) shared(cd\
            , stdout, stderr)
    {
      int const tid = omp_get_thread_num();
      RangeList *color;

#ifdef DEBUG
      int ncolors = 0;
      int scolors = 0;
      for (color = get_color(); color != NULL
	     ; color = get_next_color(color)) 
	{
	  ncolors++;
	  if (color->nsendcount > 0)
	    {
	      scolors++;
	    }
	}
      printf("iProc: %6d tid: %4d ncolors: %d scolors: %d\n"
	     ,cd->iProc,tid,ncolors,scolors);
#endif

      int nsend = 0;
      for (color = get_color(); color != NULL
	     ; color = get_next_color(color)) 
	{      
	  int i;
	  for(i = 0; i < color->nsendcount; i++)
	    {
	      int i1 = color->sendpartner[i];
	      int sendcount_color = color->sendcount[i];
	      int sendcount_local = get_sendcount_local(i1);
	      if (sendcount_color > 0 && 
		  sendcount_local > 0)
		{		  
		  int inc_local = set_send_increment_local(i1, sendcount_color);
		  if(inc_local % sendcount_local == 0)
		    {
		      int inc_global = set_send_increment(i1, sendcount_local);
		      int k = cd->commpartner[i1];
		      if (inc_global % cd->sendcount[k] == 0)
			{
			  printf("iProc: %6d tid: %4d send num: %8d to: %6d -- complete. final send color: %6d\n"
				 ,cd->iProc,tid,cd->sendcount[k],k,nsend);
			}
		    }
		}
	    }
	  nsend++;
	}

    }
}


void eval_thread_rangelist(solver_data *sd)
{
  int i0;

  /* sanity check */
  int *tmp1 = check_malloc(sd->nallpoints * sizeof(int));
  int *tmp2 = check_malloc(sd->nallpoints * sizeof(int));


  /* validate all_points_of_color */
  for(i0 = 0; i0 < sd->nallpoints; i0++)   
    {
      tmp1[i0] = -1;
    }
#pragma omp parallel default (none) shared(sd\
            , tmp1, stdout, stderr)
    {
      int const tid = omp_get_thread_num();
      int    (*fpoint)[2] = solver_local.fpoint;
      RangeList *color;

      for (color = get_color(); color != NULL
	     ; color = get_next_color(color)) 
	{      
	  int  nall_points_of_color = color->nall_points_of_color;
	  int  *all_points_of_color = color->all_points_of_color;
	  int i1;

	  for(i1 = 0; i1 < nall_points_of_color; i1++) 
	    {
	      int pnt = all_points_of_color[i1];
	      ASSERT(tmp1[pnt] == -1);
	      tmp1[pnt] += 1;
	    }
	}
    }
  for(i0 = 0; i0 < sd->nallpoints; i0++) 
    {      
      ASSERT(tmp1[i0] == 0);
    }

  /* validate first_points_of_color, ftype and tid*/
  for(i0 = 0; i0 < sd->nallpoints; i0++) 
    {      
      tmp1[i0] = -1;
      tmp2[i0] = -1;
    }
#pragma omp parallel default (none) shared(sd\
            , tmp1, tmp2, stdout, stderr)
    {

      int const tid = omp_get_thread_num();
      int    (*fpoint)[2] = solver_local.fpoint;
      RangeList *color;

      for (color = get_color(); color != NULL
	     ; color = get_next_color(color)) 
	{      
	  int  nfirst_points_of_color = color->nfirst_points_of_color;
	  int  *first_points_of_color = color->first_points_of_color;
	  int i1, face;

	  for(i1 = 0; i1 < nfirst_points_of_color; i1++) 
	    {
	      int pnt = first_points_of_color[i1];
	      ASSERT(tmp1[pnt] == -1);
	      tmp1[pnt] = tid;
	    }
	  
	  /* validate tid */
	  ASSERT(color->tid == tid);

	  /* validate ftype */
	  ASSERT(color->ftype != 0);
	  for(face = color->start; face < color->stop; face++)
	    {
	      const int  p0    = fpoint[face][0];
	      const int  p1    = fpoint[face][1];

	      if (color->ftype == 2 || 
		  color->ftype == 3)
		{
		  if (tmp2[p0] == -1)
		    {
		      tmp2[p0] = tid;
		    }
		  else 
		    {
		      ASSERT(tmp2[p0] == tid);
		    }
		}
	      if (color->ftype == 1 || 
		  color->ftype == 3)
		{
		  if (tmp2[p1] == -1)
		    {
		      tmp2[p1] = tid;
		    }
		  else
		    {
		      ASSERT(tmp2[p1] == tid);
		    }
		}
	    }
	}
    }
  for(i0 = 0; i0 < sd->nownpoints; i0++)   
    {
      ASSERT(tmp1[i0] != -1);
      ASSERT(tmp2[i0] != -1);
    }
  for(i0 = sd->nownpoints; i0 < sd->nallpoints; i0++)   
    {
      ASSERT(tmp1[i0] == -1);
      ASSERT(tmp2[i0] == -1);
    }

  /* validate last_points_of_color */
  for(i0 = 0; i0 < sd->nallpoints; i0++)   
    {
      tmp1[i0] = -1;
    }
#pragma omp parallel default (none) shared(sd\
            , tmp1, stdout, stderr)
    {

      int const tid = omp_get_thread_num();
      int    (*fpoint)[2] = solver_local.fpoint;
      RangeList *color;
      for (color = get_color(); color != NULL
	     ; color = get_next_color(color)) 
	{      
	  int  nlast_points_of_color = color->nlast_points_of_color;
	  int  *last_points_of_color = color->last_points_of_color;
	  int i1;
	  for(i1 = 0; i1 < nlast_points_of_color; i1++) 
	    {
	      int pnt = last_points_of_color[i1];
	      ASSERT(tmp1[pnt] == -1);
	      tmp1[pnt] += 1;
	    }
	}
    }
  for(i0 = 0; i0 < sd->nownpoints; i0++) 
    {      
      ASSERT(tmp1[i0] == 0);
    }
  for(i0 = sd->nownpoints; i0 < sd->nallpoints; i0++)   
    {
      ASSERT(tmp1[i0] == -1);
    }

  check_free(tmp1);
  check_free(tmp2);
}   


static void gather_sendcount(comm_data *cd
			     , solver_data *sd)
{
  int i, j, i0;

  int **idx = check_malloc(sd->nallpoints * sizeof(int*));
  for(i = 0; i < sd->nallpoints; i++)
    {
      idx[i] = NULL;
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
	      if (idx[pnt] == NULL)
		{
		  idx[pnt] = check_malloc(cd->ncommdomains * sizeof(int));
		  for(i0 = 0; i0 < cd->ncommdomains; i0++)   
		    {
		      idx[pnt][i0] = -1;
		    }
 		}             
	      // assert initial state per pnt, per target
	      ASSERT(idx[pnt][i] == -1);
	      idx[pnt][i] = j;
	    }
	}
    }

  /* assemble partial sendcounts per color, per target */
#pragma omp parallel default (none) shared(cd, sd\
            , idx, stdout, stderr)
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
	      if (idx[pnt] != NULL)
		{
		  for(i3 = 0; i3 < cd->ncommdomains; i3++)
		    {
		      int j1 = idx[pnt][i3];
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
                          if (idx[pnt] != NULL)
                            {
                              int j1 = idx[pnt][i1];
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
      if (idx[i] != NULL)
	{
	  check_free(idx[i]);
	}
    }
  check_free(idx);


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
	      int i1 = color->sendpartner[i2];
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

  int *idx = check_malloc(sd->nallpoints * sizeof(int));
  int *own  = check_malloc(sd->nallpoints * sizeof(int));
  for(i = 0; i < sd->nallpoints; i++)
    {
      idx[i] = -1;
      own[i] = -1;
    }

  /* gather recvindex data per pnt - similar to addpoint own */
  for(i = 0; i < cd->ncommdomains; i++)
    {
      int k = cd->commpartner[i];
      if(cd->recvcount[k] > 0)
	{
	  for(j = 0; j < cd->recvcount[k]; j++)
	    {
	      int pnt = cd->recvindex[k][j];
	      ASSERT(pnt >= sd->nownpoints);
	      ASSERT(idx[pnt] == -1);
	      ASSERT(own[pnt] == -1);
	      idx[pnt] = j;	      
	      own[pnt] = i;
	    }
	}
    }


  /* assemble partial recvcounts per color, per target */
#pragma omp parallel default (none) shared(cd, sd			\
					   , idx, own, stdout, stderr)
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
		ASSERT(idx[pnt] != -1);
		ASSERT(own[pnt] != -1);
		i3 = own[pnt];
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
			    own[pnt] == i1)
			  {
			    int j1 = idx[pnt];
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
	   , flag, own, stdout, stderr)
  {
    RangeList *color;
    int  *nrecvcount = check_malloc(cd->ncommdomains * sizeof(int));

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


  check_free(idx);
  check_free(own);


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
	    int i1 = color->recvpartner[i2];
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

solver_data_local* get_solver_data(void)
{
  return &solver_local;
}


int get_ncolors(void)
{
  return ncolors_local;
}

RangeList* private_get_color(RangeList *const prev)
{

  RangeList *color = NULL;

  if(prev != NULL)
    {
      color = prev->succ;      
    }
  else
    {
      color = color_local;
    }

  return color;

}




