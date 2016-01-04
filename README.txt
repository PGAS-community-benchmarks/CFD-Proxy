                            CFD Proxy
                          Version 1.0.1
                             README

                          December 2014

==============================================================================
Table of contents
==============================================================================

 1.  Overview
 
 2.  Contents of this Distribution
 3.  Hardware and Software Requirements
 4.  Configuration
 5.  Related Documentation - MPI (Message Passing Interface)
 6.  Documentation - GASPI (Global Address Space Programming Interface)
 7.  Implementation details
 8.  Results
 9.  Community involvement

==============================================================================
1. Overview
==============================================================================

This is the release of the hybrid.f6 CFD Proxy.  The CFD proxy kernel 
implements and evaluates the overlap efficiency for halo exchanges in 
unstructured meshes. 

The application first reads one of the multigrid mesh levels of a 
graph-partitioned and preprocessed F6 airplane mesh (dualgrid.[12-192].tgz). 

The actual CFD kernel we have implemented is a multi-threaded green gauss
gradient calculation with a subsequent halo exchange.  We believe that -
as far as the ratio of computation and communication is concerned - this 
kernel is representative for a much broader class of applications which
make use of unstructured meshes.

This benchmark aims at strong scaling scenarios with ~ 100 mesh points per 
(x86 Ivy Bridge) core or less. For more details please see the documentation. 
For even more details please have a look at the code.

==============================================================================
2.  Contents of this Distribution
==============================================================================

This release includes the actual mesh data in ./f6 directory and source 
code for various MPI and GASPI implementations of the halo exchange pattern
in ./src directory.

==============================================================================
3. Hardware and Software Requirements
==============================================================================

1) Server platform with InfiniBand HCA or ROCE Ethernet.

2) Linux operating system 

3) MPI. Download and install. This kernel requires MPI to be interoperable 
   with GASPI. So far this has been verified to work with with Openmpi, 
   Mvapich, Mvapich2, Intel MPI and IBM Platform MPI. 

4) GASPI. For installation and download instructions of GPI2 (currently 
   the only full implementation of GASPI) please see 
   https://github.com/cc-hpc-itwm/GPI-2

5) For reading the NetCDF dualgrid meshes you will need the NetCDF library
   http://www.unidata.ucar.edu/netcdf

==============================================================================
4. Configuration
==============================================================================

1) Download and install the required software packages, netcdf, MPI, GPI2.
   Note : GPI2 needs to be compiled against your MPI library of choice in 
   order to use the  MPI interoperabilty mode. 
   (./install.sh --with-mpi=/../../foo -p PREFIX)

2) Adapt the Makefile in src. (NETCDF_DIR, MPI_DIR, GPI2_DIR, CFLAGS, 
   (USE_MPI_MULTI_THREADED, USE_GASPI, USE_NTHREADS)). The MPI lib will 
   require support for either MPI_THREAD_MULTIPLE or MPI_THREAD_SERIALIZED.
   There are quite a few possible combinations of the provided various 
   settings. Pick the one which works best for your architecture, report 
   back and/or provide your own implementation. 
 
3) Unpack the dualgrid meshes in f6. Github restrictions currently only allows 
   for rather moderate mesh size, so we are in the process of looking for a 
   better solution.

3) GASPI can make use of the startup mechanisms of MPI. Start the
   hybrid.f6.exe hybrid MPI/GASPI executable as a regular hybrid OpenMP/MPI 
   application, e.g  mpirun -np 12 -machinefile machines -perhost 2 
   ../src/hybrid.f6.exe -lvl 2 dualgrid

==============================================================================
5. MPI
==============================================================================

   For MPI Documentation
   http://www.mpi-forum.org/docs/mpi-3.0/mpi30-report.pdf

==============================================================================
6. GASPI
==============================================================================

   For GASPI Documentation
   http://www.gpi-site.com/gpi2/gaspi/

==============================================================================
7. Implementation details
==============================================================================

Threading model
---------------
We have adopted a threading strategy which allows read access only to points
belonging to other thread domains and read/write access to the points the
thread actually owns.  The CFD proxy hence implements 3 different face types:
faces which reside entirely within the thread domain (and which are allowed 
to update both attached data points, ftype 1), faces which only write to the
left data point (ftype 2) and faces which only write to the right data point 
(ftype 3). 
In order to maximize scalar efficiency we have split the thread domains into 
sub domains (colors) which fit in the L2 cache.  The threads then  iterate 
over these colors until the respective thread domain has been  completed. 
We perform strip mining both with respect to initialized and finalized 
points. The colors in this implementation hence require a substantial 
amount of meta data about e.g about which data points are visited for the
first time (for each color), or which points are visited for the last time.  

Overlapping communication and computation
-----------------------------------------
For an efficient overlap of computation with communication we need to trigger
the communication as early as possible. When preprocessing the mesh we hence 
mark up all finalized points per color which belong to the mesh halo. 
The mesh faces are reordered such that halo points are updated first during
the gradient reconstruction. The thread which completes the final update 
(for a specific communication  partner) on these halo points then triggers 
the communication – either via  MPI_Isend, MPI_Put or gaspi_write_notify. 
In order to achieve this, the code additionally keeps track of the status 
of the inner halo (ghost cell) points for every color and maintains a list 
which halo points (per color) have to be send to which neighboring rank. 
We note that while this method allows for a maximal overlap of communication 
and computation, it either requires a full MPI_THREAD_MULTIPLE or a 
MPI_THREAD_SERIALIZED MPI version.  For the latter version we have 
encapsulated the actual MPI_Isend and MPI_Put in an OpenMP critical section.
We have enhanced the CFD proxy with a multithreaded gather/scatter operation
from/into the MPI/GASPI buffers. For this all colors use additional meta 
data with respect to required points (per color) in their outer halo. 
We note that for high scalablity on Xeon Phi  (and probably most other 
many-core architectures) this kind of optimization (multihreaded pack/unpack)
appears to be mandatory.

==============================================================================
8. Results
==============================================================================

For current results on a Fat Tree FDR/Ivy Bridge please have a look at the 
documentation. New, shiny and much better results for Xeon Phi have been  
included in the wiki.

https://github.com/PGAS-community-benchmarks/CFD-Proxy/wiki#results

==============================================================================
9. Community involvement
==============================================================================

We encourage the HPC community to provide new patterns or improve existing
ones. No restrictions apply. Instead we encourage you to improve
this kernel via better threadings models, and/or better communication and/or
more scalable communication API’s and/or better programming languages. Bonus
points are granted for highly scalable implementations which also feature
better programmability.


