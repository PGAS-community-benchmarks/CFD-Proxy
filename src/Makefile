GPI2_DIR = $(HOME)/GPI2-1.1.1_intel-5.0.1.035.mic
NETCDF_DIR = $(HOME)/netcdf-4.2.1.1.mic
MPI_DIR=/export/opt/intel2015/impi/5.0.1.035/mic
CC = $(MPI_DIR)/bin/mpiicc

#GPI2_DIR = $(HOME)/GPI2-1.1.1_intel-4.1.0
#NETCDF_DIR = $(HOME)/netcdf-4.2.1.1.mic
#MPI_DIR=/export/opt/intel2013/impi/4.1.0.024/mic
#CC = $(MPI_DIR)/bin/mpiicc

#GPI2_DIR = $(HOME)/GPI2-1.1.1_intel-5.0.1.035
#NETCDF_DIR = $(HOME)/netcdf-4.2.1.1
#MPI_DIR=/export/opt/intel2015/impi/5.0.1.035
#CC = $(MPI_DIR)/bin64/mpiicc

#GPI2_DIR = $(HOME)/GPI2-1.1.1_openmpi-1.8.4_enable-mpi-thread-multiple
#NETCDF_DIR = $(HOME)/netcdf-4.2.1.1
#MPI_DIR=$(HOME)/openmpi-1.8.4_enable-mpi-thread-multiple
#CC = $(MPI_DIR)/bin/mpicc

#GPI2_DIR = $(HOME)/GPI2-1.1.1_mvapich2-2.1a_enable-threads-multiple
#NETCDF_DIR = $(HOME)/netcdf-4.2.1.1
#MPI_DIR=$(HOME)/mvapich2-2.1a_enable-threads-multiple
#CC = $(MPI_DIR)/bin/mpicc

BIN += hybrid.f6.exe

CFLAGS += -Wall
CFLAGS += -Wextra
CFLAGS += -Wshadow
CFLAGS += -O2 -g -mmic
#CFLAGS += -O2 -g 
CFLAGS += -std=c99
#CFLAGS += -fopenmp
CFLAGS += -openmp
CFLAGS += -DDEBUG 

CFLAGS += -DGCC_EXTENSION
CFLAGS += -DUSE_NTHREADS=240
CFLAGS += -DUSE_MPI_MULTI_THREADED
CFLAGS += -DUSE_MPI_1_SIDED
CFLAGS += -DUSE_GASPI
#CFLAGS += -DUSE_GASPI_TEST
#CFLAGS += -DUSE_MPI_TEST_ANY
#CFLAGS += -DUSE_MPI_WAIT_ANY
#CFLAGS += -DUSE_MPI_IMMEDIATE_WAIT
#CFLAGS += -DUSE_MPI_EARLY_WAIT
#CFLAGS += -DUSE_MPI_TEST
#CFLAGS += -DUSE_MPI_TEST_MASTER_ONLY
CFLAGS += -DUSE_PACK_IN_BULK_SYNC
CFLAGS += -DUSE_PARALLEL_GATHER
CFLAGS += -DUSE_PARALLEL_SCATTER

###############################################################################

INCLUDE_DIR += $(NETCDF_DIR)/include 
INCLUDE_DIR += $(MPI_DIR)/include 
INCLUDE_DIR += $(GPI2_DIR)/include 
INCLUDE_DIR += ..

LIBRARY_DIR += $(NETCDF_DIR)/lib
#LIBRARY_DIR += $(MPI_DIR)/lib64
LIBRARY_DIR += $(MPI_DIR)/lib
LIBRARY_DIR += $(GPI2_DIR)/lib64/mic
#LIBRARY_DIR += $(GPI2_DIR)/lib64

LDFLAGS += $(addprefix -L,$(LIBRARY_DIR))
CFLAGS += $(addprefix -I,$(INCLUDE_DIR))

OBJ += solver
OBJ += comm_data
OBJ += solver_data
OBJ += error_handling
OBJ += read_netcdf
OBJ += exchange_data_mpi
OBJ += exchange_data_gaspi
OBJ += exchange_data_mpidma
OBJ += gradients
OBJ += flux
OBJ += rangelist
OBJ += threads
OBJ += eval
OBJ += thread_comm
OBJ += points_of_color
OBJ += waitsome
OBJ += queue
OBJ += util

LIB += GPI2
LIB += ibverbs
#LIB += mpich
LIB += mpi_mt
#LIB += mpi
LIB += netcdf
LIB += m

###############################################################################

default: $(BIN)

%.exe: %.o $(addsuffix .o, $(OBJ))
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(addprefix -l, $(LIB))

###############################################################################

.PHONY: clean objclean

objclean:
	rm -f *.o

clean: objclean
	rm -f $(BIN)
