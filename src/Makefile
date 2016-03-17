GPI2_DIR = /opt/GPI-2-1.3.0
NETCDF_DIR = $(HOME)/intel64/netcdf-3.6.3
MPI_DIR= /opt/intel/impi/5.1.2.150
CC = $(MPI_DIR)/intel64/bin/mpiicc

BIN += hybrid.f6.exe

CFLAGS += -Wall
CFLAGS += -Wextra
CFLAGS += -Wshadow
CFLAGS += -O3 -g 
CFLAGS += -std=c99
CFLAGS += -openmp
#CFLAGS += -DDEBUG 

CFLAGS += -DGCC_EXTENSION
CFLAGS += -DUSE_MPI_MULTI_THREADED
CFLAGS += -DUSE_MPI_1_SIDED
CFLAGS += -DUSE_GASPI
#CFLAGS += -DUSE_GASPI_TEST
#CFLAGS += -DUSE_MPI_TEST_ANY
CFLAGS += -DUSE_MPI_WAIT_ANY
CFLAGS += -DUSE_MPI_IMMEDIATE_WAIT
#CFLAGS += -DUSE_MPI_EARLY_WAIT
#CFLAGS += -DUSE_MPI_TEST
#CFLAGS += -DUSE_MPI_TEST_MASTER_ONLY
CFLAGS += -DUSE_PACK_IN_BULK_SYNC
CFLAGS += -DUSE_PARALLEL_GATHER
CFLAGS += -DUSE_PARALLEL_SCATTER
CFLAGS += -DUSE_NTHREADS=2

###############################################################################

INCLUDE_DIR += /usr/include
INCLUDE_DIR += $(NETCDF_DIR)/include 
INCLUDE_DIR += $(MPI_DIR)/include 
INCLUDE_DIR += $(GPI2_DIR)/include 
INCLUDE_DIR += ..

LIBRARY_DIR += $(NETCDF_DIR)/lib
LIBRARY_DIR += $(MPI_DIR)/lib
LIBRARY_DIR += $(GPI2_DIR)/lib64

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

LIB += GPI2-dbg
#LIB += ibverbs
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
