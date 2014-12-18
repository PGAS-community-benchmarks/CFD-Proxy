GPI2_DIR = $(HOME)/GPI2-1.1.1_intel-5.0.1.035.mic
NETCDF_DIR = $(HOME)/netcdf-4.2.1.1.mic
MPI_DIR=/export/opt/intel2015/impi/5.0.1.035/mic
CC = $(MPI_DIR)/bin/mpiicc

BIN += hybrid.f6.exe

CFLAGS += -Wall
CFLAGS += -Wextra
CFLAGS += -Wshadow
CFLAGS += -O2 -g -mmic
CFLAGS += -std=c99
CFLAGS += -openmp
#CFLAGS += -DDEBUG 
CFLAGS += -DGCC_EXTENSION
CFLAGS += -DUSE_NTHREADS=240
CFLAGS += -DUSE_MPI_MULTI_THREADED
CFLAGS += -DUSE_GASPI
CFLAGS += -DUSE_MPI_WAIT_ANY
CFLAGS += -DUSE_PSCW_EARLY_WAIT
CFLAGS += -DUSE_MPI_PARALLEL_GATHER
CFLAGS += -DUSE_MPI_PARALLEL_SCATTER
CFLAGS += -DUSE_GASPI_PARALLEL_GATHER
CFLAGS += -DUSE_GASPI_PARALLEL_SCATTER

###############################################################################

INCLUDE_DIR += $(NETCDF_DIR)/include 
INCLUDE_DIR += $(MPI_DIR)/include 
INCLUDE_DIR += $(GPI2_DIR)/include 
INCLUDE_DIR += ..

LIBRARY_DIR += $(NETCDF_DIR)/lib
LIBRARY_DIR += $(MPI_DIR)/lib
LIBRARY_DIR += $(GPI2_DIR)/lib64/mic


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
OBJ += rangelist
OBJ += threads
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
