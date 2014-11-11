GPI2_DIR = $(HOME)/GPI2-1.1.0_mvapich2-2.1a
NETCDF_DIR = $(HOME)/netcdf-4.2.1.1
MPI_DIR=$(HOME)/mvapich2-2.1a
CC = $(MPI_DIR)/bin/mpicc

BIN += hybrid.f6.exe

CFLAGS += -Wall
CFLAGS += -Wextra
CFLAGS += -Wshadow
CFLAGS += -O2 -g
CFLAGS += -std=c99
CFLAGS += -fopenmp
#CFLAGS += -DDEBUG 
CFLAGS += -DGCC_EXTENSION
CFLAGS += -DUSE_NTHREADS=12
#CFLAGS += -DUSE_MPI_MULTI_THREADED
CFLAGS += -DUSE_GASPI

###############################################################################

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
OBJ += rangelist
OBJ += threads
OBJ += waitsome
OBJ += queue
OBJ += util

LIB += GPI2
LIB += ibverbs
#LIB += mpich
#LIB += mpi_mt
LIB += mpi
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
