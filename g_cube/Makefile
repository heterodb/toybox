PG_CONFIG := pg_config

NAME = g_cube

MODULES = $(NAME)
MODULEDIR = pg_strom
DATA = cudax_$(NAME).h
DATA_built = cudax_$(NAME).fatbin cudax_$(NAME).gfatbin

NVCC = nvcc
NVCC_FLAGS = -I $(shell $(PG_CONFIG) --includedir-server) \
             -I $(shell $(PG_CONFIG) --sharedir)/$(MODULEDIR) \
             --relocatable-device-code=true \
             --fatbin \
             --maxrregcount=128 \
             --gpu-architecture=compute_60
NVCC_FLAGS += --gpu-code=sm_60,sm_61,sm_70,sm_75,sm_80

NVCC_DEBUG_FLAGS = $(NVCC_FLAGS) --source-in-ptx --device-debug

PG_CFLAGS = -I $(shell $(PG_CONFIG) --sharedir)/$(MODULEDIR)

#
# Build chain of PostgreSQL
#
PGXS = $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

%.fatbin:  %.cu $(GPU_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

%.gfatbin: %.cu $(GPU_HEADERS)
	$(NVCC) $(NVCC_DEBUG_FLAGS) -o $@ $<
