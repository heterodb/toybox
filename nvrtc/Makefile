PG_CONFIG := pg_config
PROGRAM = nvrtcc

CUDA_PATH := /usr/local/cuda

OBJS = nvrtcc.o

PG_CFLAGS := -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 \
             -DCUDA_INCLUDE_DIR=\"$(CUDA_PATH)/include\"
PG_LIBS   := -lnvrtc

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
