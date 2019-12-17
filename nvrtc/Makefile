PG_CONFIG := pg_config
CUDA_PATH := /usr/local/cuda
CFLAGS    := -Wall -g -O2 \
             -I $(CUDA_PATH)/include \
             -L $(CUDA_PATH)/lib64 \
             -DCPU_ARCH=\"$(shell uname -m)\" \
             -DCUDA_INCLUDE_DIR=\"$(CUDA_PATH)/include\"
#ifeq ($(shell ls $(PG_CONFIG) 2>/dev/null),$(PG_CONFIG))
ifneq ($(shall $(PG_CONFIG)),'')
CFLAGS    += -DPGSHAREDIR=\"$(shell $(PG_CONFIG) --sharedir)\" \
             -DPGSERV_INCLUDEDIR=\"$(shell $(PG_CONFIG) --includedir-server)\"
endif

all: nvrtcc

nvrtcc: nvrtcc.c
	$(CC) $(CFLAGS) -o nvrtcc nvrtcc.c -lnvrtc

clean:
	rm -f nvrtcc
