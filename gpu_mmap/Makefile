CUDA_PATH := /usr/local/cuda
CUDA_INCLUDE = $(CUDA_PATH)/include
CUDA_LIBS = $(CUDA_PATH)/lib64
CUDA_BIN  = $(CUDA_PATH)/bin
NVCC = $(CUDA_BIN)/nvcc

CFLAGS = -Wall -I$(CUDA_INCLUDE) -L$(CUDA_LIBS)

all: gpu_mmap

gpu_mmap_kernel.ptx: gpu_mmap_kernel.cu
	$(NVCC) -o $@ -ptx $< -arch sm_60

gpu_mmap_kernel.h: gpu_mmap_kernel.ptx
	@(echo 'const char *gpu_mmap_kernel =';			\
	  sed -e 's/\\/\\\\/g' -e 's/\t/\\t/g' -e 's/"/\\"/g' \
	      -e 's/^/  "/g'   -e 's/$$/\\n"/g' < $<;	\
	   echo ';') > $@

gpu_mmap: gpu_mmap.c gpu_mmap_kernel.h
	$(CC) $(CFLAGS) -o $@ $< -lcuda -lpthread

clean:
	rm -f gpu_mmap gpu_mmap_kernel.h gpu_mmap_kernel.ptx
