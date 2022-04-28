include ../Makefile.common

__NVCC_FLAGS += --fatbin \
                --maxrregcount=128 \
                --gpu-architecture=compute_60 \
                --gpu-code=sm_60,sm_61,sm_70,sm_75,sm_80 \
                --source-in-ptx \
                --relocatable-device-code=true

all: cuda_fnptr sample.fat

cuda_fnptr: cuda_fnptr.c
	$(CC) -I$(IPATH) -L$(LPATH) -D_GNU_SOURCE -g $^ -o $@ -lcuda -lrt -lpthread

sample.fat: sample.cu
	$(NVCC) $(__NVCC_FLAGS) -o $@ $<

clean:
	rm -f cuda_fnptr sample.fat
