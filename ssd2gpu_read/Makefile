CUDA_PATH := /usr/local/cuda
GDS_PATH := /usr/local/gds

CFLAGS := -O3 -g -I. -D_GNU_SOURCE \
          -I $(CUDA_PATH)/include -I $(GDS_PATH)/lib \
          -L $(CUDA_PATH)/lib -L $(GDS_PATH)/lib \
          -lcuda -lcufile -lpthread

all: ssd2gpu_read

ssd2gpu_read: ssd2gpu_read.c
	$(CC) $^ -o $@ $(CFLAGS)
