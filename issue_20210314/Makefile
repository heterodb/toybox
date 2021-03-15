include ../Makefile.common

cuda_ipc_mem: cuda_ipc_mem.c
	$(CC) -I$(IPATH) -L$(LPATH) -D_GNU_SOURCE -g $^ -o $@ -lcuda -lrt -lpthread -Wall

clean:
	rm -f cuda_ipc_mem
