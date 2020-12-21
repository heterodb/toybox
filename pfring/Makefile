CC	:= gcc
CFLAGS	:= -g -O3 -Wall

pfring_test: pfring_test.c
	$(CC) $(CFLAGS) $< -o $@ -lpcap -lpfring -lpthread

clean:
	rm -f pfring_test
