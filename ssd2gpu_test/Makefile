# HeteroDB Extra Package
MODULE    = heterodb_extra.so
OBJS      = license.o gpudirect.o sysfs.o misc.o
SSD2GPU   = ssd2gpu_test
SSD2GPU_OBJ = $(SSD2GPU).o
HEADERS   = heterodb_extra.h nvme_strom.h
EXTRA_VERSION ?= $(shell cat ../VERSION)

CUDA_PATH_LIST := /usr/local/cuda /usr/local/cuda-*
CUDA_PATH   := $(shell for x in $(CUDA_PATH_LIST);	\
    do test -e "$$x/include/cuda.h" && echo $$x; done | head -1)

CC        := gcc
CFLAGS    := -Wall -g -O2 -fPIC -D_GNU_SOURCE -I. \
             -Wno-stringop-truncation \
             -Wno-format-truncation \
             -I $(CUDA_PATH)/include \
             -L $(CUDA_PATH)/lib64   \
             -DHETERODB_EXTRA_VERSION="$(EXTRA_VERSION)"
ifneq ($(EXTRA_RELEASE),)
CFLAGS    += -DHETERODB_EXTRA_RELEASE="$(EXTRA_RELEASE)"
endif
LDFLAGS   := $(CFLAGS) -shared -lgmp -lcurl -lcufile \
             -Wl,-rpath,'$(CUDA_PATH)/lib64'

PREFIX    ?= /usr
DEST_INC  := $(DEST)$(PREFIX)/include
DEST_BIN  := $(DEST)$(PREFIX)/bin
DEST_LIB  := $(DEST)$(PREFIX)/lib64

all: $(MODULE) $(SSD2GPU)

.c.o: $(HEADERS)
	$(CC) -c $< $(CFLAGS)

$(MODULE): $(OBJS) $(HEADERS)
	$(CC) $(LDFLAGS) -shared -o $(MODULE) $(OBJS)

$(SSD2GPU): $(SSD2GPU_OBJ) $(HEADERS)
	$(CC) $(CFLAGS) -o $(SSD2GPU) $(SSD2GPU_OBJ) -O0 -g \
	    -L$(CUDA_PATH)/lib64 \
	    -ldl -lcuda -lnvidia-ml -lpthread

install: $(MODULE) $(SSD2GPU)
	mkdir -p $(DEST_INC) $(DEST_BIN) $(DEST_LIB)
	install -m 0755 $(MODULE) $(DEST_LIB)/$(MODULE).$(EXTRA_VERSION)
	ln -sf $(MODULE).$(EXTRA_VERSION) $(DEST_LIB)/$(MODULE)
	install -m 0755 $(SSD2GPU) $(DEST_BIN)

uninstall:
	rm -f $(DEST_LIB)/$(MODULE).$(EXTRA_VERSION) \
          $(DEST_LIB)/$(MODULE) \
          $(DEST_BIN)/$(SSD2GPU)

clean:
	rm -f $(MODULE) $(OBJS) $(SSD2GPU) $(SSD2GPU_OBJ)

all: $(MODULE)
