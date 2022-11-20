/* ----------------------------------------------------------------
 *
 * ssd2gpu_test
 *
 * Test program for SSD-to-GPU Direct Loading
 * --------
 * Copyright 2016-2018 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2017-2018 (C) The PG-Strom Development Team
 *
 * See LICENSE for the terms to use this software.
 * ----------------------------------------------------------------
 */
#include <assert.h>
#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <cuda.h>
#include <cufile.h>
#include <nvml.h>
#include "heterodb_extra_internal.h"
#include "nvme_strom.h"

/* misc macros */
#ifndef offsetof
#define offsetof(type, field)	((long) &((type *)0)->field)
#endif

/* command line options */
static int		cuda_dindex = -1;
static char	   *kernel_driver = NULL;
static int		nr_segments = 6;
static size_t	segment_sz = 32UL << 20;
static size_t	address_randomize = 0;
static int		enable_checks = 0;

/* static variables */
static long				PAGE_SIZE;
static CUdevice			cuda_device;
static CUcontext		cuda_context;
static unsigned long	curr_fpos = 0;	/* to be updated by atomic add */
static int				filedesc = -1;	/* buffered i/o */
static const char	   *filename = NULL;
static GPUDirectFileDesc gds_fdesc;

typedef struct
{
	pthread_t		thread;
	char		   *src_buffer;
	char		   *dst_buffer;
	CUdeviceptr		dev_buffer;
	unsigned long	dev_mhandle;
	size_t			dev_moffset;
	long			nr_ram2gpu;
	long			nr_ssd2gpu;
	long			nr_dma_submit;
	long			nr_dma_blocks;
} worker_context;

static const char *
cuErrorName(CUresult code)
{
	const char *error_name;

	if (cuGetErrorName(code, &error_name) != CUDA_SUCCESS)
		error_name = "unknown error";
	return error_name;
}

/*
 * gpuDirectInitDriver
 */
static int		(*p_gpudirect_init_driver)() = NULL;

static void
gpuDirectInitDriver(void)
{
	if (p_gpudirect_init_driver &&
		p_gpudirect_init_driver() != 0)
		__Elog("failed on gpuDirectInitDriver");
}

/*
 * gpuDirectCloseDriver
 */
static int		(*p_gpudirect_close_driver)() = NULL;

static void
gpuDirectCloseDriver(void)
{
	if (p_gpudirect_close_driver &&
		p_gpudirect_close_driver() != 0)
		__Elog("failed on gpuDirectCloseDriver");
}

/*
 * gpuDirectFileDescOpenByPath
 */
static int (*p_gpudirect_file_desc_open)(
	GPUDirectFileDesc *gds_fdesc,
	int rawfd,
	const char *pathname) = NULL;

static void
gpuDirectFileDescOpen(GPUDirectFileDesc *gds_fdesc,
					  int rawfd,
					  const char *pathname)
{
	if (p_gpudirect_file_desc_open)
	{
		if (p_gpudirect_file_desc_open(gds_fdesc, rawfd, pathname))
			__Elog("failed on gpuDirectFileDescOpen");
	}
	else
	{
		/* by VFS */
		struct stat st_buf;

		if (fstat(rawfd, &st_buf) != 0)
			__Elog("failed on fstat('%s'): %m", pathname);
		gds_fdesc->rawfd = rawfd;
        gds_fdesc->bytesize = st_buf.st_size;
	}
}

/*
 * gpuDirectFileDescClose
 */
static void (*p_gpudirect_file_desc_close)(
    const GPUDirectFileDesc *gds_fdesc) = NULL;

static void
gpuDirectFileDescClose(const GPUDirectFileDesc *gds_fdesc)
{
	if (p_gpudirect_file_desc_close)
	{
		p_gpudirect_file_desc_close(gds_fdesc);
	}
}

/*
 * gpuDirectMapGpuMemory
 */
static CUresult (*p_gpudirect_map_gpu_memory)(
    CUdeviceptr m_segment,
    size_t m_segment_sz,
    unsigned long *p_iomap_handle) = NULL;

static void
gpuDirectMapGpuMemory(CUdeviceptr m_segment,
					  size_t m_segment_sz,
					  unsigned long *p_iomap_handle)
{
	CUresult	rc;

	if (p_gpudirect_map_gpu_memory)
	{
		rc = p_gpudirect_map_gpu_memory(m_segment,
										m_segment_sz,
										p_iomap_handle);
		if (rc != CUDA_SUCCESS)
			__Elog("failed on gpuDirectMapGpuMemory: %s",
				 cuErrorName(rc));
	}
}

/*
 * gpuDirectUnmapGpuMemory
 */
static CUresult (*p_gpudirect_unmap_gpu_memory)(
    CUdeviceptr m_segment,
    unsigned long iomap_handle) = NULL;

static void
gpuDirectUnmapGpuMemory(CUdeviceptr m_segment,
                        unsigned long iomap_handle)
{
	CUresult	rc;

	if (p_gpudirect_unmap_gpu_memory)
	{
		rc = p_gpudirect_unmap_gpu_memory(m_segment,
										  iomap_handle);
		if (rc != CUDA_SUCCESS)
			__Elog("failed on gpuDirectUnmapGpuMemory: %s",
				 cuErrorName(rc));
	}
}

/*
 * gpuDirectFileReadIOV
 */
static int (*p_gpudirect_file_read_iov)(
    const GPUDirectFileDesc *gds_fdesc,
    CUdeviceptr m_segment,
    unsigned long iomap_handle,
    off_t m_offset,
    strom_io_vector *iovec) = NULL;

static void
gpuDirectFileReadIOV(const GPUDirectFileDesc *gds_fdesc,
					 CUdeviceptr m_segment,
					 unsigned long iomap_handle,
					 off_t m_offset,
					 strom_io_vector *iovec)
{
	if (p_gpudirect_file_read_iov)
	{
		if (p_gpudirect_file_read_iov(gds_fdesc,
									  m_segment,
									  iomap_handle,
									  m_offset,
									  iovec))
			__Elog("failed on gpuDirectFileReadIOV: %m");
	}
	else
	{
		/* by VFS */
		char   *buffer = alloca(segment_sz);
		int		i;

		for (i=0; i < iovec->nr_chunks; i++)
		{
			strom_io_chunk *ioc = &iovec->ioc[i];
			size_t		length   = ioc->nr_pages * PAGE_SIZE;
			off_t		file_pos = ioc->fchunk_id * PAGE_SIZE;
			off_t		dest_pos = m_offset + ioc->m_offset;
			char	   *host_pos = buffer;
			size_t		remained = length;
			ssize_t		nbytes;
			CUresult	rc;

			/* read from the file */
			if (file_pos >= gds_fdesc->bytesize)
				continue;
			if (file_pos + remained > gds_fdesc->bytesize)
				remained = gds_fdesc->bytesize - file_pos;
			while (remained > 0)
			{
				nbytes = pread(gds_fdesc->rawfd, host_pos, file_pos, remained);
				if (nbytes <= 0)
					__Elog("failed on pread(fpos=%lu, sz=%lu): %m",
						 file_pos, remained);
				host_pos += nbytes;
				file_pos += nbytes;
				remained -= nbytes;
			}
			/* RAM-to-GPU copy */
			rc = cuMemcpyHtoD(m_segment + dest_pos, buffer, length);
			if (rc != CUDA_SUCCESS)
				__Elog("failed on cuMemcpyHtoD: %s", cuErrorName(rc));
		}
	}
}

static void
memdump_on_corruption(const char *src_buffer,
					  const char *dst_buffer,
					  loff_t fpos, size_t total_length)
{
	long	unitsz = 16;
	long	pos;
	int		enable_dump = 0;
	int		i;

	for (pos=0; pos < total_length; pos += unitsz)
	{
		const char *src_ptr = src_buffer + pos;
		const char *dst_ptr = dst_buffer + pos;

		if (memcmp(src_ptr, dst_ptr, unitsz) != 0)
		{
			if (!enable_dump)
			{
				enable_dump = 1;
				total_length = Min(total_length, pos + 8 * unitsz);
				pos = Max(pos - 4 * unitsz, -unitsz);
				continue;
			}
			printf("- 0x%08lx ", (long)(fpos + pos));
			for (i=0; i < unitsz; i++)
			{
				if (i == unitsz / 2)
					putchar(' ');
				printf(" %02x", (int)(src_ptr[i] & 0xff));
			}
			putchar('\n');
			printf("+ 0x%08lx ", (long)(fpos + pos));
			for (i=0; i < unitsz; i++)
			{
				if (i == unitsz / 2)
					putchar(' ');
				printf(" %02x", (int)(dst_ptr[i] & 0xff));
			}
			putchar('\n');
		}
		else if (enable_dump)
		{
			printf("  0x%08lx ", (long)(fpos + pos));
			for (i=0; i < unitsz; i++)
			{
				if (i == unitsz / 2)
					putchar(' ');
				printf(" %02x", (int)(src_ptr[i] & 0xff));
			}
			putchar('\n');
		}
	}
	fprintf(stderr, "memory corruption detected\n");
	abort();
	exit(1);
}

static void
show_throughput(const char *filename, size_t file_size,
				struct timeval tv1, struct timeval tv2,
				long nr_ram2gpu, long nr_ssd2gpu,
				long nr_dma_submit, long nr_dma_blocks)
{
	long		time_ms;
	double		throughput;

	time_ms = ((tv2.tv_sec * 1000 + tv2.tv_usec / 1000) -
			   (tv1.tv_sec * 1000 + tv1.tv_usec / 1000));
	throughput = (double)file_size / ((double)time_ms / 1000.0);

	if (file_size < (4UL << 10))
		printf("read: %zuBytes", file_size);
	else if (file_size < (4UL << 20))
		printf("read: %.2fKB", (double)file_size / (double)(1UL << 10));
	else if (file_size < (4UL << 30))
		printf("read: %.2fMB", (double)file_size / (double)(1UL << 20));
	else
		printf("read: %.2fGB", (double)file_size / (double)(1UL << 30));

	if (time_ms < 4000UL)
		printf(", time: %lums", time_ms);
	else
		printf(", time: %.2fsec", (double)time_ms / 1000.0);

	if (throughput < (double)(4UL << 10))
		printf(", throughput: %zuB/s\n", (size_t)throughput);
	else if (throughput < (double)(4UL << 20))
		printf(", throughput: %.2fKB/s\n", throughput / (double)(1UL << 10));
	else if (throughput < (double)(4UL << 30))
		printf(", throughput: %.2fMB/s\n", throughput / (double)(1UL << 20));
	else
		printf(", throughput: %.2fGB/s\n", throughput / (double)(1UL << 30));

	if (nr_ram2gpu > 0 || nr_ssd2gpu > 0)
	{
		printf("nr_ram2gpu: %ld, nr_ssd2gpu: %ld",
			   nr_ram2gpu, nr_ssd2gpu);
	}
	if (nr_dma_submit > 0)
	{
		double	avg_dma_sz = ((double)(nr_dma_blocks << 9) /
							  (double)(nr_dma_submit));
		if (avg_dma_sz > 4194304.0)
			printf(", average DMA size: %.1fMB", avg_dma_sz / 1048576.0);
		else if (avg_dma_sz > 4096.0)
			printf(", average DMA size: %.1fKB", avg_dma_sz / 1024);
		else
			printf(", average DMA size: %.0fb", avg_dma_sz);
	}
	putchar('\n');
}

static void *
exec_gpudirect_test(void *private)
{
	worker_context *wcontext = private;
	strom_io_vector *iovec;
	unsigned long	f_pos;
	CUresult		rc;
	ssize_t			nbytes, sz;
	unsigned int	seed = (unsigned long)&seed ^ (unsigned long)time(NULL);

	rc = cuCtxSetCurrent(cuda_context);
	if (rc != CUDA_SUCCESS)
		__Elog("failed on cuCtxSetCurrent: %s", cuErrorName(rc));

	iovec = alloca(offsetof(strom_io_vector, ioc[1]));
	for (;;)
	{
		size_t			drift = 0;

		if (address_randomize)
			drift = (rand_r(&seed) % address_randomize) & ~7UL;

		f_pos = __atomic_fetch_add(&curr_fpos,
								   segment_sz,
								   __ATOMIC_SEQ_CST);
		assert((f_pos & (PAGE_SIZE-1)) == 0);
		if (f_pos >= gds_fdesc.bytesize)
			break;	/* end of the source file */
		nbytes = gds_fdesc.bytesize - f_pos;
		if (nbytes > segment_sz)
			nbytes = segment_sz;
		iovec->nr_chunks = 1;
		iovec->ioc[0].m_offset  = drift;
		iovec->ioc[0].fchunk_id = f_pos / PAGE_SIZE;
		iovec->ioc[0].nr_pages  = (nbytes + PAGE_SIZE - 1) / PAGE_SIZE;

		gpuDirectFileReadIOV(&gds_fdesc,
							 wcontext->dev_buffer,
							 wcontext->dev_mhandle,
							 wcontext->dev_moffset,
							 iovec);
		/* corruption checks? */
		if (enable_checks)
		{
			rc = cuMemcpyDtoH(wcontext->dst_buffer,
							  wcontext->dev_buffer + wcontext->dev_moffset + drift,
							  nbytes);
			if (rc != CUDA_SUCCESS)
				__Elog("failed on cuMemcpyDtoH: %s", cuErrorName(rc));

			/* read file via VFS */
			sz = pread(filedesc,
					   wcontext->src_buffer,
					   nbytes,
					   f_pos);
			if (sz < 0)
				__Elog("failed on pread(%d, %p, %ld, %ld): %m", gds_fdesc.rawfd, wcontext->src_buffer, nbytes, f_pos);
			if (sz < nbytes)
				__Elog("pread(2) read shorter than the required (%ld of %ld)",
					 sz, nbytes);
			if (memcmp(wcontext->dst_buffer,
					   wcontext->src_buffer,
					   nbytes) != 0)
			{
				memdump_on_corruption(wcontext->src_buffer,
									  wcontext->dst_buffer,
									  f_pos,
									  nbytes);
			}
		}
	}
	return NULL;
}

/*
 * usage
 */
static void usage(const char *cmdname)
{
	fprintf(stderr,
			"usage: %s [OPTIONS] <filename>\n"
			"    -d <device index>:          (default 0)\n"
			"    -k (nvme_strom|cufile|vfs)  (default: auto)\n"
			"    -n <num of segments>:       (default 6)\n"
			"    -s <segment size in MB>:    (default 32MB)\n"
			"    -c : Enables corruption check (default off)\n"
			"    -r : Enables address randomization (default off)\n"
			"    -h : Print this message   (default off)\n"
			"    -f([<i/o size in KB>]): Test by VFS access (default off)\n",
			basename(strdup(cmdname)));
	exit(1);
}

/*
 * parse_options
 */
static const char *
parse_options(int argc, char * const argv[])
{
	int		code;
	char   *end;

	while ((code = getopt(argc, argv, "d:k:n:s:crh")) >= 0)
	{
		switch (code)
		{
			case 'd':		/* device index */
				cuda_dindex = strtol(optarg, &end, 10);
				if (*end != '\0')
					usage(argv[0]);
				break;
			case 'k':		/* kernel driver */
				if (strcmp(optarg, "nvme_strom") == 0 ||
					strcmp(optarg, "cufile") == 0 ||
					strcmp(optarg, "vfs") == 0)
					kernel_driver = optarg;
				else
					usage(argv[0]);
				break;
			case 'n':		/* number of chunks */
				nr_segments = atoi(optarg);
				break;
			case 's':		/* size of chunks */
				segment_sz = strtoul(optarg, &end, 10);
				if (*end != '\0')
				{
					if (strcasecmp(end, "k") == 0)
						segment_sz <<= 10;
					else if (strcasecmp(end, "m") == 0)
						segment_sz <<= 20;
					else
						usage(argv[0]);
				}
				if ((segment_sz & (PAGE_SIZE-1)) != 0)
					__Elog("segment size must be multiple of PAGE_SIZE: %s", optarg);
				break;
			case 'c':
				enable_checks = 1;
				break;
			case 'r':
				address_randomize = 4 * PAGE_SIZE;
				break;
			default:
				usage(argv[0]);
				break;
		}
	}

	if (optind + 1 != argc)
		usage(argv[0]);
	return argv[optind];
}

/* ----------------------------------------------------------------
 *
 * load_heterodb_extra
 *
 * ----------------------------------------------------------------
 */
static void *
lookup_heterodb_extra_function(void *handle, const char *symbol)
{
	void   *fn_addr;

	fn_addr = dlsym(handle, symbol);
	if (!fn_addr)
		__Elog("could not find extra symbol \"%s\" - %s", symbol, dlerror());
	//printf("[%s] at %p\n", symbol, fn_addr);
	return fn_addr;
}

#define LOOKUP_HETERODB_EXTRA_FUNCTION(symbol) \
	p_##symbol = lookup_heterodb_extra_function(handle, #symbol)

static void *
lookup_gpudirect_function(void *handle, const char *prefix, const char *symbol)
{
	char	temp[128];

	snprintf(temp, sizeof(temp), "%s__%s", prefix, symbol);

	return lookup_heterodb_extra_function(handle, temp);
}

#define LOOKUP_GPUDIRECT_EXTRA_FUNCTION(prefix,func_name) \
	p_gpudirect_##func_name = lookup_gpudirect_function(handle, prefix, #func_name)

static void
load_heterodb_extra(void)
{
	void	   *handle;

	handle = dlopen(HETERODB_EXTRA_PATHNAME,
					RTLD_NOW | RTLD_LOCAL);
	if (!handle)
		__Elog("unable to load [%s]: %m", HETERODB_EXTRA_PATHNAME);

	//LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_extra_error_data);
	//LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_extra_module_init);
	//LOOKUP_HETERODB_EXTRA_FUNCTION(heterodb_license_reload);

	if (!kernel_driver)
	{
		if (access("/proc/driver/nvidia-fs/version", F_OK) == 0)
			kernel_driver = "cufile";
		else if (access("/proc/nvme-strom", F_OK) == 0)
			kernel_driver = "nvme_strom";
		else
			kernel_driver = "vfs";
	}

	if (strcmp(kernel_driver, "cufile") == 0 ||
		strcmp(kernel_driver, "nvme_strom") == 0)
	{
		LOOKUP_GPUDIRECT_EXTRA_FUNCTION(kernel_driver,init_driver);
		LOOKUP_GPUDIRECT_EXTRA_FUNCTION(kernel_driver,close_driver);
		LOOKUP_GPUDIRECT_EXTRA_FUNCTION(kernel_driver,file_desc_open);
		LOOKUP_GPUDIRECT_EXTRA_FUNCTION(kernel_driver,file_desc_close);
		LOOKUP_GPUDIRECT_EXTRA_FUNCTION(kernel_driver,map_gpu_memory);
		LOOKUP_GPUDIRECT_EXTRA_FUNCTION(kernel_driver,unmap_gpu_memory);
		LOOKUP_GPUDIRECT_EXTRA_FUNCTION(kernel_driver,file_read_iov);
	}
	/* license reload */
	//heterodbLicenseReload();
}

/*
 * setup_cuda_context
 */
static void
setup_cuda_context(char *namebuf, size_t namebuf_sz)
{
	CUresult		rc;
	nvmlReturn_t	rv;
	size_t			largest_bar1 = 0;
	int				count;
	unsigned int	__count;
	int				start, end;
	int				i, __dindex = -1;
	char			temp[200];
	int				values[5];
	static int		attrs[5] = { CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
								 CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
								 CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
								 CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD,
								 CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID };

	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		__Elog("failed on cuInit: %s", cuErrorName(rc));

	rv = nvmlInit();
	if (rv != NVML_SUCCESS)
		__Elog("failed on nvmlInit: %s", nvmlErrorString(rv));

	rc = cuDeviceGetCount(&count);
	if (rc != CUDA_SUCCESS)
		__Elog("failed on cuDeviceGetCount: %s", cuErrorName(rc));
	rv = nvmlDeviceGetCount(&__count);
	if (rv != NVML_SUCCESS)
		__Elog("failed on nvmlDeviceGetCount: %s", nvmlErrorString(rv));
	if (count != __count)
		__Elog("CUDA and NVML returned inconsistent number of devices (%d,%d)",
			 count, __count);
	if (count == 0)
		__Elog("No GPU devices are available");

	/* check GPU brand and BAR1 memory size */
	if (cuda_dindex < 0)
	{
		start = 0;
		end = count;
	}
	else
	{
		if (cuda_dindex >= count)
			__Elog("'-d %d' is out of range", cuda_dindex);
		start = cuda_dindex;
		end = cuda_dindex + 1;
	}

	for (i=start; i < end; i++)
	{
		nvmlDevice_t    nvml_device;
		nvmlBrandType_t nvml_brand;
		nvmlBAR1Memory_t nvml_bar1;

		rv = nvmlDeviceGetHandleByIndex(i, &nvml_device);
		if (rv != NVML_SUCCESS)
			__Elog("failed on nvmlDeviceGetHandleByIndex: %s",
				 nvmlErrorString(rv));
		rv = nvmlDeviceGetBrand(nvml_device, &nvml_brand);
		if (rv != NVML_SUCCESS)
			__Elog("failed on nvmlDeviceGetBrand: %s",
				 nvmlErrorString(rv));
		rv = nvmlDeviceGetBAR1MemoryInfo(nvml_device, &nvml_bar1);
		if (rv != NVML_SUCCESS)
			__Elog("failed on nvmlDeviceGetBAR1MemoryInfo: %s",
				 nvmlErrorString(rv));
		if ((nvml_brand == NVML_BRAND_QUADRO ||
			 nvml_brand == NVML_BRAND_TESLA ||
			 nvml_brand == NVML_BRAND_NVIDIA) &&
			nvml_bar1.bar1Total > largest_bar1)
		{
			__dindex = i;
			largest_bar1 = nvml_bar1.bar1Total;
		}
	}

	if (__dindex < 0)
	{
		if (cuda_dindex < 0)
			__Elog("No Tesla or Quadro GPUs are installed");
		else
			__Elog("GPU-%d is neither Tesla nor Quadro", cuda_dindex);
	}
	cuda_dindex = __dindex;

	/* setup CUDA context */
	rc = cuDeviceGet(&cuda_device, cuda_dindex);
	if (rc != CUDA_SUCCESS)
		__Elog("failed on cuDeviceGet: %s", cuErrorName(rc));

	rc = cuDeviceGetName(temp, sizeof(temp), cuda_device);
	if (rc != CUDA_SUCCESS)
		__Elog("failed on cuDeviceGetName: %s", cuErrorName(rc));

	for (i=0; i < 5; i++)
	{
		rc = cuDeviceGetAttribute(&values[i], attrs[i], cuda_device);
		if (rc != CUDA_SUCCESS)
			__Elog("failed on cuDeviceGetAttribute: %s", cuErrorName(rc));
	}
	snprintf(namebuf, namebuf_sz,
			 "GPU%d %s (%04x:%02x:%02x.%d)",
			 cuda_dindex, temp,
			 values[0],
			 values[1],
			 values[2],
			 (values[3] != 0 ? values[4] : 0));

	rc = cuCtxCreate(&cuda_context, CU_CTX_SCHED_AUTO, cuda_dindex);
	if (rc != CUDA_SUCCESS)
		__Elog("failed on cuCtxCreate: %s", cuErrorName(rc));
}

/*
 * entrypoint of driver_test
 */
int main(int argc, char * const argv[])
{
	size_t			buffer_sz;
	CUresult		rc;
	CUdeviceptr		dev_buffer;
	void		   *src_buffer;
	void		   *dst_buffer;
	unsigned long	dev_mhandle;
	char			devname[256];
	worker_context **wcontext;
	unsigned int	i;
	long			nr_ram2gpu = 0;
	long			nr_ssd2gpu = 0;
	long			nr_dma_submit = 0;
	long			nr_dma_blocks = 0;
	struct timeval	tv1, tv2;

	PAGE_SIZE = sysconf(_SC_PAGESIZE);
	filename = parse_options(argc, argv);
	filedesc = open(filename, O_RDONLY);
	if (filedesc < 0)
		__Elog("failed on open('%s'): %m", filename);
	load_heterodb_extra();

	/* open source file */
	gpuDirectInitDriver();
	gpuDirectFileDescOpen(&gds_fdesc, filedesc, filename);
	/* setup CUDA context */
	setup_cuda_context(devname, sizeof(devname));

	/* print test scenario */
	printf("[%s] %s\nfile: %s", kernel_driver, devname, filename);
	if (gds_fdesc.bytesize < (4UL<<10))
		printf(", size: %zuB", gds_fdesc.bytesize);
	else if (gds_fdesc.bytesize < (4UL<<20))
		printf(", size: %.2fKB", (double)gds_fdesc.bytesize / (double)(1UL<<10));
	else if (gds_fdesc.bytesize < (4UL<<30))
		printf(", size: %.2fMB", (double)gds_fdesc.bytesize / (double)(1UL<<20));
	else
		printf(", size: %.2fGB", (double)gds_fdesc.bytesize / (double)(1UL<<30));

	if (segment_sz < (1UL << 20))
		printf(", buffer: %zuKB x %d\n", segment_sz >> 10, nr_segments);
	else
		printf(", buffer: %zuMB x %d\n", segment_sz >> 20, nr_segments);

	/* allocate and map device memory */
	buffer_sz = (segment_sz + address_randomize) * nr_segments;

	rc = cuMemAlloc(&dev_buffer, buffer_sz);
	if (rc != CUDA_SUCCESS)
		__Elog("failed on cuMemAlloc: %s", cuErrorName(rc));

	rc = cuMemHostAlloc(&src_buffer, buffer_sz,
						CU_MEMHOSTALLOC_PORTABLE);
	if (rc != CUDA_SUCCESS)
		__Elog("failed on cuMemHostAlloc: %s", cuErrorName(rc));

	rc = cuMemHostAlloc(&dst_buffer, buffer_sz,
						CU_MEMHOSTALLOC_PORTABLE);
	if (rc != CUDA_SUCCESS)
		__Elog("failed on cuMemHostAlloc: %s", cuErrorName(rc));

	gpuDirectMapGpuMemory(dev_buffer, buffer_sz, &dev_mhandle);

	/* set up worker's context */
	wcontext = alloca(nr_segments * sizeof(worker_context *));
	gettimeofday(&tv1, NULL);
	for (i=0; i < nr_segments; i++)
	{
		size_t	offset = i * (segment_sz + address_randomize);

		wcontext[i] = alloca(sizeof(worker_context));
		memset(wcontext[i], 0, sizeof(worker_context));
		wcontext[i]->src_buffer		= (char *)src_buffer + offset;
		wcontext[i]->dst_buffer		= (char *)dst_buffer + offset;
		wcontext[i]->dev_buffer		= dev_buffer;
		wcontext[i]->dev_mhandle	= dev_mhandle;
		wcontext[i]->dev_moffset	= offset;

		errno = pthread_create(&wcontext[i]->thread, NULL,
							   exec_gpudirect_test,
							   wcontext[i]);
		if (errno != 0)
			__Elog("failed on pthread_create: %m");
	}

	/* wait for threads completion */
	for (i=0; i < nr_segments; i++)
	{
		pthread_join(wcontext[i]->thread, NULL);
		nr_ram2gpu += wcontext[i]->nr_ram2gpu;
		nr_ssd2gpu += wcontext[i]->nr_ssd2gpu;
		nr_dma_submit += wcontext[i]->nr_dma_submit;
		nr_dma_blocks += wcontext[i]->nr_dma_blocks;
	}
	gettimeofday(&tv2, NULL);

	gpuDirectUnmapGpuMemory(dev_buffer, dev_mhandle);
	gpuDirectFileDescClose(&gds_fdesc);
//	gpuDirectCloseDriver();

	show_throughput(filename, gds_fdesc.bytesize,
					tv1, tv2,
					nr_ram2gpu, nr_ssd2gpu,
					nr_dma_submit, nr_dma_blocks);
	return 0;
}
