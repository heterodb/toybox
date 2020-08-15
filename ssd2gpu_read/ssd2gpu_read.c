/*
 * ssd2gpu_read
 *
 * Test program for SSD-to-GPU Direct
 * ------
 * Copyright 2016-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2017-2020 (C) The PG-Strom Development Team
 */
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#include <assert.h>
#include <cuda.h>
#include <cufile.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include "nvme_strom.h"

/* command line options */
static int			cuda_dindex = 0;
static int			nr_segments = 6;
static size_t		segment_sz = (64UL << 20);	/* 64MB in default */
static size_t		io_unit_sz = (512UL << 10);	/* 512kB in default */
static int			use_nvidia_driver = 1;
/* misc variables */
static long			PAGE_SIZE;
static char			cuda_devname[256];
static CUdevice		cuda_device;
static CUcontext	cuda_context;
static CUdeviceptr	cuda_dbuffer;
static size_t		cuda_dbuffer_sz;
static const char  *filename;
static int			fdesc;
static size_t		filesize;
static unsigned int	file_pos = 0;
/* for nvidia driver */
static CUfileHandle_t	nvidia_file_handle;
/* for heterodb driver */
static unsigned long	heterodb_dbuffer_handle = 0UL;

#ifndef offsetof
#define offsetof(type, field)		((long) &((type *)0)->field)
#endif

#define Elog(fmt, ...)								\
	do {											\
		fprintf(stderr,"%s:%d  " fmt "\n",			\
				__FILE__,__LINE__, ##__VA_ARGS__);	\
		exit(1);									\
	} while(0)

static const char *
cudaError(CUresult rc)
{
	const char *error_name;

	if (cuGetErrorName(rc, &error_name) == CUDA_SUCCESS)
		return error_name;
	return "unknown error";
}

static const char *
cufileError(CUfileError_t rc)
{
	return cufileop_status_error(rc.err);
}





static void
usage(void)
{
	fprintf(stderr,
			"usage: ssd2gpu_read [OPTIONS] <filename>\n"
			"    -D (nvidia|heterodb):   (default: 'nvidia')\n"
			"    -d <device index>:      (default: 0)\n"
			"    -n <# of segments>:     (default: 6)\n"
			"    -s <segment size [MB]>: (default: 64)\n"
			"    -u <unit size of I/O> [kB]: (default: 512)\n");
	exit(1);
}

static int
nvme_strom_ioctl(int cmd, const void *arg)
{
	static __thread int fdesc_nvme_strom = -1;
	int		retry_count = 0;

	if (fdesc_nvme_strom < 0)
	{
	retry:
		fdesc_nvme_strom = open(NVME_STROM_IOCTL_PATHNAME, O_RDONLY);
		if (fdesc_nvme_strom < 0)
		{
			if (retry_count == 0 && errno == ENOENT)
			{
				if (system("/usr/bin/nvme_strom-modprobe") == 0)
				{
					retry_count++;
					goto retry;
				}
				errno = ENOENT;
            }
			Elog("failed on open('%s'): %m", NVME_STROM_IOCTL_PATHNAME);
        }
    }
    return ioctl(fdesc_nvme_strom, cmd, arg);
}

static void
init_nvidia_resources(void)
{
	CUfileDrvProps_t props;
	CUfileDescr_t	desc;
	CUfileError_t	rc;
	unsigned int	dstatus;
	unsigned int	dcontrol;
	unsigned int	fflags;
	struct timeval	tv1, tv2;

	rc = cuFileDriverOpen();
	if (rc.err != CU_FILE_SUCCESS)
		Elog("failed on cuFileDriverOpen: %s", cufileError(rc));

	rc = cuFileDriverGetProperties(&props);
	if (rc.err != CU_FILE_SUCCESS)
        Elog("failed on cuFileDriverGetProperties: %s", cufileError(rc));

	dstatus  = props.nvfs.dstatusflags;
	dcontrol = props.nvfs.dcontrolflags;
	fflags   = props.fflags;
	printf("GPUDirect Driver Status\n"
		   "version: %u.%u\n"
		   "poll_thresh_size: %zu\n"
		   "max_direct_io_size: %zu\n"
		   "dstatusflags: (%s%s%s%s%s%s%s )\n"
		   "dcontrolflags: (%s%s )\n"
		   "fflags: (%s%s%s )\n"
		   "max_device_cache_size: %u\n"
		   "per_buffer_cache_size: %u\n"
		   "max_device_pinned_mem_size: %u\n"
		   "max_batch_io_timeout_msecs: %u\n",
		   props.nvfs.major_version,
		   props.nvfs.minor_version,
		   props.nvfs.poll_thresh_size,
		   props.nvfs.max_direct_io_size,
		   (dstatus & (1<<CU_FILE_LUSTRE_SUPPORTED)) != 0 ? " Luster"  : "",
		   (dstatus & (1<<CU_FILE_WEKAFS_SUPPORTED)) != 0 ? " WakeFs"  : "",
		   (dstatus & (1<<CU_FILE_NFS_SUPPORTED))    != 0 ? " NFS"     : "",
		   (dstatus & (1<<CU_FILE_GPFS_SUPPORTED))   != 0 ? " GPFS"    : "",
		   (dstatus & (1<<CU_FILE_NVME_SUPPORTED))   != 0 ? " NVME"    : "",
		   (dstatus & (1<<CU_FILE_NVMEOF_SUPPORTED)) != 0 ? " NVMEoF"  : "",
		   (dstatus & (1<<CU_FILE_SCSI_SUPPORTED))   != 0 ? " SCSI"    : "",
		   (dcontrol & (1<<CU_FILE_USE_POLL_MODE))     != 0 ? " Polling" : "",
		   (dcontrol & (1<<CU_FILE_ALLOW_COMPAT_MODE)) != 0 ? " Compat"  : "",
		   (fflags & (1<<CU_FILE_DYN_ROUTING_SUPPORTED)) != 0 ? " DynRoute" : "",
		   (fflags & (1<<CU_FILE_BATCH_IO_SUPPORTED)) != 0 ? " BatchIO" : "",
		   (fflags & (1<<CU_FILE_STREAMS_SUPPORTED)) != 0 ? " Streams" : "",
		   props.max_device_cache_size,
		   props.per_buffer_cache_size,
		   props.max_device_pinned_mem_size,
		   props.max_batch_io_timeout_msecs);

	gettimeofday(&tv1, NULL);
	memset(&desc, 0, sizeof(CUfileDescr_t));
	desc.handle.fd = fdesc;
	desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	rc = cuFileHandleRegister(&nvidia_file_handle, &desc);
	if (rc.err != CU_FILE_SUCCESS)
		Elog("failed on cuFileHandleRegister: %s", cufileError(rc));
	gettimeofday(&tv2, NULL);

	printf("time to cuFileHandleRegister: %.3fms\n",
		   (double)((tv2.tv_sec - tv1.tv_sec) * 1000000 +
					(tv2.tv_usec - tv1.tv_usec)) / 1000.0);	

	gettimeofday(&tv1, NULL);
	rc = cuFileBufRegister((void *)cuda_dbuffer, cuda_dbuffer_sz, 0);
	if (rc.err != CU_FILE_SUCCESS)
		Elog("failed on cuFileBufRegister: %s", cufileError(rc));
	gettimeofday(&tv2, NULL);

	printf("time to cuFileBufRegister: %.3fms\n",
		   (double)((tv2.tv_sec - tv1.tv_sec) * 1000000 +
                    (tv2.tv_usec - tv1.tv_usec)) / 1000.0);
}

static void
init_heterodb_resources(void)
{
	StromCmd__CheckFile		uarg_check;
	StromCmd__MapGpuMemory	uarg_map;

	/*
	 * check whether file is on the supported filesystem, or not.
	 */
	memset(&uarg_check, 0, sizeof(StromCmd__CheckFile));
	uarg_check.fdesc = fdesc;
	uarg_check.nrooms = 0;	/* not see underlying devices */
	if (nvme_strom_ioctl(STROM_IOCTL__CHECK_FILE, &uarg_check) != 0)
		Elog("failed on ioctl(STROM_IOCTL__CHECK_FILE): %m");

	/*
	 * map GPU device memory
	 */
	memset(&uarg_map, 0, sizeof(StromCmd__MapGpuMemory));
	uarg_map.vaddress = cuda_dbuffer;
	uarg_map.length   = nr_segments * segment_sz;
	if (nvme_strom_ioctl(STROM_IOCTL__MAP_GPU_MEMORY, &uarg_map) != 0)
		Elog("failed on ioctl(STROM_IOCTL__MAP_GPU_MEMORY)");

	heterodb_dbuffer_handle = uarg_map.handle;
}

static void
init_common_resources(void)
{
	CUresult	rc;

	/* init CUDA Driver */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuInit: %s", cudaError(rc));
	/* init CUDA Device */
	rc = cuDeviceGet(&cuda_device, cuda_dindex);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuDeviceGet: %s", cudaError(rc));
	rc = cuDeviceGetName(cuda_devname, sizeof(cuda_devname),
						 cuda_device);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuDeviceGetName: %s", cudaError(rc));
	/* init CUDA Context */
	rc = cuCtxCreate(&cuda_context,
					 CU_CTX_SCHED_AUTO, cuda_device);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuCtxCreate: %s", cudaError(rc));
	/* allocation of device memory */
	cuda_dbuffer_sz = nr_segments * segment_sz;
	rc = cuMemAlloc(&cuda_dbuffer, cuda_dbuffer_sz);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuMemAlloc(%zu): %s", cuda_dbuffer_sz, cudaError(rc));
}


static void *
exec_heterodb_p2pread(void *arg)
{
	int		index = (uintptr_t)arg;

	

	return NULL;
}

static void *
exec_nvidia_p2pread(void *arg)
{
	off_t		doffset = (off_t)arg * segment_sz;

	for (;;)
	{
		unsigned int curr_pos;
		off_t		doffset;
		off_t		foffset;
		ssize_t		i, nbytes;

		curr_pos = __atomic_fetch_add(&file_pos, 1,
									  __ATOMIC_SEQ_CST);
		foffset = curr_pos * segment_sz;
		for (i=0; i < segment_sz; i+=io_unit_sz)
		{
			size_t		sz = io_unit_sz;

			if (foffset + i >= filesize)
				return NULL;
			if (foffset + i + sz > filesize)
				sz = filesize - (foffset + i);
			nbytes = cuFileRead(nvidia_file_handle,
								(void *)cuda_dbuffer,
								sz,
								foffset + i,
								doffset + i);
		}
	}
	return NULL;
}

static const char *
parse_options(int argc, char * const argv[])
{
	int		c;

	PAGE_SIZE = sysconf(_SC_PAGESIZE);
	while ((c = getopt(argc, argv, "D:d:n:s:u:")) >= 0)
	{
		switch (c)
		{
			case 'D':
				if (strcmp(optarg, "nvidia") == 0)
					use_nvidia_driver = 1;
				else if (strcmp(optarg, "heterodb") == 0)
					use_nvidia_driver = 0;
				else
					Elog("unknown driver type");
				break;

			case 'd':
				cuda_dindex = atoi(optarg);
				break;
			case 'n':
				nr_segments = atoi(optarg);
				break;
			case 's':
				segment_sz = ((size_t)atoi(optarg)) << 20;
				break;
			case 'u':
				io_unit_sz = ((size_t)atoi(optarg)) << 10;
				break;
			default:
				usage();
		}
	}
	if (optind + 1 != argc)
		usage();
	return argv[optind];
}

int main(int argc, char * const argv[])
{
	int				flags = O_RDONLY;
	uintptr_t		i;
	pthread_t	   *children;
	struct stat		stat_buf;
	struct timeval	tv1, tv2;
	double			duration;

	filename = parse_options(argc, argv);
	if (use_nvidia_driver)
		flags |= O_DIRECT;
	fdesc = open(filename, flags);
	if (fdesc < 0)
		Elog("failed on open('%s'): %m", filename);
	if (fstat(fdesc, &stat_buf) != 0)
		Elog("failed on fstat('%s'): %m", filename);
	filesize = (stat_buf.st_size + PAGE_SIZE - 1) & ~PAGE_SIZE;
	
	/* initialization */
	init_common_resources();
	if (use_nvidia_driver)
		init_nvidia_resources();
	else
		init_heterodb_resources();

	/* kick worker threads */
	children = alloca(sizeof(children) * nr_segments);
	gettimeofday(&tv1, NULL);
	for (i=0; i < nr_segments; i++)
	{
		if (pthread_create(&children[i], NULL,
						   use_nvidia_driver
						   ? exec_nvidia_p2pread
						   : exec_heterodb_p2pread,
						   (void *)i) != 0)
			Elog("failed on pthread_create: %m");
	}
	for (i=0; i < nr_segments; i++)
	{
		if (pthread_join(children[i], NULL) != 0)
			Elog("failed on pthread_join: %m");
	}
	gettimeofday(&tv2, NULL);
	duration = (double)((tv2.tv_sec - tv1.tv_sec) * 1000000 +
						(tv2.tv_usec - tv1.tv_usec)) / 1000.0;
	printf("file [%s] read %.3fms throughput: %.2fMB/s\n",
		   filename, duration, (double)filesize / (1000.0 * duration));
}






















