/*
 * gpudirect.c
 *
 * HeteroDB Extras for GPUDirect SQL
 * --
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda.h>
#include <cufile.h>

#include "nvme_strom.h"
#include "heterodb_extra_internal.h"

/* static variables */
static size_t	PAGE_SIZE;

/* ------------------------------------------------------------
 *
 * Thin wrapper for cuFile APIs
 *
 * ------------------------------------------------------------
 */
#define CUFILE_ERROR__DRIVER_NOT_INITIALIZED						\
	((CUfileError_t){CU_FILE_DRIVER_NOT_INITIALIZED,CUDA_SUCCESS})

/* __cuFileErrorText */
static const char *
__cuFileErrorText(CUfileError_t rv)
{
#define __ENT(x)	case x: return #x
	switch (rv.cu_err)
	{
		case CUDA_SUCCESS:
			return cufileop_status_error(rv.err);
		__ENT(CUDA_ERROR_INVALID_VALUE);
		__ENT(CUDA_ERROR_OUT_OF_MEMORY);
		__ENT(CUDA_ERROR_NOT_INITIALIZED);
		__ENT(CUDA_ERROR_DEINITIALIZED);
		__ENT(CUDA_ERROR_PROFILER_DISABLED);
		__ENT(CUDA_ERROR_PROFILER_NOT_INITIALIZED);
		__ENT(CUDA_ERROR_PROFILER_ALREADY_STARTED);
		__ENT(CUDA_ERROR_PROFILER_ALREADY_STOPPED);
		__ENT(CUDA_ERROR_STUB_LIBRARY);
		__ENT(CUDA_ERROR_NO_DEVICE);
		__ENT(CUDA_ERROR_INVALID_DEVICE);
		__ENT(CUDA_ERROR_DEVICE_NOT_LICENSED);
		__ENT(CUDA_ERROR_INVALID_IMAGE);
		__ENT(CUDA_ERROR_INVALID_CONTEXT);
		__ENT(CUDA_ERROR_CONTEXT_ALREADY_CURRENT);
		__ENT(CUDA_ERROR_MAP_FAILED);
		__ENT(CUDA_ERROR_UNMAP_FAILED);
		__ENT(CUDA_ERROR_ARRAY_IS_MAPPED);
		__ENT(CUDA_ERROR_ALREADY_MAPPED);
		__ENT(CUDA_ERROR_NO_BINARY_FOR_GPU);
		__ENT(CUDA_ERROR_ALREADY_ACQUIRED);
		__ENT(CUDA_ERROR_NOT_MAPPED);
		__ENT(CUDA_ERROR_NOT_MAPPED_AS_ARRAY);
		__ENT(CUDA_ERROR_NOT_MAPPED_AS_POINTER);
		__ENT(CUDA_ERROR_ECC_UNCORRECTABLE);
		__ENT(CUDA_ERROR_UNSUPPORTED_LIMIT);
		__ENT(CUDA_ERROR_CONTEXT_ALREADY_IN_USE);
		__ENT(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED);
		__ENT(CUDA_ERROR_INVALID_PTX);
		__ENT(CUDA_ERROR_INVALID_GRAPHICS_CONTEXT);
		__ENT(CUDA_ERROR_NVLINK_UNCORRECTABLE);
		__ENT(CUDA_ERROR_JIT_COMPILER_NOT_FOUND);
		__ENT(CUDA_ERROR_UNSUPPORTED_PTX_VERSION);
		__ENT(CUDA_ERROR_JIT_COMPILATION_DISABLED);
		__ENT(CUDA_ERROR_INVALID_SOURCE);
		__ENT(CUDA_ERROR_FILE_NOT_FOUND);
		__ENT(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND);
		__ENT(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED);
		__ENT(CUDA_ERROR_OPERATING_SYSTEM);
		__ENT(CUDA_ERROR_INVALID_HANDLE);
		__ENT(CUDA_ERROR_ILLEGAL_STATE);
		__ENT(CUDA_ERROR_NOT_FOUND);
		__ENT(CUDA_ERROR_NOT_READY);
		__ENT(CUDA_ERROR_ILLEGAL_ADDRESS);
		__ENT(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
		__ENT(CUDA_ERROR_LAUNCH_TIMEOUT);
		__ENT(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING);
		__ENT(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED);
		__ENT(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED);
		__ENT(CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE);
		__ENT(CUDA_ERROR_CONTEXT_IS_DESTROYED);
		__ENT(CUDA_ERROR_ASSERT);
		__ENT(CUDA_ERROR_TOO_MANY_PEERS);
		__ENT(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED);
		__ENT(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED);
		__ENT(CUDA_ERROR_HARDWARE_STACK_ERROR);
		__ENT(CUDA_ERROR_ILLEGAL_INSTRUCTION);
		__ENT(CUDA_ERROR_MISALIGNED_ADDRESS);
		__ENT(CUDA_ERROR_INVALID_ADDRESS_SPACE);
		__ENT(CUDA_ERROR_INVALID_PC);
		__ENT(CUDA_ERROR_LAUNCH_FAILED);
		__ENT(CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE);
		__ENT(CUDA_ERROR_NOT_PERMITTED);
		__ENT(CUDA_ERROR_NOT_SUPPORTED);
		__ENT(CUDA_ERROR_SYSTEM_NOT_READY);
		__ENT(CUDA_ERROR_SYSTEM_DRIVER_MISMATCH);
		__ENT(CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE);
		__ENT(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
		__ENT(CUDA_ERROR_STREAM_CAPTURE_INVALIDATED);
		__ENT(CUDA_ERROR_STREAM_CAPTURE_MERGE);
		__ENT(CUDA_ERROR_STREAM_CAPTURE_UNMATCHED);
		__ENT(CUDA_ERROR_STREAM_CAPTURE_UNJOINED);
		__ENT(CUDA_ERROR_STREAM_CAPTURE_ISOLATION);
		__ENT(CUDA_ERROR_STREAM_CAPTURE_IMPLICIT);
		__ENT(CUDA_ERROR_CAPTURED_EVENT);
		__ENT(CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD);
		__ENT(CUDA_ERROR_TIMEOUT);
		__ENT(CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE);
		default:
			return "CUDA_ERROR_UNKNOWN";
	}
#undef __ENT
}

#if 0
/* cuFileDriverOpen */
static CUfileError_t (*p_cuFileDriverOpen)(void) = NULL;

CUfileError_t
cuFileDriverOpen(void)
{
	if (!p_cuFileDriverOpen)
		return CUFILE_ERROR__DRIVER_NOT_INITIALIZED;
	return p_cuFileDriverOpen();
}

/* cuFileDriverClose */
static CUfileError_t (*p_cuFileDriverClose)(void) = NULL;

CUfileError_t
cuFileDriverClose(void)
{
	if (!p_cuFileDriverClose)
		return CUFILE_ERROR__DRIVER_NOT_INITIALIZED;
	return p_cuFileDriverClose();
}

/* cuFileDriverGetProperties */
static CUfileError_t (*p_cuFileDriverGetProperties)(
	CUfileDrvProps_t *props) = NULL;

CUfileError_t
cuFileDriverGetProperties(CUfileDrvProps_t *props)
{
	if (!p_cuFileDriverGetProperties)
		return CUFILE_ERROR__DRIVER_NOT_INITIALIZED;
	return p_cuFileDriverGetProperties(props);
}

/* cuFileDriverSetPollMode */
static CUfileError_t (*p_cuFileDriverSetPollMode)(
	bool poll,
	size_t poll_threshold_size) = NULL;

CUfileError_t
cuFileDriverSetPollMode(bool poll, size_t poll_threshold_size)
{
	if (!p_cuFileDriverSetPollMode)
		return CUFILE_ERROR__DRIVER_NOT_INITIALIZED;
	return p_cuFileDriverSetPollMode(poll, poll_threshold_size);
}

/* cuFileDriverSetMaxDirectIOSize */
static CUfileError_t (*p_cuFileDriverSetMaxDirectIOSize)(
	size_t max_direct_io_size) = NULL;

CUfileError_t
cuFileDriverSetMaxDirectIOSize(size_t max_direct_io_size)
{
	if (!p_cuFileDriverSetMaxDirectIOSize)
		return CUFILE_ERROR__DRIVER_NOT_INITIALIZED;
	return p_cuFileDriverSetMaxDirectIOSize(max_direct_io_size);
}

/* cuFileDriverSetMaxCacheSize */
static CUfileError_t (*p_cuFileDriverSetMaxCacheSize)(
	size_t max_cache_size) = NULL;

CUfileError_t
cuFileDriverSetMaxCacheSize(size_t max_cache_size)
{
	if (!p_cuFileDriverSetMaxCacheSize)
		return CUFILE_ERROR__DRIVER_NOT_INITIALIZED;
	return p_cuFileDriverSetMaxCacheSize(max_cache_size);
}

/* cuFileDriverSetMaxPinnedMemSize */
static CUfileError_t (*p_cuFileDriverSetMaxPinnedMemSize)(
	size_t max_pinned_size) = NULL;

CUfileError_t
cuFileDriverSetMaxPinnedMemSize(size_t max_pinned_size)
{
	if (!p_cuFileDriverSetMaxPinnedMemSize)
		return CUFILE_ERROR__DRIVER_NOT_INITIALIZED;
	return p_cuFileDriverSetMaxPinnedMemSize(max_pinned_size);
}

/* cuFileHandleRegister */
static CUfileError_t (*p_cuFileHandleRegister)(
	CUfileHandle_t *fh,
	CUfileDescr_t *descr) = NULL;

CUfileError_t
cuFileHandleRegister(CUfileHandle_t *fh, CUfileDescr_t *descr)
{
	if (!p_cuFileHandleRegister)
		return CUFILE_ERROR__DRIVER_NOT_INITIALIZED;
	return p_cuFileHandleRegister(fh, descr);
}

/* cuFileHandleDeregister */
static void (*p_cuFileHandleDeregister)(
	CUfileHandle_t fh) = NULL;

void
cuFileHandleDeregister(CUfileHandle_t fh)
{
	if (p_cuFileHandleDeregister)
		return p_cuFileHandleDeregister(fh);
}

/* cuFileBufRegister */
static CUfileError_t (*p_cuFileBufRegister)(
	const void *devPtr_base,
	size_t length,
	int flags) = NULL;

CUfileError_t
cuFileBufRegister(const void *devPtr_base, size_t length, int flags)
{
	if (!p_cuFileBufRegister)
		return CUFILE_ERROR__DRIVER_NOT_INITIALIZED;
	return p_cuFileBufRegister(devPtr_base, length, flags);
}

/* cuFileBufDeregister */
static CUfileError_t (*p_cuFileBufDeregister)(
	const void *devPtr_base) = NULL;

CUfileError_t cuFileBufDeregister(const void *devPtr_base)
{
	if (!p_cuFileBufDeregister)
		return CUFILE_ERROR__DRIVER_NOT_INITIALIZED;
	return p_cuFileBufDeregister(devPtr_base);
}

/* cuFileRead */
static ssize_t (*p_cuFileRead)(
	CUfileHandle_t fh,
	void *devPtr_base,
	size_t size,
	off_t file_offset,
	off_t devPtr_offset) = NULL;

ssize_t
cuFileRead(CUfileHandle_t fh,
		   void *devPtr_base,
		   size_t size,
		   off_t file_offset,
		   off_t devPtr_offset)
{
	if (!p_cuFileRead)
		return -CU_FILE_DRIVER_NOT_INITIALIZED;
	return p_cuFileRead(fh, devPtr_base, size, file_offset, devPtr_offset);
}

/* cuFileWrite */
static ssize_t (*p_cuFileWrite)(
	CUfileHandle_t fh,
	const void *devPtr_base,
	size_t size,
	off_t file_offset,
	off_t devPtr_offset) = NULL;

ssize_t cuFileWrite(CUfileHandle_t fh,
					const void *devPtr_base,
					size_t size,
					off_t file_offset,
					off_t devPtr_offset)
{
	if (!p_cuFileWrite)
		return -CU_FILE_DRIVER_NOT_INITIALIZED;
	return p_cuFileWrite(fh,devPtr_base,size,file_offset,devPtr_offset);
}

/* lookup_cufile_function */
static void *
lookup_cufile_function(void *handle, const char *func_name)
{
	void   *func_addr = dlsym(handle, func_name);

	if (!func_addr)
		Elog("could not find cuFile symbol \"%s\" - %s", func_name, dlerror());
	return func_addr;
}
#define LOOKUP_CUFILE_FUNCTION(func_name)		\
	(p_##func_name = lookup_cufile_function(handle, #func_name)) != NULL
#endif

/* ------------------------------------------------------------
 *
 * Thin wrapper for NVME-Strom APIs
 *
 * ------------------------------------------------------------
 */

/*
 * nvme_strom_ioctl
 */
static int		fdesc_nvme_strom = -1;
static int
nvme_strom_ioctl(int cmd, void *arg)
{
	int		fdesc = fdesc_nvme_strom;

	if (fdesc < 0)
		return -1;
	return ioctl(fdesc, cmd, arg);
}

/*
 * GPUDirect SQL APIs
 */

/*
 * PREFIX_file_desc_open_by_path
 */
int
cufile__file_desc_open_by_path(GPUDirectFileDesc *gds_fdesc,
							   const char *pathname)
{
	CUfileDescr_t desc;
	CUfileError_t rv;
	struct stat	st_buf;
	int			rawfd;

	rawfd = open(pathname, O_RDONLY | O_DIRECT);
	if (rawfd < 0)
	{
		Elog("failed on open('%s'): %m", pathname);
		return -1;
	}
	if (fstat(rawfd, &st_buf) != 0)
	{
		Elog("failed on fstat('%s'): %m", pathname);
		close(rawfd);
		return -1;
	}
	memset(&desc, 0, sizeof(CUfileDescr_t));
	desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
	desc.handle.fd = rawfd;
	rv = cuFileHandleRegister(&gds_fdesc->fhandle, &desc);
	if (rv.cu_err != CUDA_SUCCESS || rv.err != CU_FILE_SUCCESS)
	{
		close(rawfd);
		Elog("failed on cuFileHandleRegister('%s'): %s",
			 pathname, __cuFileErrorText(rv));
		return -1;
	}
	gds_fdesc->rawfd = rawfd;
	gds_fdesc->bytesize = st_buf.st_size;
	
	return 0;
}

static int
__nvme_strom__file_desc_open_common(GPUDirectFileDesc *gds_fdesc,
                                    int rawfd,
                                    const char *pathname)
{
	StromCmd__CheckFile	cmd;
	struct stat		st_buf;

	if (fstat(rawfd, &st_buf) != 0)
	{
		Elog("failed on fstat('%s'): %m", pathname);
	}
	else if (!S_ISREG(st_buf.st_mode))
	{
		Elog("'%s' is not a regular file", pathname);
	}
	else
	{
		memset(&cmd, 0, sizeof(StromCmd__CheckFile));
		cmd.fdesc = rawfd;
		cmd.nrooms = 0;
		if (nvme_strom_ioctl(STROM_IOCTL__CHECK_FILE, &cmd) == 0)
		{
			gds_fdesc->rawfd = rawfd;
			gds_fdesc->bytesize = st_buf.st_size;
			return 0;
		}
		Elog("nvme_strom does not support '%s'", pathname);
	}
	close(rawfd);
	return -1;
}

int
nvme_strom__file_desc_open_by_path(GPUDirectFileDesc *gds_fdesc,
								   const char *pathname)
{
	int			rawfd;

	rawfd = open(pathname, O_RDONLY, 0600);
	if (rawfd < 0)
	{
		Elog("failed on open('%s'): %m", pathname);
		return -1;
	}
	return __nvme_strom__file_desc_open_common(gds_fdesc, rawfd, pathname);
}

/*
 * PREFIX__file_desc_open
 */
int
cufile__file_desc_open(GPUDirectFileDesc *gds_fdesc,
					   int rawfd, const char *pathname)
{
	/*
     * NVIDIA GPUDirect Storage (cuFile) requires to open the
     * source file with O_DIRECT, and it ignores the flags
     * changed by dup3(). So, we always tried to open a new
     * file descriptor with O_DIRECT flag.
     */
	return cufile__file_desc_open_by_path(gds_fdesc, pathname);
}

int
nvme_strom__file_desc_open(GPUDirectFileDesc *gds_fdesc,
						   int rawfd, const char *pathname)
{
	if (rawfd < 0)
		return nvme_strom__file_desc_open_by_path(gds_fdesc, pathname);

	rawfd = dup(rawfd);
	if (rawfd < 0)
	{
		Elog("failed on dup(2): %m");
		return -1;
	}
	return __nvme_strom__file_desc_open_common(gds_fdesc, rawfd, pathname);
}

/*
 * PREFIX__file_desc_close
 */
void
cufile__file_desc_close(const GPUDirectFileDesc *gds_fdesc)
{
	cuFileHandleDeregister(gds_fdesc->fhandle);
	if (close(gds_fdesc->rawfd))
		Elog("failed on close(2): %m\n");
}

void
nvme_strom__file_desc_close(const GPUDirectFileDesc *gds_fdesc)
{
	if (close(gds_fdesc->rawfd))
		Elog("failed on close(2): %m\n");
}

/*
 * PREFIX__map_gpu_memory
 */
CUresult
cufile__map_gpu_memory(CUdeviceptr m_segment,
					   size_t m_segment_sz,
					   unsigned long *p_iomap_handle)
{
	CUfileError_t	rv;

	rv = cuFileBufRegister((void *)m_segment, m_segment_sz, 0);
	if (rv.err != CU_FILE_SUCCESS)
		return CUDA_ERROR_MAP_FAILED;
	if (rv.cu_err != CUDA_SUCCESS)
		return rv.cu_err;
	*p_iomap_handle = 0UL;		/* unused, in cuFile mode */

	return CUDA_SUCCESS;
}

CUresult
nvme_strom__map_gpu_memory(CUdeviceptr m_segment,
						   size_t m_segment_sz,
						   unsigned long *p_iomap_handle)
{
	StromCmd__MapGpuMemory cmd;

	memset(&cmd, 0, sizeof(StromCmd__MapGpuMemory));
	cmd.vaddress = m_segment;
	cmd.length = m_segment_sz;
	if (nvme_strom_ioctl(STROM_IOCTL__MAP_GPU_MEMORY, &cmd) != 0)
		return CUDA_ERROR_MAP_FAILED;
	*p_iomap_handle = cmd.handle;
	return CUDA_SUCCESS;
}

/*
 * PREFIX__unmap_gpu_memory
 */
CUresult
cufile__unmap_gpu_memory(CUdeviceptr m_segment,
						 unsigned long iomap_handle)
{
	CUfileError_t rv;

	assert(iomap_handle == 0);
	rv = cuFileBufDeregister((void *)m_segment);
	if (rv.err != CU_FILE_SUCCESS)
		return CUDA_ERROR_UNMAP_FAILED;
	return rv.cu_err;
}

CUresult
nvme_strom__unmap_gpu_memory(CUdeviceptr m_segment,
							 unsigned long iomap_handle)
{
	/* cuMemFree() invokes callback to unmap device memory in the kernel side */
	return CUDA_SUCCESS;
}

/*
 * PREFIX__file_read_iov
 */
int
cufile__file_read_iov(const GPUDirectFileDesc *gds_fdesc,
					  CUdeviceptr m_segment,
					  unsigned long iomap_handle,
					  off_t m_offset,
					  strom_io_vector *iovec)
{
	static size_t io_unitsz = (32UL << 20);	/* 32MB */
	int		i;

	for (i=0; i < iovec->nr_chunks; i++)
	{
		strom_io_chunk *ioc = &iovec->ioc[i];
		size_t	remained = ioc->nr_pages * PAGE_SIZE;
		off_t	file_pos = ioc->fchunk_id * PAGE_SIZE;
		off_t	dest_pos = m_offset + ioc->m_offset;

		/* cut off the file tail */
		if (file_pos >= gds_fdesc->bytesize)
			continue;
		if (file_pos + remained > gds_fdesc->bytesize)
			remained = gds_fdesc->bytesize - file_pos;

		while (remained > 0)
		{
			ssize_t	sz, nbytes;

			sz = Min(remained, io_unitsz);
			nbytes = cuFileRead(gds_fdesc->fhandle,
								(void *)m_segment,
								sz,
								file_pos,
								dest_pos);
			if (nbytes != sz)
			{
				Elog("failed on cuFileRead: nbytes=%zd of len=%zd, at %lu",
					 nbytes, sz, file_pos);
				return -1;
			}
			file_pos += sz;
			dest_pos += sz;
			remained -= sz;
		}
	}
	return 0;
}

int
nvme_strom__file_read_iov(const GPUDirectFileDesc *gds_fdesc,
						  CUdeviceptr m_segment,
						  unsigned long iomap_handle,
						  off_t m_offset,
						  strom_io_vector *iovec)
{
	StromCmd__MemCopySsdToGpuRaw cmd;
	StromCmd__MemCopyWait __cmd;

	assert(iomap_handle != 0UL);
	memset(&cmd, 0, sizeof(StromCmd__MemCopySsdToGpuRaw));
	cmd.handle    = iomap_handle;
	cmd.offset    = m_offset;
	cmd.file_desc = gds_fdesc->rawfd;
	cmd.nr_chunks = iovec->nr_chunks;
	cmd.page_sz   = PAGE_SIZE;
	cmd.io_chunks = iovec->ioc;

	if (nvme_strom_ioctl(STROM_IOCTL__MEMCPY_SSD2GPU_RAW, &cmd) != 0)
	{
		Elog("failed on STROM_IOCTL__MEMCPY_SSD2GPU_RAW: %m\n");
		return -1;
	}

	memset(&__cmd, 0, sizeof(StromCmd__MemCopyWait));
	__cmd.dma_task_id = cmd.dma_task_id;
	//TODO: stat
	while (nvme_strom_ioctl(STROM_IOCTL__MEMCPY_WAIT, &__cmd) != 0)
	{
		if (errno != EINTR)
		{
			Elog("failed on STROM_IOCTL__MEMCPY_WAIT): %m\n");
			return -1;
		}
	}
	return 0;
}

/*
 * PREFIX__init_driver
 */
int
cufile__init_driver(void)
{
	/* system properties */
	PAGE_SIZE = sysconf(_SC_PAGESIZE);

	/* check nvidia-fs */
	if (access("/proc/driver/nvidia-fs/version", F_OK) != 0)
	{
		Elog("it looks nvidia-fs kernel module is not loaded");
		return 1;
	}
#if 0
	handle = dlopen("libcufile.so", RTLD_NOW | RTLD_LOCAL);
	if (!handle)
	{
		handle = dlopen("/usr/local/cuda/lib64/libcufile.so",
						RTLD_NOW | RTLD_LOCAL);
		if (!handle)
		{
			Elog("failed on dlopen('libcufile.so'): %m");
			return 1;
		}
	}

	if (LOOKUP_CUFILE_FUNCTION(cuFileDriverOpen) &&
		LOOKUP_CUFILE_FUNCTION(cuFileDriverClose) &&
		LOOKUP_CUFILE_FUNCTION(cuFileDriverGetProperties) &&
		LOOKUP_CUFILE_FUNCTION(cuFileDriverSetPollMode) &&
		LOOKUP_CUFILE_FUNCTION(cuFileDriverSetMaxDirectIOSize) &&
		LOOKUP_CUFILE_FUNCTION(cuFileDriverSetMaxCacheSize) &&
		LOOKUP_CUFILE_FUNCTION(cuFileDriverSetMaxPinnedMemSize) &&
		LOOKUP_CUFILE_FUNCTION(cuFileHandleRegister) &&
		LOOKUP_CUFILE_FUNCTION(cuFileHandleDeregister) &&
		LOOKUP_CUFILE_FUNCTION(cuFileBufRegister) &&
		LOOKUP_CUFILE_FUNCTION(cuFileBufDeregister) &&
		LOOKUP_CUFILE_FUNCTION(cuFileRead) &&
		LOOKUP_CUFILE_FUNCTION(cuFileWrite))
	{
		return 0;	/* Ok */
	}
	/* symbol not found - cleanup */
	dlclose(handle);

	p_cuFileDriverOpen = NULL;
	p_cuFileDriverClose = NULL;
	p_cuFileDriverGetProperties = NULL;
	p_cuFileDriverSetPollMode = NULL;
	p_cuFileDriverSetMaxDirectIOSize = NULL;
	p_cuFileDriverSetMaxCacheSize = NULL;
	p_cuFileDriverSetMaxPinnedMemSize = NULL;
	p_cuFileHandleRegister = NULL;
	p_cuFileHandleDeregister = NULL;
	p_cuFileBufRegister = NULL;
	p_cuFileBufDeregister = NULL;
	p_cuFileRead = NULL;
	p_cuFileWrite = NULL;

	return 1;

	p_cuFileDriverOpen = cuFileDriverOpen;
	p_cuFileDriverClose = cuFileDriverClose;
	p_cuFileDriverGetProperties = cuFileDriverGetProperties;
	p_cuFileDriverSetPollMode = cuFileDriverSetPollMode;
	p_cuFileDriverSetMaxDirectIOSize = cuFileDriverSetMaxDirectIOSize;
	p_cuFileDriverSetMaxCacheSize = cuFileDriverSetMaxCacheSize;
	p_cuFileDriverSetMaxPinnedMemSize = cuFileDriverSetMaxPinnedMemSize;
	p_cuFileHandleRegister = cuFileHandleRegister;
	p_cuFileHandleDeregister = cuFileHandleDeregister;
	p_cuFileBufRegister = cuFileBufRegister;
	p_cuFileBufDeregister = cuFileBufDeregister;
	p_cuFileRead = cuFileRead;
	p_cuFileWrite = cuFileWrite;
#endif
	return 0;
}

int
nvme_strom__init_driver(void)
{
	/* system properties */
	PAGE_SIZE = sysconf(_SC_PAGESIZE);
	return 0;
}

/*
 * PREFIX__open_driver
 */
static bool		cufile_driver_opened = false;

int
cufile__open_driver(void)
{
	CUfileError_t	rv;

	if (cufile_driver_opened)
	{
		Elog("cuFileDriverOpen is called twice");
		return -1;
	}

	rv = cuFileDriverOpen();
	if (rv.err != CU_FILE_SUCCESS || rv.cu_err != CUDA_SUCCESS)
	{
		Elog("failed on cuFileDriverOpen: %s", __cuFileErrorText(rv));
		return -1;
	}
	cufile_driver_opened = true;
	return 0;
}

int
nvme_strom__open_driver(void)
{
	if (fdesc_nvme_strom < 0)
	{
		fdesc_nvme_strom = open(NVME_STROM_IOCTL_PATHNAME, O_RDONLY);
		if (fdesc_nvme_strom < 0)
			return -1;
	}
	return 0;
}

/*
 * PREFIX__close_driver
 */
int
cufile__close_driver(void)
{
	fprintf(stderr, "cufile__close_driver is called\n");
	if (cufile_driver_opened)
	{
		CUfileError_t	rv;

		rv = cuFileDriverClose();
		fprintf(stderr, "cuFileDriverClose = %d %d\n", rv.err, rv.cu_err);
		if (rv.err != CU_FILE_SUCCESS || rv.cu_err != CUDA_SUCCESS)
		{
			Elog("failed on cuFileDriverClose: %s", __cuFileErrorText(rv));
			return -1;
		}
		cufile_driver_opened = false;
	}
	return 0;
}

int
nvme_strom__close_driver(void)
{
	if (fdesc_nvme_strom >= 0)
	{
		close(fdesc_nvme_strom);
		fdesc_nvme_strom = -1;
	}
	return 0;
}
