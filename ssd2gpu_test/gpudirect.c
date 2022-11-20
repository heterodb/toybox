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
	const char *str;

	if (rv.cu_err != CUDA_SUCCESS)
		cuGetErrorName(rv.cu_err, &str);
	else
		str = cufileop_status_error(rv.err);
	return str;
}

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
		__Elog("failed on open('%s'): %m", pathname);
		return -1;
	}
	if (fstat(rawfd, &st_buf) != 0)
	{
		__Elog("failed on fstat('%s'): %m", pathname);
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
		__Elog("failed on cuFileHandleRegister('%s'): %s",
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
		__Elog("failed on fstat('%s'): %m", pathname);
	}
	else if (!S_ISREG(st_buf.st_mode))
	{
		__Elog("'%s' is not a regular file", pathname);
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
		__Elog("nvme_strom does not support '%s'", pathname);
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
		__Elog("failed on open('%s'): %m", pathname);
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
		__Elog("failed on dup(2): %m");
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
		__Elog("failed on close(2): %m\n");
}

void
nvme_strom__file_desc_close(const GPUDirectFileDesc *gds_fdesc)
{
	if (close(gds_fdesc->rawfd))
		__Elog("failed on close(2): %m\n");
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
				__Elog("failed on cuFileRead: nbytes=%zd of len=%zd, at %lu",
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
		__Elog("failed on STROM_IOCTL__MEMCPY_SSD2GPU_RAW: %m\n");
		return -1;
	}

	memset(&__cmd, 0, sizeof(StromCmd__MemCopyWait));
	__cmd.dma_task_id = cmd.dma_task_id;
	//TODO: stat
	while (nvme_strom_ioctl(STROM_IOCTL__MEMCPY_WAIT, &__cmd) != 0)
	{
		if (errno != EINTR)
		{
			__Elog("failed on STROM_IOCTL__MEMCPY_WAIT): %m\n");
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
		__Elog("it looks nvidia-fs kernel module is not loaded");
		return 1;
	}
	return 0;
}

int
nvme_strom__init_driver(void)
{
	/* system properties */
	PAGE_SIZE = sysconf(_SC_PAGESIZE);
	/* check nvme-strom */
	if (access("/proc/nvme-strom", F_OK) != 0)
	{
		__Elog("it looks nvme-strom kernel module is not loaded");
		return 1;
	}
	return 0;
}

/*
 * PREFIX__open_driver
 */
int
cufile__open_driver(void)
{
	CUfileError_t	rv;

	rv = cuFileDriverOpen();
	if (rv.err != CU_FILE_SUCCESS || rv.cu_err != CUDA_SUCCESS)
	{
		__Elog("failed on cuFileDriverOpen: %s", __cuFileErrorText(rv));
		return -1;
	}
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
	CUfileError_t	rv;

	rv = cuFileDriverClose();
	if (rv.err != CU_FILE_SUCCESS || rv.cu_err != CUDA_SUCCESS)
	{
		__Elog("failed on cuFileDriverClose: %s", __cuFileErrorText(rv));
		return -1;
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
