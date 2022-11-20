/*
 * heterodb_extra_internal.h
 *
 * Internal Definitions of HeteroDB Extra Package
 * --
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef __HETERODB_EXTRA_INTERNAL_H__
#define __HETERODB_EXTRA_INTERNAL_H__
#include <errno.h>
#include <stdio.h>
#include <stdbool.h>
#include <cuda.h>
#include "heterodb_extra.h"

#ifndef offsetof
#define offsetof(type, field)   ((long) &((type *)0)->field)
#endif

#ifndef lengthof
#define lengthof(array) (sizeof (array) / sizeof ((array)[0]))
#endif

#ifndef Min
#define Min(x,y)       ((x) < (y) ? (x) : (y))
#endif

#ifndef Max
#define Max(x,y)		((x) > (y) ? (x) : (y))
#endif

#ifndef MAXALIGN
#define MAXALIGN(x)		(((uintptr_t)(x) + sizeof(void *) - 1) & ~((uintptr_t)(sizeof(void *) - 1)))
#endif

/* license.c */
extern int	heterodb_license_check(unsigned int flags);
extern int	heterodb_license_reload(void);
extern ssize_t heterodb_license_query(char *buf, size_t bufsz);
extern int	heterodb_validate_device(int gpu_device_id,
									 const char *gpu_device_name,
									 const char *gpu_device_uuid);
/* gpudirect.c */
extern int cufile__file_desc_open_by_path(GPUDirectFileDesc *gds_fdesc,
										  const char *pathname);
extern int nvme_strom__file_desc_open_by_path(GPUDirectFileDesc *gds_fdesc,
											  const char *pathname);
extern int cufile__file_desc_open(GPUDirectFileDesc *gds_fdesc,
								  int rawfd, const char *pathname);
extern int nvme_strom__file_desc_open(GPUDirectFileDesc *gds_fdesc,
									  int rawfd, const char *pathname);
extern void cufile__file_desc_close(const GPUDirectFileDesc *gds_fdesc);
extern void nvme_strom__file_desc_close(const GPUDirectFileDesc *gds_fdesc);
extern CUresult cufile__map_gpu_memory(CUdeviceptr m_segment,
									   size_t m_segment_sz,
									   unsigned long *p_iomap_handle);
extern CUresult nvme_strom__map_gpu_memory(CUdeviceptr m_segment,
										   size_t m_segment_sz,
										   unsigned long *p_iomap_handle);
extern CUresult cufile__unmap_gpu_memory(CUdeviceptr m_segment,
										 unsigned long iomap_handle);
extern CUresult nvme_strom__unmap_gpu_memory(CUdeviceptr m_segment,
											 unsigned long iomap_handle);
extern int cufile__file_read_iov(const GPUDirectFileDesc *gds_fdesc,
								 CUdeviceptr m_segment,
								 unsigned long iomap_handle,
								 off_t m_offset,
								 strom_io_vector *iovec);
extern int nvme_strom__file_read_iov(const GPUDirectFileDesc *gds_fdesc,
									 CUdeviceptr m_segment,
									 unsigned long iomap_handle,
									 off_t m_offset,
									 strom_io_vector *iovec);
extern int cufile__init_driver(void);
extern int nvme_strom__init_driver(void);

/* sysfs.c */
extern int sysfs_setup_distance_map(int gpu_count,
									GpuPciDevItem *gpu_array,
									const char *manual_config);
extern int sysfs_lookup_optimal_gpus(int fdesc, int nrooms, int *optimal_gpus);
extern ssize_t sysfs_print_nvme_info(int index, char *buffer, ssize_t buffer_sz);
extern void sysfs_print_pci_tree(FILE *out);

/* misc.c */
extern char	*heterodb_extra_module_init(unsigned int pg_version_num);
extern unsigned int		heterodb_extra_pg_version_num;
extern __thread heterodb_extra_error_info heterodb_extra_error_data;

#define __Elog(fmt,...)								\
	do {											\
		fprintf(stderr, "[%s:%d@%s]" fmt "\n",		\
				__FILE__, __LINE__, __FUNCTION__,	\
				##__VA_ARGS__);						\
		exit(1);									\
	} while(0)

#endif	/* __HETERODB_EXTRA_INTERNAL_H__ */
