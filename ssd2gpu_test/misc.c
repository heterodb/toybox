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
#include "heterodb_extra_internal.h"
#include <string.h>
#include <unistd.h>

__thread heterodb_extra_error_info	heterodb_extra_error_data;
unsigned int	heterodb_extra_pg_version_num = 0;

char *
heterodb_extra_module_init(unsigned int pg_version_num)
{
	char		buf[1024];
	size_t		off = 0;
	int			rv;

	/* PostgreSQL version that loaded this extra module */
	heterodb_extra_pg_version_num = pg_version_num;

	off += snprintf(buf+off, sizeof(buf)-off,
					"api_version=%08u",
					HETERODB_EXTRA_API_VERSION);

	/* check cufile status */
	rv = access("/proc/driver/nvidia-fs/version", F_OK);
	off += snprintf(buf+off, sizeof(buf)-off,
					",cufile=%s", (rv == 0 ? "on" : "off"));

	/* check nvme_strom status */
	rv = access("/proc/nvme-strom", F_OK);
	off += snprintf(buf+off, sizeof(buf)-off,
					",nvme_strom=%s", (rv == 0 ? "on" : "off"));

	return strdup(buf);
}
