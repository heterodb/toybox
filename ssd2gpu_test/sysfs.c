/*
 * sysfs.c
 *
 * routines to traverse sysfs tree
 * --
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include <assert.h>
#include <ctype.h>
#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include "heterodb_extra_internal.h"

/* GPU attributes */
static GpuPciDevItem	   *gpuArray = NULL;
static int					gpuNitems = 0;

/* NVME attributes */
typedef struct NvmePciDevItem
{
	int			nvme_major;		/* major of /dev/nvmeXXn1 */
	int			nvme_minor;		/* minor of /dev/nvmeXXn1 */
	char		nvme_name[40];	/* nvmeXXnX */
	char		nvme_model[120];/* model name, if any */
	const char *cpu_affinity;	/* CPU of the NVME device */
	int			pci_domain;		/* DDDD of DDDD:bb:dd.f */
	int			pci_bus_id;		/* bb of DDDD:bb:dd.f */
	int			pci_dev_id;		/* dd of DDDD:bb:dd.f */
	int			pci_func_id;	/* f of DDDD:bb:dd.f */
	int			__distance;		/* internal use */
	int			num_optimal_gpus;
	int			optimal_gpus[1];/* optimal GPU's cuda_dindex */
} NvmePciDevItem;

static NvmePciDevItem	  **nvmeArray = NULL;
static int					nvmeNrooms = 0;
static int					nvmeNitems = 0;

/* PCI-E device tree */
typedef struct PCIDevEntry
{
	struct PCIDevEntry *parent;
	struct PCIDevEntry *siblings;
	struct PCIDevEntry *children;
	int			domain;
	int			bus_id;
	int			dev_id;
	int			func_id;
	int			depth;
	GpuPciDevItem  *gpu;	/* if GPU device */
	NvmePciDevItem *nvme;	/* if NVME device */
	char		cpu_affinity[1];
} PCIDevEntry;

static PCIDevEntry		   *pcieRoot = NULL;

/* NFS Volumes */
typedef struct NfsVolumeItem
{
	int			mount_id;
	char	   *mount_path;
	char	   *mount_fs;
	char	   *mount_dev;
	int			num_optimal_gpus;
	int			optimal_gpus[1];
} NfsVolumeItem;

static NfsVolumeItem	  **nfsvArray = NULL;
static int					nfsvNrooms = 0;
static int					nfsvNitems = 0;

static struct BlockDevItem *
__sysfs_lookup_optimal_gpu(unsigned int major, unsigned int minor);

/*
 * __trim
 */
static inline char *
__trim(char *token)
{
	char   *tail = token + strlen(token) - 1;

	while (isspace(*token))
		token++;
	while (tail >= token && isspace(*tail))
		*tail-- = '\0';
	
	return token;
}

/*
 * __strisdigit
 */
static inline bool
__strisdigit(const char *s)
{
	const char *c = s;

	while (isdigit(*c))
		c++;
	return (c != s && *c == '\0');
}

/*
 * sysfs_read_line
 */
static const char *
__sysfs_read_line(const char *path, char *buffer, size_t buflen)
{
	int			fdesc;
	ssize_t		sz;

	fdesc = open(path, O_RDONLY);
	if (fdesc < 0)
		return NULL;
retry:
	sz = read(fdesc, buffer, buflen-1);
	if (sz < 0)
	{
		if (errno == EINTR)
			goto retry;
		close(fdesc);
		return NULL;
	}
	buffer[sz] = '\0';
	close(fdesc);

	return __trim(buffer);
}

static const char *
sysfs_read_line(const char *path)
{
	static char	linebuf[2048];

	return __sysfs_read_line(path, linebuf, sizeof(linebuf));
}

/*
 * __nvmeDevComp
 */
static int
__nvmeDevComp(const void *__a, const void *__b)
{
	NvmePciDevItem *a = *((NvmePciDevItem **)__a);
	NvmePciDevItem *b = *((NvmePciDevItem **)__b);

	if (a->nvme_major < b->nvme_major)
		return -1;
	else if (a->nvme_major > b->nvme_major)
		return 1;
	else if (a->nvme_minor < b->nvme_minor)
		return -1;
	else if (a->nvme_minor > b->nvme_minor)
		return 1;
	return 0;
}

/*
 * __nvmeLookupByDevNumber
 *
 * XXX - must be called after qsort()
 */
static NvmePciDevItem *
__nvmeLookupByDevNumber(unsigned int major, unsigned int minor)
{
	if (nvmeNitems > 0)
	{
		NvmePciDevItem *nvme;
		int		head = 0;
		int		tail = nvmeNitems - 1;
		int		curr;

		while (head < tail)
		{
			curr = (head + tail) / 2;
			nvme = nvmeArray[curr];

			if (major < nvme->nvme_major)
				tail = Max(curr - 1, head);
			else if (major > nvme->nvme_major)
				head = Min(curr + 1, tail);
			else if (minor < nvme->nvme_minor)
				tail = Max(curr - 1, head);
			else if (minor > nvme->nvme_minor)
				head = Min(curr + 1, tail);
			else
				return nvme;
		}
		assert(head == tail);
		nvme = nvmeArray[head];
		if (major == nvme->nvme_major &&
			minor == nvme->nvme_minor)
			return nvme;
	}
	return NULL;
}

/*
 * sysfs_read_nvme_attrs
 */
static bool
sysfs_read_nvme_attrs(NvmePciDevItem *nvme,
					  const char *sysfs_base,
					  const char *sysfs_name)
{
	const char *value;
	char		namebuf[1024];

	/* fetch major:minor */
	snprintf(namebuf, sizeof(namebuf),
			 "%s/%s/dev",
			 sysfs_base, sysfs_name);
	value = sysfs_read_line(namebuf);
	if (!value || sscanf(value, "%u:%u",
						 &nvme->nvme_major,
						 &nvme->nvme_minor) != 2)
		return false;

	/* copy name */
	strncpy(nvme->nvme_name, sysfs_name,
			sizeof(nvme->nvme_name)-1);

	/* fetch model */
	snprintf(namebuf, sizeof(namebuf),
			 "%s/%s/device/model",
			 sysfs_base, sysfs_name);
	value = sysfs_read_line(namebuf);
	if (!value)
		value = "unknown";
	strncpy(nvme->nvme_model, value, sizeof(nvme->nvme_model)-1);

	/* check whether it is local NVME-SSD or not */
	snprintf(namebuf, sizeof(namebuf),
			 "%s/%s/device/transport",
			 sysfs_base, sysfs_name);
	value = sysfs_read_line(namebuf);
	if (!value || strcmp(value, "pcie") != 0)
	{
		/* likely, NVME-oF device. Put fake PCI-E address */
		nvme->pci_domain = UINT_MAX;
		nvme->pci_bus_id = UINT_MAX;
		nvme->pci_dev_id = UINT_MAX;
		nvme->pci_func_id = UINT_MAX;
	}
	else
	{
		/* fetch PCI-E Bus-Id */
		snprintf(namebuf, sizeof(namebuf),
				 "%s/%s/device/address",
				 sysfs_base, sysfs_name);
		value = sysfs_read_line(namebuf);
		if (!value || sscanf(value, "%x:%02x:%02x.%d",
							 &nvme->pci_domain,
							 &nvme->pci_bus_id,
							 &nvme->pci_dev_id,
							 &nvme->pci_func_id) != 4)
			return false;
	}
	return true;	/* Ok */
}

/*
 * sysfs_read_sfdv_attrs
 */
static bool
sysfs_read_sfdv_attrs(NvmePciDevItem *nvme,
					  const char *sysfs_base,
					  const char *sysfs_name)
{
	const char *value;
	char		namebuf[1024];

	/* fetch major:minor */
	snprintf(namebuf, sizeof(namebuf),
			 "%s/%s/dev",
			 sysfs_base, sysfs_name);
	value = sysfs_read_line(namebuf);
	if (!value || sscanf(value, "%u:%u",
						 &nvme->nvme_major,
						 &nvme->nvme_minor) != 2)
		return false;
	/* copy name */
	strncpy(nvme->nvme_name, sysfs_name,
			sizeof(nvme->nvme_name)-1);
	/* fetch model */
	snprintf(namebuf, sizeof(namebuf),
			 "%s/%s/device/model",
			 sysfs_base, sysfs_name);
	value = sysfs_read_line(namebuf);
	if (!value)
		value = "unknown";
	strncpy(nvme->nvme_model, value,
			sizeof(nvme->nvme_model)-1);

	/* fetch PCI-E Bus ID */
	snprintf(namebuf, sizeof(namebuf),
			 "%s/%s/bus_info",
			 sysfs_base, sysfs_name);
	value = sysfs_read_line(namebuf);
	if (!value || sscanf(value, "%x:%02x:%02x.%d",
						 &nvme->pci_domain,
						 &nvme->pci_bus_id,
						 &nvme->pci_dev_id,
						 &nvme->pci_func_id) != 4)
		return false;

	return true;
}

/*
 * sysfs_read_block_attrs
 */
static int
sysfs_read_block_attrs(void)
{
	const char	   *dirname = "/sys/class/block";
	DIR			   *dir;
	struct dirent  *dent;
	NvmePciDevItem *nvme;
	NvmePciDevItem	temp;

	dir = opendir(dirname);
	if (!dir)
	{
		Elog("failed on opendir('%s'): %m", dirname);
		return -1;
	}

	while ((dent = readdir(dir)) != NULL)
	{
		const char *start, *pos;

		if (strncmp("nvme", dent->d_name, 4) == 0)
		{
			/* only nvme[0-9]+n[0-9]+ devices */
			pos = start = dent->d_name + 4;
			while (isdigit(*pos))
				pos++;
			if (start == pos || *pos != 'n')
				continue;
			start = ++pos;
			while (isdigit(*pos))
				pos++;
			if (start == pos || *pos != '\0')
				continue;
			memset(&temp, 0, sizeof(temp));
			if (!sysfs_read_nvme_attrs(&temp, dirname, dent->d_name))
				continue;
		}
		else if (strncmp("sfdv", dent->d_name, 4) == 0)
		{
			/* only sfdv[0-9]+n[0-9]+ devices */
			pos = start = dent->d_name + 4;
			while (isdigit(*pos))
				pos++;
			if (start == pos || *pos != 'n')
				continue;
			start = ++pos;
			while (isdigit(*pos))
				pos++;
			if (start == pos || *pos != '\0')
				continue;
			memset(&temp, 0, sizeof(temp));
			if (!sysfs_read_sfdv_attrs(&temp, dirname, dent->d_name))
				continue;
		}
		else
		{
			/* not a supported device */
			continue;
		}

		/* expand array on demand */
		if (nvmeNitems >= nvmeNrooms)
		{
			NvmePciDevItem **__nvmeArray;
			int		__nrooms = (2 * nvmeNrooms + 20);

			__nvmeArray = realloc(nvmeArray, sizeof(NvmePciDevItem *) * __nrooms);
			if (!__nvmeArray)
			{
				Elog("out of memory: %m");
				return -1;
			}
			nvmeNrooms = __nrooms;
			nvmeArray = __nvmeArray;
		}
		nvme = calloc(1, offsetof(NvmePciDevItem,
								  optimal_gpus[gpuNitems]));
		if (!nvme)
		{
			Elog("out of memory: %m");
			return -1;
		}
		memcpy(nvme, &temp, offsetof(NvmePciDevItem, optimal_gpus));
		nvmeArray[nvmeNitems++] = nvme;
	}
	closedir(dir);

	/*
	 * Sort nvmeArray by major/minor for O(logN) search
	 */
	qsort(nvmeArray, nvmeNitems,
		  sizeof(NvmePciDevItem *),
		  __nvmeDevComp);
	
	return nvmeNitems;
}

/*
 * sysfs_print_nvme_info
 */
ssize_t
sysfs_print_nvme_info(int index, char *buffer, ssize_t buffer_sz)
{
	if (index < nvmeNitems)
	{
		NvmePciDevItem *nvme = nvmeArray[index];
		ssize_t		off = 0;
		int			i, cuda_dindex;

		off += snprintf(buffer + off, buffer_sz - off,
						"%s (%s", nvme->nvme_name, nvme->nvme_model);
		if (nvme->pci_domain != UINT_MAX)
			off += snprintf(buffer + off, buffer_sz - off,
							"; %04x:%02x:%02x.%d",
							nvme->pci_domain,
							nvme->pci_bus_id,
							nvme->pci_dev_id,
							nvme->pci_func_id);
		off += snprintf(buffer + off, buffer_sz - off, ")");
		if (nvme->num_optimal_gpus > 0)
		{
			off += snprintf(buffer + off, buffer_sz - off, " --> ");
			if (nvme->num_optimal_gpus > 1)
				off += snprintf(buffer + off, buffer_sz - off, "{ ");
			for (i=0; i < nvme->num_optimal_gpus; i++)
			{
				cuda_dindex = nvme->optimal_gpus[i];
				assert(cuda_dindex < gpuNitems);
				if (i > 0)
					off += snprintf(buffer + off, buffer_sz - off, ", ");
				off += snprintf(buffer + off, buffer_sz - off, "GPU%d",
								gpuArray[cuda_dindex].device_id);
			}
			if (nvme->num_optimal_gpus > 1)
				off += snprintf(buffer + off, buffer_sz - off, " }");
		}
		return off;
	}

	index -= nvmeNitems;
	if (index < nfsvNitems)
	{
		NfsVolumeItem  *nfsv = nfsvArray[index];
		ssize_t		off = 0;
		int			i, cuda_dindex;

		off += snprintf(buffer + off, buffer_sz - off,
						"%s (%s - %s; mnt_id=%d)",
						nfsv->mount_path,
						nfsv->mount_fs,
						nfsv->mount_dev,
						nfsv->mount_id);
		if (nfsv->num_optimal_gpus > 0)
		{
			off += snprintf(buffer + off, buffer_sz - off, " --> ");
			if (nfsv->num_optimal_gpus > 1)
				off += snprintf(buffer + off, buffer_sz - off, "{ ");
			for (i=0; i < nfsv->num_optimal_gpus; i++)
			{
				cuda_dindex = nfsv->optimal_gpus[i];
				assert(cuda_dindex < gpuNitems);
				if (i > 0)
					off += snprintf(buffer + off, buffer_sz - off, ", ");
				off += snprintf(buffer + off, buffer_sz - off, "GPU%d",
								gpuArray[cuda_dindex].device_id);
			}
			if (nfsv->num_optimal_gpus > 1)
				off += snprintf(buffer + off, buffer_sz - off, " }");
		}
		return off;
	}
	return -1;
}

/*
 * print_pcie_device_tree
 */
static void
__print_pcie_device_tree(FILE *out, PCIDevEntry *pcie, int indent)
{
	PCIDevEntry	   *curr;

	if (!pcie->parent)
		fprintf(out, "PCIe[%04x:%02x]\n",
				pcie->domain,
				pcie->bus_id);
	else if (pcie->gpu)
	{
		GpuPciDevItem  *gpu = pcie->gpu;

		fprintf(out, "%*s PCIe(%04x:%02x:%02x.%d) GPU%d (%s)\n",
				2 * indent, "- ",
				pcie->domain,
				pcie->bus_id,
				pcie->dev_id,
				pcie->func_id,
				gpu->device_id,
				gpu->device_name);
	}
	else if (pcie->nvme)
	{
		NvmePciDevItem *nvme = pcie->nvme;
		GpuPciDevItem  *gpu = NULL;
		size_t		off = 0;
		size_t		bufsz = 40 * nvme->num_optimal_gpus + 100;
		char	   *buf = alloca(bufsz);
		int			i, cuda_dindex;

		if (nvme->num_optimal_gpus == 0)
			buf[0] = '\0';
		else
		{
			off = snprintf(buf, bufsz, " (optimal: ");
			if (nvme->num_optimal_gpus > 1)
				off += snprintf(buf+off, bufsz-off, "{ ");
			for (i=0; i < nvme->num_optimal_gpus; i++)
			{
				if (i > 0)
					off += snprintf(buf+off, bufsz-off, ", ");
				cuda_dindex = nvme->optimal_gpus[i];
				gpu = &gpuArray[cuda_dindex];
				off += snprintf(buf+off, bufsz-off, "GPU%d", gpu->device_id);
			}
			if (nvme->num_optimal_gpus > 1)
                off += snprintf(buf+off, bufsz-off, " }");
			off += snprintf(buf+off, bufsz-off, ")");
		}
		fprintf(out, "%*s PCIe(%04x:%02x:%02x.%d) %s (%s)%s\n",
				2 * indent, "- ",
				pcie->domain,
				pcie->bus_id,
				pcie->dev_id,
				pcie->func_id,
				nvme->nvme_name,
				nvme->nvme_model,
				buf);
	}
	else
	{
		fprintf(out, "%*s PCIe(%04x:%02x:%02x.%d)\n",
				2 * indent, "- ",
				pcie->domain,
				pcie->bus_id,
				pcie->dev_id,
				pcie->func_id);
	}

	for (curr = pcie->children; curr; curr = curr->siblings)
		__print_pcie_device_tree(out, curr, indent+1);
}

void
sysfs_print_pci_tree(FILE *out)
{
	PCIDevEntry *curr;

	for (curr = pcieRoot; curr; curr = curr->siblings)
		__print_pcie_device_tree(out, curr, 0);
}

/*
 * sysfs_free_pcie_subtree
 */
static void
sysfs_free_pcie_subtree(PCIDevEntry *pcie)
{
	PCIDevEntry *next;

	while (pcie)
	{
		next = pcie->siblings;

		sysfs_free_pcie_subtree(pcie->children);

		free(pcie);

		pcie = next;
	}
}

/*
 * sysfs_read_pcie_subtree
 */
static int
__sysfs_read_pcie_subtree(PCIDevEntry *pcie,
						  const char *dirname,
						  const char *my_name)
{
	char	   *path;
	DIR		   *dir;
	struct dirent *dent;
	int			i, has_gpu_or_nvme = 0;

	/* Is it a GPU device? */
	for (i=0; i < gpuNitems; i++)
	{
		GpuPciDevItem  *gpu = &gpuArray[i];

		if (pcie->domain == gpu->pci_domain &&
			pcie->bus_id == gpu->pci_bus_id &&
			pcie->dev_id == gpu->pci_dev_id &&
			pcie->func_id == gpu->pci_func_id)
		{
			has_gpu_or_nvme = 1;
			pcie->gpu = gpu;
			gpu->cpu_affinity = pcie->cpu_affinity;
			break;
		}
	}

	/* Is it a NVME device? */
	for (i=0; i < nvmeNitems; i++)
	{
		NvmePciDevItem *nvme = nvmeArray[i];

		if (pcie->domain == nvme->pci_domain &&
			pcie->bus_id == nvme->pci_bus_id &&
			pcie->dev_id == nvme->pci_dev_id &&
			pcie->func_id == nvme->pci_func_id)
		{
			has_gpu_or_nvme = 1;
			pcie->nvme = nvme;
			nvme->cpu_affinity = pcie->cpu_affinity;
			break;
		}
	}

	/* walk down the PCIe device tree */
	path = alloca(strlen(dirname) + strlen(my_name) + 2);
	sprintf(path, "%s/%s", dirname, my_name);
	dir = opendir(path);
	if (!dir)
	{
		Elog("failed on opendir('%s'): %m", path);
		return -1;
	}

	while ((dent = readdir(dir)) != NULL)
	{
		PCIDevEntry *child;
		const char *delim = "::.";
		char	   *pos;
		int			rv;

		/* xxxx:xx:xx.x? */
		for (pos = dent->d_name; *pos != '\0'; pos++)
		{
			if (*pos == *delim)
				delim++;
			else if (*delim != '\0' ? !isxdigit(*pos) : !isdigit(*pos))
				break;
		}

		if (*pos == '\0' && *delim == '\0')
		{
			child = calloc(1, offsetof(PCIDevEntry,
									   cpu_affinity) + strlen(pcie->cpu_affinity) + 1);
			if (!child)
			{
				Elog("out of memory: %m");
				return -1;
			}

			if (sscanf(dent->d_name,
					   "%x:%02x:%02x.%d",
					   &child->domain,
					   &child->bus_id,
					   &child->dev_id,
					   &child->func_id) != 4)
			{
				Elog("bug? cannot parse [%s]", dent->d_name);
				return -1;
			}
			strcpy(child->cpu_affinity, pcie->cpu_affinity);
			child->parent = pcie;
			child->depth = pcie->depth + 1;

			rv = __sysfs_read_pcie_subtree(child,
										   path,
										   dent->d_name);
			if (rv < 0)
				return -1;
			else if (rv == 0)
				sysfs_free_pcie_subtree(child);
			else
			{
				child->siblings = pcie->children;
				pcie->children = child;
				has_gpu_or_nvme = 1;
			}
		}
	}
	closedir(dir);

	return has_gpu_or_nvme;
}

static int
sysfs_read_pcie_subtree(void)
{
	/* Walks on the PCI-E bus tree for each root complex */
	const char *dirname = "/sys/devices";
	DIR		   *dir;
	struct dirent *dent;
	int			rv;
	PCIDevEntry *result = NULL;

	dir = opendir(dirname);
	if (!dir)
	{
		Elog("failed on opendir('%s'): %m", dirname);
		return -1;
	}

	while ((dent = readdir(dir)) != NULL)
	{
		PCIDevEntry	   *curr;
		int				pci_domain;
		int				pci_bus_id;
		char			path[1024];
		const char	   *line;

		/* only /sys/devices/pciXXXX:XX */
		if (sscanf(dent->d_name,
				   "pci%04x:%02x",
				   &pci_domain,
				   &pci_bus_id) != 2)
			continue;
		/* fetch CPU affinity if any */
		snprintf(path, sizeof(path),
				 "%s/%s/pci_bus/%04x:%02x/cpuaffinity",
				 dirname,
				 dent->d_name,
				 pci_domain,
				 pci_bus_id);
		line = sysfs_read_line(path);
		if (!line)
			line = "unknown";
		curr = calloc(1, offsetof(PCIDevEntry,
								  cpu_affinity) + strlen(line) + 1);
		if (!curr)
		{
			Elog("out of memory");
			sysfs_free_pcie_subtree(result);
			return -1;
		}
		curr->domain = pci_domain;
		curr->bus_id = pci_bus_id;
		strcpy(curr->cpu_affinity, line);

		rv = __sysfs_read_pcie_subtree(curr,
									   dirname,
									   dent->d_name);
		if (rv < 0)
		{
			sysfs_free_pcie_subtree(result);
			return -1;
		}
		else if (rv == 0)
		{
			sysfs_free_pcie_subtree(curr);
		}
		else
		{
			if (!result)
				result = curr;
			else
			{
				curr->siblings = result;
				result = curr;
			}
		}
	}
	closedir(dir);

	pcieRoot = result;

	return 0;
}

/*
 * sysfs_calculate_distance
 */
static int
__sysfs_calculate_distance(PCIDevEntry *pcie,
						   GpuPciDevItem *gpu, bool *p_gpu_found,
						   NvmePciDevItem *nvme, bool *p_nvme_found)
{
	PCIDevEntry	*curr;
	int		gpu_depth = -1;
	int		nvme_depth = -1;
	int		dist;
	
	if (pcie->gpu == gpu)
		gpu_depth = 0;
	if (pcie->nvme == nvme)
		nvme_depth = 0;

	assert(gpu_depth < 0 || nvme_depth < 0);
	for (curr = pcie->children; curr; curr = curr->siblings)
	{
		bool	gpu_found = false;
		bool	nvme_found = false;

		dist = __sysfs_calculate_distance(curr,
										  gpu, &gpu_found,
										  nvme, &nvme_found);
		if (gpu_found && nvme_found)
		{
			*p_gpu_found = true;
			*p_nvme_found = true;
			return dist;
		}
		else if (gpu_found)
		{
			assert(gpu_depth < 0);
			gpu_depth = dist + 1;
		}
		else if (nvme_found)
		{
			assert(nvme_depth < 0);
			nvme_depth = dist + 1;
		}
	}

	if (gpu_depth >= 0 && nvme_depth >= 0)
		dist = (gpu_depth + 1 + nvme_depth);
	else if (gpu_depth >= 0)
		dist = gpu_depth;
	else if (nvme_depth >= 0)
		dist = nvme_depth;
	else
		dist = -1;

	*p_gpu_found = (gpu_depth >= 0);
	*p_nvme_found = (nvme_depth >= 0);
	return dist;
}

static int
sysfs_calculate_distance_root(GpuPciDevItem *gpu,
							  NvmePciDevItem *nvme)
{
	PCIDevEntry *curr;
	int		gpu_depth = -1;
	int		nvme_depth = -1;
	int		root_gap = 5;
	int		dist;

	for (curr = pcieRoot; curr; curr = curr->siblings)
	{
		bool	gpu_found = false;
		bool	nvme_found = false;

		dist = __sysfs_calculate_distance(curr,
										  gpu, &gpu_found,
										  nvme, &nvme_found);
		if (gpu_found && nvme_found)
		{
			return dist;
		}
		else if (gpu_found)
		{
			assert(gpu_depth < 0);
			gpu_depth = dist;
		}
		else if (nvme_found)
		{
			assert(nvme_depth < 0);
			nvme_depth = dist;
		}
	}

	if (gpu_depth < 0 || nvme_depth < 0)
		return -1;		/* no optimal GPU/NVME */
	if (strcmp(gpu->cpu_affinity, nvme->cpu_affinity) == 0)
		root_gap = 99;
	return (gpu_depth + root_gap + nvme_depth);
}



/*
 * __nfsVolumeComp
 */
static int
__nfsVolumeComp(const void *__a, const void *__b)
{
	NfsVolumeItem  *a = *((NfsVolumeItem **)__a);
	NfsVolumeItem  *b = *((NfsVolumeItem **)__b);

	if (a->mount_id < b->mount_id)
		return -1;
	else if (a->mount_id > b->mount_id)
		return 1;
	else
		return 0;
}

/*
 * __nfsVolumeSetup
 */
static int
__nfsVolumeSetup(FILE *mount_info,
				 const char *mount_point,
				 int num_optimal_gpus,
				 int *optimal_gpus)
{
	NfsVolumeItem *nfsv;
	int		mount_id;
	char	linebuf[2048];
	char	mount_path[512];
	char	mount_fs[100];
	char	mount_dev[512];
	int		major, minor;
	size_t	sz, off;

	rewind(mount_info);
	while (fgets(linebuf, sizeof(linebuf), mount_info) != NULL)
	{
		if (sscanf(linebuf, "%u %*u %u:%u %*s %512s %*s %*s - %100s %512s",
				   &mount_id,
				   &major,
				   &minor,
				   mount_path,
				   mount_fs,
				   mount_dev) == 6)
		{
			if (strcmp(mount_path, mount_point) == 0)
				goto found;
		}
	}
	Elog("distance map: [%s] is not a mount point", mount_point);
	return -1;

found:
	if (nfsvNitems >= nfsvNrooms)
	{
		nfsvNrooms += (nfsvNrooms + 40);

		nfsvArray = realloc(nfsvArray, sizeof(NfsVolumeItem *) * nfsvNrooms);
		if (!nfsvArray)
		{
			Elog("out of memory");
			return -1;
		}
	}
	off = MAXALIGN(offsetof(NfsVolumeItem,
								 optimal_gpus[num_optimal_gpus]));
	sz = off + (MAXALIGN(strlen(mount_path) + 1) +
				MAXALIGN(strlen(mount_fs) + 1) +
				MAXALIGN(strlen(mount_dev) + 1));
	nfsv = calloc(1, sz);
	if (!nfsv)
	{
		Elog("out of memory");
		return -1;
	}
	nfsv->mount_id = mount_id;

	nfsv->mount_path = (char *)nfsv + off;
	strcpy(nfsv->mount_path, mount_path);
	off += MAXALIGN(strlen(mount_path) + 1);

	nfsv->mount_fs   = (char *)nfsv + off;
	strcpy(nfsv->mount_fs, mount_fs);
	off += MAXALIGN(strlen(mount_fs) + 1);

	nfsv->mount_dev  = (char *)nfsv + off;
	strcpy(nfsv->mount_dev, mount_dev);
	off += MAXALIGN(strlen(mount_dev) + 1);

	nfsv->num_optimal_gpus = num_optimal_gpus;
	if (num_optimal_gpus > 0)
		memcpy(nfsv->optimal_gpus, optimal_gpus, sizeof(int) * num_optimal_gpus);

	nfsvArray[nfsvNitems++] = nfsv;
	return 0;
}

/*
 * sysfs_apply_manual_config
 */
static int
__apply_manual_config(char **items, int nitems, FILE **p_mount_info)
{
	int		num_optimal_gpus = 0;
	int	   *optimal_gpus = alloca(sizeof(int) * gpuNitems);
	int		i, j, k;
	
	/* step-1: setup GPUs list */
	for (i=0; i < nitems; i++)
	{
		char   *ident = items[i];

		if (strncasecmp(ident, "gpu", 3) == 0 && __strisdigit(ident+3))
		{
			int		device_id = atoi(ident+3);
			int		cuda_dindex = -1;

			for (j=0; j < gpuNitems; j++)
			{
				if (gpuArray[j].device_id == device_id)
				{
					cuda_dindex = i;
					break;
				}
			}
			if (cuda_dindex < 0)
			{
				Elog("distance map: [%s] was not found", ident);
				return -1;
			}
			items[i] = NULL;	/* ignore on the step-2 */

			for (k=0; k < num_optimal_gpus; k++)
			{
				if (optimal_gpus[k] == cuda_dindex)
					break;
			}
			if (k == num_optimal_gpus)
				optimal_gpus[num_optimal_gpus++] = cuda_dindex;
		}
	}

	/* step-2: apply optimal GPUs to devices */
	for (i=0; i < nitems; i++)
	{
		char   *ident = items[i];

		if (!ident)
			continue;

		if ((strncasecmp(ident, "nvme", 4) == 0 && __strisdigit(ident+4)) ||
			(strncasecmp(ident, "sfdv", 4) == 0 && __strisdigit(ident+4)))
		{
			/* overwrite optimal gpu */
			bool	found = false;
			char	temp[100];
			int		sz;

			sz = snprintf(temp, sizeof(temp), "%sn", ident);
			for (j=0; j < nvmeNitems; j++)
			{
				NvmePciDevItem *nvme = nvmeArray[j];

				if (strncasecmp(temp, nvme->nvme_name, sz) == 0)
				{
					nvme->num_optimal_gpus = num_optimal_gpus;
					if (num_optimal_gpus > 0)
						memcpy(nvme->optimal_gpus, optimal_gpus,
							   sizeof(int) * num_optimal_gpus);
					found = true;
					break;
				}
			}
			if (!found)
			{
				Elog("distance map: [%s] was not found", ident);
				return -1;
			}
		}
		else if (*ident == '/')
		{
			FILE   *mount_info = *p_mount_info;
			char   *c;

			/* remove '/' on the tail */
			for (c = ident + strlen(ident) - 1; c > ident && *c == '/'; c--)
				*c = '\0';
			/* open /proc/self/mountinfo, if not yet */
			if (!mount_info)
			{
				mount_info = fopen("/proc/self/mountinfo","r");
				if (!mount_info)
				{
					Elog("failed on fopen('/proc/self/mountinfo'): %m");
					return -1;
				}
				*p_mount_info = mount_info;
			}
			/* setup NfsVolumeItem */
			if (__nfsVolumeSetup(mount_info, ident,
								 num_optimal_gpus,
								 optimal_gpus) != 0)
				return -1;
		}
		else
		{
			Elog("distance map: unexpected token [%s]", ident);
			return -1;
		}
	}
	/* sort NFS volumes by mount_id */
	if (nfsvArray)
		qsort(nfsvArray, nfsvNitems,
			  sizeof(NfsVolumeItem *),
			  __nfsVolumeComp);
	return 0;
}

static int
sysfs_apply_manual_config(const char *__config)
{
	char	   *config;
	char	  **items = NULL;
	int			nitems = 0;
	int			phase = 0;
	char	   *pos, *tok;
	int			c, rv = -1;
	size_t		n = strlen(__config);
	FILE	   *mount_info = NULL;

	config = alloca(n + 1);
	strcpy(config, __config);
	items = alloca(sizeof(char *) * n);		// enough space

	for (pos = config; (c = *pos) != '\0'; pos++)
	{
		switch (phase)
		{
			case 0:
				if (c == '{')
				{
					nitems = 0;
					tok = NULL;
					phase = 1;
				}
				else if (!isspace(c))
				{
					Elog("distance map: syntax error [%s]", __config);
					goto bailout;
				}
				break;

			case 1:		/* inside of '{' */
				if (c == '}')
				{
					/* apply manual configurations, if any */
					if (__apply_manual_config(items, nitems, &mount_info) < 0)
						goto bailout;
					phase = 0;
				}
				else if (c == '"')
				{
					/* begin token with quotation */
					tok = pos + 1;
					phase = 3;
				}
				else if (!isspace(c))
				{
					/* begin token without quotation */
					tok = pos;
					phase = 2;
				}
				break;

			case 2:		/* inside of token, without quotation */
				if (isspace(c) || c == ',' || c == '}')
				{
					*pos = '\0';
					items[nitems++] = tok;
					if (c == '}')
					{
						if (__apply_manual_config(items, nitems, &mount_info) < 0)
							goto bailout;
						phase = 0;
					}
					else if (c == ',')
						phase = 1;	/* wait for next token */
					else
						phase = 4;	/* wait for comma */
				}
				break;

			case 3:		/* inside of token, with quotation */
				if (c == '"')
				{
					*pos = '\0';
					items[nitems++] = tok;
					phase = 4;
				}
				break;

			case 4:		/* wait for comma, until white-spaces */
				if (c == ',')
					phase = 1;
				else if (c == '}')
				{
					if (__apply_manual_config(items, nitems, &mount_info) < 0)
						goto bailout;
					phase = 0;
				}
				else if (!isspace(c))
				{
					Elog("distance map: syntax error [%s] %c", __config, c);
					goto bailout;
				}
				break;
			default:
				Elog("distance map: syntax error [%s]", __config);
				goto bailout;
		}
	}

	if (phase != 0)
	{
		Elog("distance map: syntax error [%s]", __config);
		goto bailout;
	}
	rv = 0;		/* success */
bailout:
	if (mount_info)
		fclose(mount_info);
	return rv;
}

/*
 * sysfs_setup_distance_map
 */
static int
__sysfs_setup_distance_map(unsigned int gpu_count,
						   GpuPciDevItem *gpu_array,
						   const char *manual_config)
{
	DIR		   *dir;
	struct dirent *dent;
	int			__major;
	int			__minor;
	int			i, j;
	int			dist;

	/* copy GpuPciDevItem */
	if (gpu_count > 0)
	{
		gpuArray = calloc(gpu_count, sizeof(GpuPciDevItem));
		if (!gpuArray)
		{
			Elog("out of memory");
			goto bailout;
		}
		memcpy(gpuArray, gpu_array, sizeof(GpuPciDevItem) * gpu_count);
	}
	gpuNitems = gpu_count;

	/* setup nvmeArray */
	if (sysfs_read_block_attrs() < 0)
		goto bailout;

	/* setup pcieRoot */
	if (sysfs_read_pcie_subtree() < 0)
		goto bailout;

	/* calculation of SSD<->GPU distance for each pair */
	for (i=0; i < nvmeNitems; i++)
	{
		NvmePciDevItem *nvme = nvmeArray[i];

		nvme->__distance = INT_MAX;
		for (j=0; j < gpuNitems; j++)
		{
			GpuPciDevItem *gpu = &gpuArray[j];

			dist = sysfs_calculate_distance_root(gpu, nvme);
			if (dist >= 0)
			{
				if (dist < nvme->__distance)
					nvme->num_optimal_gpus = 0;
				nvme->optimal_gpus[nvme->num_optimal_gpus++] = j;
			}
		}
	}
	/* overwrite optimal GPUs, by manual configuration */
	if (manual_config &&
		sysfs_apply_manual_config(manual_config) < 0)
		goto bailout;

	/* preload existing devices */
	dir = opendir("/sys/dev/block");
	if (dir)
	{
		while ((dent = readdir(dir)) != NULL)
		{
			if (sscanf(dent->d_name,
					   "%u:%u",
					   &__major,
					   &__minor) != 2)
				continue;
			__sysfs_lookup_optimal_gpu(__major, __minor);
		}
		closedir(dir);
	}
	return nvmeNitems + nfsvNitems;

bailout:
	if (nfsvArray)
	{
		for (i=0; i < nfsvNitems; i++)
			free(nfsvArray[i]);
		free(nfsvArray);
	}
	if (pcieRoot)
		sysfs_free_pcie_subtree(pcieRoot);
	if (nvmeArray)
	{
		for (i=0; i < nvmeNitems; i++)
			free(nvmeArray[i]);
		free(nvmeArray);
	}
	if (gpuArray)
		free(gpuArray);
	return -1;
}

int
sysfs_setup_distance_map(int gpu_count,
						 GpuPciDevItem *gpu_array,
						 const char *manual_config)
{
	GpuPciDevItem  *gpuArray_saved = gpuArray;
	int				gpuNitems_saved = gpuNitems;
	NvmePciDevItem **nvmeArray_saved = nvmeArray;
	int				nvmeNrooms_saved = nvmeNrooms;
	int				nvmeNitems_saved = nvmeNitems;
	NfsVolumeItem **nfsvArray_saved = nfsvArray;
	int				nfsvNrooms_saved = nfsvNrooms;
	int				nfsvNitems_saved = nfsvNitems;
	PCIDevEntry	   *pcieRoot_saved = pcieRoot;
	int				i, rv;

	gpuArray = NULL;
	gpuNitems = 0;
	nvmeArray = NULL;
	nvmeNrooms = 0;
	nvmeNitems = 0;
	pcieRoot = NULL;

	rv = __sysfs_setup_distance_map(gpu_count, gpu_array,
									manual_config);
	if (rv < 0)
	{
		/* restore */
		gpuArray	= gpuArray_saved;
		gpuNitems	= gpuNitems_saved;
		nvmeArray	= nvmeArray_saved;
		nvmeNrooms	= nvmeNrooms_saved;
		nvmeNitems	= nvmeNitems_saved;
		nfsvArray	= nfsvArray_saved;
		nfsvNrooms	= nfsvNrooms_saved;
		nfsvNitems	= nfsvNitems_saved;
		pcieRoot	= pcieRoot_saved;
	}
	else
	{
		/* release saved resources */
		if (gpuArray_saved)
			free(gpuArray_saved);
		if (nvmeArray_saved)
			free(nvmeArray_saved);
		if (nfsvArray_saved)
		{
			for (i=0; i < nfsvNitems_saved; i++)
				free(nfsvArray_saved[i]);
			free(nfsvArray_saved);
		}
		if (pcieRoot_saved)
			sysfs_free_pcie_subtree(pcieRoot_saved);
	}
	return rv;
}

/* ------------------------------------------------------------
 *  block device --> optimal GPU hash
 * ------------------------------------------------------------
 */
typedef struct BlockDevItem
{
	struct BlockDevItem *next;
	unsigned int major;
	unsigned int minor;
	bool		is_licensed;
	bool		is_nvme_dev;
	int			num_optimal_gpus;
	int			optimal_gpus[1];
} BlockDevItem;

#define BLOCK_DEV_HASH_NSLOTS_BITS		9
#define BLOCK_DEV_HASH_NSLOTS			(1<<BLOCK_DEV_HASH_NSLOTS_BITS)
static BlockDevItem		  **bdevHash = NULL;

/*
 * blockDevHashIndex
 */
#define GOLDEN_RATIO_PRIME_32 0x9e370001UL
static int
blockDevHashIndex(unsigned int major, unsigned int minor)
{
	uint32_t	hash = 0x5c5c5c5c;
	int			shift = (32 - BLOCK_DEV_HASH_NSLOTS_BITS);

	hash ^= ((major * GOLDEN_RATIO_PRIME_32) ^
			 (minor * GOLDEN_RATIO_PRIME_32)) >> shift;
	return hash & ((1U << BLOCK_DEV_HASH_NSLOTS_BITS) - 1);
}

/*
 * __sysfs_lookup_optimal_gpu_partition
 */
static BlockDevItem *
__sysfs_bdev_make_partition(unsigned int major, unsigned int minor)
{
	BlockDevItem   *parent;
	BlockDevItem   *bdev;
	unsigned int	__major, __minor;
	char			namebuf[1024];
	const char	   *value;

	snprintf(namebuf, sizeof(namebuf),
			 "/sys/dev/block/%u:%u/../dev",
			 major, minor);
	value = sysfs_read_line(namebuf);
	if (!value || sscanf(value, "%u:%u",
						 &__major,
						 &__minor) != 2)
	{
		Elog("failed on read '%s'", namebuf);
		return NULL;
	}
	parent = __sysfs_lookup_optimal_gpu(__major, __minor);
	if (!parent)
		return NULL;

	bdev = calloc(1, offsetof(BlockDevItem,
							  optimal_gpus[parent->num_optimal_gpus]));
	if (!bdev)
	{
		Elog("out of memory: %m");
		return NULL;
	}
	bdev->major = major;
	bdev->minor = minor;
	bdev->is_licensed = parent->is_licensed;
	bdev->is_nvme_dev = parent->is_nvme_dev;
	bdev->num_optimal_gpus = parent->num_optimal_gpus;
	memcpy(bdev->optimal_gpus, parent->optimal_gpus,
		   sizeof(int) * bdev->num_optimal_gpus);
	return bdev;
}

static BlockDevItem *
__sysfs_bdev_make_mdraid(unsigned int major, unsigned int minor,
						 const char *dirname)
{
	long			PAGE_SIZE = sysconf(_SC_PAGESIZE);
	char			namebuf[1024];
	const char	   *value;
	DIR			   *dir;
	struct dirent  *dent;
	int				num_optimal_gpus = -1;
	int			   *optimal_gpus = NULL;
	int				meet_non_nvme = -1;
	size_t			chunk_sz;
	BlockDevItem   *bdev;

	/* raid level */
	snprintf(namebuf, sizeof(namebuf), "%s/level", dirname);
	value = sysfs_read_line(namebuf);
	if (!value || strcmp(value, "raid0") != 0)
		goto unsupported;

	/* raid chunk-size */
	snprintf(namebuf, sizeof(namebuf),
			 "%s/chunk_size", dirname);
	value = sysfs_read_line(namebuf);
	if (!value)
		goto unsupported;
	chunk_sz = atol(value);
	if (chunk_sz < PAGE_SIZE || (chunk_sz & (PAGE_SIZE-1)) != 0)
		goto unsupported;

	/* fetch sub-volumes */
	dir = opendir(dirname);
	if (!dir)
	{
		Elog("failed on opendir('%s'): %m", namebuf);
		return NULL;
	}

	while ((dent = readdir(dir)) != NULL)
	{
		BlockDevItem   *child;
		unsigned int	__major;
		unsigned int	__minor;

		if (strncmp("rd", dent->d_name, 2) != 0 ||
			!__strisdigit(dent->d_name + 2))
			continue;
		snprintf(namebuf, sizeof(namebuf),
				 "%s/%s/block/dev", dirname, dent->d_name);
		value = sysfs_read_line(namebuf);
		if (!value || sscanf(value, "%u:%u",
							 &__major,
							 &__minor) != 2)
		{
			Elog("failed on read '%s' [%s]", namebuf, value);
			closedir(dir);
			return NULL;
		}
		child = __sysfs_lookup_optimal_gpu(__major, __minor);
		if (!child)
		{
			closedir(dir);
			return NULL;
		}

		if (num_optimal_gpus < 0)
		{
			num_optimal_gpus = child->num_optimal_gpus;
			optimal_gpus = alloca(sizeof(int) * num_optimal_gpus);
			memcpy(optimal_gpus, child->optimal_gpus,
				   sizeof(int) * num_optimal_gpus);
			meet_non_nvme = (child->is_nvme_dev ? 0 : 1);
		}
		else
		{
			int		i, j, k = 0;

			for (i=0; i < num_optimal_gpus; i++)
			{
				int		cuda_dindex = optimal_gpus[i];

				for (j=0; j < child->num_optimal_gpus; j++)
				{
					if (cuda_dindex == child->optimal_gpus[j])
					{
						optimal_gpus[k++] = cuda_dindex;
						break;
					}
				}
			}
			num_optimal_gpus = k;
			if (!child->is_nvme_dev)
				meet_non_nvme = 1;
		}
	}
	closedir(dir);
unsupported:
	bdev = calloc(1, sizeof(BlockDevItem));
	if (!bdev)
	{
		Elog("out of memory: %m");
		return NULL;
	}
	bdev->major = major;
	bdev->minor = minor;
	bdev->is_licensed = (num_optimal_gpus > 0);
	bdev->is_nvme_dev = (meet_non_nvme == 0);
	bdev->num_optimal_gpus = num_optimal_gpus;
	memcpy(bdev->optimal_gpus, optimal_gpus, sizeof(int) * num_optimal_gpus);

	return bdev;
}

/*
 * sysfs_lookup_optimal_gpu
 */
static BlockDevItem *
__sysfs_lookup_optimal_gpu(unsigned int major, unsigned int minor)
{
	NvmePciDevItem *nvme = NULL;
	BlockDevItem   *bdev;
	int				hindex;
	char			namebuf[1024];
	const char	   *value;
	struct stat		stat_buf;

	/* allocation of hash slot at the first touch */
	if (!bdevHash)
	{
		bdevHash = calloc(BLOCK_DEV_HASH_NSLOTS, sizeof(BlockDevItem *));
		if (!bdevHash)
		{
			Elog("out of memory: %m");
            return NULL;
		}
	}
	
	/* lookup the hash table first */
	hindex = blockDevHashIndex(major, minor);
	for (bdev = bdevHash[hindex]; bdev; bdev = bdev->next)
	{
		if (bdev->major == major &&
			bdev->minor == minor)
			return bdev;
	}
	/* not found */

	/* check partition */
	snprintf(namebuf, sizeof(namebuf),
			 "/sys/dev/block/%u:%u/partition",
			 major, minor);
	value = sysfs_read_line(namebuf);
	if (value && atoi(value) > 0)
	{
		bdev = __sysfs_bdev_make_partition(major, minor);
		if (bdev)
			goto found;
		return NULL;
	}
	/* check md-raid */
	snprintf(namebuf, sizeof(namebuf),
			 "/sys/dev/block/%u:%u/md", major, minor);
	if (stat(namebuf, &stat_buf) == 0)
	{
		assert((stat_buf.st_mode & S_IFMT) == S_IFDIR);
		bdev = __sysfs_bdev_make_mdraid(major, minor, namebuf);
		if (bdev)
			goto found;
		return NULL;
	}
	/* build a raw block device entry */
	if (nvmeNitems > 0 &&
		(nvme = __nvmeLookupByDevNumber(major, minor)) != NULL)
	{
		bdev = calloc(1, offsetof(BlockDevItem,
								  optimal_gpus[nvme->num_optimal_gpus]));
		if (!bdev)
			return NULL;
		bdev->major = major;
		bdev->minor = minor;
		bdev->is_licensed = (nvme->pci_domain == UINT_MAX);
		bdev->is_nvme_dev = true;
		bdev->num_optimal_gpus = nvme->num_optimal_gpus;
		memcpy(bdev->optimal_gpus, nvme->optimal_gpus,
			   sizeof(int) * bdev->num_optimal_gpus);
		goto found;
	}

	bdev = calloc(1, sizeof(BlockDevItem));
	if (!bdev)
	{
		Elog("out of memory: %m");
		return NULL;
	}
	bdev->major = major;
	bdev->minor = minor;
	bdev->is_licensed = false;
	bdev->is_nvme_dev = false;
	bdev->num_optimal_gpus = 0;
found:
	/* attach to the hash table */
	bdev->next = bdevHash[hindex];
	bdevHash[hindex] = bdev;

	return bdev;
}

static int
__sysfs_lookup_optimal_gpu_by_mountpoint(int fdesc, int nrooms, int *optimal_gpus)
{
	NfsVolumeItem *nfsv;
	FILE	   *filp;
	char		path[100];
	char		linebuf[512];
	int			mount_id = -1;
	int			head, tail, curr;

	if (nfsvNitems == 0)
		return 0;	/* quick bailout; no optimal GPUs */

	snprintf(path, sizeof(path), "/proc/self/fdinfo/%d", fdesc);
	filp = fopen(path, "r");
	if (!filp)
	{
		Elog("failed on fopen('%s'): %m", path);
		return -1;
	}

	while (fgets(linebuf, sizeof(linebuf), filp) != NULL)
	{
		if (strncmp(linebuf, "mnt_id:", 7) == 0 &&
			sscanf(linebuf+7, "%u", &mount_id) == 1)
			break;
	}
	fclose(filp);

	if (mount_id < 0)
	{
		Elog("failed on lookup mount_id");
		return -1;
	}

	head = 0;
	tail = nfsvNitems;
	while (head < tail)
	{
		curr = (head + tail) / 2;
		nfsv = nfsvArray[curr];
		if (mount_id < nfsv->mount_id)
			tail = Max(curr - 1, head);
		else if (mount_id > nfsv->mount_id)
			head = Min(curr + 1, tail);
		else if (heterodb_license_check(0))
			goto found;
		else
			return 0;	/* no optimal GPUs */
	}
	assert(head == tail);
	nfsv = nfsvArray[head];
	if (nfsv->mount_id != mount_id)
		return 0;
found:
	if (heterodb_license_check(0))
	{
		int		nitems = Min(nrooms, nfsv->num_optimal_gpus);

		if (nitems > 0)
			memcpy(optimal_gpus, nfsv->optimal_gpus, sizeof(int) * nitems);
		return nfsv->num_optimal_gpus;
	}
	return 0;
}

int
sysfs_lookup_optimal_gpus(int fdesc,
						  int nrooms,
						  int *optimal_gpus)
{
	BlockDevItem   *bdev;
	struct stat		stat_buf;

	if (fstat(fdesc, &stat_buf) != 0)
	{
		Elog("failed on fstat(2): %m");
		return -1;
	}

	/* lookup block device hash table first */
	bdev = __sysfs_lookup_optimal_gpu(major(stat_buf.st_dev),
									  minor(stat_buf.st_dev));
	if (!bdev)
		return -1;
	if (!bdev->is_nvme_dev)
		return __sysfs_lookup_optimal_gpu_by_mountpoint(fdesc, nrooms, optimal_gpus);
	if (!bdev->is_licensed || heterodb_license_check(0))
	{
		int		nitems = Min(nrooms, bdev->num_optimal_gpus);

		if (nitems > 0)
			memcpy(optimal_gpus, bdev->optimal_gpus, sizeof(int) * nitems);
		return bdev->num_optimal_gpus;
	}
	return 0;	/* no optimal GPUs */
}

#if 0
/* for debug */
int main(int argc, char *argv[])
{
	const char *manual_config = (argc > 1 ? argv[1] : NULL);
	GpuPciDevItem *gpu_array;
	int			gpu_count;
	int		   *optimal_gpus;
	CUresult	rc;
	CUdevice	cuda_device;
	int			i, j;
	ssize_t		sz;
	char		buffer[1024];

	heterodb_license_reload();
	
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		return 1;

	rc = cuDeviceGetCount(&gpu_count);
	if (rc != CUDA_SUCCESS)
		return 1;

	gpu_array = alloca(gpu_count * sizeof(GpuPciDevItem));
	for (i=0; i < gpu_count; i++)
	{
		GpuPciDevItem  *gpu = &gpu_array[i];
		int		values[5];
		int		attrs[5] = {CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
							CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
							CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
							CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD,
							CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID };

		rc = cuDeviceGet(&cuda_device, i);
		if (rc != CUDA_SUCCESS)
			return 1;

		rc = cuDeviceGetName(gpu->device_name,
							 sizeof(gpu->device_name),
							 cuda_device);
		if (rc != CUDA_SUCCESS)
			return 1;

		for (j=0; j < 5; j++)
		{
			rc = cuDeviceGetAttribute(&values[j], attrs[j], cuda_device);
			if (rc != CUDA_SUCCESS)
				return 1;
		}
		gpu->device_id = i;
		gpu->pci_domain = values[0];
		gpu->pci_bus_id = values[1];
		gpu->pci_dev_id = values[2];
		gpu->pci_func_id = (values[3] != 0 ? values[4] : 0);
		printf("GPU-%d %s (%04x:%02x:%02x.%d)\n",
			   i, gpu->device_name,
			   gpu->pci_domain,
			   gpu->pci_bus_id,
			   gpu->pci_dev_id,
			   gpu->pci_func_id);
	}
	if (sysfs_setup_distance_map(gpu_count, gpu_array, manual_config) < 0)
		goto error;

	sysfs_print_pci_tree(stdout);
	for (i=0; (sz = sysfs_print_nvme_info(i, buffer, 1024)) > 0; i++)
		puts(buffer);

	optimal_gpus = alloca(sizeof(int) * gpu_count);
	for (i=2; i < argc; i++)
	{
		const char *fname = argv[i];
		int		fdesc;
		int		nitems;

		fdesc = open(fname, O_RDONLY);
		if (fdesc < 0)
		{
			fprintf(stderr, "failed on open('%s'): %m\n", fname);
			continue;
		}
		nitems = sysfs_lookup_optimal_gpus(fdesc, gpu_count, optimal_gpus);
		if (nitems < 0)
			goto error;
		if (nitems == 0)
			printf("[%s] -> No optimal GPU\n", fname);
		else
		{
			printf("[%s] -> ", fname);
			for (j=0; j < nitems; j++)
				printf("%sGPU%d", j > 0 ? ", " : "", optimal_gpus[j]);
			putchar('\n');
		}
		close(fdesc);
	}
	return 0;

error:
	fprintf(stderr, "(%s:%d) %s\n",
			heterodb_extra_error_data.filename,
			heterodb_extra_error_data.lineno,
			heterodb_extra_error_data.message);
	return 1;
}
#endif
