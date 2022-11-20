/*
 * extra/license.c
 *
 * HeteroDB Enterprise License Validation Logic
 *
 * Copyright (C) 2020 HeteroDB,Inc
 */
#include <assert.h>
#include <curl/curl.h>	/* libcurl-devel or libcurl-dev package */
#include <fcntl.h>
#include <gmp.h>		/* gmp-devel or libgmp-dev package */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "heterodb_extra_internal.h"
#include "nvme_strom.h"
#include "license.h"

/*
 * heterodb_license_info
 */
typedef struct
{
	uint32_t	version;		/* == 2 */
	time_t		timestamp;
	const char *serial_nr;
	uint32_t	issued_at;		/* YYYYMMDD */
	uint32_t	expired_at;		/* YYYYMMDD */
	const char *licensee_org;
	const char *licensee_name;
	const char *licensee_mail;
	const char *description;
	uint32_t	nr_gpus;
	const char *gpu_uuid[1];	/* variable length */
} heterodb_license_info_v2;

typedef struct
{
	uint32_t	version;		/* == 3 */
	time_t		timestamp;
	const char *serial_nr;
	uint32_t	issued_at;		/* YYYYMMDD */
	uint32_t	expired_at;		/* YYYYMMDD */
	const char *cloud_vendor;
	const char *vm_type_id;
	const char *vm_image_id;
	const char *description;
	uint32_t	nr_gpus;
	char	   *__gpu_uuid_list;	
	const char *gpu_uuid[1];	/* variable length */
} heterodb_license_info_v3;

typedef union
{
	uint32_t	version;
	heterodb_license_info_v2 v2;
	heterodb_license_info_v3 v3;
} heterodb_license_info;

/* misc definitions */
#define __UUID_LEN				16
#define LICENSE_EXPIRE_MARGIN		(25 * 60 * 60)	/* 1day + 1hour */

static heterodb_license_info *current_license_info = NULL;
static time_t		current_license_expired_at = (time_t)0;
static int			current_free_gpu_count = 0;
static char			current_free_gpu_uuid[__UUID_LEN];

/*
 * heterodb_license_check
 */
int
heterodb_license_check(unsigned int reserved)
{
	static time_t system_uptime_expired_at = 0;
	time_t		timestamp = time(NULL);

	assert(reserved == 0);
	if (timestamp < current_license_expired_at)
		return 1;		/* ok, valid */
	if (system_uptime_expired_at == 0)
	{
		FILE	   *filp;
		double		elapsed, dummy;

		system_uptime_expired_at = (time_t) 1;
		filp = fopen("/proc/uptime", "rb");
		if (filp)
		{
			if (fscanf(filp, "%lf %lf", &elapsed, &dummy) == 2)
			{
				/*
				 * commercial features are available within 3-hours from
				 * the system startup time.
				 */
				system_uptime_expired_at = timestamp - (time_t)elapsed + 10800;
			}
			fclose(filp);
		}
	}
	return timestamp < system_uptime_expired_at;
}

/*
 * store_new_license_info
 */
static void
store_new_license_info(heterodb_license_info *linfo, time_t expired_at)
{
	if (current_license_info)
		free(current_license_info);
	current_license_info = linfo;
	current_license_expired_at = expired_at;
	current_free_gpu_count = 0;
	memset(current_free_gpu_uuid, 0, __UUID_LEN);
}

/*
 * build_license_info_v2
 */
static int
build_license_info_v2(uint32_t version,
					  const char *serial_nr,
					  int i_year, int i_mon, int i_day,
					  int e_year, int e_mon, int e_day,
					  const char *licensee_org,
					  const char *licensee_name,
					  const char *licensee_mail,
					  const char *description,
					  int nr_gpus,
					  const char **gpu_uuid)
{
	static int	__mdays[] = {31,29,30,30,31,30,31,31,30,31,30,31};
	heterodb_license_info_v2 *linfo;
	time_t		timestamp = time(NULL);
	time_t		expired_at;
	char	   *pos;
	struct tm	tm;
	int			i, extra = 0;

	/* sanity checks */
	if (version != 2 || !serial_nr || nr_gpus == 0 ||
		i_year < 2000 || i_year >= 3000 ||
		i_mon  < 1    || i_mon  >  12   ||
		i_day  < 1    || i_day  >  __mdays[i_mon-1] ||
		e_year < 2000 || e_year >= 3000 ||
		e_mon  < 1    || e_mon  >  12   ||
		e_day  < 1    || e_day  >  __mdays[e_mon-1])
		return 0;

	/* license expire checks */
	memset(&tm, 0, sizeof(struct tm));
	tm.tm_mday = e_day;
	tm.tm_mon  = e_mon - 1;
	tm.tm_year = e_year;
	expired_at = mktime(&tm) + (24 * 60 * 60);		/* a day margin */
	if (timestamp > expired_at)
		return 0;

	/* build heterodb_license_info_v2 */
	if (serial_nr)
		extra += strlen(serial_nr) + 1;
	if (licensee_org)
		extra += strlen(licensee_org) + 1;
	if (licensee_name)
		extra += strlen(licensee_name) + 1;
	if (licensee_mail)
		extra += strlen(licensee_mail) + 1;
	if (description)
		extra += strlen(description) + 1;
	for (i=0; i < nr_gpus; i++)
		extra += strlen(gpu_uuid[i]) + 1;

	linfo = calloc(1, offsetof(heterodb_license_info_v2,
							   gpu_uuid[nr_gpus]) + extra);
	pos = (char *)&linfo->gpu_uuid[nr_gpus];
	linfo->version = version;
	linfo->timestamp = timestamp;

	linfo->serial_nr = pos;
	strcpy(pos, serial_nr);
	pos += strlen(serial_nr) + 1;

	linfo->issued_at  = i_year * 10000 + i_mon * 100 + i_day;
	linfo->expired_at = e_year * 10000 + e_mon * 100 + e_day;
	if (licensee_org)
	{
		linfo->licensee_org = pos;
		strcpy(pos, licensee_org);
		pos += strlen(licensee_org) + 1;
	}
	if (licensee_name)
	{
		linfo->licensee_name = pos;
		strcpy(pos, licensee_name);
		pos += strlen(licensee_name) + 1;
	}
	if (licensee_mail)
	{
		linfo->licensee_mail = pos;
		strcpy(pos, licensee_mail);
		pos += strlen(licensee_mail) + 1;
	}
	linfo->nr_gpus = nr_gpus;
	for (i=0; i < nr_gpus; i++)
	{
		linfo->gpu_uuid[i] = pos;
		strcpy(pos, gpu_uuid[i]);
		pos += strlen(gpu_uuid[i]) + 1;
	}
	assert((char *)&linfo->gpu_uuid[nr_gpus] + extra == pos);
	store_new_license_info((heterodb_license_info *)linfo, expired_at);

	return 1;
}

static int
build_license_info_v3(int version,
					  const char *serial_nr,
					  int i_year, int i_mon, int i_day,
					  int e_year, int e_mon, int e_day,
					  const char *cloud_vendor,
					  const char *vm_type_id,
					  const char *vm_image_id,
					  int nr_gpus,
					  const char *description,
					  char *gpu_uuid_list)
{
	static int	__mdays[] = {31,29,30,30,31,30,31,31,30,31,30,31};
	heterodb_license_info_v3 *linfo;
	time_t		timestamp = time(NULL);
	time_t		expired_at;
	char	   *uuid;
	char	   *pos;
	struct tm	tm;
	int			index;
	int			extra = 0;

	/* sanity checks */
	if (version != 3 ||
		!serial_nr ||
		i_year < 2000 || i_year >= 3000 ||
		i_mon  < 1    || i_mon  >  12   ||
		i_day  < 1    || i_day  >  __mdays[i_mon-1] ||
		e_year < 2000 || e_year >= 3000 ||
		e_mon  < 1    || e_mon  >  12   ||
		e_day  < 1    || e_day  >  __mdays[e_mon-1] ||
		!cloud_vendor ||
		!vm_type_id ||
		!vm_image_id ||
		nr_gpus < 1)
	{
		errno = EINVAL;
		return 0;
	}
	/* license expire checks */
	memset(&tm, 0, sizeof(struct tm));
	tm.tm_mday = e_day;
	tm.tm_mon  = e_mon - 1;
	tm.tm_year = e_year;
	expired_at = mktime(&tm) + LICENSE_EXPIRE_MARGIN;
	if (timestamp > expired_at)
	{
		errno = EKEYEXPIRED;
		return 0;
	}
	/* build heterodb_license_info_v3 */
	if (serial_nr)
		extra += strlen(serial_nr) + 1;
	if (cloud_vendor)
		extra += strlen(cloud_vendor) + 1;
	if (vm_type_id)
		extra += strlen(vm_type_id) + 1;
	if (vm_image_id)
		extra += strlen(vm_image_id) + 1;
	if (description)
		extra += strlen(description) + 1;
	linfo = calloc(1, offsetof(heterodb_license_info_v3,
							   gpu_uuid[nr_gpus]) + extra);
	if (!linfo)
	{
		errno = ENOMEM;
		return -1;
	}
	linfo->__gpu_uuid_list = gpu_uuid_list;

	for (uuid = strtok_r(gpu_uuid_list, ",", &pos), index = 0;
		 uuid != NULL && index < nr_gpus;
		 uuid = strtok_r(NULL, ",", &pos), index++)
	{
		linfo->gpu_uuid[index] = uuid;
	}

	pos = (char *)&linfo->gpu_uuid[nr_gpus];
	linfo->version = version;
	linfo->timestamp = timestamp;
	if (serial_nr)
	{
		linfo->serial_nr = pos;
		strcpy(pos, serial_nr);
		pos += strlen(serial_nr) + 1;
	}
	linfo->issued_at  = i_year * 10000 + i_mon * 100 + i_day;
	linfo->expired_at = e_year * 10000 + e_mon * 100 + e_day;
	if (cloud_vendor)
	{
		linfo->cloud_vendor = pos;
		strcpy(pos, cloud_vendor);
		pos += strlen(cloud_vendor) + 1;
	}
	if (vm_type_id)
	{
		linfo->vm_type_id = pos;
		strcpy(pos, vm_type_id);
		pos += strlen(vm_type_id) + 1;
	}
	if (vm_image_id)
	{
		linfo->vm_image_id = pos;
		strcpy(pos, vm_image_id);
		pos += strlen(vm_image_id) + 1;
	}
	if (description)
	{
		linfo->description = pos;
		strcpy(pos, description);
		pos += strlen(description) + 1;
	}
	linfo->nr_gpus = nr_gpus;

	store_new_license_info((heterodb_license_info *)linfo, expired_at);

	return 1;
}

/*
 * __trim - remove whitespace at the head/tail of cstring
 */
static inline char *
__trim(char *token)
{
	char   *tail = token + strlen(token) - 1;

	while (*token == ' ' || *token == '\t')
		token++;
	while (tail >= token && (*tail == ' ' || *tail == '\t'))
		*tail-- = '\0';
	return token;
}

/*
 * procfs_extract_gpu_uuid
 */
static char *
procfs_extract_gpu_uuid(void)
{
	const char *command;
	FILE	   *filp;
	char		linebuf[1024];
	char	   *buf;
	size_t		bufsz = 2048;
	size_t		off;

	command = "grep '^GPU UUID:' /proc/driver/nvidia/gpus/*/information";
restart:
	filp = popen(command, "r");
	if (!filp)
	{
		fprintf(stderr, "failed to fetch GPU-UUIDs: %m\n");
		return strdup("");	/* empty GPU-UUID */
	}
	buf = alloca(bufsz);
	off = 0;
	while (fgets(linebuf, sizeof(linebuf), filp) != NULL)
	{
		char   *token;

		if (strncmp(linebuf, "GPU UUID:", 9) != 0)
			continue;
		token = __trim(linebuf + 9);

		off += snprintf(buf+off, bufsz-off, "%s%s",
						(off > 0 ? "," : ""), token);
	}
	buf[off] = '\0';
	fclose(filp);

	if (off < bufsz)
		return strdup(buf);

	bufsz += bufsz;
	goto restart;
}

/*
 * vm_system_check_aws
 */
#define AWS_META_URL	"http://169.254.169.254/latest/meta-data/"
static size_t
__vm_system_check_aws_callback(char *ptr, size_t size, size_t nmemb, void *userdata)
{
	size_t		nbytes = size * nmemb;
	char	   *temp;

	if (nbytes > 0)
	{
		temp = malloc(nbytes + 1);
		if (!temp)
			fprintf(stderr, "out of memory: %m\n");
		else
		{
			memcpy(temp, ptr, nbytes);
			temp[nbytes] = '\0';

			*((char **)userdata) = temp;
		}
	}
	return nbytes;
}

static char *
vm_system_check_aws(const char *license_vm_type_id,
					const char *license_vm_image_id)
{
	char	   *meta_vm_type_id = NULL;
	char	   *meta_vm_image_id = NULL;
	char	   *gpu_uuid_list = NULL;
	CURL	   *curl;
	CURLcode	rv;

	/* sanity checks */
	if (!license_vm_type_id || !license_vm_image_id)
		return NULL;

	/* obtain vm_type_id and vm_image_id from 169.254.169.254 */
	curl = curl_easy_init();
	if (!curl)
	{
		fprintf(stderr, "failed on curl_easy_init()\n");
		return NULL;
	}

	rv = curl_easy_setopt(curl, CURLOPT_URL, AWS_META_URL "instance-type");
	if (rv != CURLE_OK)
	{
		fprintf(stderr, "failed on curl_easy_setopt(CURLOPT_URL): %s\n",
				curl_easy_strerror(rv));
		goto out;
	}

	rv = curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
						  __vm_system_check_aws_callback);
	if (rv != CURLE_OK)
	{
		fprintf(stderr, "failed on curl_easy_setopt(CURLOPT_WRITEFUNCTION): %s\n",
				curl_easy_strerror(rv));
		goto out;
	}

	rv = curl_easy_setopt(curl, CURLOPT_WRITEDATA, &meta_vm_type_id);
	if (rv != CURLE_OK)
	{
		fprintf(stderr, "failed on curl_easy_setopt(CURLOPT_WRITEDATA): %s\n",
				curl_easy_strerror(rv));
		goto out;
	}

	rv = curl_easy_perform(curl);
	if (rv != CURLE_OK)
	{
		fprintf(stderr, "failed on curl_easy_perform(): %s\n",
				curl_easy_strerror(rv));
		goto out;
	}

	rv = curl_easy_setopt(curl, CURLOPT_URL, AWS_META_URL "ami-id");
	if (rv != CURLE_OK)
	{
		fprintf(stderr, "failed on curl_easy_setopt(CURLOPT_URL): %s\n",
				curl_easy_strerror(rv));
		goto out;
	}

	rv = curl_easy_setopt(curl, CURLOPT_WRITEDATA, &meta_vm_image_id);
	if (rv != CURLE_OK)
	{
		fprintf(stderr, "failed on curl_easy_setopt(CURLOPT_WRITEDATA): %s\n",
				curl_easy_strerror(rv));
		goto out;
	}

	rv = curl_easy_perform(curl);
	if (rv != CURLE_OK)
	{
		fprintf(stderr, "failed on curl_easy_perform(): %s\n",
				curl_easy_strerror(rv));
		goto out;
	}

	if (strcmp(license_vm_type_id, meta_vm_type_id) == 0 &&
		strcmp(license_vm_image_id, meta_vm_image_id) == 0)
	{
		gpu_uuid_list = procfs_extract_gpu_uuid();
	}
out:
    curl_easy_cleanup(curl);

	return gpu_uuid_list;
}

/*
 * license_validation_by_kmod
 */
static int
__license_validation_by_kmod(int fdesc, char *license, size_t license_sz)
{
	StromCmd__LicenseInfo *cmd;
	ssize_t		bufsz;
	const char **gpu_uuid = NULL;
	int			i_year, i_mon, i_day;
	int			e_year, e_mon, e_day;
	int			i;

	bufsz = sizeof(StromCmd__LicenseInfo) + 2 * license_sz;
	cmd = (StromCmd__LicenseInfo *)alloca(bufsz);
	memset(cmd, 0, bufsz);

	/* invoke STROM_IOCTL__LICENSE_LOAD */
	cmd->buffer_sz = 2 * license_sz;
	memcpy(cmd->u.buffer, license, license_sz);
	if (ioctl(fdesc, STROM_IOCTL__LICENSE_LOAD, cmd) != 0)
		return -1;		/* license v2 is not valid, but may be v3 */
	if (cmd->nr_gpus == 0)
		return 0;		/* no GPUs are licensed */
	gpu_uuid = alloca(sizeof(char *) * cmd->nr_gpus);
	for (i=0; i < cmd->nr_gpus; i++)
	{
		gpu_uuid[i] = cmd->u.gpus[i].uuid;
	}
	i_year = (cmd->issued_at / 10000);
	i_mon  = (cmd->issued_at / 100) % 100;
	i_day  = (cmd->issued_at % 100);
	e_year = (cmd->expired_at / 10000);
	e_mon  = (cmd->expired_at / 100) % 100;
	e_day  = (cmd->expired_at % 100);

	return build_license_info_v2(cmd->version,
								 cmd->serial_nr,
								 i_year, i_mon, i_day,
								 e_year, e_mon, e_day,
								 cmd->licensee_org,
								 cmd->licensee_name,
								 cmd->licensee_mail,
								 cmd->description,
								 cmd->nr_gpus,
								 gpu_uuid);
}

static int
license_validation_by_kmod(char *license, size_t license_sz)
{
	int		fdesc;
	int		rv;

	fdesc = open(NVME_STROM_IOCTL_PATHNAME, O_RDONLY);
	if (fdesc < 0)
	{
		if (errno == ENOENT &&
			system("/usr/bin/nvme_strom-modprobe 2>/dev/null") == 0)
		{
			/* try again, if kernel module is not loaded yet */
			fdesc = open(NVME_STROM_IOCTL_PATHNAME, O_RDONLY);
		}
		if (fdesc < 0)
			return -1;
	}
	rv = __license_validation_by_kmod(fdesc, license, license_sz);
	close(fdesc);

	return rv;
}

/*
 * license_decrypt_by_pubkey
 */
static char strom_license_expo_pub[] = STROM_LICENSE_EXPO_PUB;
static char strom_license_modulus[] = STROM_LICENSE_MODULUS;

static int
license_decrypt_by_pubkey(char *buffer, char *license, size_t license_sz)
{
	char	   *curr, *tail, *wpos;
	mpz_t		expo_pub;
	mpz_t		modulus;
	mpz_t		crypt;
	mpz_t		plain;
	int			nbits;
	size_t		count;
	int			retval = EBADMSG;

	assert(strom_license_expo_pub[1] >> 3 == sizeof(strom_license_expo_pub) - 2);
	
	nbits = ((int)strom_license_modulus[0] << 8 |
			 (int)strom_license_modulus[1]);
	if (nbits != HETERODB_LICENSE_KEYBITS)
	{
		fprintf(stderr, "license public key (%ubits) is not expected\n", nbits);
		return EBADMSG;
	}
	mpz_init(expo_pub);
	mpz_init(modulus);
	mpz_init(crypt);
	mpz_init(plain);

	mpz_import(expo_pub,
			   sizeof(strom_license_expo_pub) - 2,
			   -1, 1, 0, 0,
			   strom_license_expo_pub + 2);
	mpz_import(modulus,
			   sizeof(strom_license_modulus) - 2,
			   1, 1, 0, 0,
			   strom_license_modulus + 2);

	/* decrypt */
	tail = license + license_sz;
	for (curr = license, wpos = buffer;
		 curr + HETERODB_LICENSE_KEYLEN+2 <= tail;
		 curr += HETERODB_LICENSE_KEYLEN+2, wpos += HETERODB_LICENSE_KEYLEN)
	{
		if (curr[0] != strom_license_modulus[0] ||
			curr[1] != strom_license_modulus[1])
		{
			fprintf(stderr, "license public key (%ubits) is unexpected\n",
					(int)curr[0] << 8 | (int)curr[1]);
			goto out;
		}
		mpz_import(crypt,
				   HETERODB_LICENSE_KEYLEN,
				   1, 1, 0, 0,
				   curr+2);
		/* decrypt: D = (C)^(expo) mod N */
		mpz_powm(plain, crypt, expo_pub, modulus);
		mpz_export(wpos, &count, 1, 1, 0, 0, plain);
		if (count != HETERODB_LICENSE_KEYLEN)
		{
			fprintf(stderr, "plain license has unexpected length\n");
			goto out;
		}
	}
	if (curr != tail)
	{
		fprintf(stderr, "length of encrypted license mismatch (%lu of %lu)\n",
				curr - license, license_sz);
		goto out;
	}
	*wpos = '\0';
	retval = 0;
out:
	mpz_clear(expo_pub);
	mpz_clear(modulus);
	mpz_clear(crypt);
	mpz_clear(plain);
	return retval;
}

/*
 * license_validation_version2
 */
static int
license_validation_version2(char *license)
{
	int			version = -1;
	char	   *serial_nr = NULL;
	int			max_gpus = 20;
	int			nr_gpus = 0;
	const char **gpu_uuid = alloca(sizeof(char *) * max_gpus);
	char	   *licensee_org = NULL;
	char	   *licensee_name = NULL;
	char	   *licensee_mail = NULL;
	char	   *description = NULL;
	char	   *key, *val, *pos;
	int			i_year = -1, i_mon, i_day;
	int			e_year = -1, e_mon, e_day;

	for (key = strtok_r(license, "\n", &pos);
		 key != NULL;
		 key = strtok_r(NULL, "\n", &pos))
	{
		val = strchr(key, ':');
		if (!val)
			return 0;
		*val++ = '\0';

		if (strcmp(key, "VERSION") == 0)
		{
			if (version >= 0)
				return 0;
			version = atoi(val);
		}
		else if (strcmp(key, "SERIAL_NR") == 0)
		{
			if (serial_nr)
				return 0;
			serial_nr = val;
		}
		else if (strcmp(key, "ISSUED_AT") == 0)
		{
			if (i_year >= 0)
				return 0;
			if (sscanf(val, "%d-%d-%d", &i_year, &i_mon, &i_day) != 3)
				return 0;
		}
		else if (strcmp(key, "EXPIRED_AT") == 0)
		{
			if (e_year >= 0)
				return 0;
			if (sscanf(val, "%d-%d-%d", &e_year, &e_mon, &e_day) != 3)
				return 0;
		}
		else if (strcmp(key, "GPU_UUID") == 0)
		{
			if (nr_gpus == max_gpus)
			{
				const char **tmp_uuid = alloca(sizeof(char *) * (max_gpus + 20));

				memcpy(tmp_uuid, gpu_uuid, sizeof(char *) * max_gpus);
				gpu_uuid = tmp_uuid;
				max_gpus += 20;
			}
			gpu_uuid[nr_gpus++] = val;
		}
		else if (strcmp(key, "LICENSEE_ORG") == 0)
		{
			if (licensee_org)
				return 0;
			licensee_org = val;
		}
		else if (strcmp(key, "LICENSEE_NAME") == 0)
		{
			if (licensee_name)
				return 0;
			licensee_name = val;
		}
		else if (strcmp(key, "LICENSEE_MAIL") == 0)
		{
			if (licensee_mail)
				return 0;
			licensee_mail = val;
		}
		else if (strcmp(key, "DESCRIPTION") == 0)
		{
			if (description)
				return 0;
			description = val;
		}
		else
		{
			fprintf(stderr, "unknown license field [%s]\n", key);
			return 0;
		}
	}
	return build_license_info_v2(version,
								 serial_nr,
								 i_year, i_mon, i_day,
								 e_year, e_mon, e_day,
								 licensee_org,
								 licensee_name,
								 licensee_mail,
								 description,
								 nr_gpus, gpu_uuid);
}

/*
 * license_validation_version3
 */
static int
license_validation_version3(char *license)
{
	int			version = -1;
	char	   *serial_nr = NULL;
	char	   *cloud_vendor = NULL;
	char	   *vm_type_id = NULL;
	char	   *vm_image_id = NULL;
	char	   *description = NULL;
	char	   *gpu_uuid_list = NULL;
	char	   *key, *val, *pos;
	int			nr_gpus = -1;
	int			i_year = -1, i_mon = -1, i_day = -1;
	int			e_year = -1, e_mon = -1, e_day = -1;

	for (key = strtok_r(license, "\n", &pos);
		 key != NULL;
		 key = strtok_r(NULL, "\n", &pos))
	{
		val = strchr(key, ':');
		if (!val)
			return 0;
		*val++ = '\0';

		if (strcmp(key, "VERSION") == 0)
		{
			if (version >= 0)
				return 0;
			version = atoi(val);
		}
		else if (strcmp(key, "SERIAL_NR") == 0)
		{
			if (serial_nr)
				return 0;
			serial_nr = val;
		}
		else if (strcmp(key, "ISSUED_AT") == 0)
		{
			if (i_year >= 0)
				return 0;
			if (sscanf(val, "%d-%d-%d", &i_year, &i_mon, &i_day) != 3)
				return 0;
		}
		else if (strcmp(key, "EXPIRED_AT") == 0)
		{
			if (e_year >= 0)
				return 0;
			if (sscanf(val, "%d-%d-%d", &e_year, &e_mon, &e_day) != 3)
				return 0;
		}
		else if (strcmp(key, "CLOUD_VENDOR") == 0)
		{
			if (cloud_vendor)
				return 0;
			cloud_vendor = val;
		}
		else if (strcmp(key, "VM_TYPE_ID") == 0)
		{
			if (vm_type_id)
				return 0;
			vm_type_id = val;
		}
		else if (strcmp(key, "VM_IMAGE_ID") == 0)
		{
			if (vm_image_id)
				return 0;
			vm_image_id = val;
		}
		else if (strcmp(key, "NR_GPUS") == 0)
		{
			if (nr_gpus >= 0)
				return 0;
			nr_gpus = atoi(val);
		}
		else if (strcmp(key, "DESCRIPTION") == 0)
		{
			if (description)
				return 0;
			description = val;
		}
		else
		{
			fprintf(stderr, "unknown license field [%s]\n", key);
			return 0;
		}
	}
	/* pull GPUs uuid */
	if (!cloud_vendor)
		return 0;
	else if (strcmp(cloud_vendor, "aws") == 0)
	{
		gpu_uuid_list = vm_system_check_aws(vm_type_id, vm_image_id);
		if (!gpu_uuid_list)
			return 0;
	}
	else
		return 0;	/* unknown cloud vendor */

	return build_license_info_v3(version,
								 serial_nr,
								 i_year, i_mon, i_day,
								 e_year, e_mon, e_day,
								 cloud_vendor,
								 vm_type_id,
								 vm_image_id,
								 nr_gpus,
								 description,
								 gpu_uuid_list);
}

static int
heterodb_license_validation(char *license, size_t license_sz)
{
	char	   *plain;
	int			rv;

	/*
	 * If nvme_strom.ko is available on the current operating system,
	 * we have to load the license information to the kernel space also.
	 * In this case, kernel module create "/proc/nvme_strom" pseudo file
	 * entry.
	 * Elsewhere, we are likely using GPUDirect Storage and nvidia-fs
	 * kernel module. So, userspace should do the license validation.
	 */
	rv = license_validation_by_kmod(license, license_sz);
	if (rv >= 0)
		return rv;

	/*
	 * elsewhere, we try to validate the license file by ourselves
	 */
	plain = alloca(license_sz + 1);
	if (license_decrypt_by_pubkey(plain, license, license_sz) == 0)
	{
		if (strncmp(plain, "VERSION:2\n", 10) == 0)
			return license_validation_version2(plain);
		if (strncmp(plain, "VERSION:3\n", 10) == 0)
			return license_validation_version3(plain);
		/* not a supported version */
		return 0;
	}
	return -1;
}

int
heterodb_license_reload(void)
{
	const char *path = HETERODB_LICENSE_PATHNAME;
	FILE	   *filp;
	struct stat	stat_buf;
	char	   *buffer;
	int			i, val = 0;
	int			bits = 0;

	/* open the license file */
	filp = fopen(path, "rb");
	if (!filp)
	{
		if (errno == ENOENT)
			return 0;
		fprintf(stderr, "failed on fopen('%s'): %m\n", path);
		return -1;
	}

	if (fstat(fileno(filp), &stat_buf) != 0)
	{
		fprintf(stderr, "failed on fstat('%s'): %m\n", path);
		fclose(filp);
		return -1;
	}

	/* extract base64 */
	buffer = alloca(stat_buf.st_blksize + 100);
	for (i=0;;)
	{
		int		c = fgetc(filp);

		if (c == '=' || c == EOF)
			break;
		if (c >= 'A' && c <= 'Z')
			val |= ((c - 'A') << bits);
		else if (c >= 'a' && c <= 'z')
			val |= ((c - 'a' + 26) << bits);
		else if (c >= '0' && c <= '9')
			val |= ((c - '0' + 52) << bits);
		else if (c == '+')
			val |= (62 << bits);
		else if (c == '/')
			val |= (63 << bits);
		else
		{
			fprintf(stderr, "unexpected base64 character: %c\n", c);
			fclose(filp);
			errno = EBADMSG;
			return 0;
		}
		bits += 6;
		while (bits >= 8)
		{
			assert(i < stat_buf.st_blksize);
			buffer[i++] = (val & 0xff);
			val >>= 8;
			bits -= 8;
		}
	}
	if (bits > 0)
	{
		assert(i < stat_buf.st_blksize);
		buffer[i++] = (val & 0xff);
	}
	fclose(filp);
	
	if (HETERODB_LICENSE_KEYBITS != ((int)buffer[0] << 8 | (int)buffer[1]))
	{
		fprintf(stderr, "license file corruption.\n");
		errno = EINVAL;
		return -1;
	}
	return heterodb_license_validation(buffer, i);
}

/*
 * heterodb_license_query_v2
 */
static ssize_t
heterodb_license_query_v2(heterodb_license_info_v2 *linfo, char *buf, size_t bufsz)
{
	size_t		off = 0;
	int			i;

	off += snprintf(buf + off, bufsz - off,
					"{ \"version\" : %d",
					linfo->version);
	if (linfo->serial_nr)
		off += snprintf(buf + off, bufsz - off,
						", \"serial_nr\" : \"%s\"",
                        linfo->serial_nr);
	off += snprintf(buf + off, bufsz - off,
					", \"issued_at\" : \"%04d-%02d-%02d\"",
					(linfo->issued_at / 10000),
					(linfo->issued_at / 100) % 100,
					(linfo->issued_at % 100));
	off += snprintf(buf + off, bufsz - off,
					", \"expired_at\" : \"%04d-%02d-%02d\"",
					(linfo->expired_at / 10000),
					(linfo->expired_at / 100) % 100,
					(linfo->expired_at % 100));
	if (linfo->licensee_org)
		off += snprintf(buf + off, bufsz - off,
						", \"licensee_org\" : \"%s\"",
						linfo->licensee_org);
	if (linfo->licensee_name)
		off += snprintf(buf + off, bufsz - off,
					    ", \"licensee_name\" : \"%s\"",
						linfo->licensee_name);
	if (linfo->licensee_mail)
		off += snprintf(buf + off, bufsz - off,
						", \"licensee_mail\" : \"%s\"",
						linfo->licensee_mail);
	if (linfo->description)
		off += snprintf(buf + off, bufsz - off,
						", \"description\" : \"%s\"",
						linfo->description);
	off += snprintf(buf + off, bufsz - off,
					", \"nr_gpus\" : %u",
					linfo->nr_gpus);
	if (linfo->nr_gpus > 0)
	{

		
		off += snprintf(buf + off, bufsz - off,
						", \"gpus\" : [");
		for (i=0; i < linfo->nr_gpus; i++)
		{
			off += snprintf(buf + off, bufsz -off,
							"%s{ \"uuid\" : \"%s\" }",
							i > 0 ? ", " : " ",
                            linfo->gpu_uuid[i]);
        }
		 off += snprintf(buf + off, bufsz - off, " ]");
	}
	off += snprintf(buf + off, bufsz - off, "}");

	return off;
}

/*
 * heterodb_license_query_v3
 */
static ssize_t
heterodb_license_query_v3(heterodb_license_info_v3 *linfo, char *buf, size_t bufsz)
{
	size_t		off = 0;
	int			i;

	off += snprintf(buf + off, bufsz - off,
					"{ \"version\" : %d",
					linfo->version);
	if (linfo->serial_nr)
		off += snprintf(buf + off, bufsz - off,
						", \"serial_nr\" : \"%s\"",
						linfo->serial_nr);	
	off += snprintf(buf + off, bufsz - off,
					", \"issued_at\" : \"%04d-%02d-%02d\"",
					(linfo->issued_at / 10000),
					(linfo->issued_at / 100) % 100,
					(linfo->issued_at % 100));
	off += snprintf(buf + off, bufsz - off,
					", \"expired_at\" : \"%04d-%02d-%02d\"",
					(linfo->expired_at / 10000),
					(linfo->expired_at / 100) % 100,
					(linfo->expired_at % 100));
	if (linfo->cloud_vendor)
		off += snprintf(buf + off, bufsz - off,
						", \"cloud_vendor\" : \"%s\"",
						linfo->cloud_vendor);
	if (linfo->vm_type_id)
		off += snprintf(buf + off, bufsz - off,
                        ", \"vm_type_id\" : \"%s\"",
						linfo->vm_type_id);
	if (linfo->vm_image_id)
		off += snprintf(buf + off, bufsz - off,
						", \"vm_image_id\" : \"%s\"",
						linfo->vm_image_id);
	if (linfo->description)
		 off += snprintf(buf + off, bufsz - off,
						 ", \"description\" : \"%s\"",
						 linfo->description);
	off += snprintf(buf + off, bufsz - off,
					", \"nr_gpus\" : %u",
					linfo->nr_gpus);
	if (linfo->nr_gpus > 0)
	{
		size_t	__off_saved = off;
		int		found = 0;

		off += snprintf(buf + off, bufsz - off,
						", \"gpus\" : [");
		for (i=0; i < linfo->nr_gpus; i++)
		{
			if (!linfo->gpu_uuid[i])
				continue;
			off += snprintf(buf + off, bufsz -off,
							"%s{ \"uuid\" : \"%s\" }",
							i > 0 ? ", " : " ",
							linfo->gpu_uuid[i]);
			found = 1;
		}
		off += snprintf(buf + off, bufsz - off, " ]");
		if (!found)
			off = __off_saved;
	}
	off += snprintf(buf + off, bufsz - off, "}");

	return off;
}

/*
 * heterodb_license_query
 */
ssize_t
heterodb_license_query(char *buf, size_t bufsz)
{
	if (current_license_info)
	{
		if (current_license_info->version == 2)
			return heterodb_license_query_v2(&current_license_info->v2,
											 buf, bufsz);
		if (current_license_info->version == 3)
			return heterodb_license_query_v3(&current_license_info->v3,
											 buf, bufsz);
	}
	return -1;
}

/*
 * heterodb_validate_device
 */
int
heterodb_validate_device(int gpu_device_id,
						 const char *gpu_device_name,
						 const char *gpu_device_uuid)
{
	char		uuid[100];
	int			i;

	if (!current_license_info ||
		time(NULL) > current_license_expired_at)
	{
		/* license is not valid, so only first GPU can be visible */
		if (current_free_gpu_count == 0)
		{
			memcpy(current_free_gpu_uuid, gpu_device_uuid, __UUID_LEN);
			current_free_gpu_count++;
			return 1;
		}
		return (memcmp(current_free_gpu_uuid, gpu_device_uuid, __UUID_LEN) == 0);
	}

	sprintf(uuid, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
			(unsigned char)gpu_device_uuid[0],
			(unsigned char)gpu_device_uuid[1],
			(unsigned char)gpu_device_uuid[2],
			(unsigned char)gpu_device_uuid[3],
			(unsigned char)gpu_device_uuid[4],
			(unsigned char)gpu_device_uuid[5],
			(unsigned char)gpu_device_uuid[6],
			(unsigned char)gpu_device_uuid[7],
			(unsigned char)gpu_device_uuid[8],
			(unsigned char)gpu_device_uuid[9],
			(unsigned char)gpu_device_uuid[10],
			(unsigned char)gpu_device_uuid[11],
			(unsigned char)gpu_device_uuid[12],
			(unsigned char)gpu_device_uuid[13],
			(unsigned char)gpu_device_uuid[14],
			(unsigned char)gpu_device_uuid[15]);
	if (current_license_info->version == 2)
	{
		for (i=0; i < current_license_info->v2.nr_gpus; i++)
		{
			const char *__uuid = current_license_info->v2.gpu_uuid[i];

			if (__uuid && strcmp(__uuid, uuid) == 0)
				return 1;
		}
	}
	else if (current_license_info->version == 3)
	{
		for (i=0; i < current_license_info->v3.nr_gpus; i++)
		{
			const char *__uuid = current_license_info->v3.gpu_uuid[i];

			if (__uuid && strcmp(__uuid, uuid) == 0)
				return 1;
		}
	}
	return 0;
}

#if 0
int main(int argc, char *argv[])
{
	char	buf[10240];
	size_t	bufsz = sizeof(buf);

	if (heterodb_license_reload() > 0)
	{
		if (heterodb_license_query(buf, bufsz) > 0)
			puts(buf);
		else
			fprintf(stderr, "no valid license");
	}
	return 0;
}
#endif
