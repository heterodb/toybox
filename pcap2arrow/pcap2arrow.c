/*
 * pcap2arrow.c
 *
 * multi-thread ultra fast packet capture and translator to Apache Arrow.
 *
 * Portions Copyright (c) 2021, HeteroDB Inc
 */
#include <ctype.h>
#include <fcntl.h>
#include <getopt.h>
#include <limits.h>
#include <pcap.h>
#include <pfring.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "arrow_ipc.h"

#if 1
#define Assert(x)					assert(x)
#else
#define Assert(x)
#endif

#define __PCAP_PROTO__IPv4			0x0001
#define __PCAP_PROTO__IPv6			0x0002
#define __PCAP_PROTO__TCP			0x0010
#define __PCAP_PROTO__UDP			0x0020
#define __PCAP_PROTO__ICMP			0x0040

#define PCAP_PROTO__RAW_IPv4		(__PCAP_PROTO__IPv4)
#define PCAP_PROTO__TCP_IPv4		(__PCAP_PROTO__IPv4 | __PCAP_PROTO__TCP)
#define PCAP_PROTO__UDP_IPv4		(__PCAP_PROTO__IPv4 | __PCAP_PROTO__UDP)
#define PCAP_PROTO__ICMP_IPv4		(__PCAP_PROTO__IPv4 | __PCAP_PROTO__ICMP)
#define PCAP_PROTO__RAW_IPv6		(__PCAP_PROTO__IPv6)
#define PCAP_PROTO__TCP_IPv6		(__PCAP_PROTO__IPv6 | __PCAP_PROTO__TCP)
#define PCAP_PROTO__UDP_IPv6		(__PCAP_PROTO__IPv6 | __PCAP_PROTO__UDP)
#define PCAP_PROTO__ICMP_IPv6		(__PCAP_PROTO__IPv6 | __PCAP_PROTO__ICMP)
#define PCAP_PROTO__DEFAULT			(PCAP_PROTO__TCP_IPv4 |	\
									 PCAP_PROTO__UDP_IPv4 |	\
									 PCAP_PROTO__ICMP_IPv4)

#define PCAP_SWITCH__NEVER			0
#define PCAP_SWITCH__PER_MINUTE		1
#define PCAP_SWITCH__PER_HOUR		2
#define PCAP_SWITCH__PER_DAY		3
#define PCAP_SWITCH__PER_WEEK		4
#define PCAP_SWITCH__PER_MONTH		5

typedef struct arrowFileDesc		arrowFileDesc;
typedef struct arrowChunkBuffer		arrowChunkBuffer;

#define PCAP_SCHEMA_MAX_NFIELDS		50
/*
 * arrowFileDesc
 */
struct arrowFileDesc
{
	pthread_mutex_t	mutex;		/* mutex to protect the table below */
	SQLtable		table;		/* to track RecordBatch array */
};

/*
 * arrowChunkBuffer
 */
struct arrowChunkBuffer
{
	arrowChunkBuffer *next;		/* for arrow_chunk_buffer_pending */
	size_t			usage;
	size_t			nitems;
	int				nfields;
	SQLfield		columns[FLEXIBLE_ARRAY_MEMBER];
};

/*
 * misc definitions
 */
#define MACADDR_LEN			6
#define IP4ADDR_LEN			4
#define IP6ADDR_LEN			16

/* command-line options */
static char		   *input_pathname = NULL;
static char		   *output_filename = "/tmp/pcap_%i_%y%m%d_%H:%M:%S.arrow";
static int			protocol_mask = PCAP_PROTO__DEFAULT;
static int			num_threads = -1;
static int			num_pcap_threads = -1;
static char		   *bpf_filter_rule = NULL;
static size_t		output_filesize_limit = 0UL;				/* No limit */
static size_t		record_batch_threshold = (128UL << 20);		/* 128MB */
static bool			enable_direct_io = false;
static bool			only_headers = false;

/* static variable for PF-RING capture mode */
static pfring		   *pd = NULL;
static volatile bool	do_shutdown = false;
static pthread_mutex_t	arrow_file_desc_lock;
static arrowFileDesc   *arrow_file_desc_current = NULL;
static arrowChunkBuffer *arrow_chunk_buffer_pending = NULL;
static char			   *arrow_file_head_image = NULL;
static size_t			arrow_file_head_length = 0;
static sem_t			pcap_worker_sem;

/* static variables for PCAP file scan mode */
#define PCAP_MAGIC__HOST		0xa1b2c3d4U
#define PCAP_MAGIC__SWAP		0xd4c3b2a1U
#define PCAP_MAGIC__HOST_NS		0xa1b23c4dU
#define PCAP_MAGIC__SWAP_NS		0x4d3cb2a1U

static char			   *pcap_filemap = NULL;
static uint32_t			pcap_file_magic = 0;
static size_t			pcap_file_sz = 0UL;
static uint64_t			pcap_file_read_pos = 0;

/* other static variables */
static long				PAGESIZE;
static long				NCPUS;
static __thread int		worker_id = -1;

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define __ntoh16(x)			__builtin_bswap16(x)
#define __ntoh32(x)			__builtin_bswap32(x)
#define __ntoh64(x)			__builtin_bswap64(x)
#else
#define __ntoh16(x)			(x)
#define __ntoh32(x)			(x)
#define __ntoh64(x)			(x)
#endif

static inline void
pthreadMutexInit(pthread_mutex_t *mutex)
{
	if ((errno = pthread_mutex_init(mutex, NULL)) != 0)
		Elog("failed on pthread_mutex_init: %m");
}

static inline void
pthreadMutexLock(pthread_mutex_t *mutex)
{
	if ((errno = pthread_mutex_lock(mutex)) != 0)
		Elog("failed on pthread_mutex_lock: %m");
}

static inline void
pthreadMutexUnlock(pthread_mutex_t *mutex)
{
	if ((errno = pthread_mutex_unlock(mutex)) != 0)
        Elog("failed on pthread_mutex_unlock: %m");
}

static inline void
pthreadCondInit(pthread_cond_t *cond)
{
	if ((errno = pthread_cond_init(cond, NULL)) != 0)
		Elog("failed on pthread_cond_init: %m");
}

static inline void
pthreadCondBroadcast(pthread_cond_t *cond)
{
	if ((errno = pthread_cond_broadcast(cond)) != 0)
		Elog("failed on pthread_cond_broadcast: %m");
}
/*
 * atomic operations
 */
static inline uint64_t
atomicAdd64(volatile uint64_t *addr, uint64_t value)
{
	return __atomic_fetch_add(addr, value, __ATOMIC_SEQ_CST);
}

/*
 * SIGINT handler
 */
static void
on_sigint_handler(int signal)
{
	int		errno_saved = errno;

	do_shutdown = true;
	if (pd)
		pfring_breakloop(pd);

	errno = errno_saved;
}

/*
 * pcap_file_mmap - map pcap file and check header
 */
static void *
pcap_file_mmap(const char *pathname, struct stat *stat_buf)
{
	int		fdesc;
	struct pcap_file_header *pcap_head;

	fdesc = open(pathname, O_RDONLY);
	if (fdesc < 0)
		Elog("failed to open file '%s': %m", pathname);
	pcap_head = mmap(NULL, stat_buf->st_size, PROT_READ, MAP_PRIVATE, fdesc, 0);
	if (pcap_head == MAP_FAILED)
		Elog("failed to mmap file '%s': %m", pathname);
	close(fdesc);

	/* check pcal file header */
	Elog("right now pcap file read is not implemented yet");

	pcap_file_magic = pcap_head->magic;
	pcap_file_sz = stat_buf->st_size;
	pcap_file_read_pos = sizeof(struct pcap_file_header);

	return (void *)pcap_head;
}

/* ----------------------------------------------------------------
 *
 * Routines for PCAP Arrow Schema Definition
 *
 * ----------------------------------------------------------------
 */
static inline size_t
__buffer_usage_inline_type(SQLfield *column)
{
	size_t		usage;

	usage = ARROWALIGN(column->values.usage);
	if (column->nullcount > 0)
		usage += ARROWALIGN(column->nullmap.usage);
	return usage;
}

static inline size_t
__buffer_usage_varlena_type(SQLfield *column)
{
	size_t		usage;

	usage = (ARROWALIGN(column->values.usage) +
			 ARROWALIGN(column->extra.usage));
	if (column->nullcount > 0)
		usage += ARROWALIGN(column->nullmap.usage);
	return usage;
}

static inline void
__put_inline_null_value(SQLfield *column, size_t index, int sz)
{
	column->nullcount++;
	sql_buffer_clrbit(&column->nullmap, index);
	sql_buffer_append_zero(&column->values, sz);
}

static inline size_t
__put_uint_value_common(SQLfield *column, const char *addr, int sz)
{
	size_t		index = column->nitems++;

	if (!addr)
		__put_inline_null_value(column, index, sz);
	else
	{
		sql_buffer_setbit(&column->nullmap, index);
		sql_buffer_append(&column->values, addr, sz);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_uint8_value(SQLfield *column, const char *addr, int sz)
{
	size_t		index = column->nitems++;

	Assert(!addr || sz == sizeof(uint8_t));
	if (!addr)
		__put_inline_null_value(column, index, sizeof(uint8_t));
	else
	{
		sql_buffer_setbit(&column->nullmap, index);
		sql_buffer_append(&column->values, addr, sizeof(uint8_t));
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_uint16_value(SQLfield *column, const char *addr, int sz)
{
	size_t		index = column->nitems++;

	Assert(!addr || sz == sizeof(uint16_t));
	if (!addr)
		__put_inline_null_value(column, index, sizeof(uint16_t));
	else
	{
		sql_buffer_setbit(&column->nullmap, index);
		sql_buffer_append(&column->values, addr, sizeof(uint16_t));
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_uint16_value_bswap(SQLfield *column, const char *addr, int sz)
{
	size_t		index = column->nitems++;

	Assert(!addr || sz == sizeof(uint16_t));
	if (!addr)
		__put_inline_null_value(column, index, sizeof(uint16_t));
	else
	{
		uint16_t	value = __ntoh16(*((uint16_t *)addr));

		sql_buffer_setbit(&column->nullmap, index);
		sql_buffer_append(&column->values, &value, sz);
	}
	return __buffer_usage_inline_type(column);
}

#if 0
static size_t
put_uint32_value(SQLfield *column, const char *addr, int sz)
{
	size_t		index = column->nitems++;

	Assert(!addr || sz == sizeof(uint32_t));
	if (!addr)
		__put_inline_null_value(column, index, sizeof(uint32_t));
	else
	{
		sql_buffer_setbit(&column->nullmap, index);
		sql_buffer_append(&column->values, addr, sizeof(uint32_t));
	}
	return __buffer_usage_inline_type(column);
}
#endif

static size_t
put_uint32_value_bswap(SQLfield *column, const char *addr, int sz)
{
	size_t		index = column->nitems++;

	Assert(!addr || sz == sizeof(uint32_t));
	if (!addr)
		__put_inline_null_value(column, index, sizeof(uint16_t));
	else
	{
		uint32_t	value = __ntoh32(*((uint32_t *)addr));

		sql_buffer_setbit(&column->nullmap, index);
		sql_buffer_append(&column->values, &value, sz);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_timestamp_us_value(SQLfield *column, const char *addr, int sz)
{
	size_t		index = column->nitems++;
	uint64_t	value;

	if (!addr)
		__put_inline_null_value(column, index, sizeof(uint64_t));
	else
	{
		Assert(sz == sizeof(struct timeval));
		value = (((struct timeval *)addr)->tv_sec * 1000000L +
				 ((struct timeval *)addr)->tv_usec);
		sql_buffer_setbit(&column->nullmap, index);
		sql_buffer_append(&column->values, &value, sizeof(uint64_t));
	}
    return __buffer_usage_inline_type(column);
}

static inline size_t
__put_fixed_size_binary_value_common(SQLfield *column, const char *addr, int sz)
{
	size_t		index = column->nitems++;

	Assert(column->arrow_type.FixedSizeBinary.byteWidth == sz);
	if (!addr)
		__put_inline_null_value(column, index, sz);
	else
	{
		sql_buffer_setbit(&column->nullmap, index);
		sql_buffer_append(&column->values, addr, sz);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_fixed_size_binary_macaddr_value(SQLfield *column, const char *addr, int sz)
{
	Assert(!addr || sz == MACADDR_LEN);
	return __put_fixed_size_binary_value_common(column, addr, MACADDR_LEN);
}

static size_t
put_fixed_size_binary_ip4addr_value(SQLfield *column, const char *addr, int sz)
{
	Assert(!addr || sz == IP4ADDR_LEN);
	return __put_fixed_size_binary_value_common(column, addr, IP4ADDR_LEN);
}

static size_t
put_fixed_size_binary_ip6addr_value(SQLfield *column, const char *addr, int sz)
{
	Assert(!addr || sz == IP6ADDR_LEN);
	return __put_fixed_size_binary_value_common(column, addr, IP6ADDR_LEN);
}

static size_t
put_variable_value(SQLfield *column, const char *addr, int sz)
{
	size_t		index = column->nitems++;

	if (index == 0)
		sql_buffer_append_zero(&column->values, sizeof(uint32_t));
	if (!addr)
	{
		column->nullcount++;
		sql_buffer_clrbit(&column->nullmap, index);
		sql_buffer_append(&column->values,
						  &column->extra.usage, sizeof(uint32_t));
	}
	else
	{
		sql_buffer_setbit(&column->nullmap, index);
		sql_buffer_append(&column->extra, addr, sz);
		sql_buffer_append(&column->values,
						  &column->extra.usage, sizeof(uint32_t));
	}
	return __buffer_usage_varlena_type(column);
}

static void
arrowFieldAddCustomMetadata(SQLfield *column,
							const char *key,
							const char *value)
{
	ArrowKeyValue *kv;

	if (column->numCustomMetadata == 0)
	{
		Assert(column->customMetadata == NULL);
		column->customMetadata = palloc(sizeof(ArrowKeyValue));
	}
	else
	{
		size_t	sz = sizeof(ArrowKeyValue) * (column->numCustomMetadata + 1);

		Assert(column->customMetadata != NULL);
		column->customMetadata = repalloc(column->customMetadata, sz);
	}
	kv = &column->customMetadata[column->numCustomMetadata++];
	initArrowNode(&kv->node, KeyValue);
	kv->key = pstrdup(key);
	kv->_key_len = strlen(key);
	kv->value = pstrdup(value);
	kv->_value_len = strlen(value);
}

static void
arrowFieldInitAsUint8(SQLfield *column, const char *field_name)
{
	memset(column, 0, sizeof(SQLfield));
	initArrowNode(&column->arrow_type, Int);
	column->arrow_type.Int.bitWidth = 8;
	column->arrow_type.Int.is_signed = false;
	column->put_value = put_uint8_value;
	column->field_name = pstrdup(field_name);
}

static void
arrowFieldInitAsUint16(SQLfield *column, const char *field_name)
{
	memset(column, 0, sizeof(SQLfield));
	initArrowNode(&column->arrow_type, Int);
	column->arrow_type.Int.bitWidth = 16;
	column->arrow_type.Int.is_signed = false;
	column->put_value = put_uint16_value;
	column->field_name = pstrdup(field_name);
}

static void
arrowFieldInitAsUint16Bswap(SQLfield *column, const char *field_name)
{
	memset(column, 0, sizeof(SQLfield));
	initArrowNode(&column->arrow_type, Int);
	column->arrow_type.Int.bitWidth = 16;
	column->arrow_type.Int.is_signed = false;
	column->put_value = put_uint16_value_bswap;
	column->field_name = pstrdup(field_name);
}

#if 0
static void
arrowFieldInitAsUint32(SQLfield *column, const char *field_name)
{
	memset(column, 0, sizeof(SQLfield));
	initArrowNode(&column->arrow_type, Int);
	column->arrow_type.Int.bitWidth = 32;
	column->arrow_type.Int.is_signed = false;
	column->put_value = put_uint32_value;
	column->field_name = pstrdup(field_name);
}
#endif

static void
arrowFieldInitAsUint32Bswap(SQLfield *column, const char *field_name)
{
	memset(column, 0, sizeof(SQLfield));
	initArrowNode(&column->arrow_type, Int);
	column->arrow_type.Int.bitWidth = 16;
	column->arrow_type.Int.is_signed = false;
	column->put_value = put_uint32_value_bswap;
	column->field_name = pstrdup(field_name);
}

static void
arrowFieldInitAsTimestampUs(SQLfield *column, const char *field_name)
{
	memset(column, 0, sizeof(SQLfield));
	initArrowNode(&column->arrow_type, Timestamp);
	column->arrow_type.Timestamp.unit = ArrowTimeUnit__MilliSecond;
	/* no timezone setting, right now */
	column->put_value = put_timestamp_us_value;
	column->field_name = pstrdup(field_name);
}

static void
arrowFieldInitAsMacAddr(SQLfield *column, const char *field_name)
{
	memset(column, 0, sizeof(SQLfield));
	initArrowNode(&column->arrow_type, FixedSizeBinary);
	column->arrow_type.FixedSizeBinary.byteWidth = MACADDR_LEN;
	column->put_value = put_fixed_size_binary_macaddr_value;
	column->field_name = pstrdup(field_name);
	arrowFieldAddCustomMetadata(column, "pg_type", "pg_catalog.macaddr");
}

static void
arrowFieldInitAsIP4Addr(SQLfield *column, const char *field_name)
{
	memset(column, 0, sizeof(SQLfield));
	initArrowNode(&column->arrow_type, FixedSizeBinary);
	column->arrow_type.FixedSizeBinary.byteWidth = IP4ADDR_LEN;
	column->put_value = put_fixed_size_binary_ip4addr_value;
	column->field_name = pstrdup(field_name);
	arrowFieldAddCustomMetadata(column, "pg_type", "pg_catalog.inet");
}

static void
arrowFieldInitAsIP6Addr(SQLfield *column, const char *field_name)
{
	memset(column, 0, sizeof(SQLfield));
	initArrowNode(&column->arrow_type, FixedSizeBinary);
	column->arrow_type.FixedSizeBinary.byteWidth = IP6ADDR_LEN;
	column->put_value = put_fixed_size_binary_ip6addr_value;
	column->field_name = pstrdup(field_name);
	arrowFieldAddCustomMetadata(column, "pg_type", "pg_catalog.inet");
}

static void
arrowFieldInitAsBinary(SQLfield *column, const char *field_name)
{
	memset(column, 0, sizeof(SQLfield));
    initArrowNode(&column->arrow_type, Binary);
	column->put_value = put_variable_value;
	column->field_name = pstrdup(field_name);
}

#if 0
/* IPv6 Routing Ext Header */
static size_t
put_ipv6ext_routing_value(SQLfield *column, const char *addr, int sz)
{
}

static void
arrowFieldInitAsIPv6Ext_Routing(SQLfield *column, const char *field_name)
{
	SQLfield   *subfields = palloc0(sizeof(SQLfield) * 3);

	memset(column, 0, sizeof(SQLfield));
	initArrowNode(&column->arrow_type, Struct);
	column->put_value = put_ipv6ext_routing_value;

	arrowFieldInitAsUint8(&subfields[0], "routing_type");
	arrowFieldInitAsUint8(&subfields[1], "segment_left");
	arrowFieldInitAsBinary(&subfields[2], "data");
	column->subfields = subfields;
	column->nfields = 3;
}

/* IPv6 Fragment Ext Header */
static size_t
put_ipv6ext_fragment_value(SQLfield *column, const char *addr, int sz)
{}

static void
arrowFieldInitAsIPv6Ext_Fragment(SQLfield *column, const char *field_name)
{
	SQLfield   *subfields = palloc0(sizeof(SQLfield) * 3);

	memset(column, 0, sizeof(SQLfield));
	initArrowNode(&column->arrow_type, Struct);
	column->put_value = put_ipv6ext_fragment_value;

	arrowFieldInitAsUint16(&subfields[0], "fragment_offset");
	arrowFieldInitAsBool(&subfields[1], "m_flag");
	arrowFieldInitAsUint32(&subfields[2], "identification");
	column->subfields = subfields;
	column->nfields = 3;
}

static void
arrowFieldInitAsIPv6Ext_AH(SQLfield *column, const char *field_name)
{}

static void
arrowFieldInitAsIPv6Ext_ESP(SQLfield *column, const char *field_name)
{}
#endif

/* basic ethernet frame */
static int arrow_cindex__timestamp			= -1;
static int arrow_cindex__dst_mac			= -1;
static int arrow_cindex__src_mac			= -1;
static int arrow_cindex__ether_type			= -1;
/* IPv4 headers */
static int arrow_cindex__tos				= -1;
static int arrow_cindex__length				= -1;
static int arrow_cindex__identifier			= -1;
static int arrow_cindex__fragment			= -1;
static int arrow_cindex__ttl				= -1;
static int arrow_cindex__protocol			= -1;
static int arrow_cindex__ip_checksum		= -1;
static int arrow_cindex__src_addr			= -1;
static int arrow_cindex__dst_addr			= -1;
static int arrow_cindex__ip_options			= -1;
/* TCP/UDP headers */
static int arrow_cindex__src_port			= -1;
static int arrow_cindex__dst_port			= -1;
/* TCP headers */
static int arrow_cindex__seq_nr				= -1;
static int arrow_cindex__ack_nr				= -1;
static int arrow_cindex__tcp_flags			= -1;
static int arrow_cindex__window_sz			= -1;
static int arrow_cindex__tcp_checksum		= -1;
static int arrow_cindex__urgent_ptr			= -1;
static int arrow_cindex__tcp_options		= -1;
/* UDP headers */
static int arrow_cindex__segment_sz			= -1;
static int arrow_cindex__udp_checksum		= -1;
/* ICMP headers */
static int arrow_cindex__icmp_type			= -1;
static int arrow_cindex__icmp_code			= -1;
static int arrow_cindex__icmp_checksum		= -1;
/* Payload */
static int arrow_cindex__payload			= -1;

static int
arrowSchemaInitFields(SQLfield *columns)
{
	int		j = 0;

#define __ARROW_FIELD_INIT(__NAME, __TYPE)				\
	if (arrow_cindex__##__NAME < 0)						\
		arrow_cindex__##__NAME = j;						\
	else												\
		Assert(arrow_cindex__##__NAME == j);			\
	arrowFieldInitAs##__TYPE(&columns[j++], (#__NAME))

	/* timestamp and mac-address */
    __ARROW_FIELD_INIT(timestamp,	TimestampUs);
    __ARROW_FIELD_INIT(dst_mac,		MacAddr);
    __ARROW_FIELD_INIT(src_mac,		MacAddr);
    __ARROW_FIELD_INIT(ether_type,	Uint16);	/* byte swap by caller */

	/* IPv4 */
	if ((protocol_mask & __PCAP_PROTO__IPv4) != 0)
	{
		__ARROW_FIELD_INIT(tos,			Uint8);
		__ARROW_FIELD_INIT(length,		Uint16Bswap);
		__ARROW_FIELD_INIT(identifier,	Uint16Bswap);
		__ARROW_FIELD_INIT(fragment,	Uint16Bswap);
		__ARROW_FIELD_INIT(ttl,			Uint8);
		__ARROW_FIELD_INIT(protocol,	Uint8);
		__ARROW_FIELD_INIT(ip_checksum,	Uint16Bswap);
		__ARROW_FIELD_INIT(src_addr,	IP4Addr);
		__ARROW_FIELD_INIT(dst_addr,	IP4Addr);
		__ARROW_FIELD_INIT(ip_options,	Binary);
	}
	/* IPv6 */
	if ((protocol_mask & __PCAP_PROTO__IPv6) != 0)
	{
		Elog("IPv6 is not implemented yet");
		__ARROW_FIELD_INIT(src_addr,	IP6Addr);
		__ARROW_FIELD_INIT(dst_addr,	IP6Addr);
	}
	/* TCP or UDP */
	if ((protocol_mask & (__PCAP_PROTO__TCP | __PCAP_PROTO__UDP)) != 0)
	{
		__ARROW_FIELD_INIT(src_port,	Uint16Bswap);
		__ARROW_FIELD_INIT(dst_port,	Uint16Bswap);
	}
	/* TCP */
	if ((protocol_mask & __PCAP_PROTO__TCP) != 0)
	{
		__ARROW_FIELD_INIT(seq_nr,		Uint32Bswap);
		__ARROW_FIELD_INIT(ack_nr,		Uint32Bswap);
		__ARROW_FIELD_INIT(tcp_flags,	Uint16Bswap);
		__ARROW_FIELD_INIT(window_sz,	Uint16Bswap);
		__ARROW_FIELD_INIT(tcp_checksum,Uint16Bswap);
		__ARROW_FIELD_INIT(urgent_ptr,	Uint16Bswap);
		__ARROW_FIELD_INIT(tcp_options,	Binary);
	}
	/* UDP */
	if ((protocol_mask & __PCAP_PROTO__UDP) == __PCAP_PROTO__UDP)
	{
		__ARROW_FIELD_INIT(segment_sz,   Uint16Bswap);
		__ARROW_FIELD_INIT(udp_checksum, Uint16Bswap);
	}
	/* ICMP */
	if ((protocol_mask & __PCAP_PROTO__ICMP) == __PCAP_PROTO__ICMP)
	{
		__ARROW_FIELD_INIT(icmp_type,	  Uint8);
		__ARROW_FIELD_INIT(icmp_code,	  Uint8);
		__ARROW_FIELD_INIT(icmp_checksum, Uint16Bswap);
	}
	/* remained data - payload */
	__ARROW_FIELD_INIT(payload,		Binary);
#undef __ARROW_FIELD_INIT

	return j;
}

#define __FIELD_PUT_VALUE_DECL											\
	SQLfield   *__field;												\
	size_t      usage = 0
#define __FIELD_PUT_VALUE(NAME,ADDR,SZ)									\
	Assert(arrow_cindex__##NAME >= 0);									\
	__field = &chunk->columns[arrow_cindex__##NAME];					\
	usage += __field->put_value(__field, (const char *)(ADDR),(SZ))

/*
 * handlePacketRawEthernet
 */
static u_char *
handlePacketRawEthernet(arrowChunkBuffer *chunk,
						struct pfring_pkthdr *hdr,
						u_char *buf, uint16_t *p_ether_type)
{
	__FIELD_PUT_VALUE_DECL;
	struct __raw_ether {
		u_char		dst_mac[6];
		u_char		src_mac[6];
		uint16_t	ether_type;
	}		   *raw_ether = (struct __raw_ether *)buf;

	__FIELD_PUT_VALUE(timestamp, &hdr->ts, sizeof(hdr->ts));
	if (hdr->caplen < sizeof(struct __raw_ether))
	{
		__FIELD_PUT_VALUE(dst_mac, NULL, 0);
		__FIELD_PUT_VALUE(src_mac, NULL, 0);
		__FIELD_PUT_VALUE(ether_type, NULL, 0);
		return NULL;
	}
	__FIELD_PUT_VALUE(dst_mac, raw_ether->dst_mac, MACADDR_LEN);
	__FIELD_PUT_VALUE(src_mac, raw_ether->src_mac, MACADDR_LEN);
	*p_ether_type = __ntoh16(raw_ether->ether_type);
	__FIELD_PUT_VALUE(ether_type, p_ether_type, sizeof(uint16_t));

	chunk->usage = usage;	/* raw-ethernet shall be 1st call */
	
	return buf + sizeof(struct __raw_ether);
}

/*
 * handlePacketIPv4Header
 */
static u_char *
handlePacketIPv4Header(arrowChunkBuffer *chunk,
					   u_char *buf, size_t sz, uint8_t *p_proto)
{
	__FIELD_PUT_VALUE_DECL;
	struct __ipv4_head {
		uint8_t		version;	/* 4bit of version means header-size */
		uint8_t		tos;
		uint16_t	length;
		uint16_t	identifier;
		uint16_t	fragment;
		uint8_t		ttl;
		uint8_t		protocol;
		uint16_t	ip_checksum;
		uint32_t	src_addr;
		uint32_t	dst_addr;
		char		ip_options[0];
	}		   *ipv4 = (struct __ipv4_head *)buf;
	uint16_t	head_sz;

	if (!buf || sz < 20)
		goto fillup_by_null;
	/* 1st octet is IP version (4) and header size */
	if ((ipv4->version & 0xf0) != 0x40)
		goto fillup_by_null;
	head_sz = 4 * (ipv4->version & 0x0f);
	if (head_sz > sz)
		goto fillup_by_null;

	__FIELD_PUT_VALUE(tos,         &ipv4->tos,         sizeof(uint8_t));
	__FIELD_PUT_VALUE(length,      &ipv4->length,      sizeof(uint16_t));
	__FIELD_PUT_VALUE(identifier,  &ipv4->identifier,  sizeof(uint16_t));
	__FIELD_PUT_VALUE(fragment,    &ipv4->fragment,    sizeof(uint16_t));
	__FIELD_PUT_VALUE(ttl,         &ipv4->ttl,         sizeof(uint8_t));
	*p_proto = ipv4->protocol;
	__FIELD_PUT_VALUE(protocol,    &ipv4->protocol,    sizeof(uint8_t));
	__FIELD_PUT_VALUE(ip_checksum, &ipv4->ip_checksum, sizeof(uint16_t));
	__FIELD_PUT_VALUE(src_addr,    &ipv4->src_addr,    sizeof(uint32_t));
	__FIELD_PUT_VALUE(dst_addr,    &ipv4->dst_addr,    sizeof(uint32_t));
	if (head_sz > offsetof(struct __ipv4_head, ip_options))
	{
		__FIELD_PUT_VALUE(ip_options, ipv4->ip_options,
						  head_sz - offsetof(struct __ipv4_head, ip_options));
	}
	else
	{
		__FIELD_PUT_VALUE(ip_options, NULL, 0);
	}
	chunk->usage += usage;

	return buf + head_sz;

fillup_by_null:
	__FIELD_PUT_VALUE(tos, NULL, 0);
	__FIELD_PUT_VALUE(length, NULL, 0);
	__FIELD_PUT_VALUE(identifier, NULL, 0);
	__FIELD_PUT_VALUE(fragment, NULL, 0);
	__FIELD_PUT_VALUE(ttl, NULL, 0);
	__FIELD_PUT_VALUE(protocol, NULL, 0);
	__FIELD_PUT_VALUE(ip_checksum, NULL, 0);
	__FIELD_PUT_VALUE(src_addr, NULL, 0);
	__FIELD_PUT_VALUE(dst_addr, NULL, 0);
	__FIELD_PUT_VALUE(ip_options, NULL, 0);
	chunk->usage += usage;

	return NULL;
}

/*
 * handlePacketIPv6Header
 */
static u_char *
handlePacketIPv6Header(arrowChunkBuffer *chunk,
					   u_char *buf, size_t sz, uint8_t *p_proto)
{
	Elog("handlePacketIPv6Header not implemented yet");
	return NULL;
}

/*
 * handlePacketTcpHeader
 */
static u_char *
handlePacketTcpHeader(arrowChunkBuffer *chunk,
					  u_char *buf, size_t sz, int proto)
{
	__FIELD_PUT_VALUE_DECL;
	struct __tcp_head {
		uint16_t	src_port;
		uint16_t	dst_port;
		uint32_t	seq_nr;
		uint32_t	ack_nr;
		uint16_t	tcp_flags;
		uint16_t	window_sz;
		uint16_t	tcp_checksum;
		uint16_t	urgent_ptr;
		char		tcp_options[0];
	}		   *tcp = (struct __tcp_head *)buf;
	uint16_t	head_sz;

	if (!buf || sz < offsetof(struct __tcp_head, tcp_options))
		goto fillup_by_null;
	head_sz = sizeof(uint32_t) * (__ntoh16(tcp->tcp_flags) & 0x000f);
	if (head_sz > sz)
		goto fillup_by_null;

	__FIELD_PUT_VALUE(src_port,     &tcp->src_port,     sizeof(uint16_t));
	__FIELD_PUT_VALUE(dst_port,     &tcp->dst_port,     sizeof(uint16_t));
	__FIELD_PUT_VALUE(seq_nr,       &tcp->seq_nr,       sizeof(uint32_t));
	__FIELD_PUT_VALUE(ack_nr,       &tcp->ack_nr,       sizeof(uint32_t));
	__FIELD_PUT_VALUE(tcp_flags,    &tcp->tcp_flags,    sizeof(uint16_t));
	__FIELD_PUT_VALUE(window_sz,    &tcp->window_sz,    sizeof(uint16_t));
	__FIELD_PUT_VALUE(tcp_checksum, &tcp->tcp_checksum, sizeof(uint16_t));
	__FIELD_PUT_VALUE(urgent_ptr,   &tcp->urgent_ptr,   sizeof(uint16_t));
	if (head_sz > offsetof(struct __tcp_head, tcp_options))
	{
		__FIELD_PUT_VALUE(tcp_options, tcp->tcp_options,
						  head_sz - offsetof(struct __tcp_head, tcp_options));
	}
	else
	{
		__FIELD_PUT_VALUE(tcp_options, NULL, 0);
	}
	chunk->usage += usage;
	return buf + head_sz;

fillup_by_null:
	if (proto != 0x11)		/* if not UDP */
	{
		__FIELD_PUT_VALUE(src_port, NULL, 0);
		__FIELD_PUT_VALUE(dst_port, NULL, 0);
	}
	__FIELD_PUT_VALUE(seq_nr, NULL, 0);
	__FIELD_PUT_VALUE(ack_nr, NULL, 0);
	__FIELD_PUT_VALUE(tcp_flags, NULL, 0);
	__FIELD_PUT_VALUE(window_sz, NULL, 0);
	__FIELD_PUT_VALUE(tcp_checksum, NULL, 0);
	__FIELD_PUT_VALUE(urgent_ptr, NULL, 0);
	__FIELD_PUT_VALUE(tcp_options, NULL, 0);
	chunk->usage += usage;
	return NULL;
}

/*
 * handlePacketUdpHeader
 */
static u_char *
handlePacketUdpHeader(arrowChunkBuffer *chunk,
					  u_char *buf, size_t sz, int proto)
{
	__FIELD_PUT_VALUE_DECL;
	struct __udp_head {
		uint16_t	src_port;
		uint16_t	dst_port;
		uint16_t	segment_sz;
		uint16_t	udp_checksum;
	}		   *udp = (struct __udp_head *) buf;

	if (!buf || sz < sizeof(struct __udp_head))
		goto fillup_by_null;
	__FIELD_PUT_VALUE(src_port,     &udp->src_port,     sizeof(uint16_t));
	__FIELD_PUT_VALUE(dst_port,     &udp->dst_port,     sizeof(uint16_t));
	__FIELD_PUT_VALUE(segment_sz,   &udp->segment_sz,   sizeof(uint16_t));
	__FIELD_PUT_VALUE(udp_checksum, &udp->udp_checksum, sizeof(uint16_t));
	chunk->usage += usage;
	return buf + sizeof(struct __udp_head);

fillup_by_null:
	if (proto != 0x06)		/* if not TCP */
	{
		//???
		__FIELD_PUT_VALUE(src_port, NULL, 0);
		__FIELD_PUT_VALUE(dst_port, NULL, 0);
	}
	__FIELD_PUT_VALUE(segment_sz, NULL, 0);
	__FIELD_PUT_VALUE(udp_checksum, NULL, 0);
	chunk->usage += usage;
	return NULL;
}

/*
 * handlePacketIcmpHeader
 */
static u_char *
handlePacketIcmpHeader(arrowChunkBuffer *chunk,
					   u_char *buf, size_t sz, int proto)
{
	__FIELD_PUT_VALUE_DECL;
	struct __icmp_head {
		uint8_t		icmp_type;
		uint8_t		icmp_code;
		uint16_t	icmp_checksum;
	}		   *icmp = (struct __icmp_head *) buf;

	if (!buf || sz < sizeof(struct __icmp_head))
		goto fillup_by_null;

	__FIELD_PUT_VALUE(icmp_type,     &icmp->icmp_type, sizeof(uint8_t));
	__FIELD_PUT_VALUE(icmp_code,     &icmp->icmp_code, sizeof(uint8_t));
	__FIELD_PUT_VALUE(icmp_checksum, &icmp->icmp_checksum, sizeof(uint16_t));
	chunk->usage += usage;
	return buf + sizeof(struct __icmp_head);

fillup_by_null:
	__FIELD_PUT_VALUE(icmp_type,     NULL, 0);
	__FIELD_PUT_VALUE(icmp_code,     NULL, 0);
	__FIELD_PUT_VALUE(icmp_checksum, NULL, 0);
	chunk->usage += usage;
	return NULL;
}

/*
 * handlePacketPayload
 */
static void
handlePacketPayload(arrowChunkBuffer *chunk, u_char *buf, size_t sz)
{
	__FIELD_PUT_VALUE_DECL;
	
	if (buf && sz > 0)
	{
		__FIELD_PUT_VALUE(payload, buf, sz);
	}
	else
	{
		__FIELD_PUT_VALUE(payload, NULL, 0);
	}
	chunk->usage += usage;
}

/*
 * arrowOpenOutputFile
 */
static arrowFileDesc *
arrowOpenOutputFile(void)
{
	static int	output_file_seqno = 1;
	time_t		tv = time(NULL);
	struct tm	tm;
	char	   *path, *pos;
	int			off, sz = 256;
	int			fdesc, flags;
	arrowFileDesc *outfd;

	/* build a filename */
	localtime_r(&tv, &tm);
	do {
		off = 0;
		path = alloca(sz);

		for (pos = output_filename; *pos != '\0' && off < sz; pos++)
		{
			if (*pos == '%')
			{
				pos++;
				switch (*pos)
				{
					case 'i':
						off += snprintf(path+off, sz-off, "%s", input_pathname);
						break;
					case 'Y':
						off += snprintf(path+off, sz-off, "%04d", tm.tm_year + 1900);
						break;
					case 'y':
						off += snprintf(path+off, sz-off, "%02d", tm.tm_year % 100);
						break;
					case 'm':
						off += snprintf(path+off, sz-off, "%02d", tm.tm_mon + 1);
						break;
					case 'd':
						off += snprintf(path+off, sz-off, "%02d", tm.tm_mday);
						break;
					case 'H':
						off += snprintf(path+off, sz-off, "%02d", tm.tm_hour);
						break;
					case 'M':
						off += snprintf(path+off, sz-off, "%02d", tm.tm_min);
						break;
					case 'S':
						off += snprintf(path+off, sz-off, "%02d", tm.tm_sec);
						break;
					case 'q':
						off += snprintf(path+off, sz-off, "%d", output_file_seqno);
					break;
					default:
						Elog("unexpected output file format '%%%c'", *pos);
				}
			}
			else
			{
				path[off++] = *pos;
			}
		}
		if (off < sz)
			path[off++] = '\0';
	} while (off >= sz);

	/* open file */
	flags = O_RDWR | O_CREAT | O_EXCL;
	if (enable_direct_io)
		flags |= O_DIRECT;
	fdesc = open(path, flags);
	if (fdesc < 0)
		Elog("failed to open('%s'): %m", path);

	/* setup arrowFileDesc */
	outfd = palloc0(offsetof(arrowFileDesc,
							 table.columns[PCAP_SCHEMA_MAX_NFIELDS]));
	pthreadMutexInit(&outfd->mutex);
	outfd->table.filename = pstrdup(path);
	outfd->table.fdesc = fdesc;
	outfd->table.f_pos = 0;
	outfd->table.segment_sz = record_batch_threshold;
	outfd->table.nfields = arrowSchemaInitFields(outfd->table.columns);

	/* write out header portion */
	if (arrow_file_head_image)
	{
		arrowFileWrite(&outfd->table, arrow_file_head_image, arrow_file_head_length);
	}
	else
	{
		ssize_t		nbytes, offset = 0;

		arrowFileWrite(&outfd->table, "ARROW1\0\0", 8);
		writeArrowSchema(&outfd->table);

		arrow_file_head_length = outfd->table.f_pos;
		printf("arrow_file_head_length = %zu\n", arrow_file_head_length);
		arrow_file_head_image = palloc(arrow_file_head_length);

		while (offset < arrow_file_head_length)
		{
			nbytes = pread(fdesc, (char *)arrow_file_head_image + offset,
						   arrow_file_head_length - offset, offset);
			if (nbytes < 0)
			{
				if (errno == EINTR)
					continue;
				Elog("failed on pread('%s'): %m", path);
			}
			else if (nbytes == 0)
				Elog("unexpected EOF at pread('%s')", path);
			offset += nbytes;
		}
	}
	return outfd;
}

/*
 * arrowChunkWriteOut
 */
static void
arrowChunkWriteOut(arrowChunkBuffer *chunk)
{

}

/*
 * __execCaptureOnePacket
 */
static inline void
__execCaptureOnePacket(arrowChunkBuffer *chunk,
					   struct pfring_pkthdr *hdr, u_char *pos)
{
	u_char	   *end = pos + hdr->caplen;
	u_char	   *next;
	uint16_t	ether_type;
	uint8_t		proto;

	pos = handlePacketRawEthernet(chunk, hdr, pos, &ether_type);
	if (!pos)
		goto fillup_by_null;

	if (ether_type == 0x0800)		/* IPv4 */
	{
		if ((protocol_mask & __PCAP_PROTO__IPv4) == 0)
			goto fillup_by_null;

		next = handlePacketIPv4Header(chunk, pos, end - pos, &proto);
		if (!next)
			goto fillup_by_null;
		pos = next;
		if ((protocol_mask & __PCAP_PROTO__IPv6) != 0)
			handlePacketIPv6Header(chunk, NULL, 0, NULL);
	}
	else if (ether_type == 0x86dd)	/* IPv6 */
	{
		if ((protocol_mask & __PCAP_PROTO__IPv6) == 0)
			goto fillup_by_null;
		next = handlePacketIPv6Header(chunk, pos, end - pos, &proto);
		if (!next)
			goto fillup_by_null;
		pos = next;
		if ((protocol_mask & __PCAP_PROTO__IPv4) != 0)
			handlePacketIPv4Header(chunk, NULL, 0, NULL);
	}
	else
	{
		/* neither IPv4 nor IPv6 */
		goto fillup_by_null;
	}

	/* TCP */
	if (proto == 0x06 && (protocol_mask & __PCAP_PROTO__TCP) != 0)
	{
		next = handlePacketTcpHeader(chunk, pos, end - pos, proto);
		if (!next)
			handlePacketTcpHeader(chunk, NULL, 0, -1);
		pos = next;
	}
	else
	{
		handlePacketTcpHeader(chunk, NULL, 0, proto);
	}

	/* UDP */
	if (proto == 0x11 && (protocol_mask & __PCAP_PROTO__UDP) != 0)
	{
		next = handlePacketUdpHeader(chunk, pos, end - pos, proto);
		if (!next)
			handlePacketUdpHeader(chunk, NULL, 0, 0xff);
		pos = next;
	}
	else
	{
		handlePacketUdpHeader(chunk, NULL, 0, proto);
	}

	/* ICMP */
	if (proto == 0x01 && (protocol_mask & __PCAP_PROTO__ICMP) != 0)
	{
		next = handlePacketIcmpHeader(chunk, pos, end - pos, proto);
		if (!next)
			handlePacketIcmpHeader(chunk, NULL, 0, 0xff);
		pos = next;
	}
	else
	{
		handlePacketIcmpHeader(chunk, NULL, 0, proto);
	}

	/* Payload */
	if (!only_headers)
		handlePacketPayload(chunk, pos, end - pos);
	return;

fillup_by_null:
	if ((protocol_mask & __PCAP_PROTO__IPv4) != 0)
		handlePacketIPv4Header(chunk, NULL, 0, NULL);
	if ((protocol_mask & __PCAP_PROTO__IPv6) != 0)
		handlePacketIPv6Header(chunk, NULL, 0, NULL);
	if ((protocol_mask & __PCAP_PROTO__TCP) != 0)
		handlePacketTcpHeader(chunk, NULL, 0, 0xff);
	if ((protocol_mask & __PCAP_PROTO__UDP) != 0)
		handlePacketUdpHeader(chunk, NULL, 0, 0xff);
	if ((protocol_mask & __PCAP_PROTO__ICMP) != 0)
		handlePacketIcmpHeader(chunk, NULL, 0, 0xff);
	if (!only_headers && pos != NULL)
		handlePacketPayload(chunk, pos, end - pos);
}

/*
 * execCapturePackets
 */
static int
execCapturePackets(arrowChunkBuffer *chunk)
{
	struct pfring_pkthdr hdr;
	u_char		__buffer[10000];
	u_char	   *buffer = __buffer;
	int			j, rv;

	/* clear the chunk-buffer */
	for (j=0; j < chunk->nfields; j++)
		sql_field_clear(&chunk->columns[j]);
	chunk->usage = 0;
	chunk->nitems = 0;

	while (!do_shutdown)
	{
		rv = pfring_recv(pd, &buffer, sizeof(__buffer), &hdr, 1);
		if (rv > 0)
		{
			__execCaptureOnePacket(chunk, &hdr, buffer);
			if (chunk->usage >= record_batch_threshold)
				return 1;	/* write out the buffer */
		}
	}
	/* interrupted, thus chunk-buffer is partially filled up */
	return 0;
}

/*
 * pcap_worker_main
 */
static void *
pcap_worker_main(void *__arg)
{
	arrowChunkBuffer *chunk;

	/* assign worker-id of this thread */
	worker_id = (long)__arg;

	/* allocation of local chunk buffer */
	chunk = palloc0(offsetof(arrowChunkBuffer,
							 columns[PCAP_SCHEMA_MAX_NFIELDS]));
	chunk->nfields = arrowSchemaInitFields(chunk->columns);

	while (!do_shutdown)
	{
		int		status = -1;

		if (sem_wait(&pcap_worker_sem) != 0)
		{
			if (errno == EINTR)
				continue;
			Elog("worker-%d: failed on sem_wait: %m", worker_id);
		}
		/*
		 * Ok, Go to packet capture
		 */
		if (!do_shutdown)
		{
			status = execCapturePackets(chunk);
			Assert(status >= 0);
		}
		if (sem_post(&pcap_worker_sem) != 0)
            Elog("failed on sem_post: %m");

		if (status > 0)
		{
			/*
			 * chunk-buffer consumption exceeds record-batch-threshold
			 * so, write out by itself.
			 */
			arrowChunkWriteOut(chunk);
		}
		else if (status == 0)
		{
			/*
			 * execCapturePackets() was interrupted by do_shutdown.
			 * so, final chunk shall be merged by the main thread.
			 */
			Assert(do_shutdown);
			pthreadMutexLock(&arrow_file_desc_lock);
			chunk->next = arrow_chunk_buffer_pending;
			arrow_chunk_buffer_pending = chunk;
			pthreadMutexUnlock(&arrow_file_desc_lock);

			break;
		}
	}
	return NULL;
}

/*
 * usage
 */
static int
usage(int status)
{
	fputs("usage: pcap2arrow [OPTIONS]\n"
		  "\n"
		  "OPTIONS:\n"
		  "  -i|--input=<input device or pcap file>\n"
		  "  -o|--output=<output file; with format>\n"
		  "      filename format can contains:"
		  "        %i : interface name\n"
		  "        %Y : year in 4-digits\n"
		  "        %y : year in 2-digits\n"
		  "        %m : month in 2-digits\n"
		  "        %d : day in 2-digits\n"
		  "        %H : hour in 2-digits\n"
		  "        %M : minute in 2-digits\n"
		  "        %S : second in 2-digits\n"
		  "        %q : sequence number when file is switched by -l|--limit\n"
		  "      default is '/tmp/pcap_%y%m%d_%H:%M:%S_%i_%i.arrow'\n"
		  "  -p|--protocol=<PROTO>\n"
		  "        <PROTO> is a comma separated string contains\n"
		  "        the following tokens:\n"
		  "          tcp4, udp4, icmp4, ipv4, tcp6, udp6, icmp6, ipv66\n"
		  "        (default: 'tcp4,udp4,icmp4,ipv4')\n"
		  "  -t|--threads=<NUM of threads> (default: 2 * NCPUs)\n"
		  "     --pcap-threads=<NUM of threads> (default: NCPUS)\n"
		  "  -l|--limit=<LIMIT> : (default: no limit)\n"
		  "  -s|--chunk-size=<SIZE> : size of record batch (default: 128MB)\n"
		  "  -r|--rule=<RULE> : packet filtering rules\n"
		  "     --direct-io : enables O_DIRECT for write-i/o\n"
		  "  -h|--help    : shows this message\n"
		  "\n"
		  "  Copyright 2020-2021 - HeteroDB,Inc\n",
		  stderr);
	exit(status);
}

static void
parse_options(int argc, char *argv[])
{
	static struct option long_options[] = {
		{"input",        required_argument, NULL, 'i'},
		{"output",       required_argument, NULL, 'o'},
		{"protocol",     required_argument, NULL, 'p'},
		{"threads",      required_argument, NULL, 't'},
		{"limit",        required_argument, NULL, 'l'},
		{"chunk-size",   required_argument, NULL, 's'},
		{"rule",         required_argument, NULL, 'r'},
		{"pcap-threads", required_argument, NULL, 1000},
		{"direct-io",    no_argument,       NULL, 1001},
		{"only-headers", no_argument,       NULL, 1002},
		{"help",         no_argument,       NULL, 'h'},
		{NULL, 0, NULL, 0}
	};
	int		code;
	bool	output_has_seqno = false;
	char   *pos;

	while ((code = getopt_long(argc, argv, "i:o:p:t:l:s:r:h",
							   long_options, NULL)) >= 0)
	{
		char	   *token, *end;

		switch (code)
		{
			case 'i':	/* input */
				if (input_pathname)
					Elog("-i|--input was specified twice");
				input_pathname = optarg;
				break;
			case 'o':	/* output */
				output_filename = optarg;
				break;
			case 'p':	/* protocol */
				protocol_mask = 0;
				for (token = strtok_r(optarg, ",", &pos);
					 token != NULL;
					 token = strtok_r(NULL, ",", &pos))
				{
					/* remove spaces */
					while (*token != '\0' && isspace(*token))
						token++;
					end = token + strlen(token) - 1;
					while (end >= token && isspace(*end))
						*end-- = '\0';
					if (strcmp(token, "ipv4") == 0)
						protocol_mask |= PCAP_PROTO__RAW_IPv4;
					else if (strcmp(token, "tcp4") == 0)
						protocol_mask |= PCAP_PROTO__TCP_IPv4;
					else if (strcmp(token, "udp4") == 0)
						protocol_mask |= PCAP_PROTO__UDP_IPv4;
					else if (strcmp(token, "icmp4") == 0)
						protocol_mask |= PCAP_PROTO__ICMP_IPv4;
					else if (strcmp(token, "ipv6") == 0)
						protocol_mask |= PCAP_PROTO__RAW_IPv6;
					else if (strcmp(token, "tcp6") == 0)
						protocol_mask |= PCAP_PROTO__TCP_IPv6;
					else if (strcmp(token, "udp6") == 0)
						protocol_mask |= PCAP_PROTO__UDP_IPv6;
					else if (strcmp(token, "icmp6") == 0)
						protocol_mask |= PCAP_PROTO__ICMP_IPv6;
					else
						Elog("unknown protocol [%s]", token);
				}
				break;
			case 't':	/* threads */
				num_threads = strtol(optarg, &pos, 10);
				if (*pos != '\0')
					Elog("invalid -t|--threads argument: %s", optarg);
				if (num_threads < 1)
					Elog("invalid number of threads: %d", num_threads);
				break;
			case 1000:	/* pcap-threads */
				num_pcap_threads = strtol(optarg, &pos, 10);
				if (*pos != '\0')
					Elog("invalid --pcap-threads argument: %s", optarg);
				if (num_pcap_threads < 1)
					Elog("invalid number of pcap-threads: %d", num_pcap_threads);
				break;

			case 'l':	/* limit */
				output_filesize_limit = strtol(optarg, &pos, 10);
				if (strcasecmp(pos, "k") == 0 || strcasecmp(pos, "kb") == 0)
					output_filesize_limit <<= 10;
				else if (strcasecmp(pos, "m") == 0 || strcasecmp(pos, "mb") == 0)
					output_filesize_limit <<= 20;
				else if (strcasecmp(pos, "g") == 0 || strcasecmp(pos, "gb") == 0)
					output_filesize_limit <<= 30;
				else if (*pos != '\0')
					Elog("unknown unit size '%s' in -l|--limit option",
						 optarg);
				break;

			case 's':	/* chunk-size */
				record_batch_threshold = strtol(optarg, &pos, 10);
				if (strcasecmp(pos, "k") == 0 || strcasecmp(pos, "kb") == 0)
					record_batch_threshold <<= 10;
				else if (strcasecmp(pos, "m") == 0 || strcasecmp(pos, "mb") == 0)
					record_batch_threshold <<= 20;
				else if (strcasecmp(pos, "g") == 0 || strcasecmp(pos, "gb") == 0)
					record_batch_threshold <<= 30;
				else
					Elog("unknown unit size '%s' in -s|--chunk-size option",
						 optarg);
				break;
			case 'r':	/* rule */
				bpf_filter_rule = pstrdup(optarg);
				break;
			case 1001:	/* --direct-io */
				enable_direct_io = true;
				break;
			case 1002:	/* --only-headers */
				only_headers = true;
				break;
			default:
				usage(code == 'h' ? 0 : 1);
				break;
		}
	}
	if (argc != optind)
		Elog("unexpected tokens in the command line argument");
	if (!input_pathname)
		Elog("no input device or file was specified by -i|--input");
	for (pos = output_filename; *pos != '\0'; pos++)
	{
		if (*pos == '%')
		{
			pos++;
			switch (*pos)
			{
				case 'q':
					output_has_seqno = true;
					break;
				case 'i':
				case 'Y':
				case 'y':
				case 'm':
				case 'd':
				case 'H':
				case 'M':
				case 'S':
					/* ok supported */
					break;
				default:
					Elog("unknown format string '%c' in '%s'",
						 *pos, output_filename);
					break;
			}
		}
	}
	if (output_filesize_limit != 0 && !output_has_seqno)
		Elog("-o|--output must has '%%q' to split files when it exceeds the threshold");
	if (num_threads < 0)
		num_threads = 2 * NCPUS;
	if (num_pcap_threads < 0)
		num_pcap_threads = NCPUS;
}

int main(int argc, char *argv[])
{
	struct stat	stat_buf;
	pthread_t  *workers;
	long		i, rv;

	/* init misc variables */
	PAGESIZE = sysconf(_SC_PAGESIZE);
	NCPUS = sysconf(_SC_NPROCESSORS_ONLN);

	/* parse command line options */
	parse_options(argc, argv);

	/*
	 * If -i|--input is a regular file, we try to open & mmap pcap file,
	 * for parallel transformation to Apache Arrow.
	 * Elsewhere, we assume the input_file is device name to be captured
	 * using PF_RING module.
	 */
	if (stat(input_pathname, &stat_buf) == 0 &&
		S_ISREG(stat_buf.st_mode))
	{
		pcap_filemap = pcap_file_mmap(input_pathname, &stat_buf);
	}
	else
	{
		int		flags = (PF_RING_REENTRANT |
						 PF_RING_TIMESTAMP |
						 PF_RING_PROMISC);

		pd = pfring_open(input_pathname, 65536, flags);
		if (!pd)
			Elog("failed on pfring_open: %m - "
				 "pf_ring not loaded or interface %s is down?",
				 input_pathname);
		rv = pfring_set_application_name(pd, "pcap2arrow");
		if (rv)
			Elog("failed on pfring_set_application_name");

		//NOTE: Is rx_only_direction right?
		rv = pfring_set_direction(pd, rx_only_direction);
		if (rv)
			Elog("failed on pfring_set_direction");

		rv = pfring_set_poll_duration(pd, 50);
		if (rv)
			Elog("failed on pfring_set_poll_duration");
		
		rv = pfring_set_socket_mode(pd, recv_only_mode);
		if (rv)
			Elog("failed on pfring_set_socket_mode");

		if (bpf_filter_rule)
		{
			rv = pfring_set_bpf_filter(pd, bpf_filter_rule);
			if (rv)
				Elog("failed on pfring_set_bpf_filter");
		}

		rv = pfring_enable_ring(pd);
		if (rv)
			Elog("failed on pfring_enable_ring");
	}
	/* open the output file, and misc init stuff */
	arrow_file_desc_current = arrowOpenOutputFile();
	pthreadMutexInit(&arrow_file_desc_lock);
	if (sem_init(&pcap_worker_sem, 0, num_pcap_threads) != 0)
		Elog("failed on sem_init: %m");

	/* ctrl-c handler */
	signal(SIGINT, on_sigint_handler);
	signal(SIGTERM, on_sigint_handler);
	
	/* launch worker threads */
	workers = alloca(sizeof(pthread_t) * num_threads);
	for (i=0; i < num_threads; i++)
	{
		rv = pthread_create(&workers[i], NULL, pcap_worker_main, (void *)i);
		if (rv != 0)
			Elog("failed on pthread_create: %s", strerror(rv));
	}
	/* wait for completion */
	for (i=0; i < num_threads; i++)
	{
		rv = pthread_join(workers[i], NULL);
		if (rv != 0)
			Elog("failed on pthread_join: %s", strerror(rv));
	}

	/* wait for write out footers for any remaining */

	return 0;
}

/*
 * memory allocation handlers
 */
void *
palloc(size_t sz)
{
	void   *ptr = malloc(sz);

	if (!ptr)
		Elog("out of memory");
	return ptr;
}

void *
palloc0(size_t sz)
{
	void   *ptr = malloc(sz);

	if (!ptr)
		Elog("out of memory");
	memset(ptr, 0, sz);
	return ptr;
}

char *
pstrdup(const char *str)
{
	char   *ptr = strdup(str);

	if (!ptr)
		Elog("out of memory");
	return ptr;
}

void *
repalloc(void *old, size_t sz)
{
	char   *ptr = realloc(old, sz);

	if (!ptr)
		Elog("out of memory");
	return ptr;
}
