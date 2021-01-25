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

#define PCAP_PROTO__PACKET			0
#define PCAP_PROTO__RAW_IPv4		1
#define PCAP_PROTO__TCP_IPv4		2
#define PCAP_PROTO__UDP_IPv4		3
#define PCAP_PROTO__ICMP_IPv4		4
#define PCAP_PROTO__RAW_IPv6		5
#define PCAP_PROTO__TCP_IPv6		6
#define PCAP_PROTO__UDP_IPv6		7
#define PCAP_PROTO__ICMP_IPv6		8
#define PCAP_PROTO_NUMS				9

#define PCAP_PROTO_MASK__PACKET			(1U << PCAP_PROTO__PACKET)
#define PCAP_PROTO_MASK__RAW_IPv4		(1U << PCAP_PROTO__RAW_IPv4)
#define PCAP_PROTO_MASK__TCP_IPv4		(1U << PCAP_PROTO__TCP_IPv4)
#define PCAP_PROTO_MASK__UDP_IPv4		(1U << PCAP_PROTO__UDP_IPv4)
#define PCAP_PROTO_MASK__ICMP_IPv4		(1U << PCAP_PROTO__ICMP_IPv4)
#define PCAP_PROTO_MASK__RAW_IPv6		(1U << PCAP_PROTO__RAW_IPv6)
#define PCAP_PROTO_MASK__TCP_IPv6		(1U << PCAP_PROTO__TCP_IPv6)
#define PCAP_PROTO_MASK__UDP_IPv6		(1U << PCAP_PROTO__UDP_IPv6)
#define PCAP_PROTO_MASK__ICMP_IPv6		(1U << PCAP_PROTO__ICMP_IPv6)
#define PCAP_PROTO_MASK__DEFAULT		(PCAP_PROTO_MASK__PACKET |		\
										 PCAP_PROTO_MASK__TCP_IPv4 |	\
										 PCAP_PROTO_MASK__UDP_IPv4 |	\
										 PCAP_PROTO_MASK__ICMP_IPv4)
#define PCAP_PROTO_MAX_NFIELDS		35


#define PCAP_OUTPUT__PER_HOUR		(-1)
#define PCAP_OUTPUT__PER_DAY		(-2)
#define PCAP_OUTPUT__PER_WEEK		(-3)
#define PCAP_OUTPUT__PER_MONTH		(-4)

typedef struct pcapFileBuffer		pcapFileBuffer;
typedef struct pcapChunkBuffer		pcapChunkBuffer;
typedef struct pcapWorkerTask		pcapWorkerTask;

/*
 * pcapFileBuffer
 */
struct pcapFileBuffer
{
	int			proto;		/* one of PCAP_PROTO_INDEX__xxx */
	pthread_mutex_t list_mutex;
	pcapChunkBuffer *free_list;
	pcapChunkBuffer *pending_list;
	pthread_mutex_t table_mutex;
	SQLtable	table;		/* to track RecordBatches */
};

/*
 * pcapChunkBuffer
 */
struct pcapChunkBuffer
{
	pcapFileBuffer *f_buf;
	pcapChunkBuffer *next;	/* for free-list */
	int				nfields;
	size_t			nitems;
	SQLfield		columns[FLEXIBLE_ARRAY_MEMBER];
};

/*
 * pcapWorkerTask
 */
struct pcapWorkerTask
{
	pcapWorkerTask	   *next;	/* for pending-list */
	pcapChunkBuffer	   *chunks[PCAP_PROTO_NUMS];	/* for each protocols */
};

/* command-line options */
static char		   *input_pathname = NULL;
static char		   *output_filename = "/tmp/pcap_%y%m%d_%H:%M:%S_%i_%p.arrow";
static long			duration_to_switch = 0;			/* not switch */
static int			protocol_mask = PCAP_PROTO_MASK__DEFAULT;
static bool			pcapture_raw_ipv4;
static bool			pcapture_tcp_ipv4;
static bool			pcapture_udp_ipv4;
static bool			pcapture_icmp_ipv4;
static bool			pcapture_raw_ipv6;
static bool			pcapture_tcp_ipv6;
static bool			pcapture_udp_ipv6;
static bool			pcapture_icmp_ipv6;
static int			num_threads = -1;
static int			num_pcap_threads = -1;
static size_t		output_filesize_limit = 0UL;				/* No limit */
static size_t		record_batch_threshold = (128UL << 20);		/* 128MB */
static bool			enable_direct_io = false;
static bool			enable_hugetlb = false;

/* static variable for PF-RING capture mode */
static pfring		   *pd = NULL;
static sem_t			pcap_worker_sem;
static pthread_mutex_t	pcap_task_mutex = PTHREAD_MUTEX_INITIALIZER;
static pcapWorkerTask  *pcap_task_list = NULL;
static pcapFileBuffer  *pcap_file_array[PCAP_PROTO_NUMS];

/* static variables for PCAP file scan mode */
#define PCAP_MAGIC__HOST		0xa1b2c3d4U
#define PCAP_MAGIC__SWAP		0xd4c3b2a1U
#define PCAP_MAGIC__HOST_NS		0xa1b23c4dU
#define PCAP_MAGIC__SWAP_NS		0x4d3cb2a1U

static char		   *pcap_filemap = NULL;
static uint32_t		pcap_file_magic = 0;
static size_t		pcap_file_sz = 0UL;
static uint64_t		pcap_file_read_pos = 0;

/* other static variables */
static long			PAGESIZE;
static long			NCPUS;
static bool			do_shutdown = false;
static __thread int	worker_id = -1;

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define __ntoh16(x)			__builtin_bswap16(x)
#define __ntoh32(x)			__builtin_bswap32(x)
#define __ntoh64(x)			__builtin_bswap64(x)
#else
#define __ntoh16(x)			(x)
#define __ntoh32(x)			(x)
#define __ntoh64(x)			(x)
#endif

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
	int		flags = MAP_PRIVATE;
	struct pcap_file_header *pcap_head;

	fdesc = open(pathname, O_RDONLY);
	if (fdesc < 0)
		Elog("failed to open file '%s': %m", pathname);
	if (enable_hugetlb)
		flags |= MAP_HUGETLB;
	pcap_head = mmap(NULL, stat_buf->st_size, PROT_READ, flags, fdesc, 0);
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

/*
 * atomic operations
 */
static inline uint32_t
atomicAdd32(uint32_t *addr, uint32_t value)
{
	return __atomic_fetch_add(addr, value, __ATOMIC_SEQ_CST);
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
__put_uint_value_common(SQLfield *column, const char *addr, int sz, int width)
{
	size_t		index = column->nitems++;

	if (!addr)
		__put_inline_null_value(column, index, width);
	else
	{
		assert(sz == width);
		sql_buffer_setbit(&column->nullmap, index);
		sql_buffer_append(&column->values, addr, sz);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_uint8_value(SQLfield *column, const char *addr, int sz)
{
	return __put_uint_value_common(column, addr, sz, sizeof(uint8_t));
}

static size_t
put_uint16_value(SQLfield *column, const char *addr, int sz)
{
	return __put_uint_value_common(column, addr, sz, sizeof(uint16_t));
}

#if 0
static size_t
put_uint32_value(SQLfield *column, const char *addr, int sz)
{
	return __put_uint_value_common(column, addr, sz, sizeof(uint32_t));
}

static size_t
put_uint64_value(SQLfield *column, const char *addr, int sz)
{
	return __put_uint_value_common(column, addr, sz, sizeof(uint64_t));
}
#endif

static size_t
put_timestamp_us_value(SQLfield *column, const char *addr, int sz)
{
	size_t		index = column->nitems++;
	uint64_t	value;

	if (!addr)
		__put_inline_null_value(column, index, sizeof(uint64_t));
	else
	{
		assert(sz == sizeof(struct timeval));
		value = (((struct timeval *)addr)->tv_sec * 1000000L +
				 ((struct timeval *)addr)->tv_usec);
		sql_buffer_setbit(&column->nullmap, index);
		sql_buffer_append(&column->values, &value, sizeof(uint64_t));
	}
    return __buffer_usage_inline_type(column);
}

static size_t
put_fixed_size_binary_value(SQLfield *column, const char *addr, int sz)
{
	size_t		index = column->nitems++;
	int			byteWidth = column->arrow_type.FixedSizeBinary.byteWidth;

	if (!addr)
		__put_inline_null_value(column, index, byteWidth);
	else
	{
		if (sz > byteWidth)
			sz = byteWidth;
		sql_buffer_setbit(&column->nullmap, index);
		sql_buffer_append(&column->values, addr, sz);
		if (sz < byteWidth)
			sql_buffer_append_zero(&column->values, byteWidth - sz);
	}
	return __buffer_usage_inline_type(column);
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
		assert(column->customMetadata == NULL);
		column->customMetadata = palloc(sizeof(ArrowKeyValue));
	}
	else
	{
		size_t	sz = sizeof(ArrowKeyValue) * (column->numCustomMetadata + 1);

		assert(column->customMetadata != NULL);
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
arrowFieldInitAsUint32(SQLfield *column, const char *field_name)
{
	memset(column, 0, sizeof(SQLfield));
	initArrowNode(&column->arrow_type, Int);
	column->arrow_type.Int.bitWidth = 16;
	column->arrow_type.Int.is_signed = false;
	column->put_value = put_uint16_value;
	column->field_name = pstrdup(field_name);
}

#if 0
static void
arrowFieldInitAsUint64(SQLfield *column, const char *field_name)
{
	memset(column, 0, sizeof(SQLfield));
	initArrowNode(&column->arrow_type, Int);
	column->arrow_type.Int.bitWidth = 16;
	column->arrow_type.Int.is_signed = false;
	column->put_value = put_uint16_value;
	column->field_name = pstrdup(field_name);
}
#endif

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
	column->arrow_type.FixedSizeBinary.byteWidth = 6;
	column->put_value = put_fixed_size_binary_value;
	column->field_name = pstrdup(field_name);
	arrowFieldAddCustomMetadata(column, "pg_type", "pg_catalog.macaddr");
}

static void
arrowFieldInitAsIP4Addr(SQLfield *column, const char *field_name)
{
	memset(column, 0, sizeof(SQLfield));
	initArrowNode(&column->arrow_type, FixedSizeBinary);
	column->arrow_type.FixedSizeBinary.byteWidth = 4;
	column->put_value = put_fixed_size_binary_value;
	column->field_name = pstrdup(field_name);
	arrowFieldAddCustomMetadata(column, "pg_type", "pg_catalog.inet");
}

static void
arrowFieldInitAsIP6Addr(SQLfield *column, const char *field_name)
{
	memset(column, 0, sizeof(SQLfield));
	initArrowNode(&column->arrow_type, FixedSizeBinary);
	column->arrow_type.FixedSizeBinary.byteWidth = 16;
	column->put_value = put_fixed_size_binary_value;
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







#define __ARROW_FIELD_INIT(__index, __name, __type)		\
	arrowFieldInitAs##__type(&columns[(__index)], (__name))

static int
arrowSchemaInitForRawPacket(SQLfield *columns)
{
	int		j = 0;

	__ARROW_FIELD_INIT(j++, "timestamp",    TimestampUs);
	__ARROW_FIELD_INIT(j++, "dst_mac",      MacAddr);
	__ARROW_FIELD_INIT(j++, "src_mac",      MacAddr);
	__ARROW_FIELD_INIT(j++, "ether_type",   Uint16);
	__ARROW_FIELD_INIT(j++, "payload",      Binary);

	return j;
}

static int
arrowSchemaInitForRawIPv4(SQLfield *columns)
{
	int		j = arrowSchemaInitForRawPacket(columns) - 1;

	__ARROW_FIELD_INIT(j++, "tos",          Uint8);
	__ARROW_FIELD_INIT(j++, "length",       Uint16);
	__ARROW_FIELD_INIT(j++, "identifier",   Uint16);
	__ARROW_FIELD_INIT(j++, "fragment",     Uint16);
	__ARROW_FIELD_INIT(j++, "ttl",          Uint8);
	__ARROW_FIELD_INIT(j++, "protocol",     Uint8);
	__ARROW_FIELD_INIT(j++, "ip_checksum",  Uint16);
	__ARROW_FIELD_INIT(j++, "src_addr",     IP4Addr);
	__ARROW_FIELD_INIT(j++, "dst_addr",     IP4Addr);
	__ARROW_FIELD_INIT(j++, "ip_options",   Binary);
	__ARROW_FIELD_INIT(j++, "payload",      Binary);

	return j;
}

static int
arrowSchemaInitForTcpIPv4(SQLfield *columns)
{
	int		j = arrowSchemaInitForRawIPv4(columns) - 1;

	__ARROW_FIELD_INIT(j++, "src_port",     Uint16);
	__ARROW_FIELD_INIT(j++, "dst_port",     Uint16);
	__ARROW_FIELD_INIT(j++, "seq_nr",       Uint32);
	__ARROW_FIELD_INIT(j++, "ack_nr",       Uint32);
	__ARROW_FIELD_INIT(j++, "tcp_flags",    Uint8);
	__ARROW_FIELD_INIT(j++, "window_sz",    Uint16);
	__ARROW_FIELD_INIT(j++, "tcp_checksum", Uint16);
	__ARROW_FIELD_INIT(j++, "urgent_ptr",   Uint16);
	__ARROW_FIELD_INIT(j++, "tcp_options",  Binary);
	__ARROW_FIELD_INIT(j++, "payload",      Binary);

	return j;
}

static int
arrowSchemaInitForUdpIPv4(SQLfield *columns)
{
	int		j = arrowSchemaInitForRawIPv4(columns) - 1;

	__ARROW_FIELD_INIT(j++, "src_port",     Uint16);
	__ARROW_FIELD_INIT(j++, "dst_port",     Uint16);
	__ARROW_FIELD_INIT(j++, "segment_sz",   Uint16);
	__ARROW_FIELD_INIT(j++, "udp_checksum", Uint16);
	__ARROW_FIELD_INIT(j++, "payload",      Binary);

	return j;
}

static int
arrowSchemaInitForIcmpIPv4(SQLfield *columns)
{
	int		j = arrowSchemaInitForRawIPv4(columns) - 1;

	__ARROW_FIELD_INIT(j++, "icmp_type",    Uint8);
	__ARROW_FIELD_INIT(j++, "icmp_code",    Uint8);
	__ARROW_FIELD_INIT(j++, "icmp_checksum",Uint8);
	__ARROW_FIELD_INIT(j++, "payload",      Binary);

	return j;
}

static int
arrowSchemaInitForRawIPv6(SQLfield *columns)
{
	int		j = arrowSchemaInitForRawPacket(columns) - 1;
	__ARROW_FIELD_INIT(j++, "traffic_class", Uint8);
//	__ARROW_FIELD_INIT(j++, "flow_label",    Uint32);
	__ARROW_FIELD_INIT(j++, "hop_limit",     Uint8);
	__ARROW_FIELD_INIT(j++, "src_addr",      IP6Addr);
	__ARROW_FIELD_INIT(j++, "dst_addr",      IP6Addr);

	__ARROW_FIELD_INIT(j++, "hop_by_hop",    Binary);
//	__ARROW_FIELD_INIT(j++, "routing",       IPv6Ext_Routing);	

	Elog("arrowSchemaInitForRawIPv6 not implemented yet");
	return j;
}

static int
arrowSchemaInitForTcpIPv6(SQLfield *columns)
{
	int		j = arrowSchemaInitForRawIPv4(columns) - 1;

	__ARROW_FIELD_INIT(j++, "src_port",     Uint16);
	__ARROW_FIELD_INIT(j++, "dst_port",     Uint16);
	__ARROW_FIELD_INIT(j++, "seq_nr",       Uint32);
	__ARROW_FIELD_INIT(j++, "ack_nr",       Uint32);
	__ARROW_FIELD_INIT(j++, "tcp_flags",    Uint8);
	__ARROW_FIELD_INIT(j++, "window_sz",    Uint16);
	__ARROW_FIELD_INIT(j++, "tcp_checksum", Uint16);
	__ARROW_FIELD_INIT(j++, "urgent_ptr",   Uint16);
	__ARROW_FIELD_INIT(j++, "tcp_options",  Binary);
	__ARROW_FIELD_INIT(j++, "payload",      Binary);

	return j;
}

static int
arrowSchemaInitForUdpIPv6(SQLfield *columns)
{
	int		j = arrowSchemaInitForRawIPv6(columns) - 1;

	__ARROW_FIELD_INIT(j++, "src_port",     Uint16);
	__ARROW_FIELD_INIT(j++, "dst_port",     Uint16);
	__ARROW_FIELD_INIT(j++, "segment_sz",   Uint16);
	__ARROW_FIELD_INIT(j++, "udp_checksum", Uint16);
	__ARROW_FIELD_INIT(j++, "payload",      Binary);

	return j;
}

static int
arrowSchemaInitForIcmpIPv6(SQLfield *columns)
{
	int		j = arrowSchemaInitForRawIPv4(columns) - 1;

	__ARROW_FIELD_INIT(j++, "icmp_type",    Uint8);
	__ARROW_FIELD_INIT(j++, "icmp_code",    Uint8);
	__ARROW_FIELD_INIT(j++, "icmp_checksum",Uint8);
	__ARROW_FIELD_INIT(j++, "payload",      Binary);

	return j;
}

static const struct {
	int			proto;
	const char *proto_name;
	int		  (*proto_schema)(SQLfield *columns);
} pcap_protocol_catalog[] = {
	{ PCAP_PROTO__PACKET,   "packet", arrowSchemaInitForRawPacket },
	{ PCAP_PROTO__RAW_IPv4,  "ipv4",  arrowSchemaInitForRawIPv4 },
	{ PCAP_PROTO__TCP_IPv4,  "tcp4",  arrowSchemaInitForTcpIPv4 },
	{ PCAP_PROTO__UDP_IPv4,  "udp4",  arrowSchemaInitForUdpIPv4 },
	{ PCAP_PROTO__ICMP_IPv4, "icmp4", arrowSchemaInitForIcmpIPv4 },
	{ PCAP_PROTO__RAW_IPv6,  "ipv6",  arrowSchemaInitForRawIPv6 },
	{ PCAP_PROTO__TCP_IPv6,  "tcp6",  arrowSchemaInitForTcpIPv6 },
	{ PCAP_PROTO__UDP_IPv6,  "udp6",  arrowSchemaInitForUdpIPv6 },
	{ PCAP_PROTO__ICMP_IPv6, "icmp6", arrowSchemaInitForIcmpIPv6 },
	{ 0, NULL, NULL },
};

#undef __ARROW_FIELD_INIT

/*
 * handlePacketRawEthernet
 */
static void
handlePacketRawEthernet(pcapWorkerTask *pw_task,
						struct pfring_pkthdr *hdr, u_char *buffer)
{
}

/*
 * handlePacketRawIPv4
 */
static bool
handlePacketRawIPv4(pcapWorkerTask *pw_task,
					struct pfring_pkthdr *hdr, u_char *buffer)
{
	return true;
}

/*
 * handlePacketTcpIPv4
 */
static bool
handlePacketTcpIPv4(pcapWorkerTask *pw_task,
					struct pfring_pkthdr *hdr, u_char *buffer)
{
	return true;
}

/*
 * handlePacketUdpIPv4
 */
static bool
handlePacketUdpIPv4(pcapWorkerTask *pw_task,
                    struct pfring_pkthdr *hdr, u_char *buffer)
{
	return true;
}

/*
 * handlePacketIcmpIPv4
 */
static bool
handlePacketIcmpIPv4(pcapWorkerTask *pw_task,
					 struct pfring_pkthdr *hdr, u_char *buffer)
{
	return true;
}

/*
 * pcapCapturePackets
 */
static int
pcapCapturePackets(pcapWorkerTask *pw_task)
{
	struct pfring_pkthdr hdr;
	u_char		__buffer[5000];
	u_char	   *buffer = __buffer;
	int			rv;

	while (!do_shutdown)
	{
		rv = pfring_recv(pd, &buffer, sizeof(__buffer), &hdr, 1);
		if (rv > 0)
		{
			uint16_t	ether_type = __ntoh16(*((uint16_t *)(buffer + 12)));
			uint8_t		ip_version = (buffer[14] & 0xf0);

			printf("ether_type = %04x ip_version = %02x\n", ether_type, ip_version);
			if (ether_type == 0x0800 && ip_version == 0x40)
			{
				uint8_t		proto = buffer[14 + 9];		/* IPv4 Protocol */

				if (proto == 0x06 && pcapture_tcp_ipv4 &&
					hdr.caplen >= 14 + 20 + 20)			/* TCP/IPv4 */
				{
					if (handlePacketTcpIPv4(pw_task, &hdr, buffer))
						continue;
				}
				else if (proto == 0x11 && pcapture_udp_ipv4 &&
						 hdr.caplen >= 14 + 20 + 8)		/* UDP/IPv4 */
				{
					if (handlePacketUdpIPv4(pw_task, &hdr, buffer))
						continue;
				}
				else if (proto == 0x01 && pcapture_icmp_ipv4 &&
						 hdr.caplen >= 14 + 20 + 8)		/* ICMP/IPv4 */
				{
					if (handlePacketIcmpIPv4(pw_task, &hdr, buffer))
						continue;
				}
				/* not TCP/UDP/ICMP, so save as Raw IPv4 */
				if (handlePacketRawIPv4(pw_task, &hdr, buffer))
					continue;
			}
#if 0
			else if (ether_type == 0x86dd && ip_version == 0x60)	/* IPv6 */
			{
				//do IPv6 handling
			}
#endif
			if (hdr.caplen < 14)
			{
				fprintf(stderr, "worker-%d captured too short packet (sz=%u), ignored\n",
						worker_id, hdr.caplen);
			}
			/* elsewhere, all we can capture is hardware ether net packets */
			handlePacketRawEthernet(pw_task, &hdr, buffer);
		}
	}
	//flush pending packets
	return -1;
}

/*
 * pcapAllocChunkBuffer
 */
static pcapChunkBuffer *
pcapAllocChunkBuffer(int proto)
{
	pcapFileBuffer	*f_buf;
	pcapChunkBuffer	*chunk;

	assert(proto >= 0 && proto < PCAP_PROTO_NUMS);
	f_buf = pcap_file_array[proto];
	assert(f_buf != NULL);

	if (pthread_mutex_lock(&f_buf->list_mutex) != 0)
		Elog("failed on pthread_mutex_lock: %m");
	chunk = f_buf->free_list;
	if (chunk)
	{
		f_buf->free_list = chunk->next;
		chunk->next = NULL;
	}
	if (pthread_mutex_unlock(&f_buf->list_mutex) != 0)
		Elog("failed on pthread_mutex_unlock: %m");
	if (!chunk)
	{
		chunk = palloc0(offsetof(pcapChunkBuffer,
								 columns[PCAP_PROTO_MAX_NFIELDS]));
		chunk->f_buf = f_buf;
		chunk->next = NULL;
		chunk->nitems = 0;
		chunk->nfields =
			pcap_protocol_catalog[proto].proto_schema(chunk->columns);
	}
	return chunk;
}

/*
 * pcap_worker_main
 */
static void *
pcap_worker_main(void *__arg)
{
	worker_id = (long)__arg;

	while (!do_shutdown)
	{
		pcapWorkerTask *pw_task;
		pcapChunkBuffer *chunk = NULL;
		int			j, proto;

		if (sem_wait(&pcap_worker_sem) != 0)
		{
			if (errno == EINTR)
				continue;
			Elog("worker-%d: failed on sem_wait: %m", worker_id);
			break;
		}
		fprintf(stderr, "worker-%d: entered to pcap_worker_sem\n", worker_id);

		if (pthread_mutex_lock(&pcap_task_mutex) != 0)
			Elog("worker-%d: failed on pthread_mutex_lock: %m", worker_id);
		pw_task = pcap_task_list;
		if (pw_task)
		{
			pcap_task_list = pw_task->next;
			pw_task->next = NULL;
		}
		if (pthread_mutex_unlock(&pcap_task_mutex) != 0)
			Elog("worker-%d: failed on pthread_mutex_unlock: %m", worker_id);

		/* software may be going to exit, check do_shutdown */
		if (!pw_task)
		{
			if (sem_post(&pcap_worker_sem) != 0)
				Elog("failed on sem_post: %m");
			continue;
		}

		/* allocation of pcapChunkBuffer */
		for (proto=0; proto < PCAP_PROTO_NUMS; proto++)
		{
			if (!pw_task->chunks[proto] && (protocol_mask & (1U << proto)) != 0)
			{
				pw_task->chunks[proto] = pcapAllocChunkBuffer(proto);
			}
		}
		proto = pcapCapturePackets(pw_task);
		if (proto >= 0)
		{
			/*
			 * pcapChunkBuffer exceeds the threshold to write,
			 * so go to file i/o and release pcapWorkerTask from
			 * this thread.
			 */
			chunk = pw_task->chunks[proto];
			assert(chunk != NULL);
			pw_task->chunks[proto] = NULL;

			if (pthread_mutex_lock(&pcap_task_mutex) != 0)
				Elog("worker-%d: failed on pthread_mutex_lock: %m", worker_id);
			pw_task->next = pcap_task_list;
			pcap_task_list = pw_task;
			if (pthread_mutex_unlock(&pcap_task_mutex) != 0)
				Elog("worker-%d: failed on pthread_mutex_unlock: %m", worker_id);
		}
		if (sem_post(&pcap_worker_sem) != 0)
			Elog("failed on sem_post: %m");
		fprintf(stderr, "worker-%d: exit from pcap_worker_sem\n", worker_id);

		if (chunk)
		{
			pcapFileBuffer *f_buf = chunk->f_buf;

			
			//goto file i/o

			/* clear the buffer for reuse */
			for (j=0; j < chunk->nfields; j++)
				sql_field_clear(&chunk->columns[j]);

			/* attach to the free-list of FileBuffer */
			if (pthread_mutex_lock(&f_buf->list_mutex) != 0)
				Elog("failed on pthread_mutex_lock: %m");
			chunk->next = f_buf->free_list;
			f_buf->free_list = chunk;
			if (pthread_mutex_lock(&f_buf->list_mutex) != 0)
				Elog("failed on pthread_mutex_unlock: %m");
		}
	}
	fprintf(stderr, "worker-%d: exit\n", worker_id);
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
		  "        %p : protocol specified by -p\n"
		  "        %q : sequence number if file is switched by -l|--limit\n"
		  "      default is '/tmp/pcap_%y%m%d_%H:%M:%S_%i_%i.arrow'\n"
		  "  -p|--protocol=<PROTO>\n"
		  "        <PROTO> is a comma separated string contains\n"
		  "        the following tokens:\n"
		  "          tcp4, udp4, icmp4, ipv4, tcp6, udp6, icmp6, ipv66\n"
		  "          (*) any packets other than above protocols are\n"
		  "              categorized to 'raw'\n"
		  "        default protocol selection is:\n"
		  "          'tcp4,udp4,icmp4,tcp6,udp6,icmp6'\n"
		  "  -t|--threads=<NUM of threads> (default: 2 * NCPUs)\n"
		  "     --pcap-threads=<NUM of threads> (default: NCPUS)\n"
		  "  -d|--duration=<DURATION> : duration until output switch\n"
		  "      <DURATION> is one of the below:\n"
		  "        <NUM>s : switch per NUM seconds (e.g: 30s)\n"
		  "        <NUM>m : switch per NUM minutes (e.g: 5m)\n"
		  "        <NUM>h : switch per NUM hours (e.g: 2h)\n"
		  "        hour   : switch at the next hour\n"
		  "        day    : switch at the next day\n"
		  "        week   : switch at the next week\n"
		  "        months : switch at the next month\n"
		  "  -l|--limit=<LIMIT> : (default: no limit)\n"
		  "  -s|--chunk-size=<SIZE> : size of record batch (default: 128MB)\n"
#if 0
		  "  -r|--rule=<RULE> : packet filtering rules\n"
#endif
		  "     --direct-io : enables O_DIRECT for write-i/o\n"
		  "     --hugetlb : try to use hugepages for buffers\n"
		  "  -h|--help    : shows this message"
		  "\n"
		  "  Copyright 2020-2021 - HeteroDB,Inc\n",
		  stderr);
	exit(status);
}

static void
parse_options(int argc, char *argv[])
{
	static struct option long_options[] = {
		{"input",      required_argument, NULL, 'i'},
		{"output",     required_argument, NULL, 'o'},
		{"protocol",   required_argument, NULL, 'p'},
		{"threads",    required_argument, NULL, 't'},
		{"duration",   required_argument, NULL, 'd'},
		{"limit",      required_argument, NULL, 'l'},
		{"chunk-size", required_argument, NULL, 's'},
#if 0
		{"rule",       required_argument, NULL, 'r'},
#endif
		{"pcap-threads", required_argument, NULL, 1000},
		{"direct-io",  no_argument,       NULL, 1001},
		{"hugetlb",    no_argument,       NULL, 1002},
		{"help",       no_argument,       NULL, 'h'},
		{NULL, 0, NULL, 0}
	};
	int		code;
	bool	output_has_proto = false;
	bool	output_has_seqno = false;
	char   *pos;

	while ((code = getopt_long(argc, argv, "i:o:p:t:d:l:s:",
							   long_options, NULL)) >= 0)
	{
		char	   *token, *end;
		int			i, __mask;

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
				__mask = PCAP_PROTO_MASK__PACKET;
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
					for (i=1; pcap_protocol_catalog[i].proto_name != NULL; i++)
					{
						if (strcmp(pcap_protocol_catalog[i].proto_name, token) == 0)
						{
							__mask |= (1U << pcap_protocol_catalog[i].proto);
							break;
						}
					}
					if (i >= PCAP_PROTO_NUMS)
						Elog("Unknown protocol [%s]", token);
				}
				protocol_mask = __mask;
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

			case 'd':	/* duration */
				if (strcmp(optarg, "hour") == 0)
					duration_to_switch = PCAP_OUTPUT__PER_HOUR;
				else if (strcmp(optarg, "day") == 0)
					duration_to_switch = PCAP_OUTPUT__PER_DAY;
				else if (strcmp(optarg, "week") == 0)
					duration_to_switch = PCAP_OUTPUT__PER_WEEK;
				else if (strcmp(optarg, "month") == 0)
					duration_to_switch = PCAP_OUTPUT__PER_MONTH;
				else
				{
					duration_to_switch = strtol(optarg, &pos, 10);
					if (strcasecmp(pos, "m") == 0)
						duration_to_switch *= 60;
					else if (strcasecmp(pos, "h") == 0)
						duration_to_switch *= 3600;
					else if (strcasecmp(pos, "d") == 0)
						duration_to_switch *= 86400;
					else
						Elog("unknown unit size '%s' in -d|--duration option",
							 optarg);
				}
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
#if 0
			case 'r':	/* rule */
				/* TODO, in the future version */
				break;
#endif
			case 1001:	/* --direct-io */
				enable_direct_io = true;
				break;
			case 1002:	/* --hugetlb */
				enable_hugetlb = true;
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
				case 'p':
					output_has_proto = true;
					break;
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
	if (protocol_mask != PCAP_PROTO_MASK__PACKET && !output_has_proto)
		Elog("-o|--output must has '%%p' to distribute packet based on protocols");
	if (output_filesize_limit != 0 && !output_has_seqno)
		Elog("-o|--output must has '%%q' to split files when it exceeds the threshold");
	if (num_threads < 0)
		num_threads = 2 * NCPUS;
	if (num_pcap_threads < 0)
		num_pcap_threads = NCPUS;

	/* just for convenience */
	pcapture_raw_ipv4 = ((protocol_mask & PCAP_PROTO_MASK__RAW_IPv4) != 0);
	pcapture_tcp_ipv4 = ((protocol_mask & PCAP_PROTO_MASK__TCP_IPv4) != 0);
	pcapture_udp_ipv4 = ((protocol_mask & PCAP_PROTO_MASK__UDP_IPv4) != 0);
	pcapture_icmp_ipv4 = ((protocol_mask & PCAP_PROTO_MASK__ICMP_IPv4) != 0);
	pcapture_raw_ipv6 = ((protocol_mask & PCAP_PROTO_MASK__RAW_IPv4) != 0);
	pcapture_tcp_ipv6 = ((protocol_mask & PCAP_PROTO_MASK__TCP_IPv4) != 0);
	pcapture_udp_ipv6 = ((protocol_mask & PCAP_PROTO_MASK__UDP_IPv4) != 0);
	pcapture_icmp_ipv6 = ((protocol_mask & PCAP_PROTO_MASK__ICMP_IPv4) != 0);
}

/*
 * open_output_files
 */
static char *
build_output_pathname(int proto, struct tm *tm)
{
	char	   *path, *pos;
	int			off, sz = 256;

	assert(proto >= 0 && proto < PCAP_PROTO_NUMS);
	/* construct output path name */
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
						off += snprintf(path+off, sz-off,
										"%s", input_pathname);
						break;
					case 'Y':
						off += snprintf(path+off, sz-off,
										"%04d", tm->tm_year + 1900);
						break;
					case 'y':
						off += snprintf(path+off, sz-off,
										"%02d", tm->tm_year % 100);
						break;
					case 'm':
						off += snprintf(path+off, sz-off,
										"%02d", tm->tm_mon + 1);
						break;
					case 'd':
						off += snprintf(path+off, sz-off,
										"%02d", tm->tm_mday);
						break;
					case 'H':
						off += snprintf(path+off, sz-off,
										"%02d", tm->tm_hour);
						break;
					case 'M':
						off += snprintf(path+off, sz-off,
										"%02d", tm->tm_min);
						break;
					case 'S':
						off += snprintf(path+off, sz-off,
										"%02d", tm->tm_sec);
						break;
					case 'p':
						off += snprintf(path+off, sz-off,
										"%s", pcap_protocol_catalog[proto].proto_name);
						break;
					case 'q':
						//currently no seqno support
						off += snprintf(path+off, sz-off, "0");
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
		path[off++] = '\0';
	} while (*pos != '\0');

	return pstrdup(path);
}

static void
pcap2arrow_open_files(void)
{
	time_t		t = time(NULL);
	struct tm	tm;
	int			i, flags = O_RDWR | O_CREAT | O_EXCL;

	localtime_r(&t, &tm);
	if (enable_direct_io)
		flags |= O_DIRECT;

	assert((protocol_mask & PCAP_PROTO__PACKET) != 0);
	for (i=0; pcap_protocol_catalog[i].proto_name != NULL; i++)
	{
		int		proto = pcap_protocol_catalog[i].proto;
		char   *filename;
		int		fdesc;
		pcapFileBuffer *f_buf;

		if ((protocol_mask & (1U << proto)) == 0)
			continue;
		
		/* open file */
		filename = build_output_pathname(proto, &tm);
		fdesc = open(filename, flags, 0644);
		if (fdesc < 0)
			Elog("failed on open('%s'): %m", filename);

		/* allocation */
		f_buf = palloc0(offsetof(pcapFileBuffer,
								 table.columns[PCAP_PROTO_MAX_NFIELDS]));
		f_buf->proto = proto;
		if (pthread_mutex_init(&f_buf->list_mutex, NULL) != 0 ||
			pthread_mutex_init(&f_buf->table_mutex, NULL) != 0)
			Elog("failed on pthread_mutex_init");
		f_buf->table.filename = filename;
		f_buf->table.fdesc = fdesc;
		f_buf->table.f_pos = 0;
		f_buf->table.nfields =
			pcap_protocol_catalog[i].proto_schema(f_buf->table.columns);

		/* write header portion */
		arrowFileWrite(&f_buf->table, "ARROW1\0\0", 8);
		writeArrowSchema(&f_buf->table);

		pcap_file_array[i] = f_buf;
	}
}

int main(int argc, char *argv[])
{
	struct stat	stat_buf;
	pthread_t  *workers;
	long		i;
	int			rv;

	/* init misc variables */
	PAGESIZE = sysconf(_SC_PAGESIZE);
	NCPUS = sysconf(_SC_NPROCESSORS_ONLN);

	/* parse command line options */
	parse_options(argc, argv);

	/* open output files and allocate buffers */
	pcap2arrow_open_files();

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

		rv = pfring_enable_ring(pd);
		if (rv)
			Elog("failed on pfring_enable_ring");
	}
	/* init other stuff */
	if (sem_init(&pcap_worker_sem, 0, num_pcap_threads) != 0)
		Elog("failed on sem_init: %m");
	for (i=0; i < num_pcap_threads; i++)
	{
		pcapWorkerTask *pw_task = palloc0(sizeof(pcapWorkerTask));

		pw_task->next = pcap_task_list;
		pcap_task_list = pw_task;
	}

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

	//write out arrow footer

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
