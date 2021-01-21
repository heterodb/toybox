/*
 * pcap2arrow.c
 *
 * multi-thread ultra fast packet capture and translator to Apache Arrow.
 *
 * Portions Copyright (c) 2021, HeteroDB Inc
 */
#include <ctype.h>
#include <getopt.h>
#include <pcap.h>
#include <pfring.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include "arrow_ipc.h"









#define PCAP_PROTO__TCPv4		0x0001U
#define PCAP_PROTO__UDPv4		0x0002U
#define PCAP_PROTO__ICMPv4		0x0004U
#define PCAP_PROTO__IPv4		0x0008U
#define PCAP_PROTO__TCPv6		0x0100U
#define PCAP_PROTO__UDPv6		0x0200U
#define PCAP_PROTO__ICMPv6		0x0400U
#define PCAP_PROTO__IPv6		0x0800U
#define PCAP_PROTO__DEFAULT		(PCAP_PROTO__TCPv4 | \
								 PCAP_PROTO__UDPv4 | \
								 PCAP_PROTO__ICMPv4| \
								 PCAP_PROTO__TCPv6 | \
								 PCAP_PROTO__UDPv6 | \
								 PCAP_PROTO__ICMPv6)
#define PCAP_OUTPUT__PER_HOUR		(-1)
#define PCAP_OUTPUT__PER_DAY		(-2)
#define PCAP_OUTPUT__PER_WEEK		(-3)
#define PCAP_OUTPUT__PER_MONTH		(-4)

/* command-line options */
static char		   *input_pathname = NULL;
static char		   *output_filename = NULL;
static long			duration_to_switch = 0;			/* not switch */
static int			protocol_mask = PCAP_PROTO__DEFAULT;
static int			num_threads = -1;
static int			num_pcap_threads = -1;
static size_t		output_filesize_limit = 0UL;				/* No limit */
static size_t		record_batch_threshold = (128UL << 20);		/* 128MB */
static bool			enable_direct_io = false;
static bool			enable_hugetlb = false;

/* static variable for PF-RING capture mode */
static pfring	   *pd = NULL;

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






/*
 * pcap_worker_main
 */
static void *
pcap_worker_main(void *__arg)
{
	uint32_t	worker_id = (long)__arg;
	struct pfring_pkthdr hdr;
	u_char		__buffer[5000];
	u_char	   *buffer = __buffer;
	int			rv;

	while (!do_shutdown)
	{
		rv = pfring_recv(pd, &buffer, sizeof(__buffer), &hdr, 1);
		if (rv > 0)
		{
			struct tm	__tm;
			uint32_t	v0, v1, v2, v3, v4, v5, v6, v7;

			localtime_r(&hdr.ts.tv_sec, &__tm);

			v0 = ((uint32_t *)buffer)[0];
			v1 = ((uint32_t *)buffer)[1];
			v2 = ((uint32_t *)buffer)[2];
			v3 = ((uint32_t *)buffer)[3];
			v4 = ((uint32_t *)buffer)[4];
			v5 = ((uint32_t *)buffer)[5];
			v6 = ((uint32_t *)buffer)[6];
			v7 = ((uint32_t *)buffer)[7];

			printf("worker-%d: [%02d:%02d:%02d] packet cap=%u len=%u [%08x,%08x,%08x,%08x %08x,%08x,%08x,%08x]\n",
				   worker_id,
				   __tm.tm_hour,
				   __tm.tm_min,
				   __tm.tm_sec,
				   hdr.caplen,
				   hdr.len,
				   v0,v1,v2,v3,v4,v5,v6,v7);
		}
	}
	//flush pending packets
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
		  "        %Y : year in 4-digits\n"
		  "        %y : year in 2-digits\n"
		  "        %m : month in 2-digits\n"
		  "        %d : day in 2-digits\n"
		  "        %H : hour in 2-digits\n"
		  "        %M : minute in 2-digits\n"
		  "        %S : second in 2-digits\n"
		  "        %p : protocol specified by -p\n"
		  "        %q : sequence number if file is switched by -l|--limit\n"
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

	while ((code = getopt_long(argc, argv, "i:o:p:t:d:l:s:",
							   long_options, NULL)) >= 0)
	{
		char	   *token, *pos, *end;
		int			__mask;

		switch (code)
		{
			case 'i':	/* input */
				if (input_pathname)
					Elog("-i|--input was specified twice");
				input_pathname = optarg;
				break;
			case 'o':	/* output */
				if (output_filename)
					Elog("-o|--output was specified twice");
				/* check format string */
				for (pos = optarg; *pos != '\0'; pos++)
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
									 *pos, optarg);
								break;
						}
					}
				}
				output_filename = optarg;
				break;
			case 'p':	/* protocol */
				__mask = 0;
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
					if (strcmp(token, "tcp4") == 0)
						__mask |= PCAP_PROTO__TCPv4;
					else if (strcmp(token, "udp4") == 0)
						__mask |= PCAP_PROTO__UDPv4;
					else if (strcmp(token, "icmp4") == 0)
						__mask |= PCAP_PROTO__ICMPv4;
					else if (strcmp(token, "ipv4") == 0)
						__mask |= PCAP_PROTO__IPv4;
					else if (strcmp(token, "tcp6") == 0)
						__mask |= PCAP_PROTO__TCPv6;
					else if (strcmp(token, "udp6") == 0)
						__mask |= PCAP_PROTO__UDPv6;
					else if (strcmp(token, "icmp6") == 0)
						__mask |= PCAP_PROTO__ICMPv6;
					else if (strcmp(token, "ipv6") == 0)
						__mask |= PCAP_PROTO__IPv6;
					else
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
	if (!output_filename)
		Elog("no output file name was specified by -o|--output");
	if (protocol_mask != 0 && !output_has_proto)
		Elog("-o|--output must has '%%p' to distribute packet based on protocols");
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
	long		i;
	int			rv;

	/* init misc variables */
	PAGESIZE = sysconf(_SC_PAGESIZE);
	NCPUS = sysconf(_SC_NPROCESSORS_ONLN);

	/* parse command line options */
	parse_options(argc, argv);

	/* setup arrow write buffer */


	
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

		rv = pfring_set_socket_mode(pd, recv_only_mode);
		if (rv)
			Elog("failed on pfring_set_socket_mode");

//		rv = pfring_set_poll_duration(pd, 50);
//		if (rv)
//			Elog("failed on pfring_set_poll_duration");

		/*
		rv = pfring_set_poll_watermark_timeout(pd, 500);
		if (rv)
			Elog("failed on pfring_set_poll_watermark_timeout");
		*/			
		rv = pfring_enable_ring(pd);
		if (rv)
			Elog("failed on pfring_enable_ring");

	}
	/* ctrl-c handler */
	signal(SIGINT, on_sigint_handler);
	
	/* launch worker threads */
	workers = alloca(sizeof(pthread_t) * num_threads);
	printf("pd = %p\n", pd);
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
