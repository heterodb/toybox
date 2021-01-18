/*
 * pcap2arrow.c
 *
 * multi-thread ultra fast packet capture and translator to Apache Arrow.
 *
 * Portions Copyright (c) 2021, HeteroDB Inc
 */
#include <ctype.h>
#include <getopt.h>
#include <pfring.h>
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

static char	   *input_filename = NULL;
static char	   *output_filename = NULL;
static long		duration_to_switch = 0;			/* not switch */
static int		protocol_mask = PCAP_PROTO__DEFAULT;
static int		num_threads = 8;
static size_t	output_filesize_limit = 0UL;				/* No limit */
static size_t	record_batch_threshold = (128UL << 20);		/* 128MB */











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
		  "  -t|--threads=<NUM of threads> (default: 8)\n"
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
				if (input_filename)
					Elog("-i|--input was specified twice");
				input_filename = optarg;
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
					Elog("invalid -t|--thread argument: %s", optarg);
				if (num_threads < 1)
					Elog("invalid number of threads: %d", num_threads);
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
			default:
				usage(code == 'h' ? 0 : 1);
				break;
		}
	}
	if (argc != optind)
		Elog("unexpected tokens in the command line argument");

	if (!input_filename)
		Elog("no input device or file was specified by -i|--input");
	if (!output_filename)
		Elog("no output file name was specified by -o|--output");
	if (protocol_mask != 0 && !output_has_proto)
		Elog("-o|--output must has '%%p' to distribute packet based on protocols");
	if (output_filesize_limit != 0 && !output_has_seqno)
		Elog("-o|--output must has '%%q' to split files when it exceeds the threshold");
}

int main(int argc, char *argv[])
{
	parse_options(argc, argv);







	return 0;
}






