/*
 * Simple Test Program for PF-RING
 */
#include <pfring.h>
#include <stdio.h>
#include <unistd.h>
#include <arpa/inet.h>

static int			thread_count = 0;
static __thread int	thread_id;

#define Elog(fmt, ...)                              \
	do {                                            \
		fprintf(stderr,"%s:%d  " fmt "\n",			\
				__FILE__, __LINE__, ##__VA_ARGS__);	\
		exit(1);                                    \
	} while(0)


static void worker_handler(const struct pfring_pkthdr *pkthdr,
						   const u_char *buf, const u_char *__private)
{
	struct tm	tm;

	localtime_r(&pkthdr->ts.tv_sec, &tm);

	if (pkthdr->len >= sizeof(uint32_t) * 6)
	{
		int			version = (buf[0] & 0x0f);
		int			head_sz = (buf[0] >> 4);
		int			length = ntohs(((uint16_t *)buf)[1]);
		int			ttl		= buf[8];
		int			proto	= buf[9];
		uint32_t	src_ip	= ntohl(((uint32_t *)buf)[3]);
		uint32_t	dst_ip	= ntohl(((uint32_t *)buf)[4]);
		uint32_t	extra	= ntohl(((uint32_t *)buf)[5]);
		
		printf("thread[%d] %02d:%02d:%02d.%06ld LEN=%u ipv%d head_sz=%d length=%d ttl=%d proto=%d src_ip=%08x dst_ip=%08x extra=%08x\n",
			   thread_id,
			   tm.tm_hour,
			   tm.tm_min,
			   tm.tm_sec,
			   pkthdr->ts.tv_usec,
			   pkthdr->len,
			   version,
			   head_sz,
			   length,
			   ttl,
			   proto,
			   src_ip,
			   dst_ip,
			   extra);
	}
	else
	{
		printf("thread[%d] %02d:%02d:%02d.%06ld caplen=%u len=%u\n",
			   thread_id,
			   tm.tm_hour,
			   tm.tm_min,
			   tm.tm_sec,
			   pkthdr->ts.tv_usec,
			   pkthdr->caplen,
			   pkthdr->len);
	}
}

static void *worker_main(void *__private)
{
	pfring	   *pfring = __private;

	thread_id = __atomic_fetch_add(&thread_count, 1, __ATOMIC_SEQ_CST);
	while (pfring_loop(pfring, worker_handler, NULL, 1) == 0);

	return NULL;
}

int main(int argc, char *argv[])
{
	char	   *dev_name = "any";
	int			num_threads = 6;
	int			i, c;
	pthread_t  *threads;
	pfring	   *pfring;

	while ((c = getopt(argc, argv, "d:j:")) >= 0)
	{
		switch (c)
		{
			case 'd':
				dev_name = optarg;
				break;
			case 'j':
				num_threads = atoi(optarg);
				break;
			default:
				fputs("usage: pfring_test -d DEV -j N-threads", stderr);
				return 1;
		}
	}
	pfring = pfring_open(dev_name, 65536, (PF_RING_PROMISC |
										   PF_RING_TIMESTAMP |
										   PF_RING_HW_TIMESTAMP));
	if (!pfring)
		Elog("failed on pfring_open");

	threads = alloca(sizeof(pthread_t) * num_threads);
	for (i=0; i < num_threads; i++)
	{
		if (pthread_create(&threads[i], NULL, worker_main, pfring) != 0)
			Elog("failed on pthread_create");
	}
	for (i=0; i < num_threads; i++)
		pthread_join(threads[i], NULL);
	pfring_close(pfring);
	return 0;
}
