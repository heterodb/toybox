#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>

#define Elog(fmt, ...)                              \
	do {											\
		fprintf(stderr,"%s:%d  " fmt "\n",          \
				__FILE__,__LINE__, ##__VA_ARGS__);  \
		exit(1);                                    \
	} while(0)

static long			PAGE_SIZE;
static size_t		fread_chunk_sz = (64UL << 20);	/* 64MB */
static int			fread_num_threads = 6;			/* # of threads */
static bool			fread_use_direct_io = false;
typedef struct
{
	const char	   *fname;
	int				fdesc;
	uint64_t		f_pos;
} fread_state;

static void *fread_worker_main(void *__priv)
{
	fread_state	   *state = __priv;
	struct stat		stat_buf;
	uint64_t		f_pos;
	size_t			length;
	size_t			offset;
	ssize_t			nbytes;
	char		   *buffer;

	if (fstat(state->fdesc, &stat_buf) != 0)
		Elog("failed on fstat('%s'): %m", state->fname);
	length = stat_buf.st_size & ~(PAGE_SIZE - 1);

	buffer = malloc(fread_chunk_sz + PAGE_SIZE);
	if (!buffer)
		Elog("out of memory: %m");
	buffer = (void *)((uintptr_t)buffer & ~(PAGE_SIZE - 1));
	for (;;)
	{
		size_t		chunk_sz = fread_chunk_sz;

		f_pos = __atomic_fetch_add(&state->f_pos, chunk_sz, __ATOMIC_SEQ_CST);
		if (f_pos >= length)
			break;
		if (f_pos + chunk_sz > length)
			chunk_sz = length - f_pos;

		offset = 0;
		while (offset < fread_chunk_sz)
		{
			nbytes = pread(state->fdesc,
						   buffer + offset,
						   chunk_sz - offset,
						   f_pos + offset);
			if (nbytes > 0)
				offset += nbytes;
			else
				break;
		}
	}
	return NULL;
}

int main(int argc, char *argv[])
{
	pthread_t  *workers;
	int			nworkers;
	int			c, i, k;

	PAGE_SIZE = sysconf(_SC_PAGESIZE);
	
	while ((c = getopt(argc, argv, "s:n:d")) >= 0)
	{
		switch (c)
		{
			case 's':
				fread_chunk_sz = ((size_t)atoi(optarg) << 20);
				break;
			case 'n':
				fread_num_threads = atoi(optarg);
				break;
			case 'd':
				fread_use_direct_io = true;
				break;
			default:
				fputs("usage: freads [-s <chunk_sz>][-n <num threads>][-d] FILES...\n",
					  stderr);
				return 1;
		}
	}
	if (optind == argc)
		Elog("no filename is given");
	nworkers = fread_num_threads * (argc - optind);
	workers = alloca(sizeof(pthread_t) * nworkers);

	i = k = 0;
	while (optind < argc)
	{
		const char	   *fname = argv[optind++];
		fread_state	   *state = alloca(sizeof(fread_state));
		int				flags = O_RDONLY;
		int				fdesc;

		if (fread_use_direct_io)
			flags |= O_DIRECT;
		fdesc = open(fname, flags, 0600);
		if (fdesc < 0)
			Elog("failed to open('%s'): %m", fname);
		
		state->fname = fname;
		state->fdesc = fdesc;
		state->f_pos = 0;
		for (k=0; k < fread_num_threads; k++)
		{
		    if (pthread_create(&workers[i++], NULL,
							   fread_worker_main, state) != 0)
				Elog("failed on pthread_create");
		}
	}
	assert(i == nworkers);

	for (i=0; i < nworkers; i++)
	{
		pthread_join(workers[i], NULL);
	}
	return 0;
}
