#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <unistd.h>

#define get_local_id()          (threadIdx.x)
#define get_local_size()        (blockDim.x)
#define get_global_id()         (threadIdx.x + blockIdx.x * blockDim.x)
#define get_global_size()       (blockDim.x * gridDim.x)

#define ELOG(rc,fmt,...)									\
	do {													\
		if (rc != cudaSuccess)								\
		{													\
			fprintf(stderr, "(%s:%d, %s) " fmt "\n",		\
					__FUNCTION__, __LINE__,					\
					cudaGetErrorName(rc), ##__VA_ARGS__);	\
				exit(1);									\
		}													\
	} while(0)

#define USE_SYSV_SHMEM		0

__global__ void makeSumKernel(const unsigned char *x, size_t nitems,
							  unsigned long long *p_sum)
{
	size_t		index;
	unsigned long long	sum = 0;
	__shared__ unsigned long long lsum;

	if (get_local_id() == 0)
		lsum = 0;
	__syncthreads();

	for (index = get_global_id();
		 index < nitems;
		 index += get_global_size())
	{
		sum += x[index];
	}
	atomicAdd(&lsum, sum);
	__syncthreads();
	if (get_local_id() == 0)
		atomicAdd(p_sum, lsum);
}

static void usage(const char *argv0)
{
	const char *pos = strrchr(argv0, '/');
	const char *command = (pos ? pos + 1 : argv0);

	fprintf(stderr, "usage: %s [options]\n"
			"    -n <num of processes> (default: 1)\n"
			"    -s <buffer size in MB> (default: 256)\n\n"
			"    -d <dir of temporary file> (default: /dev/shm)\n",
			command);
	exit(1);
}

static int child_main(int fdesc, size_t length)
{
	cudaExternalMemoryHandleDesc mdesc;
	cudaExternalMemoryBufferDesc bdesc;
	cudaExternalMemory_t extMem;
	cudaError_t	rc;
	void	   *buf;
	unsigned long long *sum;
	int			gridSz, blockSz = 1024;

	/* cudaImportExternalMemory */
	memset(&mdesc, 0, sizeof(mdesc));
	mdesc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
	mdesc.handle.fd = fdesc;
	mdesc.size      = length;
	mdesc.flags     = 0;
	rc = cudaImportExternalMemory(&extMem, &mdesc);
	ELOG(rc, "failed on cudaImportExternalMemory");

	/* cudaExternalMemoryGetMappedBuffer */
	memset(&bdesc, 0, sizeof(bdesc));
	bdesc.offset = 0;
	bdesc.size   = length;
	bdesc.flags  = 0;
	rc = cudaExternalMemoryGetMappedBuffer(&buf, extMem, &bdesc);
	ELOG(rc, "failed on cudaExternalMemoryGetMappedBuffer");

	/* result buffer */
	rc = cudaMallocManaged(&sum, sizeof(unsigned long long),
						   cudaMemAttachHost);
    ELOG(rc, "failed on cudaMallocManaged");
	memset(sum, 0, sizeof(unsigned long long));

	/* kernel invocation */
	blockSz = 1024;
	gridSz = (length + blockSz - 1) / blockSz;

	makeSumKernel<<<gridSz, blockSz>>>((const unsigned char *)buf,
									   length, sum);

	rc = cudaStreamSynchronize(NULL);
	ELOG(rc, "failed on cudaStreamSynchronize");

	printf("sum = %lu\n", sum[0]);

	sleep(10);

	return 0;
}

int main(int argc, char * const argv[])
{
	int			nprocs = 1;
	size_t		length = (256 << 20);
	const char *dirname = "/dev/shm";
	char		path[1024];
	int			fdesc, c, _len;

	while ((c = getopt(argc, argv, "n:s:d:")) >= 0)
	{
		switch (c)
		{
			case 'n':
				nprocs = atoi(optarg);
				if (nprocs < 1)
					usage(argv[0]);
				break;
			case 's':
				_len = atoi(optarg);
				if (_len < 0)
					usage(argv[0]);
				length = (size_t)_len << 20;
				break;
			case 'd':
				dirname = optarg;
				break;
			default:
				usage(argv[0]);
				break;
		}
	}
	snprintf(path, sizeof(path), "%s/hogeXXXXXX", dirname);
	fdesc = mkstemp(path);
	if (fdesc < 0)
	{
		fprintf(stderr, "failed on mkstemp: %m\n");
		return 1;
	}

	if (ftruncate(fdesc, length))
	{
		fprintf(stderr, "failed on ftruncate: %m\n");
		return 1;
	}
	if (nprocs == 0)
		child_main(fdesc, length);
	else
	{
		int		i, status;
		pid_t	child;

		for (i=1; i <= nprocs; i++)
		{
			child = fork();
			if (child == 0)
				return child_main(fdesc, length);
			else if (child < 0)
			{
				fprintf(stderr, "failed on fork: %m\n");
				return 1;
			}
		}

		for (i=1; i <= nprocs; i++)
		{
			child = wait(&status);
		}
	}
	if (unlink(path))
	{
		fprintf(stderr, "failed on unlink: %m\n");
		return 1;
	}
	return 0;
}
