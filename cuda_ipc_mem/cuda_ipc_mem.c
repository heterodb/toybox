#include <stdio.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <errno.h>
#include <string.h>
#include <cuda.h>

typedef struct
{
	sem_t		sem;
	CUipcMemHandle	ipc_mhandle[1];
} cudaIpcMemInfo;

static int		num_segments = 4;
static size_t	segment_sz = 128UL << 20;	/* 128MB */
static cudaIpcMemInfo *cuda_ipc_mem_info = MAP_FAILED;

#define Elog(fmt,...)										\
	do {                                                    \
		fprintf(stderr, "%s:%d: " fmt "\n",					\
				__FUNCTION__, __LINE__, ##__VA_ARGS__);		\
		exit(1);											\
    } while(0)

static const char *
cudaErrorName(CUresult rc)
{
	const char *result;

	if (cuGetErrorName(rc, &result) != CUDA_SUCCESS)
		return "unknown error";
	return result;
}

static const char *
ipcHandleCString(CUipcMemHandle *ipc_mhandle)
{
	static char temp[200];
	uint64_t   *values = (uint64_t *)ipc_mhandle;
	int			i, ofs = 0;

	for (i=0; i < CU_IPC_HANDLE_SIZE / sizeof(uint64_t); i++)
	{
		ofs += snprintf(temp + ofs, sizeof(temp) - ofs,
						"%s%ld", i==0 ? "" : ":", values[i]);
	}
	return temp;
}

static int child_main(void)
{
	CUresult	rc;
	CUdevice	cuda_device;
	CUcontext	cuda_context;
	int			i;

	/* setup device context */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuInit: %s", cudaErrorName(rc));
	rc = cuDeviceGet(&cuda_device, 0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuDeviceGet: %s", cudaErrorName(rc));
	rc = cuCtxCreate(&cuda_context, 0, cuda_device);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuCtxCreate: %s", cudaErrorName(rc));

	system("nvidia-smi");

	for (i=0; i < num_segments; i++)
	{
		CUdeviceptr dptr;

		if (sem_wait(&cuda_ipc_mem_info->sem) != 0)
			Elog("failed on sem_wait: %m");
		rc = cuIpcOpenMemHandle(&dptr, cuda_ipc_mem_info->ipc_mhandle[i],
								CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
		if (rc != CUDA_SUCCESS)
			Elog("failed on cuIpcOpenMemHandle: %s", cudaErrorName(rc));
		printf("segment[%d] dptr=%016lx [%s]\n", i, dptr,
			   ipcHandleCString(&cuda_ipc_mem_info->ipc_mhandle[i]));
	}
	return 0;
}

int main(int argc, const char *argv[])
{
	CUresult	rc;
	CUdevice	cuda_device;
	CUcontext	cuda_context;
	pid_t		child;
	size_t		PAGE_SIZE = getpagesize();
	size_t		length;
	int			i, status;
	int			fdesc;
	char		path[256];

	if (argc == 2)
		num_segments = atoi(argv[1]);
	else if (argc > 2)
	{
		fputs("cuda_ipc_mem [num_segments]", stderr);
		return 1;
	}
	length = sizeof(cudaIpcMemInfo) + sizeof(CUipcMemHandle) * num_segments;
	length = ((length + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1));
	
	srandom(time(NULL));
	for (;;)
	{
		snprintf(path, sizeof(path),
				 "cuda_ipc_mem.%lu.temp", random());
		fdesc = shm_open(path, O_RDWR | O_CREAT | O_EXCL, 0600);
		if (fdesc < 0)
		{
			if (errno == EEXIST)
				continue;
			Elog("failed on shm_open: %m");
		}
		if (fallocate(fdesc, 0, 0, length) != 0)
			Elog("failed on fallocate: %m");
		break;
	}
	cuda_ipc_mem_info = mmap(NULL, length,
							 PROT_READ | PROT_WRITE,
							 MAP_SHARED, fdesc, 0);
	if (cuda_ipc_mem_info == MAP_FAILED)
		Elog("failed on mmap: %m");

	memset(cuda_ipc_mem_info, 0, length);
	if (sem_init(&cuda_ipc_mem_info->sem, 1, 0) != 0)
		Elog("failed on sem_init: %m");
	close(fdesc);
	shm_unlink(path);

	child = fork();
	if (child == 0)
		return child_main();
	else if (child < 0)
		Elog("failed on fork: %m");

	/* allocation and export */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuInit: %s", cudaErrorName(rc));
	rc = cuDeviceGet(&cuda_device, 0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuDeviceGet: %s", cudaErrorName(rc));
	rc = cuCtxCreate(&cuda_context, 0, cuda_device);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuCtxCreate: %s", cudaErrorName(rc));
	
	for (i=0; i < num_segments; i++)
	{
		CUdeviceptr	dptr;

		rc = cuMemAlloc(&dptr, segment_sz);
		if (rc != CUDA_SUCCESS)
			Elog("failed on cuMemAlloc: %s", cudaErrorName(rc));
		rc = cuIpcGetMemHandle(&cuda_ipc_mem_info->ipc_mhandle[i], dptr);
		if (rc != CUDA_SUCCESS)
			Elog("failed on cuIpcGetMemHandle: %s", cudaErrorName(rc));
		if (sem_post(&cuda_ipc_mem_info->sem) != 0)
			Elog("failed on sem_post: %m");
	}
	waitpid(child, &status, 0);

	return 0;
}
