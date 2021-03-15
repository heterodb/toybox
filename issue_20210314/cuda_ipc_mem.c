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

#define NUM_IPC_MEM_SEGMENTS	3
typedef struct
{
	sem_t		psem;
//	sem_t		csem;
	CUipcMemHandle	ipc_mhandle[NUM_IPC_MEM_SEGMENTS];
} cudaIpcMemInfo;

static int				drop_context = 1;
static pid_t			child_pid = 0;
static cudaIpcMemInfo  *cuda_ipc_mem_info = MAP_FAILED;

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

static int child_main(void)
{
	CUresult	rc;
	CUdevice	cuda_device;
	CUcontext	cuda_context = NULL;
	int			i, k;

	/* setup device context */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuInit: %s", cudaErrorName(rc));
	rc = cuDeviceGet(&cuda_device, 0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuDeviceGet: %s", cudaErrorName(rc));

	system("nvidia-smi");

	if (sem_wait(&cuda_ipc_mem_info->psem) != 0)
		Elog("failed on sem_wait: %m");
	for (k=0; k < 2; k++)
	{
		CUdeviceptr		m_addr[NUM_IPC_MEM_SEGMENTS];
		
		if (!cuda_context)
		{
			rc = cuCtxCreate(&cuda_context, 0, cuda_device);
			if (rc != CUDA_SUCCESS)
				Elog("failed on cuCtxCreate: %s", cudaErrorName(rc));
			printf("child: cuCtxCreate done\n");
		}

		for (i=0; i < NUM_IPC_MEM_SEGMENTS; i++)
		{
			rc = cuIpcOpenMemHandle(&m_addr[i],
									cuda_ipc_mem_info->ipc_mhandle[i],
									CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
			printf("IpcOpen[%d] rc=%d dptr=%llx [%lx,%lx,%lx,%lx %lx,%lx,%lx,%lx]\n",
				   i, rc, m_addr[i],
				   ((uint64_t *)(&cuda_ipc_mem_info->ipc_mhandle[i]))[0],
				   ((uint64_t *)(&cuda_ipc_mem_info->ipc_mhandle[i]))[1],
				   ((uint64_t *)(&cuda_ipc_mem_info->ipc_mhandle[i]))[2],
				   ((uint64_t *)(&cuda_ipc_mem_info->ipc_mhandle[i]))[3],
				   ((uint64_t *)(&cuda_ipc_mem_info->ipc_mhandle[i]))[4],
				   ((uint64_t *)(&cuda_ipc_mem_info->ipc_mhandle[i]))[5],
				   ((uint64_t *)(&cuda_ipc_mem_info->ipc_mhandle[i]))[6],
				   ((uint64_t *)(&cuda_ipc_mem_info->ipc_mhandle[i]))[7]);
			if (rc != CUDA_SUCCESS)
				Elog("failed on cuIpcOpenMemHandle: %s", cudaErrorName(rc));
		}

		for (i=0; i < NUM_IPC_MEM_SEGMENTS; i++)
		{
			rc = cuIpcCloseMemHandle(m_addr[i]);
			if (rc != CUDA_SUCCESS)
				Elog("failed on cuIpcCloseMemHandle: %s", cudaErrorName(rc));
		}

		if (drop_context)
		{
			rc = cuCtxDestroy(cuda_context);
			if (rc != CUDA_SUCCESS)
				Elog("failed on cuCtxDestroy: %s", cudaErrorName(rc));
			cuda_context = NULL;
			printf("child: cuCtxDestroy done\n");
		}
	}
	return 0;
}

static int parent_main(void)
{
	CUdevice	cuda_device;
	CUcontext	cuda_context;
	CUdeviceptr	m_addr[NUM_IPC_MEM_SEGMENTS];
	CUresult	rc;
	int			i, status;
	size_t		__segment_sz[] = {146064, 226656, 154909264};

	/* init cuda context */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuInit: %s", cudaErrorName(rc));
	rc = cuDeviceGet(&cuda_device, 0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuDeviceGet: %s", cudaErrorName(rc));
	rc = cuCtxCreate(&cuda_context, 0, cuda_device);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuCtxCreate: %s", cudaErrorName(rc));

	/* allocation and export */
	for (i=0; i < NUM_IPC_MEM_SEGMENTS; i++)
	{
		size_t	segment_sz = __segment_sz[i];

		rc = cuMemAlloc(&m_addr[i], segment_sz);
		if (rc != CUDA_SUCCESS)
			Elog("failed on cuMemAlloc: %s", cudaErrorName(rc));
		rc = cuIpcGetMemHandle(&cuda_ipc_mem_info->ipc_mhandle[i], m_addr[i]);
		if (rc != CUDA_SUCCESS)
			Elog("failed on cuIpcGetMemHandle: %s", cudaErrorName(rc));
		printf("GPUmem[%d] ptr=%llx sz=%zu handle[%lx,%lx,%lx,%lx %lx,%lx,%lx,%lx]\n",
			   i, m_addr[i], segment_sz,
			   ((uint64_t *)&cuda_ipc_mem_info->ipc_mhandle[i])[0],
			   ((uint64_t *)&cuda_ipc_mem_info->ipc_mhandle[i])[1],
			   ((uint64_t *)&cuda_ipc_mem_info->ipc_mhandle[i])[2],
			   ((uint64_t *)&cuda_ipc_mem_info->ipc_mhandle[i])[3],
			   ((uint64_t *)&cuda_ipc_mem_info->ipc_mhandle[i])[4],
			   ((uint64_t *)&cuda_ipc_mem_info->ipc_mhandle[i])[5],
			   ((uint64_t *)&cuda_ipc_mem_info->ipc_mhandle[i])[6],
			   ((uint64_t *)&cuda_ipc_mem_info->ipc_mhandle[i])[7]);
	}
	if (sem_post(&cuda_ipc_mem_info->psem) != 0)
		Elog("failed on sem_post: %m");
	waitpid(child_pid, &status, 0);

	return 0;
}

int main(int argc, const char *argv[])
{
	int			fdesc;
	size_t		length = sizeof(cudaIpcMemInfo);
	char		path[256];

	/* command line option */
	if (argc > 1)
		drop_context = atoi(argv[1]);

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
	if (sem_init(&cuda_ipc_mem_info->psem, 1, 0) != 0)
		Elog("failed on sem_init: %m");
	close(fdesc);
	shm_unlink(path);

	child_pid = fork();
	if (child_pid == 0)
		return child_main();
	else if (child_pid < 0)
		Elog("failed on fork: %m");
	return parent_main();
}
