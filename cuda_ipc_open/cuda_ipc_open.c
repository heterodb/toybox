#include <libgen.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cuda.h>

static size_t		segment_sz = (256UL << 20);		/* 256MB */


#define Elog(fmt,...)                                       \
	do {                                                    \
		fprintf(stderr, "%s:%d: " fmt "\n",                 \
				__FUNCTION__, __LINE__, ##__VA_ARGS__);     \
		exit(1);                                            \
	} while(0)

static const char *
cudaErrorName(CUresult rc)
{
	const char *result;

	if (cuGetErrorName(rc, &result) != CUDA_SUCCESS)
		return "unknown error";
	return result;
}

static int
child_main(int pipefd, int cleanup)
{
	CUipcMemHandle	ipc_mhandle;
	CUdevice		cuda_device;
	CUcontext		cuda_context;
	CUdeviceptr		cuda_devptr;
	CUresult		rc;

	/* init CUDA driver APIs */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuInit: %s", cudaErrorName(rc));
	rc = cuDeviceGet(&cuda_device, 0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuDeviceGet: %s", cudaErrorName(rc));

	/* import device memory at the 1st context */
	if (read(pipefd,
			 &ipc_mhandle,
			 sizeof(ipc_mhandle)) != sizeof(ipc_mhandle))
		Elog("failed on read(&ipc_mhandle): %m");
	rc = cuCtxCreate(&cuda_context, 0, cuda_device);
	if (rc != CUDA_SUCCESS)
		Elog("1st: failed on cuCtxCreate: %s", cudaErrorName(rc));
	rc = cuIpcOpenMemHandle(&cuda_devptr, ipc_mhandle,
							CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
	if (rc != CUDA_SUCCESS)
		Elog("1st: failed on cuIpcOpenMemHandle: %s", cudaErrorName(rc));

	if (cleanup)
	{
		rc = cuIpcCloseMemHandle(cuda_devptr);
		if (rc != CUDA_SUCCESS)
			Elog("1st: failed on cuIpcCloseMemHandle: %s", cudaErrorName(rc));
	}
	
	/*
	 * Destroy CUDA context - we expect all the related resources
	 * shall be cleaned up, as if they are individually released.
	 */
	rc = cuCtxDestroy(cuda_context);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuCtxDestroy: %s", cudaErrorName(rc));

	/* import device memory at the 2nd context */
	rc = cuCtxCreate(&cuda_context, 0, cuda_device);
	if (rc != CUDA_SUCCESS)
		Elog("2nd: failed on cuCtxCreate: %s", cudaErrorName(rc));
	rc = cuIpcOpenMemHandle(&cuda_devptr, ipc_mhandle,
							CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
	if (rc != CUDA_SUCCESS)
		Elog("2nd: failed on cuIpcOpenMemHandle: %s", cudaErrorName(rc));
	
	return 0;
}

int main(int argc, const char *argv[])
{
	int			pipefd[2];
	pid_t		child;
	int			status;
	int			cleanup;
	CUresult	rc;
	CUdevice	cuda_device;
	CUcontext	cuda_context;
	CUdeviceptr	cuda_devptr;
	CUipcMemHandle ipc_mhandle;

	/* parse command line option */
	switch (argc)
	{
		case 1:
			cleanup = 0;
			break;
		case 2:
			cleanup = atoi(argv[1]);
			if (cleanup == 0 || cleanup == 1)
				break;
		default:
			fprintf(stderr, "usage: %s [0|1]\n", strdup(argv[0]));
			return 1;
	}

	/* kick a child process */
	if (pipe(pipefd) != 0)
		Elog("failed on pipe(2): %m");

	child = fork();
	if (child == 0)
	{
		close(pipefd[1]);
		return child_main(pipefd[0], cleanup);
	}
	else if (child < 0)
		Elog("failed on fork(2): %m");
	close(pipefd[0]);

	/* init CUDA context */
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
	rc = cuMemAlloc(&cuda_devptr, segment_sz);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuMemAlloc: %s", cudaErrorName(rc));
	rc = cuIpcGetMemHandle(&ipc_mhandle, cuda_devptr);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuIpcGetMemHandle: %s", cudaErrorName(rc));
	if (write(pipefd[1],
			  &ipc_mhandle,
			  sizeof(ipc_mhandle)) != sizeof(ipc_mhandle))
		Elog("failed on write(&ipc_mhandle): %m");

	/* wait for child exit */
	waitpid(child, &status, 0);

	return 0;
}
