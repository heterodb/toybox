#include <stdio.h>
#define CUDA_API_PER_THREAD_DEFAULT_STREAM      1
#include <cuda.h>

#define Elog(fmt,...)                                       \
	do {                                                    \
		fprintf(stderr, "%s:%d: " fmt "\n",                 \
				__FUNCTION__, __LINE__, ##__VA_ARGS__);     \
		exit(1);                                            \
	} while(0)

int main(int argc, char *argv[])
{
	CUdevice	cuda_device;
	CUcontext	cuda_context;
	CUmodule	cuda_module;
	CUfunction	f_sample;
	CUfunction	f_setup;
	CUlinkState	lstate;
	CUresult	rc;
	CUdeviceptr	dptr;
	size_t		sz;
	const char *module_name;
	void	   *image;
	size_t		length;
	void	   *kern_args[10];
	char		buffer[1024];

	if (argc < 3)
		Elog("usage: %s <cuda_module> [symbols]");
	module_name = argv[1];
	
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuInit: %d", rc);
	rc = cuDeviceGet(&cuda_device, 0);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuDeviceGet: %d", rc);
	rc = cuCtxCreate(&cuda_context, CU_CTX_SCHED_AUTO, cuda_device);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuCtxCreate: %d", rc);
	rc = cuLinkCreate(0, NULL, NULL, &lstate);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuLinkCreate: %d", rc);
	rc = cuLinkAddFile(lstate, CU_JIT_INPUT_FATBINARY, module_name, 0, NULL, NULL);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuLinkAddFile: %d", rc);
	rc = cuLinkComplete(lstate, &image, &length);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuLinkComplete: %d", rc);
	rc = cuModuleLoadData(&cuda_module, image);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuModuleLoadData: %d", rc);

	rc = cuModuleGetGlobal(&dptr, &sz, cuda_module, "func_map_catalog");
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuModuleGetGlobal('func_map_catalog'): %d", rc);
	printf("foo1 --> %p, sz=%zu\n", (void *)dptr, sz);

	rc = cuModuleGetFunction(&f_sample, cuda_module, "kern_sample");
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuModuleGetFunction('kern_sample'): %d", rc);
	rc = cuLaunchKernel(f_sample,
						1, 1, 1,
						64, 1, 1,
						1024,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuLaunchKernel: %d", rc);
	rc = cuStreamSynchronize(CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuStreamSynchronize: %d", rc);

	return 0;
}
