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
#if 0
	rc = cuModuleGetGlobal(&dptr, &sz, cuda_module, "foo1");
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuModuleGetGlobal('foo1'): %d", rc);
	printf("foo1 --> %p, sz=%zu\n", (void *)dptr, sz);
	rc = cuModuleGetGlobal(&dptr, &sz, cuda_module, "foo2");
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuModuleGetGlobal('foo2'): %d", rc);
	printf("foo2 --> %p, sz=%zu\n", (void *)dptr, sz);
#endif
	rc = cuModuleGetFunction(&f_setup, cuda_module, "kern_setup");
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuModuleGetFunction('kern_setup'): %d", rc);
	rc = cuModuleGetFunction(&f_sample, cuda_module, "kern_sample");
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuModuleGetFunction('kern_sample'): %d", rc);
	rc = cuMemAllocManaged(&dptr, 1024, CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuMemAllocManaged: %d", rc);
	kern_args[0] = &dptr;
	rc = cuLaunchKernel(f_setup,
						1, 1, 1,
						1, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		Elog("failed on cuLaunchKernel: %d", rc);
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
