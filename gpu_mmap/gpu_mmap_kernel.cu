#define get_global_size()		(blockDim.x * gridDim.x)
#define get_global_id()			(threadIdx.x + blockIdx.x * blockDim.x)
#define get_local_id()			(threadIdx.x)
#define get_local_size()		(blockDim.x)

__device__ double	gpu_mmap_result = 0.0;

extern "C" __global__ __launch_bounds__(1024)
void gpu_mmap_init(char *buffer, size_t buffer_sz)
{
	float		   *values = (float *)buffer;
	size_t			i, N = buffer_sz / sizeof(float);
	unsigned int	seed = 0xdeadbeaf + get_global_id();

	for (i=get_global_id(); i < N; i += get_global_size())
	{
		unsigned long	next = (unsigned long)seed * (unsigned long)seed;

		seed = (next >> 16) & 0x7fffffffU;
		values[i] = 100.0 * ((double)seed / (double)UINT_MAX);
	}
	if (get_global_id() == 0)
		printf("buffer = %lx buffer_sz = %lu\n", buffer, buffer_sz);
}

extern "C" __global__ __launch_bounds__(1024)
void gpu_mmap_kernel(char *buffer, size_t buffer_sz)
{
	__shared__ double sum;
	float		   *values = (float *)buffer;
	size_t			i, N = buffer_sz / sizeof(float);
	float			__sum = 0.0;

	if (get_local_id() == 0)
		sum = 0.0;

	for (i=get_global_id(); i < N; i += get_global_size())
		__sum += values[i];

	__syncthreads();
	atomicAdd(&sum, __sum);
	__syncthreads();
	if (get_local_id() == 0)
		atomicAdd(&gpu_mmap_result, sum);
}

