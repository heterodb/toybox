#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define get_global_id()         (threadIdx.x + blockIdx.x * blockDim.x)

extern "C" __device__ void
foo1(void)
{
	printf("foo1 thread: %u\n", get_global_id());
}

extern "C" __device__ void
foo2(void)
{
	printf("foo2 thread: %u\n", get_global_id());
}

typedef struct {
	void	(*fn1)(void);
	void	(*fn2)(void);
} func_map;


extern "C" __global__ void
kern_setup(func_map *f_map)
{
	printf("foo1 = %p foo2 = %p\n", foo1, foo2);
	f_map->fn1 = foo1;
	f_map->fn2 = foo2;
}

extern "C" __global__ void
kern_sample(func_map *f_map)
{
	if (get_global_id() % 2 == 0)
		f_map->fn1();
	else
		f_map->fn2();
}
