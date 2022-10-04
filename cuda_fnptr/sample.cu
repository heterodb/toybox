#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define get_global_id()         (threadIdx.x + blockIdx.x * blockDim.x)

extern "C" __device__ int
foo1(int *rv, ...)
{
	va_list		ap;
	const char *arg1;
	double		arg2;

	va_start(ap, rv);
	arg1 = va_arg(ap, const char *);
	arg2 = va_arg(ap, double);
	va_end(ap);

	printf("foo1 thread=%u arg1=[%s] arg2=[%f]\n", get_global_id(), arg1, arg2);
	*rv = 1234;

	return 0;
}

extern "C" __device__ int
foo2(int *rv, ...)
{
	va_list		ap;
	int			arg1;

	va_start(ap, rv);
    arg1 = va_arg(ap, int);
	va_end(ap);

	printf("foo2 thread=%u arg1=[%d]\n", get_global_id(), arg1);
	*rv = 2345;

	return 0;
}

__device__ struct {
	uint32_t	opcode;
	int		  (*func)(int *rv,...);
} func_map_catalog[] = {
	{ 1234, foo1 },
	{ 2345, foo2 },
	{NULL, NULL},
};

extern "C" __global__ void
kern_sample(char *f_map)
{
	const char *str = "hello world";
	double		fval = 345.678;
	int			ival = 12345;
	int			rv;

	if (get_global_id() % 2 == 0)
		func_map_catalog[0].func(&rv, str, fval);
	else
		func_map_catalog[1].func(&rv, ival);

	printf("thread=%u rv=%d\n", get_global_id(), rv);
}
