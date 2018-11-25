CREATE OR REPLACE FUNCTION
gpu_dot_product(real[], real[])
RETURNS float
AS $$
#plcuda_decl
#include "cuda_matrix.h"

KERNEL_FUNCTION_MAXTHREADS(void)
gpu_dot_product(double *p_dot,
				VectorTypeFloat *X,
				VectorTypeFloat *Y)
{
	size_t		index = get_global_id();
	size_t		nitems = X->height;
	float		v[MAXTHREADS_PER_BLOCK];
	float		sum;

	if (index < nitems)
		v[get_local_id()] = X->values[index] * Y->values[index];
	else
		v[get_local_id()] = 0.0;

	sum = pgstromTotalSum(v, MAXTHREADS_PER_BLOCK);
	if (get_local_id() == 0)
		atomicAdd(p_dot, (double)sum);
	__syncthreads();
}
#plcuda_begin
{
	size_t		nitems;
	int			blockSz;
	int			gridSz;
	double	   *dot;
	cudaError_t	rc;

	if (!VALIDATE_ARRAY_VECTOR_TYPE_STRICT(arg1, PG_FLOAT4OID) ||
		!VALIDATE_ARRAY_VECTOR_TYPE_STRICT(arg2, PG_FLOAT4OID))
		EEXIT("arguments are not vector like array");
	nitems = ARRAY_VECTOR_HEIGHT(arg1);
	if (nitems != ARRAY_VECTOR_HEIGHT(arg2))
		EEXIT("length of arguments mismatch");

	rc = cudaMallocManaged(&dot, sizeof(double));
	if (rc != cudaSuccess)
		CUEXIT(rc, "failed on cudaMallocManaged");
	memset(dot, 0, sizeof(double));

	blockSz = MAXTHREADS_PER_BLOCK;
	gridSz = (nitems + MAXTHREADS_PER_BLOCK - 1) / MAXTHREADS_PER_BLOCK;
	gpu_dot_product<<<gridSz,blockSz>>>(dot,
										(VectorTypeFloat *)arg1,
										(VectorTypeFloat *)arg2);
	rc = cudaStreamSynchronize(NULL);
	if (rc != cudaSuccess)
		CUEXIT(rc, "failed on cudaStreamSynchronize");

	return *dot;
}
#plcuda_end
$$ LANGUAGE 'plcuda';
