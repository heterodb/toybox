CREATE OR REPLACE FUNCTION
logregr_train(reggstore,      -- source table (only Gstore foreign table)
              smallint,       -- column number of dependent variable
              smallint[],     -- columns number of independent variables
              int = 20,       -- max number of iteration
              real = 0.0001)  -- threashold to stop iteration
RETURNS real[]
AS $$
#plcuda_library cublas
#plcuda_decl
#include <cublas_v2.h>
#include "cuda_matrix.h"
#include <unistd.h>

KERNEL_FUNCTION_MAXTHREADS(void)
logregr_update_Z(VectorTypeFloat *Z,
				 cl_float  **Xp,
				 VectorTypeFloat *W)
{
	cl_uint		nitems = Z->height;
	cl_uint		width = W->height;
	cl_uint		i, j;

	for (i = get_global_id(); i < nitems; i += get_global_size())
	{
		cl_float	x, w, sum = 0.0;
		for (j=0; j < width; j++)
		{
			w = W->values[j];
			x = (j == 0 ? 1.0 : Xp[j-1][i]);
			sum += w * x;
		}
		Z->values[i] = 1.0 / (1.0 + __expf(-sum));
	}
}

KERNEL_FUNCTION_MAXTHREADS(void)
logregr_update_P(cl_double **Preg,	/* out */
				 cl_float  **Xp,
				 cl_int      width,
				 VectorTypeFloat *Z)
{
	cl_double  *P = Preg[0];
	cl_uint		nitems = Z->height;
	cl_uint		nitems_bs;
	cl_uint		i, j, k;
	size_t		loop, nloops;
	__shared__ cl_float v[MAXTHREADS_PER_BLOCK];

	/* block size must be 1024 */
	assert(get_local_size() == MAXTHREADS_PER_BLOCK);
	nitems_bs = TYPEALIGN(get_local_size(), nitems);
	nloops = width * width * nitems_bs;
	for (loop = get_global_id();
		 loop < nloops;
		 loop += get_global_size())
	{
		cl_float	sum;

		k = loop % nitems_bs;
		i = (loop / nitems_bs) % width;
		j = loop / (nitems_bs * width);
		assert(i < width && j < width);
		if (k < nitems)
		{
			cl_double	z = Z->values[k];
			cl_double	x1 = (i == 0 ? 1.0 : Xp[i-1][k]);
			cl_double	x2 = (j == 0 ? 1.0 : Xp[j-1][k]);

			v[get_local_id()] = x1 * z * (1.0 - z) * x2;
		}
		else
			v[get_local_id()] = 0.0;

		sum = pgstromTotalSum(v,MAXTHREADS_PER_BLOCK);
		if (get_local_id() == 0)
			atomicAdd(&P[i + j * width], sum);
		__syncthreads();
	}
}

KERNEL_FUNCTION_MAXTHREADS(void)
logregr_update_Wd(VectorTypeFloat *Wd,	/* out */
				  cl_float       **Xp,
				  cl_bool         *Tp,
				  cl_double      **Pinv,
				  VectorTypeFloat *Z)
{
	cl_double  *P = Pinv[0];
	cl_uint		width = Wd->height;
	cl_uint		nitems = Z->height;
	cl_uint		nitems_bs;
	size_t		loop, nloops;
	__shared__ cl_float v[MAXTHREADS_PER_BLOCK];

	assert(get_local_size() == MAXTHREADS_PER_BLOCK);
	nitems_bs = TYPEALIGN(get_global_size(), nitems);
	nloops = width * nitems_bs;
	for (loop = get_global_id();
		 loop < nloops;
		 loop += get_global_size())
	{
		cl_uint		i, j, k;
		cl_double	sum, val = 0.0;

		i = loop % nitems_bs;
		j = loop / nitems_bs;

		if (i < nitems)
		{
			for (k=0; k < width; k++)
			{
				val += P[j*width+k] * (k == 0 ? 1.0 : Xp[k-1][i]);
			}
			val *= (Z->values[i] - (Tp[i] ? 1.0 : 0.0));
		}
		v[get_local_id()] = val;
		sum = pgstromTotalSum(v, MAXTHREADS_PER_BLOCK);
		if (get_local_id() == 0)
			atomicAdd(&Wd->values[j], sum);
		__syncthreads();
	}
}

static VectorTypeFloat  *Z     = NULL;
static VectorTypeFloat  *W     = NULL;
static VectorTypeFloat  *Wd    = NULL;
static cl_double       **Preg  = NULL;
static cl_double       **Pinv  = NULL;
static cl_int           *Pivot = NULL; 	/* for cuBlas */
static cl_int		    *Info = NULL;	/* for cuBlas */

static void
logregr_alloc_buffer(cl_uint width, cl_uint nitems)
{
	cudaError_t		rc;

	rc = cudaMallocManaged(&Z, offsetof(VectorTypeFloat,
										values[nitems]));
	if (rc != cudaSuccess)
		CUEXIT(rc, "failed on cudaMallocManaged");
	INIT_ARRAY_VECTOR(Z, PG_FLOAT4OID, sizeof(float), nitems);

	rc = cudaMallocManaged(&W, offsetof(VectorTypeFloat,
										values[width]));
	if (rc != cudaSuccess)
		CUEXIT(rc, "failed on cudaMallocManaged");
	INIT_ARRAY_VECTOR(W, PG_FLOAT4OID, sizeof(float), width);

	rc = cudaMallocManaged(&Wd, offsetof(VectorTypeFloat,
										 values[width]));
	if (rc != cudaSuccess)
		CUEXIT(rc, "failed on cudaMallocManaged");
	INIT_ARRAY_VECTOR(Wd, PG_FLOAT4OID, sizeof(float), width);

	rc = cudaMallocManaged(&Preg, (sizeof(cl_double *) +
								   sizeof(cl_double) * width * width));
	if (rc != cudaSuccess)
		CUEXIT(rc, "failed on cudaMallocManaged");
	Preg[0] = (cl_double *)(Preg+1);

	rc = cudaMallocManaged(&Pinv, (sizeof(cl_double *) +
								   sizeof(cl_double) * width * width));
	if (rc != cudaSuccess)
		CUEXIT(rc, "failed on cudaMallocManaged");
	Pinv[0] = (cl_double *)(Pinv+1);

	rc = cudaMallocManaged(&Pivot, sizeof(int) * width);
	if (rc != cudaSuccess)
		CUEXIT(rc, "failed on cudaMallocManaged");

	rc = cudaMallocManaged(&Info, sizeof(int));
	if (rc != cudaSuccess)
		CUEXIT(rc, "failed on cudaMallocManaged");
}

static cl_int	gridSz_Z;
static cl_int	gridSz_P;
static cl_int	gridSz_Wd;
static cl_int	blockSz = MAXTHREADS_PER_BLOCK;

static void
logregr_optimal_gridsz(void)
{
	cl_int		device_id;
	cl_int		mp_count;
	cl_int		num_blocks;
	cudaError_t	rc;

	rc = cudaGetDevice(&device_id);
	if (rc != cudaSuccess)
		CUEXIT(rc, "failed on cudaGetDevice");
	rc = cudaDeviceGetAttribute(&mp_count,
								cudaDevAttrMultiProcessorCount,
								device_id);
	if (rc != cudaSuccess)
		CUEXIT(rc, "failed on cudaDeviceGetAttribute");

	rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks,
													   logregr_update_Z,
													   blockSz, 0);
	if (rc != cudaSuccess)
		CUEXIT(rc, "failed on cudaOccupancyMaxActiveBlocksPerMultiprocessor");
	gridSz_Z = mp_count * num_blocks;

	rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks,
													   logregr_update_P,
													   blockSz, 0);
	if (rc != cudaSuccess)
		CUEXIT(rc, "failed on cudaOccupancyMaxActiveBlocksPerMultiprocessor");
	gridSz_P = mp_count * num_blocks;

	rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks,
													   logregr_update_Wd,
													   blockSz, 0);
	if (rc != cudaSuccess)
		CUEXIT(rc, "failed on cudaOccupancyMaxActiveBlocksPerMultiprocessor");
	gridSz_Wd = mp_count * num_blocks;	
}

static VectorTypeFloat *
logregr_train(cl_char   *Tp,		/* dependent data */
			  cl_float **Xp,		/* independent data */
			  cl_int     width,
			  cl_uint    nitems,
			  cl_int     nloops,
			  cl_float   threshold)
{
	cl_int		i, loop;
	cl_float	delta, denom;
	cublasHandle_t handle;
	cublasStatus_t status;
	cudaError_t	rc;

	/* allocation of vector/matrix buffer */
	logregr_alloc_buffer(width, nitems);
	/* calculation of optimal grid size for each kernel */
	logregr_optimal_gridsz();
	/* assign random initial value of W */
	for (i=0; i < width; i++)
		W->values[i] = (i == 0 ? 1.0 : 0.0);
	/* init cuBlas library */
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
		EEXIT("failed on cublasCreate: %d", (int)status);

	for (loop=0; loop < nloops; loop++)
	{
		memset(Wd->values, 0, sizeof(cl_float) * width);
		memset(Preg[0], 0, sizeof(cl_double) * width * width);
		memset(Pinv[0], 0, sizeof(cl_double) * width * width);
		/* compute Z vector */
		logregr_update_Z<<<gridSz_Z, blockSz>>>(Z, Xp, W);
#if 0
		{
			rc = cudaStreamSynchronize(NULL);
			if (rc != cudaSuccess)
				CUEXIT(rc, "failed on cudaStreamSynchronize 1");

			for (i=0; i < width; i++)
			{
				fprintf(stderr, "%s%f",
						i==0 ? "Z: " : "  ", Z->values[i]);
			}
			fputc('\n', stderr);
		}
#endif
		/* compute P matrix */
		logregr_update_P<<<gridSz_P, blockSz>>>(Preg, Xp, width, Z);
#if 0
		{
			rc = cudaStreamSynchronize(NULL);
			if (rc != cudaSuccess)
				CUEXIT(rc, "failed on cudaStreamSynchronize 1");

			cl_double *pr = Preg[0];

			fprintf(stderr, "Preg");
			for (i=0; i < width * width; i++)
			{
				fprintf(stderr, "%s%f",
						i % width == 0 ? "\n" : "  ", pr[i]);
			}
			fputc('\n', stderr);
		}
#endif
		/* compute P-inverse */
		status = cublasDgetrfBatched(handle,
									 width,
									 Preg,
									 width,
									 Pivot,
									 Info,
									 1);
		if (status != CUBLAS_STATUS_SUCCESS)
			EEXIT("failed on cublasSgetrfBatched: %d", (int)status);
		status = cublasDgetriBatched(handle,
									 width,
									 Preg,
									 width,
									 Pivot,
									 Pinv,
									 width,
									 Info,
									 1);
		if (status != CUBLAS_STATUS_SUCCESS)
			EEXIT("failed on cublasSgetriBatched: %d", (int)status);
#if 0
		{
			rc = cudaStreamSynchronize(NULL);
			if (rc != cudaSuccess)
				CUEXIT(rc, "failed on cudaStreamSynchronize 1");

			cl_double *pi = Pinv[0];

			fprintf(stderr, "Pinv");
			for (i=0; i < width * width; i++)
			{
				fprintf(stderr, "%s%f",
						i % width == 0 ? "\n" : "  ", pi[i]);
			}
			fputc('\n', stderr);
		}
#endif
		/* compute Wd vector */
		logregr_update_Wd<<<gridSz_Wd, blockSz>>>(Wd, Xp, Tp, Pinv, Z);
		/* compute delta */
		rc = cudaStreamSynchronize(NULL);
		if (rc != cudaSuccess)
			CUEXIT(rc, "failed on cudaStreamSynchronize 5");
		delta = denom = 0.0;
		for (i=0; i < width; i++)
		{
			cl_float	wd = Wd->values[i];
			cl_float	wo = W->values[i];

			delta += wd * wd;
			denom += wo * wo;

			W->values[i] -= wd;
		}
		/* check delta */
		if (delta / denom < threshold)
			break;
	}
	return W;
}
#plcuda_begin
{
	char			buf[KDS_LEAST_LENGTH];
	kern_data_store *kds_h = (kern_data_store *)buf;
	kern_data_store *kds_d;
	VectorTypeShort *iattrs;
	cl_float	  **Xp;
	cl_char		   *Tp;
	kern_colmeta   *cmeta;
	size_t			length = KDS_LEAST_LENGTH;
	int				i, j;
	cudaError_t		rc;

	if (!arg1)
		EEXIT("source table is empty");
	/* assign CUDA context on the preferable device */
	rc = cudaSetDevice(arg1->h.device_id);
	if (rc != cudaSuccess)
		CUEXIT(rc, "failed on cudaSetDevice");

	/* fetch header of the source table */
	kds_d = (kern_data_store *)arg1->map;
retry:
	rc = cudaMemcpy(kds_h, kds_d, length,
					cudaMemcpyDeviceToHost);
	if (rc != cudaSuccess)
		CUEXIT(rc, "failed on cudaMemcpy");
	if (length < KERN_DATA_STORE_HEAD_LENGTH(kds_h))
	{
		length = KERN_DATA_STORE_HEAD_LENGTH(kds_h);
		kds_h = (kern_data_store *)malloc(length);
		if (!kds_h)
			EEXIT("out of host memory");
		goto retry;
	}

	/* simple array of dependent data */
	if (arg2 <= 0 || arg2 > kds_h->ncols)
		EEXIT("training variable is out of range");
	cmeta = &kds_h->colmeta[arg2-1];
	if (cmeta->atttypid != PG_BOOLOID)
		EEXIT("training variable must be 'bool' %u", cmeta->atttypid);
	Tp = (char *)kds_d + __kds_unpack(cmeta->va_offset);

	/* set of simple array of independent data */
	if (!VALIDATE_ARRAY_VECTOR_TYPE_STRICT(arg3, PG_INT2OID))
		EEXIT("independent variables must be 'smallint[]'");
	iattrs = (VectorTypeShort *)arg3;
	rc = cudaMallocManaged(&Xp, sizeof(float *) * iattrs->height);
	if (rc != cudaSuccess)
		CUEXIT(rc, "failed on cudaMallocManaged");
	for (i=0; i < iattrs->height; i++)
	{
		j = iattrs->values[i];
		if (j <= 0 || j > kds_h->ncols)
			EEXIT("independent variable is out of range");
		cmeta = &kds_h->colmeta[j-1];
		if (cmeta->atttypid != PG_FLOAT4OID)
			EEXIT("independent variables must be 'real'");
		Xp[i] = (float *)((char *)kds_d + __kds_unpack(cmeta->va_offset));
	}

	/* sanity-check of other arguments */
	if (arg4 <= 0)
		EEXIT("minimum number of iteration must be positive integer");
	if (arg5 <= 0)
		EEXIT("threshold must be positive number");
	/* run main logic */
    return (varlena *)logregr_train(Tp, Xp,
									iattrs->height + 1,	/* width */
									kds_h->nitems,		/* nitems */
									arg4, arg5);
}
#plcuda_end
$$ LANGUAGE 'plcuda' STRICT;

CREATE OR REPLACE FUNCTION
logregr_predict(real[],		  -- result of the training
                real[],       -- independent variables
                float = 0.5)  -- threshold of true/false
RETURNS bool
AS 'MODULE_PATHNAME','logregr_predict'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION
logregr_predict_prob(real[],  -- result of the training
                     real[])  -- independent variables
RETURNS float
AS 'MODULE_PATHNAME','logregr_predict_prob'
LANGUAGE C STRICT;
