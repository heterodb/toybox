#include <stdio.h>
#include <unistd.h>
#include <cublas_v2.h>

#define EEXIT(fmt, ...)                             \
	do {											\
		fprintf(stderr, "L%d: " fmt "\n",           \
				__LINE__, ##__VA_ARGS__);           \
		exit(2);                                    \
	} while(0)

static float M[] = {  3,  4,  5,  6,  7,
					  2,  2,  6,  8, 10,
					  5,  4,  3,  2,  1,
					  3,  2,  3,  2,  3,
					  4,  5,  3,  7,  8 };
int main (int argc, char *argv[])
{
	cublasHandle_t handle;
	cublasStatus_t status;
	int		n = 5;
	int		lda = n;
	int	   *P;
	int	   *INFO;
	float **AP;
	float **BP;
	float  *A;
	float  *B;
	int		i,j,k;

	status =  cublasCreate_v2(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
		EEXIT("failed on cublasCreate");

	cudaMallocManaged<float *>(&AP, sizeof(float *));
	cudaMallocManaged<float *>(&BP, sizeof(float *));
	cudaMallocManaged<float>(&A, n*n*sizeof(float));
	cudaMallocManaged<float>(&B, n*n*sizeof(float));
	cudaMallocManaged<int>(&INFO, sizeof(int));
	cudaMallocManaged<int>(&P, n*sizeof(int));

	memcpy(A, M, n*n*sizeof(float));
	AP[0] = A;
	BP[0] = B;

	puts("\n ---- A ----");
	for (i=0; i < n*n; i++)
		printf("%s  %.4f", i%n==0 ? "\n" : "  ", A[i]);

	status = cublasSgetrfBatched(handle,
								 n,
								 AP, lda,
								 P,
								 INFO,
								 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		EEXIT("failed on cublasSgetrfBatched");

	status = cublasSgetriBatched(handle,
								 n,
								 AP, lda,
								 P,
								 BP, lda,
								 INFO,
								 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		EEXIT("failed on cublasSgetriBatched");

	cudaStreamSynchronize(NULL);

	puts("\n ---- B ----");
	for (i=0; i < n*n; i++)
		printf("%s  %.4f", i%n==0 ? "\n" : "  ", B[i]);

	puts("\n ---- C ----");
	for (i=0; i < n; i++)
	{
		for (j=0; j < n; j++)
		{
			float sum=0.0;
			for (k=0; k < n; k++)
			{
				sum += M[i*n+k] * B[k*n+j];
				if (i==0 && j==0)
					printf("M=%f B=%f\n", M[i*n+k], B[k*n+j]);
			}
			printf("%s  %.4f", j==0 ? "\n" : "  ", sum);
		}
	}
	putchar('\n');
	cublasDestroy_v2(handle);

	return 0;
}
