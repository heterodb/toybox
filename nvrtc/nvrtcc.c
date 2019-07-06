#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <nvrtc.h>

#define Elog(fmt, ...)								\
	do {											\
		fprintf(stderr, "(%s:%d) " fmt "\n",			\
				__FILE__, __LINE__, ##__VA_ARGS__);	\
		exit(1);									\
	} while(0)

static void *__malloc(size_t sz)
{
	void   *ptr = malloc(sz);
	if (!ptr)
		Elog("out of memory");
	return ptr;
}

static char *argv0 = NULL;
static void usage(void)
{
	fprintf(stderr, "usage: %s [-I <dir>][-D <def>][-o <ptx>] <source>\n",
			argv0);
	exit(1);
}

int main(int argc, char *argv[])
{
	const char	   *src_file = NULL;
	const char	   *ptx_file = NULL;
	nvrtcProgram	prog;
	nvrtcResult		rc, status;
	const char	   *options[40];
	int				numopts = 0;
	int				c;
	size_t			sz;
	char		   *temp;
	struct stat		st_buf;
	int				src_fdesc;
	void		   *src_mmap;

	argv0 = basename(strdup(argv[0]));

	options[numopts++] = "-I " CUDA_INCLUDE_DIR;
	options[numopts++] = "--gpu-architecture=compute_60";
	while ((c = getopt(argc, argv, "o:I:D:")) != -1)
	{
		switch (c)
		{
			case 'I':
				temp = __malloc(strlen(optarg) + 10);
				sprintf(temp, "-I '%s'", optarg);
				options[numopts++] = temp;
				break;
			case 'D':
				temp = __malloc(strlen(optarg) + 10);
				sprintf(temp, "-D '%s'", optarg);
				options[numopts++] = temp;
				break;
			case 'o':
				ptx_file = optarg;
				break;
			default:
				usage();
				break;
		}
	}
	if (optind + 1 != argc)
		usage();
	src_file = argv[optind];
	src_fdesc = open(src_file, O_RDONLY);
	if (src_fdesc < 0)
		Elog("failed on open('%s'): %m", src_file);
	if (fstat(src_fdesc, &st_buf) != 0)
		Elog("failed on fstat('%s'): %m", src_file);
	src_mmap = mmap(NULL, st_buf.st_size,
					PROT_READ, MAP_PRIVATE,
					src_fdesc, 0);
	if (src_mmap == MAP_FAILED)
		Elog("failed on mmap('%s'): %m", src_file);

	rc = nvrtcCreateProgram(&prog, src_mmap, src_file, 0, NULL, NULL);
	if (rc != NVRTC_SUCCESS)
		Elog("failed on nvrtcCreateProgram: %d", (int)rc);

	status = nvrtcCompileProgram(prog, numopts, options);
	if (status != NVRTC_SUCCESS &&
		status != NVRTC_ERROR_COMPILATION)
		Elog("failed on nvrtcCompileProgram: %d", (int)status);

	rc = nvrtcGetProgramLogSize(prog, &sz);
	if (rc != NVRTC_SUCCESS)
		Elog("failed on nvrtcGetProgramLogSize: %d", (int)rc);
	temp = __malloc(sz+1);
	rc =  nvrtcGetProgramLog(prog, temp);
	if (rc != NVRTC_SUCCESS)
		Elog("failed on nvrtcGetProgramLog: %d", (int)rc);
	puts(temp);
	if (status != NVRTC_SUCCESS)
		return 1;

	if (ptx_file)
	{
		FILE   *filp;

		rc = nvrtcGetPTXSize(prog, &sz);
		if (rc != NVRTC_SUCCESS)
			Elog("failed on nvrtcGetPTXSize: %d", (int)rc);
		temp = __malloc(sz + 1);
		rc = nvrtcGetPTX(prog, temp);
		if (rc != NVRTC_SUCCESS)
			Elog("failed on nvrtcGetPTX: %d", (int)rc);
		filp = fopen(ptx_file, "wb");
		if (!filp)
			Elog("failed on open('%s'): %m", ptx_file);
		fwrite(temp, sz, 1, filp);
		fclose(filp);
	}
	return 0;
}
