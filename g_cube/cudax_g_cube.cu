#include "cuda_common.h"
#include "cudax_g_cube.h"

/*
 * Arrow->Array support function
 */
DEVICE_FUNCTION(cl_uint)
pg_cube_array_from_arrow(kern_context *kcxt,
						 char *dest,
						 kern_colmeta *cmeta,
						 char *base,
						 cl_uint start,
						 cl_uint end)
{
	return pg_varlena_array_from_arrow<pg_cube_t>
		(kcxt, dest, cmeta, base, start, end);
}

typedef struct __NDBOX
{
	cl_uint		header;
	double		x[1];
} __NDBOX;

#define POINT_BIT			0x80000000
#define DIM_MASK			0x7fffffff

#define IS_POINT(header)	(((header) & POINT_BIT) != 0)
#define DIM(header)			((header) & DIM_MASK)

STATIC_FUNCTION(cl_bool)
cube_contains_v0(cl_uint header_a, double *ll_coord_a,
				 cl_uint header_b, double *ll_coord_b)
{
	cl_uint		dim_a = DIM(header_a);
	cl_uint		dim_b = DIM(header_b);
	double	   *ur_coord_a = (IS_POINT(header_a) ? NULL : ll_coord_a + dim_a);
	double	   *ur_coord_b = (IS_POINT(header_b) ? NULL : ll_coord_b + dim_b);
	double		aval, bval, temp;
	int			i, n;

	if (dim_a < dim_b)
	{
		for (i=dim_a; i < dim_b; i++)
		{
			if (__Fetch(&ll_coord_b[i]) != 0.0 ||
				__Fetch(&ur_coord_b[i]) != 0.0)
				return false;
		}
	}

	n = Min(dim_a, dim_b);
	for (i=0; i < n; i++)
	{
		aval = __Fetch(&ll_coord_a[i]);
		if (ur_coord_a)
		{
			temp = __Fetch(&ur_coord_a[i]);
			if (aval > temp)
				aval = temp;
		}
		bval = __Fetch(&ll_coord_b[i]);
		if (ur_coord_b)
		{
			temp = __Fetch(&ur_coord_b[i]);
			if (bval > temp)
				bval = temp;
		}
		if (aval > bval)
			return false;

		aval = __Fetch(&ll_coord_a[i]);
		if (ur_coord_a)
		{
			temp = __Fetch(&ur_coord_a[i]);
			if (aval < temp)
				aval = temp;
		}
		bval = __Fetch(&ll_coord_b[i]);
		if (ur_coord_b)
		{
			temp = __Fetch(&ur_coord_b[i]);
			if (bval < temp)
				bval = temp;
		}
		if (aval < bval)
			return false;
	}
	return true;
}

DEVICE_INLINE(cl_bool)
pg_cube_datum_extract(kern_context *kcxt, pg_cube_t arg,
					  cl_uint *p_header, cl_double **p_values)
{
	char	   *pos;
	cl_uint		sz;
	cl_uint		header;
	cl_uint		nitems;

	if (arg.isnull)
		return false;
	if (arg.length < 0)
	{
		if (VARATT_IS_COMPRESSED(arg.value) ||
			VARATT_IS_EXTERNAL(arg.value))
		{
			STROM_CPU_FALLBACK(kcxt, ERRCODE_STROM_VARLENA_UNSUPPORTED,
							   "varlena datum is compressed or external");
			return false;
		}
		pos = VARDATA_ANY(arg.value);
		sz = VARSIZE_ANY_EXHDR(arg.value);
	}
	else
	{
		pos = arg.value;
		sz = arg.length;
	}

	if (sz < sizeof(cl_uint))
	{
		STROM_CPU_FALLBACK(kcxt, ERRCODE_INVALID_BINARY_REPRESENTATION,
						   "cube datum is too small");
		return false;
	}
	memcpy(&header, pos, sizeof(cl_uint));
	nitems = (header & DIM_MASK);
	if ((header & POINT_BIT) == 0)
		nitems += nitems;
	if (sz < sizeof(cl_uint) + sizeof(cl_double) * nitems)
	{
		STROM_CPU_FALLBACK(kcxt, ERRCODE_INVALID_BINARY_REPRESENTATION,
						   "cube datum is too small");
		return false;
	}
	*p_header = header;
	*p_values = (cl_double *)(pos + sizeof(cl_uint));	/* unaligned */
	return true;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_cube_contains(kern_context *kcxt, pg_cube_t arg1, pg_cube_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		cl_uint		header1;
		cl_uint		header2;
		cl_double  *values1;
		cl_double  *values2;

		if (pg_cube_datum_extract(kcxt, arg1, &header1, &values1) &&
			pg_cube_datum_extract(kcxt, arg2, &header2, &values2))
		{
			result.value = cube_contains_v0(header1, values1,
											header2, values2);
		}
		else
		{
			result.isnull = true;
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_cube_contained(kern_context *kcxt, pg_cube_t arg1, pg_cube_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		cl_uint		header1;
		cl_uint		header2;
		cl_double  *values1;
		cl_double  *values2;

		if (pg_cube_datum_extract(kcxt, arg2, &header2, &values2) &&
			pg_cube_datum_extract(kcxt, arg1, &header1, &values1))
		{
			result.value = cube_contains_v0(header2, values2,
											header1, values1);
		}
		else
		{
			result.isnull = true;
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_float8_t)
pgfn_cube_ll_coord(kern_context *kcxt, pg_cube_t arg1, pg_int4_t arg2)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		cl_uint		header;
		cl_double  *values;

		if (pg_cube_datum_extract(kcxt, arg1, &header, &values))
		{
			cl_uint		dim = DIM(header);
			cl_int		index = arg2.value;

			if (index > 0 && index <= dim)
			{
				result.value = __Fetch(&values[index-1]);
				if (!IS_POINT(header))
				{
					cl_double	fval = __Fetch(&values[index-1 + dim]);

					if (result.value < fval)
						result.value = fval;
				}
			}
			else
			{
				result.value = 0.0;
			}
		}
		else
		{
			result.isnull = true;
		}
	}
	return result;
}
