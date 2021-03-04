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

#define LL_COORD(cube, i) ( (cube)->x[i] )
#define UR_COORD(cube, i) ( IS_POINT(cube) ? (cube)->x[i] : (cube)->x[(i) + DIM(cube)] )



STATIC_FUNCTION(cl_bool)
cube_contains_v0(__NDBOX *a, __NDBOX *b)
{
	cl_uint		header_a = __Fetch(&a->header);
	cl_uint		header_b = __Fetch(&b->header);
	cl_uint		dim_a = DIM(header_a);
	cl_uint		dim_b = DIM(header_b);
	double	   *ll_coord_a = a->x;
	double	   *ll_coord_b = b->x;
	double	   *ur_coord_a = a->x + (IS_POINT(header_a) ? 0 : dim_a);
	double	   *ur_coord_b = b->x + (IS_POINT(header_b) ? 0 : dim_b);
	double		aval, bval;
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
		aval = Min(__Fetch(&ll_coord_a[i]), __Fetch(&ur_coord_a[i]));
		bval = Min(__Fetch(&ll_coord_b[i]), __Fetch(&ur_coord_b[i]));
		if (aval > bval)
			return false;
		aval = Min(__Fetch(&ll_coord_a[i]), __Fetch(&ur_coord_a[i]));
		bval = Min(__Fetch(&ll_coord_b[i]), __Fetch(&ur_coord_b[i]));
		if (aval < bval)
			return false;
	}
	return true;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_cube_contains(kern_context *kcxt, pg_cube_t arg1, pg_cube_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = cube_contains_v0((__NDBOX *)VARDATA_ANY(arg1.value),
										(__NDBOX *)VARDATA_ANY(arg2.value));
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_cube_contained(kern_context *kcxt, pg_cube_t arg1, pg_cube_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = cube_contains_v0((__NDBOX *)VARDATA_ANY(arg2.value),
										(__NDBOX *)VARDATA_ANY(arg1.value));
	return result;
}

DEVICE_FUNCTION(pg_float8_t)
pgfn_cube_ll_coord(kern_context *kcxt, pg_cube_t arg1, pg_int4_t arg2)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		//TODO: in case when read from arrow!!
		__NDBOX	   *c = (__NDBOX *)VARDATA_ANY(arg1.value);
		cl_uint		header_c = __Fetch(&c->header);
		cl_uint		dim_c = DIM(header_c);
		cl_int		index = arg2.value;

		if (index > 0 && index <= dim_c)
		{
			if (IS_POINT(header_c))
				result.value = __Fetch(&c->x[index-1]);
			else
				result.value = Max(__Fetch(&c->x[index-1]),
								   __Fetch(&c->x[index-1 + dim_c]));
		}
		else
		{
			result.value = 0.0;
		}
	}
	return result;
}
