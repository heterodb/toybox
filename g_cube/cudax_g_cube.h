/*
 * cudax_g_cube.h
 */
#ifndef CUDAX_G_CUDA_H
#define CUDAX_G_CUDA_H

/* pg_cube_t */
#ifndef PG_CUBE_TYPE_DEFINED
#define PG_CUBE_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(cube)
STROMCL_VARLENA_COMP_HASH_TEMPLATE(cube)
STROMCL_VARLENA_ARROW_TEMPLATE(cube)
STROMCL_EXTERNAL_PGARRAY_TEMPLATE(cube)
#endif

/* pg_earth_t */
#ifndef PG_EARTH_TYPE_DEFINED
#define PG_EARTH_TYPE_DEFINED
typedef pg_cube_t				pg_earth_t;
#define pg_earth_param(a,b)		pg_cube_param(a,b)
#endif

DEVICE_INLINE(pg_cube_t)
to_cube(pg_earth_t arg)
{
	return arg;
}

DEVICE_INLINE(pg_earth_t)
to_earth(pg_cube_t arg)
{
	return arg;
}

DEVICE_INLINE(pg_earth_t)
to_earth_domain(kern_context *kcxt, pg_cube_t arg)
{
	//TODO: put domain constraint checks
	return arg;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_cube_contains(kern_context *kcxt, pg_cube_t arg1, pg_cube_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_cube_contained(kern_context *kcxt, pg_cube_t arg1, pg_cube_t arg2);
DEVICE_FUNCTION(pg_float8_t)
pgfn_cube_ll_coord(kern_context *kcxt, pg_cube_t arg1, pg_int4_t arg2);

#endif /* CUDAX_G_CUDA_H */
