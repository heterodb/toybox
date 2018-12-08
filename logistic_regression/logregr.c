/*
 * logregr.c
 */
#include "postgres.h"
#include "catalog/pg_type.h"
#include "fmgr.h"
#include "utils/array.h"
#include <math.h>

PG_MODULE_MAGIC;

Datum	logregr_predict(PG_FUNCTION_ARGS);
Datum	logregr_predict_prob(PG_FUNCTION_ARGS);
Datum	logregr_predict_fp64(PG_FUNCTION_ARGS);
Datum	logregr_predict_prob_fp64(PG_FUNCTION_ARGS);

/*
 * compute_logregr_predict_prob
 */
static double
compute_logregr_predict_prob(ArrayType *W, ArrayType *X)
{
	float  *wp;
	float  *xp;
	double	sum;
	int		i, nitems;

	/* sanity check of array */
	if (ARR_NDIM(W) != 1 ||
		ARR_HASNULL(W) ||
		ARR_ELEMTYPE(W) != FLOAT4OID)
		elog(ERROR, "invalid weight parameter");
	if (ARR_NDIM(X) != 1 ||
		ARR_HASNULL(X) ||
		ARR_ELEMTYPE(X) != FLOAT4OID)
		elog(ERROR, "invalid explanatory variables");
	if (ARR_DIMS(W)[0] != ARR_DIMS(X)[0] + 1)
		elog(ERROR, "length mismatch");
	nitems = ARR_DIMS(W)[0];
	wp = (float *)ARR_DATA_PTR(W);
	xp = (float *)ARR_DATA_PTR(X);
	for (i=1, sum=wp[0]; i < nitems; i++)
		sum += wp[i] * xp[i-1];

	return 1.0 / (1 + exp(-sum));
}

/*
 * compute_logregr_predict_prob_fp64
 */
static double
compute_logregr_predict_prob_fp64(ArrayType *W, ArrayType *X)
{
	double	   *wp;
	double	   *xp;
	double		sum;
	int			i, nitems;

	/* sanity check of array */
	if (ARR_NDIM(W) != 1 ||
		ARR_HASNULL(W) ||
		ARR_ELEMTYPE(W) != FLOAT8OID)
		elog(ERROR, "invalid weight parameter");
	if (ARR_NDIM(X) != 1 ||
		ARR_HASNULL(X) ||
		ARR_ELEMTYPE(X) != FLOAT8OID)
		elog(ERROR, "invalid explanatory variables");
	if (ARR_DIMS(W)[0] != ARR_DIMS(X)[0] + 1)
		elog(ERROR, "length mismatch");
	nitems = ARR_DIMS(W)[0];
	wp = (double *)ARR_DATA_PTR(W);
	xp = (double *)ARR_DATA_PTR(X);
	for (i=1, sum=wp[0]; i < nitems; i++)
		sum += wp[i] * xp[i-1];

	return 1.0 / (1 + exp(-sum));
}

/*
 * logregr_predict
 */
Datum
logregr_predict(PG_FUNCTION_ARGS)
{
	ArrayType  *W = PG_GETARG_ARRAYTYPE_P(0);
	ArrayType  *X = PG_GETARG_ARRAYTYPE_P(1);
	double		threshold = PG_GETARG_FLOAT8(2);
	double		prob = compute_logregr_predict_prob(W, X);

	PG_RETURN_BOOL(prob >= threshold);
}
PG_FUNCTION_INFO_V1(logregr_predict);

/*
 * logregr_predict_prob
 */
Datum
logregr_predict_prob(PG_FUNCTION_ARGS)
{
	ArrayType  *W = PG_GETARG_ARRAYTYPE_P(0);
	ArrayType  *X = PG_GETARG_ARRAYTYPE_P(1);

	PG_RETURN_FLOAT8(compute_logregr_predict_prob(W, X));
}
PG_FUNCTION_INFO_V1(logregr_predict_prob);

/*
 * logregr_predict_fp64
 */
Datum
logregr_predict_fp64(PG_FUNCTION_ARGS)
{
	ArrayType  *W = PG_GETARG_ARRAYTYPE_P(0);
	ArrayType  *X = PG_GETARG_ARRAYTYPE_P(1);
	double		threshold = PG_GETARG_FLOAT8(2);
	double		prob = compute_logregr_predict_prob_fp64(W, X);

	PG_RETURN_BOOL(prob >= threshold);
}
PG_FUNCTION_INFO_V1(logregr_predict_fp64);

/*
 * logregr_predict_prob_fp64
 */
Datum
logregr_predict_prob_fp64(PG_FUNCTION_ARGS)
{
	ArrayType  *W = PG_GETARG_ARRAYTYPE_P(0);
	ArrayType  *X = PG_GETARG_ARRAYTYPE_P(1);

	PG_RETURN_FLOAT8(compute_logregr_predict_prob_fp64(W, X));
}
PG_FUNCTION_INFO_V1(logregr_predict_prob_fp64);
