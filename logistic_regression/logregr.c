/*
 * logregr.c
 */
#include "postgres.h"
#include "fmgr.h"

PG_MODULE_MAGIC;

Datum	logregr_predict(PG_FUNCTION_ARGS);
Datum	logregr_predict_prob(PG_FUNCTION_ARGS);

/*
 * logregr_predict
 */
Datum
logregr_predict(PG_FUNCTION_ARGS)
{
	PG_RETURN_NULL();
}
PG_FUNCTION_INFO_V1(logregr_predict);

/*
 * logregr_predict_prob
 */
Datum
logregr_predict_prob(PG_FUNCTION_ARGS)
{
	PG_RETURN_NULL();
}
PG_FUNCTION_INFO_V1(logregr_predict_prob);
