#include "cuda_codegen.h"
#include "cuda_common.h"
#include "access/genam.h"
#include "access/htup_details.h"
#include "access/table.h"
#include "catalog/indexing.h"
#include "catalog/pg_namespace.h"
#include "catalog/pg_extension.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
#include "utils/catcache.h"
#include "utils/hashutils.h"
#include "utils/inval.h"
#include "utils/fmgroids.h"
#include "utils/lsyscache.h"
#include "utils/syscache.h"

PG_MODULE_MAGIC;

static pgstromUsersExtraDescriptor	g_cubeUsersExtraDesc;
static uint32	g_cube_extra_flags;
static Oid		__pg_cube_schema_oid = InvalidOid;
static Oid		__pg_cube_type_oid = InvalidOid;
static Oid		__pg_earth_schema_oid = InvalidOid;
static Oid		__pg_earth_type_oid = InvalidOid;
void			_PG_init(void);

static void
pg_type_cache_invalidator(Datum arg, int cacheid, uint32 hashvalue)
{
	__pg_cube_schema_oid = InvalidOid;
	__pg_cube_type_oid = InvalidOid;
	__pg_earth_schema_oid = InvalidOid;
	__pg_earth_type_oid = InvalidOid;
}

static Oid
__pg_extension_schema_oid(const char *extname)
{
	static bool	invalidator_registered = false;
	Relation	rel;
	SysScanDesc	sdesc;
	ScanKeyData	skeys[1];
	HeapTuple	tuple;
	Oid			schema_oid = InvalidOid;

	if (!invalidator_registered)
	{
		CacheRegisterSyscacheCallback(TYPEOID, pg_type_cache_invalidator, 0);
		invalidator_registered = true;
	}
	rel = table_open(ExtensionRelationId, AccessShareLock);

	ScanKeyInit(&skeys[0],
				Anum_pg_extension_extname,
				BTEqualStrategyNumber, F_NAMEEQ,
				CStringGetDatum(extname));
	sdesc = systable_beginscan(rel, ExtensionNameIndexId, true,
							   NULL, 1, skeys);
	tuple = systable_getnext(sdesc);
	elog(INFO, "tuple = %p", tuple);
	if (HeapTupleIsValid(tuple))
		schema_oid = ((Form_pg_extension) GETSTRUCT(tuple))->extnamespace;
	systable_endscan(sdesc);
	table_close(rel, AccessShareLock);

	return schema_oid;
}

static inline Oid
pg_cube_schema_oid(void)
{
	if (!OidIsValid(__pg_cube_schema_oid))
		__pg_cube_schema_oid = __pg_extension_schema_oid("cube");
	return __pg_cube_schema_oid;
}

static inline Oid
pg_earch_schema_oid(void)
{
	if (!OidIsValid(__pg_earth_schema_oid))
		__pg_earth_schema_oid = __pg_extension_schema_oid("earth");
	return __pg_earth_schema_oid;
}

static inline Oid
pg_cube_type_oid(void)
{
	if (!OidIsValid(__pg_cube_type_oid))
	{
		Oid		schema_oid = pg_cube_schema_oid();

		if (OidIsValid(schema_oid))
			__pg_cube_type_oid = GetSysCacheOid2(TYPENAMENSP,
#if PG_VERSION_NUM >= 120000
												 Anum_pg_type_oid,
#endif
												 PointerGetDatum("cube"),
												 ObjectIdGetDatum(schema_oid));
	}
	return __pg_cube_type_oid;
}

static inline Oid
pg_earth_type_oid(void)
{
	if (!OidIsValid(__pg_earth_type_oid))
	{
		Oid		schema_oid = pg_cube_schema_oid();

		if (OidIsValid(schema_oid))
			__pg_earth_type_oid = GetSysCacheOid2(TYPENAMENSP,
#if PG_VERSION_NUM >= 120000
												  Anum_pg_type_oid,
#endif
												  PointerGetDatum("earth"),
												  ObjectIdGetDatum(schema_oid));
	}
	return __pg_earth_type_oid;
}

static uint32
pg_cube_devtype_hashfunc(devtype_info *dtype, Datum datum)
{
	return hash_any((unsigned char *)VARDATA_ANY(datum),
					VARSIZE_ANY_EXHDR(datum));
}

static devtype_info *
g_cube_lookup_extra_devtype(MemoryContext memcxt,
							TypeCacheEntry *tcache)
{
	devtype_info *dtype = NULL;

	if (tcache->type_id == pg_cube_type_oid() ||
		tcache->type_id == pg_earth_type_oid())
	{
		MemoryContext	oldcxt;
		HeapTuple		tuple;
		const char	   *typname;

		tuple = SearchSysCache1(TYPEOID, ObjectIdGetDatum(tcache->type_id));
		if (!HeapTupleIsValid(tuple))
			return NULL;
		typname = NameStr(((Form_pg_type) GETSTRUCT(tuple))->typname);

		oldcxt = MemoryContextSwitchTo(memcxt);
		dtype = palloc0(sizeof(devtype_info));
		dtype->type_oid = tcache->type_id;
		dtype->type_flags = g_cube_extra_flags;
		dtype->type_length = tcache->typlen;
		dtype->type_align = att_align_nominal(1, tcache->typalign);
		dtype->type_byval = tcache->typbyval;
		dtype->type_name = pstrdup(typname);
		dtype->extra_sz = sizeof(pg_varlena_t);
		dtype->hash_func = pg_cube_devtype_hashfunc;
		dtype->type_eqfunc = get_opcode(tcache->eq_opr);
		dtype->type_cmpfunc = tcache->cmp_proc;

		MemoryContextSwitchTo(oldcxt);

		ReleaseSysCache(tuple);
	}
	return dtype;
}

static devfunc_info *
g_cube_lookup_extra_devfunc(MemoryContext memcxt,
							Oid proc_oid,
							Form_pg_proc proc_form,
							devtype_info *dfunc_rettype,
							int dfunc_nargs,
							devtype_info **dfunc_argtypes,
							Oid func_collid)
{
	devfunc_info   *dfunc = NULL;
	const char	   *proc_name = NameStr(proc_form->proname);

	if (proc_form->pronamespace != pg_cube_schema_oid())
		return NULL;

	if ((strcmp(proc_name, "cube_contains") == 0 &&
		 dfunc_rettype->type_oid == BOOLOID &&
		 dfunc_nargs == 2 &&
		 dfunc_argtypes[0]->type_oid == pg_cube_type_oid() &&
		 dfunc_argtypes[1]->type_oid == pg_cube_type_oid()) ||
		(strcmp(proc_name, "cube_contained") == 0 &&
		 dfunc_rettype->type_oid == BOOLOID &&
		 dfunc_nargs == 2 &&
		 dfunc_argtypes[0]->type_oid == pg_cube_type_oid() &&
		 dfunc_argtypes[1]->type_oid == pg_cube_type_oid()) ||
		(strcmp(proc_name, "cube_ll_coord") == 0 &&
		 dfunc_rettype->type_oid == FLOAT8OID &&
		 dfunc_nargs == 2 &&
		 dfunc_argtypes[0]->type_oid == pg_cube_type_oid() &&
		 dfunc_argtypes[1]->type_oid == INT4OID))
	{
		MemoryContext oldcxt = MemoryContextSwitchTo(memcxt);
		int		i;

		dfunc = palloc0(sizeof(devfunc_info));
		dfunc->func_oid = proc_oid;
		dfunc->func_collid = func_collid;
		dfunc->func_is_negative = false;
		dfunc->func_is_strict = true;
		dfunc->func_flags = g_cube_extra_flags;
		dfunc->func_rettype = dfunc_rettype;
		for (i=0; i < dfunc_nargs; i++)
			dfunc->func_args = lappend(dfunc->func_args, dfunc_argtypes[i]);
		dfunc->func_sqlname = pstrdup(proc_name);
		dfunc->func_devname = pstrdup(proc_name);
		dfunc->func_devcost = 5;

		MemoryContextSwitchTo(oldcxt);
	}
	return dfunc;
}

static devcast_info *
g_cube_lookup_extra_devcast(MemoryContext memcxt,
							devtype_info *dtype_src,
							devtype_info *dtype_dst)
{
	devcast_info   *dcast = NULL;

	if ((dtype_src->type_oid == pg_cube_type_oid() &&
		 dtype_dst->type_oid == pg_earth_type_oid()) ||
		(dtype_src->type_oid == pg_earth_type_oid() &&
		 dtype_dst->type_oid == pg_cube_type_oid()))
	{
		dcast = MemoryContextAllocZero(memcxt, sizeof(devcast_info));

		dcast->src_type = dtype_src;
		dcast->dst_type = dtype_dst;
		dcast->has_domain_checks = (dtype_dst->type_oid == pg_earth_type_oid());
	}
	return dcast;
}

void
_PG_init(void)
{
	memset(&g_cubeUsersExtraDesc, 0, sizeof(pgstromUsersExtraDescriptor));
	g_cubeUsersExtraDesc.magic = PGSTROM_USERS_EXTRA_MAGIC_V1;
	g_cubeUsersExtraDesc.pg_version = PG_VERSION_NUM;
	g_cubeUsersExtraDesc.extra_name = "cudax_g_cube";
	g_cubeUsersExtraDesc.lookup_extra_devtype = g_cube_lookup_extra_devtype;
	g_cubeUsersExtraDesc.lookup_extra_devfunc = g_cube_lookup_extra_devfunc;
	g_cubeUsersExtraDesc.lookup_extra_devcast = g_cube_lookup_extra_devcast;

	g_cube_extra_flags = pgstrom_register_users_extra(&g_cubeUsersExtraDesc);
}
