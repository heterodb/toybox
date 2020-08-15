/*
 * gist_probe.c - misc code for GiST support development
 */
#include "postgres.h"
#include "fmgr.h"
#include "access/genam.h"
#include "access/gist.h"
#include "access/gist_private.h"
#include "access/itup.h"
#include "storage/bufmgr.h"
#include "utils/fmgroids.h"
#include "utils/rel.h"

PG_MODULE_MAGIC;

Datum gist_probe(PG_FUNCTION_ARGS);

static void
dump_tupdesc(TupleDesc tupdesc)
{
	int		j;

	elog(INFO, "TupleDesc {natts=%d tdtypeid=%u tdtypmod=%u}",
		 tupdesc->natts,
		 tupdesc->tdtypeid,
		 tupdesc->tdtypmod);
	for (j=0; j < tupdesc->natts; j++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

		elog(INFO, "attr[%d] {attname='%s' atttypid=%u attlen=%d attnum=%d atttypmod=%d attbyval=%d attstorage=%c attalign=%c}", j,
			 NameStr(attr->attname),
			 attr->atttypid,
			 attr->attlen,
			 attr->attnum,
			 attr->atttypmod,
			 attr->attbyval,
			 attr->attstorage,
			 attr->attalign);
	}
}

static void
dump_gist_page(char *base, Relation irel, BlockNumber blkno,
			   int depth, double x, double y)
{
	PageHeader		page = (PageHeader)(base + BLCKSZ * blkno);
	GISTPageOpaque	op = GistPageGetOpaque((Page)page);
	OffsetNumber	i, maxoff;
	bool			is_leaf = ((op->flags & F_LEAF) != 0);

	dump_tupdesc(RelationGetDescr(irel));
	elog(INFO, "PageHeader[%d] pd_lsn=%u:%d pd_checksum=%02x pd_flags=%02x pd_pagesize_version=%d pd_prune_xid=%u",
		 blkno,
		 page->pd_lsn.xlogid,
		 page->pd_lsn.xrecoff,
		 page->pd_checksum,
		 page->pd_flags,
		 page->pd_pagesize_version,
		 page->pd_prune_xid);
	elog(INFO, "GISTPageOpaque nsn=%u:%u rightlink=%u flags=%02x gist_page_id=%d",
		 op->nsn.xlogid,
		 op->nsn.xrecoff,
		 op->rightlink,
		 op->flags,
		 op->gist_page_id);

	maxoff = PageGetMaxOffsetNumber(page);
	for (i=FirstOffsetNumber; i <= maxoff; i = OffsetNumberNext(i))
	{
		ItemId		iid = PageGetItemId(page, i);
		IndexTuple	it;
		BlockNumber	__blkno;
		struct {
			float xmin, xmax, ymin, ymax;
		} *bbox;

		if (ItemIdIsDead(iid))
			continue;
		it = (IndexTuple) PageGetItem(page, iid);
		__blkno = BlockIdGetBlockNumber(&it->t_tid.ip_blkid);;

		bbox = (void *)(it + 1);
		if (x < bbox->xmin || x > bbox->xmax || y < bbox->ymin || y > bbox->ymax)
			continue;

		elog(INFO, "IndexTuple[%d] ctid=(%u,%d) t_info=%02x (hasnull=%d, hasvarlena=%d am_priv=%d tupsz=%d)", i,
			 BlockIdGetBlockNumber(&it->t_tid.ip_blkid),
			 it->t_tid.ip_posid,
			 it->t_info,
			 (it->t_info >> 15) & 1,
			 (it->t_info >> 14) & 1,
			 (it->t_info >> 13) & 1,
			 (it->t_info & 0x1fff));
		elog(INFO, "  dump { %f %f %f %f %08x %08x }",
			 ((float4 *)(it + 1))[0],
			 ((float4 *)(it + 1))[1],
			 ((float4 *)(it + 1))[2],
			 ((float4 *)(it + 1))[3],
			 ((uint32 *)(it + 1))[4],
			 ((uint32 *)(it + 1))[5]);
		if (!is_leaf)
			dump_gist_page(base, irel, __blkno, depth+1, x, y);
	}
}

static void
gist_make_tree(char *base, BlockNumber blkno,
			   BlockNumber parent_blkno, OffsetNumber parent_offno)
{
	PageHeader		page = (PageHeader)(base + BLCKSZ * blkno);
	GISTPageOpaque	op = GistPageGetOpaque((Page)page);
    OffsetNumber    i, maxoff;

	page->pd_lsn.xlogid  = parent_blkno;
	page->pd_lsn.xrecoff = parent_offno;
	if ((op->flags & F_LEAF) != 0)
		return;

	maxoff = PageGetMaxOffsetNumber(page);
	for (i=FirstOffsetNumber; i <= maxoff; i = OffsetNumberNext(i))
	{
		ItemId		iid = PageGetItemId(page, i);
        IndexTuple	it;

		if (ItemIdIsDead(iid))
			continue;
		it = (IndexTuple) PageGetItem(page, iid);
		gist_make_tree(base, BlockIdGetBlockNumber(&it->t_tid.ip_blkid),
					   blkno, i);
	}
}

Datum gist_probe(PG_FUNCTION_ARGS)
{
	Oid			index_oid = PG_GETARG_OID(0);
	float8		x = PG_GETARG_FLOAT8(1);
	float8		y = PG_GETARG_FLOAT8(2);
	Relation	irel;
	BlockNumber	i, nblocks;
	char	   *base;
	bool		retval = false;

	irel = index_open(index_oid, AccessShareLock);
	if (irel->rd_amhandler != F_GISTHANDLER)
		elog(ERROR, "index '%s' is not GiST index",
			 RelationGetRelationName(irel));
	nblocks = RelationGetNumberOfBlocks(irel);
	elog(INFO, "index '%s' has %u blocks",
		 RelationGetRelationName(irel), nblocks);
	base = MemoryContextAllocHuge(CurrentMemoryContext, BLCKSZ * nblocks);
	for (i=0; i < nblocks; i++)
	{
		Buffer	buffer;
		Page	page;
		PageHeader hpage;

		buffer = ReadBuffer(irel, i);
		LockBuffer(buffer, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buffer);
		hpage = (PageHeader)(base + BLCKSZ * i);

		memcpy(hpage, page, BLCKSZ);
		hpage->pd_lsn.xlogid = InvalidBlockNumber;
		hpage->pd_lsn.xrecoff = InvalidOffsetNumber;

		UnlockReleaseBuffer(buffer);
	}
	gist_make_tree(base, 0, InvalidBlockNumber, InvalidOffsetNumber);
	
	dump_gist_page(base, irel, 0, 0, x, y);


	index_close(irel, AccessShareLock);

	PG_RETURN_BOOL(retval);
}
PG_FUNCTION_INFO_V1(gist_probe);





