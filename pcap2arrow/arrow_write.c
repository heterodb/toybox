/*
 * arrow_write.c - routines to write out apache arrow format
 *
 * Copyright 2018-2020 (C) KaiGai Kohei <kaigai@heterodb.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License. See the LICENSE file.
 */
#include "postgres.h"
#include <assert.h>
#include "arrow_ipc.h"

typedef struct
{
	uint16		vlen;	/* vtable length */
	uint16		tlen;	/* table length */
	uint16		offset[FLEXIBLE_ARRAY_MEMBER];
} FBVtable;

typedef struct
{
	void	  **extra_buf;		/* external buffer */
	int32	   *extra_sz;		/* size of extra data */
	uint16	   *extra_align;	/* alignment of extra data */
	uint16		nattrs;			/* number of variables */
	uint16		maxalign;		/* expected alignment of vtable base */
	int32		length;			/* length of the flat image.
								 * If -1, buffer is not flatten yet. */
	FBVtable	vtable;
} FBTableBuf;

static FBTableBuf *
__allocFBTableBuf(int nattrs, const char *func_name)
{
	FBTableBuf *buf;
	size_t		required = (MAXALIGN(offsetof(FBTableBuf,
											  vtable.offset[nattrs])) +
							MAXALIGN(sizeof(int32) +
									 sizeof(Datum) * nattrs));
	buf = palloc0(required);
	buf->extra_buf		= palloc0(sizeof(void *) * nattrs);
	buf->extra_sz		= palloc0(sizeof(int32) * nattrs);
	buf->extra_align	= palloc0(sizeof(uint16) * nattrs);
	buf->nattrs			= nattrs;
	buf->maxalign		= ALIGNOF_INT;
	buf->length			= -1;	/* not flatten yet */
	buf->vtable.vlen	= offsetof(FBVtable, offset[0]);
	buf->vtable.tlen	= sizeof(int32);

	return buf;
}
#define allocFBTableBuf(a)						\
	__allocFBTableBuf((a),__FUNCTION__)

static void
__addBufferScalar(FBTableBuf *buf, int index, void *ptr, int sz, int align)
{
	FBVtable   *vtable = &buf->vtable;

	assert(sz >= 0 && sz <= sizeof(int64));
	assert(index >= 0 && index < buf->nattrs);
	if (!ptr || sz == 0)
		vtable->offset[index] = 0;
	else
	{
		char   *table;
		int		offset;

		assert(buf->vtable.tlen >= sizeof(int32));
		table = (char *)&buf->vtable + offsetof(FBVtable, offset[buf->nattrs]);
		offset = TYPEALIGN(align, vtable->tlen);
		memcpy(table + offset, ptr, sz);
		vtable->offset[index] = offset;
		vtable->tlen = offset + sz;
		vtable->vlen = Max(vtable->vlen,
						   offsetof(FBVtable, offset[index+1]));
		buf->maxalign = Max(buf->maxalign, align);
	}
}

static void
__addBufferBinary(FBTableBuf *buf, int index, void *ptr, int sz, int align)
{
	assert(index >= 0 && index < buf->nattrs);
	if (!ptr || sz == 0)
		buf->vtable.offset[index] = 0;
	else
	{
		int32	zero = 0;

		buf->extra_buf[index]	= ptr;
		buf->extra_sz[index]	= sz;
		buf->extra_align[index] = align;
		__addBufferScalar(buf, index, &zero, sizeof(int32), ALIGNOF_INT);
	}
}

static inline void
addBufferOffset(FBTableBuf *buf, int index, FBTableBuf *sub)
{
	assert(index >= 0 && index < buf->nattrs);
	if (!sub)
		buf->vtable.offset[index] = 0;
	else
	{
		int32	shift = sub->vtable.vlen;

		if (sub->length < 0)
			Elog("Bug? FBTableBuf is not flatten");
		buf->extra_buf[index]	= &sub->vtable;
		buf->extra_sz[index]	= sub->length;
		buf->extra_align[index]	= sub->maxalign;
		__addBufferScalar(buf, index, &shift, sizeof(int32), ALIGNOF_INT);
	}
}

static inline void
__addBufferBool(FBTableBuf *buf, int index, bool value, bool __default)
{
	if (value != __default)
		__addBufferScalar(buf, index, &value, sizeof(value), 1);
}
#define addBufferBool(a,b,c)		__addBufferBool((a),(b),(c),false)

static inline void
__addBufferChar(FBTableBuf *buf, int index, int8 value, int8 __default)
{
	if (value != __default)
		__addBufferScalar(buf, index, &value, sizeof(value), 1);
}
#define addBufferChar(a,b,c)		__addBufferChar((a),(b),(c),0)

static inline void
__addBufferShort(FBTableBuf *buf, int index, int16 value, int16 __default)
{
	if (value != __default)
		__addBufferScalar(buf, index, &value, sizeof(value), ALIGNOF_SHORT);
}
#define addBufferShort(a,b,c)		__addBufferShort((a),(b),(c),0)

static inline void
__addBufferInt(FBTableBuf *buf, int index, int32 value, int32 __default)
{
	if (value != __default)
		__addBufferScalar(buf, index, &value, sizeof(value), ALIGNOF_INT);
}
#define addBufferInt(a,b,c)			__addBufferInt((a),(b),(c),0)

static inline void
__addBufferLong(FBTableBuf *buf, int index, int64 value, int64 __default)
{
	if (value != __default)
		__addBufferScalar(buf, index, &value, sizeof(value), ALIGNOF_LONG);
}
#define addBufferLong(a,b,c)		__addBufferLong((a),(b),(c),0)

static inline void
addBufferString(FBTableBuf *buf, int index, const char *cstring)
{
	int		slen, blen;
	char   *temp;

	if (cstring && (slen = strlen(cstring)) > 0)
	{
		blen = sizeof(int32) + INTALIGN(slen + 1);
		temp = palloc0(blen);
		*((int32 *)temp) = slen;
		strcpy(temp + sizeof(int32), cstring);
		__addBufferBinary(buf, index, temp, blen, ALIGNOF_INT);
	}
}

static void
addBufferVector(FBTableBuf *buf, int index, int nitems, FBTableBuf **elements)
{
	size_t	len = MAXALIGN(sizeof(int32) + sizeof(int32) * nitems);
	int32	i, *vector;
	char   *pos;
	int		maxalign = ALIGNOF_INT;

	/* calculation of flat buffer length */
	for (i=0; i < nitems; i++)
	{
		FBTableBuf *e = elements[i];

		if (e->length < 0)
			Elog("Bug? FBTableBuf is not flatten");
		len += MAXALIGN(e->length);
	}
	vector = palloc0(len);
	vector[0] = nitems;
	pos = (char *)&vector[1 + nitems];
	for (i=0; i < nitems; i++)
	{
		FBTableBuf *e = elements[i];
		int			gap = TYPEALIGN(e->maxalign,
									e->vtable.vlen) - e->vtable.vlen;

		if (gap > 0)
		{
			memset(pos, 0, gap);
			pos += gap;
		}
		memcpy(pos, &e->vtable, e->length);
		vector[i+1] = (pos + e->vtable.vlen) - (char *)&vector[i+1];
		pos += e->length;

		maxalign = Max(maxalign, e->maxalign);
	}
	__addBufferBinary(buf, index, vector, pos - (char *)vector, maxalign);
}

static FBTableBuf *
__makeBufferFlatten(FBTableBuf *buf, const char *func_name)
{
	FBVtable   *vtable = &buf->vtable;
	size_t		extra_sz = 0;
	int			i;

	assert(vtable->vlen == SHORTALIGN(vtable->vlen) &&
		   vtable->vlen <= offsetof(FBVtable, offset[buf->nattrs]));
	assert(vtable->tlen >= sizeof(int32));

	/* close up the hole between vtable tail and table head if any */
	if (buf->vtable.vlen != offsetof(FBVtable, offset[buf->nattrs]))
	{
		memmove((char *)vtable + vtable->vlen,
				(char *)vtable + offsetof(FBVtable, offset[buf->nattrs]),
				vtable->tlen);
	}
	*((int32 *)((char *)vtable + vtable->vlen)) = vtable->vlen;

	/* check extra buffer usage */
	for (i=0; i < buf->nattrs; i++)
	{
		if (buf->extra_buf[i])
			extra_sz += MAXALIGN(buf->extra_sz[i]);
	}

	if (extra_sz == 0)
		buf->length = vtable->vlen + vtable->tlen;
	else
	{
		char	   *base;
		int32	   *offset;
		size_t		usage, gap;

		buf = repalloc(buf, (offsetof(FBTableBuf, vtable) +
							 MAXALIGN(buf->vtable.vlen) +
							 MAXALIGN(buf->vtable.tlen) + extra_sz));
		vtable = &buf->vtable;
		base   = (char *)vtable + vtable->vlen;
		usage  = vtable->tlen;
		for (i=0; i < buf->nattrs; i++)
		{
			if (!buf->extra_buf[i])
				continue;
			assert(buf->extra_sz[i] > 0);
			assert(buf->extra_align[i] > 0);
			assert(vtable->offset[i] != 0);
			offset = (int32 *)(base + vtable->offset[i]);
			gap = TYPEALIGN(buf->extra_align[i],
							usage + *offset) - (usage + *offset);
			if (gap > 0)
			{
				memset(base + usage, 0, gap);
				usage += gap;
			}
			memcpy(base + usage, buf->extra_buf[i], buf->extra_sz[i]);
			*offset = (base + usage + *offset) - (char *)offset;
			usage += buf->extra_sz[i];
		}
		buf->length = buf->vtable.vlen + usage;
	}
	return buf;
}

#define makeBufferFlatten(a)	__makeBufferFlatten((a),__FUNCTION__)

/*
 * Arrow v0.15 didn't allow Filed->type is null object, even if type has
 * no parameters. So, we inject an empty flat-buffer entry.
 */
static FBTableBuf *
createArrowTypeSimple(void)
{
	FBTableBuf *buf = allocFBTableBuf(0);

	return makeBufferFlatten(buf);
}

static FBTableBuf *
createArrowTypeInt(ArrowTypeInt *node)
{
	FBTableBuf *buf = allocFBTableBuf(2);

	assert(ArrowNodeIs(node, Int));
	addBufferInt(buf, 0, node->bitWidth);
	addBufferBool(buf, 1, node->is_signed);

	return makeBufferFlatten(buf);
}

static FBTableBuf *
createArrowTypeFloatingPoint(ArrowTypeFloatingPoint *node)
{
	FBTableBuf *buf = allocFBTableBuf(1);

	assert(ArrowNodeIs(node, FloatingPoint));
	addBufferShort(buf, 0, node->precision);

	return makeBufferFlatten(buf);
}

static FBTableBuf *
createArrowTypeDecimal(ArrowTypeDecimal *node)
{
	FBTableBuf *buf = allocFBTableBuf(2);

	assert(ArrowNodeIs(node, Decimal));
	addBufferInt(buf, 0, node->precision);
	addBufferInt(buf, 1, node->scale);

	return makeBufferFlatten(buf);
}

static FBTableBuf *
createArrowTypeDate(ArrowTypeDate *node)
{
	FBTableBuf *buf = allocFBTableBuf(1);

	assert(ArrowNodeIs(node, Date));
	__addBufferShort(buf, 0, node->unit, ArrowDateUnit__MilliSecond);

	return makeBufferFlatten(buf);
}

static FBTableBuf *
createArrowTypeTime(ArrowTypeTime *node)
{
	FBTableBuf *buf = allocFBTableBuf(2);

	assert(ArrowNodeIs(node, Time));
	__addBufferShort(buf, 0, node->unit, ArrowTimeUnit__MilliSecond);
	__addBufferInt(buf, 1, node->bitWidth, 32);

	return makeBufferFlatten(buf);
}

static FBTableBuf *
createArrowTypeTimestamp(ArrowTypeTimestamp *node)
{
	FBTableBuf *buf = allocFBTableBuf(2);

	assert(ArrowNodeIs(node, Timestamp));
	addBufferShort(buf, 0, node->unit);
	addBufferString(buf, 1, node->timezone);

	return makeBufferFlatten(buf);
}

static FBTableBuf *
createArrowTypeInterval(ArrowTypeInterval *node)
{
	FBTableBuf *buf = allocFBTableBuf(1);

	assert(ArrowNodeIs(node, Interval));
	addBufferShort(buf, 0, node->unit);

	return makeBufferFlatten(buf);
}

static FBTableBuf *
createArrowTypeUnion(ArrowTypeUnion *node)
{
	FBTableBuf *buf = allocFBTableBuf(2);

	assert(ArrowNodeIs(node, Union));
	addBufferShort(buf, 0, node->mode);
	if (node->_num_typeIds > 0)
	{
		size_t	sz = sizeof(int32) * (node->_num_typeIds + 1);
		int32  *vector = alloca(sz);
		int		i;

		vector[0] = node->_num_typeIds;
		for (i=0; i < node->_num_typeIds; i++)
			vector[i+1] = node->typeIds[i];
		__addBufferBinary(buf, 1, vector, sz, ALIGNOF_INT);
	}
	return makeBufferFlatten(buf);
}

static FBTableBuf *
createArrowTypeFixedSizeBinary(ArrowTypeFixedSizeBinary *node)
{
	FBTableBuf *buf = allocFBTableBuf(1);

	assert(ArrowNodeIs(node, FixedSizeBinary));
	addBufferInt(buf, 0, node->byteWidth);

	return makeBufferFlatten(buf);
}

static FBTableBuf *
createArrowTypeFixedSizeList(ArrowTypeFixedSizeList *node)
{
	FBTableBuf *buf = allocFBTableBuf(1);

	assert(ArrowNodeIs(node, FixedSizeList));
	addBufferInt(buf, 0, node->listSize);

	return makeBufferFlatten(buf);
}

static FBTableBuf *
createArrowTypeMap(ArrowTypeMap *node)
{
	FBTableBuf *buf = allocFBTableBuf(1);

	assert(ArrowNodeIs(node, Map));
	addBufferBool(buf, 0, node->keysSorted);

	return makeBufferFlatten(buf);
}

static FBTableBuf *
createArrowType(ArrowType *node, ArrowTypeTag *p_type_tag)
{
	FBTableBuf	   *buf = NULL;
	ArrowTypeTag	tag;

	switch (ArrowNodeTag(node))
	{
		case ArrowNodeTag__Null:
			tag = ArrowType__Null;
			buf = createArrowTypeSimple();
			break;
		case ArrowNodeTag__Int:
			tag = ArrowType__Int;
			buf = createArrowTypeInt((ArrowTypeInt *)node);
			break;
		case ArrowNodeTag__FloatingPoint:
			tag = ArrowType__FloatingPoint;
			buf = createArrowTypeFloatingPoint((ArrowTypeFloatingPoint *)node);
			break;
		case ArrowNodeTag__Utf8:
			tag = ArrowType__Utf8;
			buf = createArrowTypeSimple();
			break;
		case ArrowNodeTag__Binary:
			tag = ArrowType__Binary;
			buf = createArrowTypeSimple();
			break;
		case ArrowNodeTag__Bool:
			tag = ArrowType__Bool;
			buf = createArrowTypeSimple();
			break;
		case ArrowNodeTag__Decimal:
			tag = ArrowType__Decimal;
			buf = createArrowTypeDecimal((ArrowTypeDecimal *)node);
			break;
		case ArrowNodeTag__Date:
			tag = ArrowType__Date;
			buf = createArrowTypeDate((ArrowTypeDate *)node);
			break;
		case ArrowNodeTag__Time:
			tag = ArrowType__Time;
			buf = createArrowTypeTime((ArrowTypeTime *)node);
			break;
		case ArrowNodeTag__Timestamp:
			tag = ArrowType__Timestamp;
			buf = createArrowTypeTimestamp((ArrowTypeTimestamp *) node);
			break;
		case ArrowNodeTag__Interval:
			tag = ArrowType__Interval;
			buf = createArrowTypeInterval((ArrowTypeInterval *) node);
			break;
		case ArrowNodeTag__List:
			tag = ArrowType__List;
			buf = createArrowTypeSimple();
			break;
		case ArrowNodeTag__Struct:
			tag = ArrowType__Struct;
			buf = createArrowTypeSimple();
			break;
		case ArrowNodeTag__Union:
			tag = ArrowType__Union;
			buf = createArrowTypeUnion((ArrowTypeUnion *) node);
			break;
		case ArrowNodeTag__FixedSizeBinary:
			tag = ArrowType__FixedSizeBinary;
			buf = createArrowTypeFixedSizeBinary((ArrowTypeFixedSizeBinary *)node);
			break;
		case ArrowNodeTag__FixedSizeList:
			tag = ArrowType__FixedSizeList;
			buf = createArrowTypeFixedSizeList((ArrowTypeFixedSizeList *)node);
			break;
		case ArrowNodeTag__Map:
			tag = ArrowType__Map;
			buf = createArrowTypeMap((ArrowTypeMap *)node);
			break;
		default:
			Elog("unknown ArrowNodeTag: %d", ArrowNodeTag(node));
			break;
	}
	*p_type_tag = tag;
	return buf;
}

struct ArrowBufferVector
{
	int32		nitems;
	struct {
		int64	offset;
		int64	length;
	} buffers[FLEXIBLE_ARRAY_MEMBER];
} __attribute__((packed));
typedef struct ArrowBufferVector	ArrowBufferVector;

static void
addBufferArrowBufferVector(FBTableBuf *buf, int index,
						   int nitems, ArrowBuffer *arrow_buffers)
{
	ArrowBufferVector *vector;
	size_t	sz = offsetof(ArrowBufferVector, buffers[nitems]);
	int		i;

	vector = palloc0(sz);
	vector->nitems = nitems;
	for (i=0; i < nitems; i++)
	{
		ArrowBuffer *b = &arrow_buffers[i];

		assert(ArrowNodeIs(b, Buffer));
		vector->buffers[i].offset = b->offset;
		vector->buffers[i].length = b->length;
	}
	__addBufferBinary(buf, index, vector, sz, ALIGNOF_LONG);
}

struct ArrowFieldNodeVector
{
	int32		nitems;
	struct {
		int64	length;
		int64	null_count;
	} nodes[FLEXIBLE_ARRAY_MEMBER];
} __attribute__((packed));
typedef struct ArrowFieldNodeVector	ArrowFieldNodeVector;

static void
addBufferArrowFieldNodeVector(FBTableBuf *buf, int index,
							  int nitems, ArrowFieldNode *elements)
{
	ArrowFieldNodeVector *vector;
	size_t	sz = offsetof(ArrowFieldNodeVector, nodes[nitems]);
	int		i;

	vector = palloc0(sz);
	vector->nitems = nitems;
	for (i=0; i < nitems; i++)
	{
		ArrowFieldNode *f = &elements[i];

		assert(ArrowNodeIs(f, FieldNode));
		vector->nodes[i].length		= f->length;
		vector->nodes[i].null_count	= f->null_count;
	}
	__addBufferBinary(buf, index, vector, sz, ALIGNOF_LONG);
}

static FBTableBuf *
createArrowKeyValue(ArrowKeyValue *node)
{
	FBTableBuf *buf = allocFBTableBuf(2);

	assert(ArrowNodeIs(node, KeyValue));
	addBufferString(buf, 0, node->key);
	addBufferString(buf, 1, node->value);

	return makeBufferFlatten(buf);
}

static FBTableBuf *
createArrowDictionaryEncoding(ArrowDictionaryEncoding *node)
{
	FBTableBuf *buf = allocFBTableBuf(3);
	FBTableBuf *typeInt;

	assert(ArrowNodeIs(node, DictionaryEncoding));
	addBufferLong(buf, 0, node->id);
	typeInt = createArrowTypeInt(&node->indexType);
	addBufferOffset(buf, 1, typeInt);
	addBufferBool(buf, 2, node->isOrdered);

	return makeBufferFlatten(buf);
}

static FBTableBuf *
createArrowField(ArrowField *node)
{
	FBTableBuf	   *buf = allocFBTableBuf(7);
	FBTableBuf	   *dictionary = NULL;
	FBTableBuf	   *type = NULL;
	FBTableBuf	  **vector;
	ArrowTypeTag	type_tag;
	int				i;

	assert(ArrowNodeIs(node, Field));
	addBufferString(buf, 0, node->name);
	addBufferBool(buf, 1, node->nullable);
	type = createArrowType(&node->type, &type_tag);
	addBufferChar(buf, 2, type_tag);
	if (type)
		addBufferOffset(buf, 3, type);
	if (node->dictionary)
	{
		dictionary = createArrowDictionaryEncoding(node->dictionary);
		addBufferOffset(buf, 4, dictionary);
	}
	if (node->_num_children == 0)
		vector = NULL;
	else
	{
		vector = alloca(sizeof(FBTableBuf *) * node->_num_children);
		for (i=0; i < node->_num_children; i++)
			vector[i] =  createArrowField(&node->children[i]);
	}
	addBufferVector(buf, 5, node->_num_children, vector);

	if (node->_num_custom_metadata > 0)
	{
		vector = alloca(sizeof(FBTableBuf *) * node->_num_custom_metadata);
		for (i=0; i < node->_num_custom_metadata; i++)
			vector[i] = createArrowKeyValue(&node->custom_metadata[i]);
		addBufferVector(buf, 6, node->_num_custom_metadata, vector);
	}
	return makeBufferFlatten(buf);
}


static FBTableBuf *
createArrowSchema(ArrowSchema *node)
{
	FBTableBuf	   *buf = allocFBTableBuf(3);
	FBTableBuf	  **fields;
	FBTableBuf	  **cmetadata;
	int				i;

	assert(ArrowNodeIs(node, Schema));
	addBufferBool(buf, 0, node->endianness);
	if (node->_num_fields > 0)
	{
		fields = alloca(sizeof(FBTableBuf *) * node->_num_fields);
		for (i=0; i < node->_num_fields; i++)
			fields[i] = createArrowField(&node->fields[i]);
		addBufferVector(buf, 1, node->_num_fields, fields);
	}
	if (node->_num_custom_metadata > 0)
	{
		cmetadata = alloca(sizeof(FBTableBuf *) *
						   node->_num_custom_metadata);
		for (i=0; i < node->_num_custom_metadata; i++)
			cmetadata[i] = createArrowKeyValue(&node->custom_metadata[i]);
		addBufferVector(buf, 2, node->_num_custom_metadata, cmetadata);
	}
	return makeBufferFlatten(buf);
}

static FBTableBuf *
createArrowRecordBatch(ArrowRecordBatch *node)
{
	FBTableBuf *buf = allocFBTableBuf(3);

	assert(ArrowNodeIs(node, RecordBatch));
	addBufferLong(buf, 0, node->length);
	addBufferArrowFieldNodeVector(buf, 1,
								  node->_num_nodes,
								  node->nodes);
	addBufferArrowBufferVector(buf, 2,
							   node->_num_buffers,
							   node->buffers);
	return makeBufferFlatten(buf);
}

static FBTableBuf *
createArrowDictionaryBatch(ArrowDictionaryBatch *node)
{
	FBTableBuf *buf = allocFBTableBuf(3);
	FBTableBuf *dataBuf;

	assert(ArrowNodeIs(node, DictionaryBatch));
	addBufferLong(buf, 0, node->id);
	dataBuf = createArrowRecordBatch(&node->data);
	addBufferOffset(buf, 1, dataBuf);
	addBufferBool(buf, 2, node->isDelta);

	return makeBufferFlatten(buf);
}

static FBTableBuf *
createArrowMessage(ArrowMessage *node)
{
	FBTableBuf *buf = allocFBTableBuf(4);
	FBTableBuf *data;
	ArrowMessageHeader tag;

	assert(ArrowNodeIs(node, Message));
	addBufferShort(buf, 0, node->version);
	switch (ArrowNodeTag(&node->body))
	{
		case ArrowNodeTag__Schema:
			tag = ArrowMessageHeader__Schema;
			data = createArrowSchema(&node->body.schema);
			break;
		case ArrowNodeTag__DictionaryBatch:
			tag = ArrowMessageHeader__DictionaryBatch;
			data = createArrowDictionaryBatch(&node->body.dictionaryBatch);
			break;
		case ArrowNodeTag__RecordBatch:
			tag = ArrowMessageHeader__RecordBatch;
			data = createArrowRecordBatch(&node->body.recordBatch);
			break;
		default:
			Elog("unexpexted ArrowNodeTag: %d", ArrowNodeTag(node));
			break;
	}
	addBufferChar(buf, 1, tag);
	addBufferOffset(buf, 2, data);
	addBufferLong(buf, 3, node->bodyLength);

	return makeBufferFlatten(buf);
}

struct ArrowBlockVector
{
	int32		nitems;
	struct {
		int64	offset;
		int32	metaDataLength;
		int32	__padding__;
		int64	bodyLength;
	} blocks[FLEXIBLE_ARRAY_MEMBER];
} __attribute__((packed));
typedef struct ArrowBlockVector		ArrowBlockVector;

static void
addBufferArrowBlockVector(FBTableBuf *buf, int index,
						  int nitems, ArrowBlock *arrow_blocks)
{
	ArrowBlockVector *vector;
	size_t	sz = offsetof(ArrowBlockVector, blocks[nitems]);
	int		i;

	vector = palloc0(sz);
    vector->nitems = nitems;
    for (i=0; i < nitems; i++)
	{
		ArrowBlock *b = &arrow_blocks[i];

		assert(ArrowNodeIs(b, Block));
		vector->blocks[i].offset = b->offset;
		vector->blocks[i].metaDataLength = b->metaDataLength;
		vector->blocks[i].bodyLength = b->bodyLength;
	}
	__addBufferBinary(buf, index, vector, sz, ALIGNOF_LONG);
}

static FBTableBuf *
createArrowFooter(ArrowFooter *node)
{
	FBTableBuf	   *buf = allocFBTableBuf(4);
	FBTableBuf	   *schema;

	assert(ArrowNodeIs(node, Footer));
	addBufferShort(buf, 0, node->version);
	schema = createArrowSchema(&node->schema);
	addBufferOffset(buf, 1, schema);
	addBufferArrowBlockVector(buf, 2,
							  node->_num_dictionaries,
							  node->dictionaries);
	addBufferArrowBlockVector(buf, 3,
							  node->_num_recordBatches,
							  node->recordBatches);
	return makeBufferFlatten(buf);
}

/* ----------------------------------------------------------------
 * Routines for File I/O
 * ---------------------------------------------------------------- */

/*
 * writeFlatBufferMessage
 */
typedef struct
{
	int32		continuation;
	int32		metaLength;
	int32		rootOffset;
	char		data[FLEXIBLE_ARRAY_MEMBER];
} FBMessageFileImage;

static ssize_t
writeFlatBufferMessage(int fdesc, ArrowMessage *message)
{
	FBTableBuf *payload = createArrowMessage(message);
	FBMessageFileImage *image;
	ssize_t		offset;
	ssize_t		gap;
	ssize_t		length;
	ssize_t		nbytes;

	assert(payload->length > 0);
	offset = TYPEALIGN(payload->maxalign, payload->vtable.vlen);
	gap = offset - payload->vtable.vlen;
	length = LONGALIGN(offsetof(FBMessageFileImage,
								data[gap + payload->length]));
	image = alloca(length);
	memset(image, 0, length);
	image->continuation = 0xffffffff;
	image->metaLength = length - offsetof(FBMessageFileImage, rootOffset);
	image->rootOffset = sizeof(int32) + offset;
	memcpy(image->data + gap, &payload->vtable, payload->length);

	offset = 0;
	while (offset < length)
	{
		nbytes = write(fdesc, (char *)image + offset, length - offset);
		if (nbytes < 0)
		{
			if (errno == EINTR)
				continue;
			Elog("failed on write: %m");
		}
		offset += nbytes;
	}
	return length;
}

/*
 * writeFlatBufferFooter
 */
typedef struct
{
	int32		rootOffset;
	char		data[FLEXIBLE_ARRAY_MEMBER];
} FBFooterFileImage;

typedef struct
{
	int32		metaOffset;
	char		signature[6];
} FBFooterTailImage;

static ssize_t
writeFlatBufferFooter(int fdesc, ArrowFooter *footer)
{
	FBTableBuf *payload = createArrowFooter(footer);
	FBFooterFileImage *image;
	FBFooterTailImage *tail;
	ssize_t		nbytes;
	ssize_t		offset;
	ssize_t		length;
	uint64		eos = 0xffffffffUL;

	/* put EOS and ensure 64bit alignment */
	nbytes = write(fdesc, &eos, sizeof(uint64));
	if (nbytes != sizeof(uint64))
		Elog("failed on write: %m");

	assert(payload->length > 0);
    offset = INTALIGN(payload->vtable.vlen) - payload->vtable.vlen;
	nbytes = INTALIGN(offset + payload->length);
	length = (offsetof(FBFooterFileImage, data[nbytes]) +
			  offsetof(FBFooterTailImage, signature[6]));
	image = alloca(length + 1);
	image->rootOffset = sizeof(int32) + INTALIGN(payload->vtable.vlen);
	if (offset > 0)
		memset(image->data, 0, offset);
    memcpy(image->data + offset, &payload->vtable, payload->length);
    offset += payload->length;
	if (offset < nbytes)
		memset(image->data + offset, 0, nbytes - offset);
	tail = (FBFooterTailImage *)(image->data + nbytes);
	tail->metaOffset = nbytes + sizeof(int32);
	strcpy(tail->signature, "ARROW1");
	if (write(fdesc, image, length) != length)
		Elog("failed on write: %m");
	return length;
}

static void
setupArrowDictionaryEncoding(ArrowDictionaryEncoding *dict,
							 SQLfield *column)
{
	SQLdictionary  *enumdict = column->enumdict;

	initArrowNode(dict, DictionaryEncoding);
	dict->id = enumdict->dict_id;
	/* dictionary index must be Int32 */
	initArrowNode(&dict->indexType, Int);
	dict->indexType.bitWidth = 32;
	dict->indexType.is_signed = true;
	dict->isOrdered = false;
}

static void
setupArrowField(ArrowField *field, SQLfield *column)
{
	initArrowNode(field, Field);
	field->name = column->field_name;
	field->_name_len = strlen(column->field_name);
	field->nullable = true;
	field->type = column->arrow_type;
	/* dictionary */
	if (column->enumdict)
	{
		field->dictionary = palloc0(sizeof(ArrowDictionaryEncoding));
		setupArrowDictionaryEncoding(field->dictionary, column);
	}
	/* array type */
	if (column->element)
	{
		field->children = palloc0(sizeof(ArrowField));
		field->_num_children = 1;
		setupArrowField(field->children, column->element);
	}
	/* composite type */
	if (column->subfields)
	{
		int		j;

		field->children = palloc0(sizeof(ArrowField) * column->nfields);
		field->_num_children = column->nfields;
		for (j=0; j < column->nfields; j++)
			setupArrowField(&field->children[j], &column->subfields[j]);
	}
	/* custom metadata, if any */
	field->_num_custom_metadata = column->numCustomMetadata;
	field->custom_metadata = column->customMetadata;
}

ssize_t
writeArrowSchema(SQLtable *table)
{
	ArrowMessage	message;
	ArrowSchema	   *schema;
	int32			i;

	/* setup Message of Schema */
	initArrowNode(&message, Message);
	message.version = ArrowMetadataVersion__V4;
	schema = &message.body.schema;
	initArrowNode(schema, Schema);
	schema->endianness = ArrowEndianness__Little;
	schema->fields = alloca(sizeof(ArrowField) * table->nfields);
	schema->_num_fields = table->nfields;
	for (i=0; i < table->nfields; i++)
		setupArrowField(&schema->fields[i], &table->columns[i]);
	schema->custom_metadata = table->customMetadata;
	schema->_num_custom_metadata = table->numCustomMetadata;
	/* serialization */
	return writeFlatBufferMessage(table->fdesc, &message);
}


/*
 * writeArrowDictionaryBatches
 */
static ArrowBlock
__writeArrowDictionaryBatch(int fdesc, SQLdictionary *dict)
{
	ArrowMessage	message;
	ArrowDictionaryBatch *dbatch;
	ArrowRecordBatch *rbatch;
	ArrowFieldNode	fnodes[1];
	ArrowBuffer		buffers[3];
	ArrowBlock		block;
	loff_t			currPos;
	size_t			metaLength = 0;
	size_t			bodyLength = 0;

	initArrowNode(&message, Message);

	/* DictionaryBatch portion */
	dbatch = &message.body.dictionaryBatch;
	initArrowNode(dbatch, DictionaryBatch);
	dbatch->id = dict->dict_id;
	dbatch->isDelta = (dict->nloaded > 0);

	/* ArrowFieldNode of RecordBatch */
	initArrowNode(&fnodes[0], FieldNode);
	fnodes[0].length = dict->nitems - dict->nloaded;
	fnodes[0].null_count = 0;

	/* ArrowBuffer[0] of RecordBatch -- nullmap */
	initArrowNode(&buffers[0], Buffer);
	buffers[0].offset = bodyLength;
	buffers[0].length = 0;

	/* ArrowBuffer[1] of RecordBatch -- offset to extra buffer */
	initArrowNode(&buffers[1], Buffer);
	buffers[1].offset = bodyLength;
	buffers[1].length = ARROWALIGN(dict->values.usage);
	bodyLength += buffers[1].length;

	/* ArrowBuffer[2] of RecordBatch -- extra buffer */
	initArrowNode(&buffers[2], Buffer);
	buffers[2].offset = bodyLength;
	buffers[2].length = ARROWALIGN(dict->extra.usage);
	bodyLength += buffers[2].length;

	/* RecordBatch portion */
	rbatch = &dbatch->data;
	initArrowNode(rbatch, RecordBatch);
	rbatch->length = dict->nitems - dict->nloaded;
	rbatch->_num_nodes = 1;
    rbatch->nodes = fnodes;
	rbatch->_num_buffers = 3;	/* empty nullmap + offset + extra buffer */
	rbatch->buffers = buffers;

	/* final wrap-up message */
    message.version = ArrowMetadataVersion__V4;
	message.bodyLength = bodyLength;

	currPos = lseek(fdesc, 0, SEEK_CUR);
	if (currPos < 0)
		Elog("unable to get current position of the file");
	metaLength = writeFlatBufferMessage(fdesc, &message);
	sql_buffer_write(fdesc, &dict->values);
	sql_buffer_write(fdesc, &dict->extra);

	/* setup Block of Footer */
	initArrowNode(&block, Block);
	block.offset = currPos;
	block.metaDataLength = metaLength;
	block.bodyLength = bodyLength;

	return block;
}

void
writeArrowDictionaryBatches(SQLtable *table)
{
	SQLdictionary  *dict;
	ArrowBlock		block;
	int				index = table->numDictionaries;

	for (dict = table->sql_dict_list; dict; dict = dict->next)
	{
		if (dict->nloaded > 0 && dict->nloaded == dict->nitems)
			continue;		/* nothing to be written */

		if (!table->dictionaries)
			table->dictionaries = palloc0(sizeof(ArrowBlock) * (index+1));
		else
			table->dictionaries = repalloc(table->dictionaries,
										   sizeof(ArrowBlock) * (index+1));
		block = __writeArrowDictionaryBatch(table->fdesc, dict);
		table->dictionaries[index++] = block;
	}
	table->numDictionaries = index;
}

/*
 * setupArrowFieldNode
 */
static int
setupArrowFieldNode(ArrowFieldNode *fnode, SQLfield *column)
{
	SQLfield   *element = column->element;
	int			j, count = 1;

	initArrowNode(fnode, FieldNode);
	fnode->length = column->nitems;
	fnode->null_count = column->nullcount;
	/* array types */
	if (element)
		count += setupArrowFieldNode(fnode + count, element);
	/* composite types */
	if (column->subfields)
	{
		for (j=0; j < column->nfields; j++)
			count += setupArrowFieldNode(fnode + count, &column->subfields[j]);
	}
	return count;
}

/*
 * setupArrowBuffer
 */
static inline size_t
__setup_arrow_buffer(ArrowBuffer *bnode, size_t offset, size_t length)
{
	initArrowNode(bnode, Buffer);
	bnode->offset = offset;
	bnode->length = ARROWALIGN(length);

	return bnode->length;
}

static int
setupArrowBuffer(ArrowBuffer *bnode, SQLfield *column, size_t *p_offset)
{
	size_t		offset = *p_offset;
	int			j, retval = -1;

	if (column->enumdict)
	{
		/* Enum data types */
		assert(column->arrow_type.node.tag == ArrowNodeTag__Utf8);
		/* nullmap */
		if (column->nullcount == 0)
			offset += __setup_arrow_buffer(bnode, offset, 0);
		else
			offset += __setup_arrow_buffer(bnode, offset,
										   column->nullmap.usage);
		/* dictionary indexes (int32) */
		offset += __setup_arrow_buffer(bnode+1, offset,
									   column->values.usage);
		retval = 2;
	}
	else if (column->element)
	{
		/* Array data types */
		retval = 2;
		assert(column->arrow_type.node.tag == ArrowNodeTag__List ||
			   column->arrow_type.node.tag == ArrowNodeTag__LargeList);
		/* nullmap */
		if (column->nullcount == 0)
			offset += __setup_arrow_buffer(bnode, offset, 0);
		else
			offset += __setup_arrow_buffer(bnode, offset,
										   column->nullmap.usage);
		/* array index values */
		offset += __setup_arrow_buffer(bnode+1, offset,
									   column->values.usage);
		retval += setupArrowBuffer(bnode+2, column->element, &offset);
	}
	else if (column->subfields)
	{
		/* Composite data types */
		retval = 1;
		assert(column->arrow_type.node.tag == ArrowNodeTag__Struct);
		/* nullmap */
		if (column->nullcount == 0)
			offset += __setup_arrow_buffer(bnode, offset, 0);
		else
			offset += __setup_arrow_buffer(bnode, offset,
										   column->nullmap.usage);
		/* for each sub-fields */
		for (j=0; j < column->nfields; j++)
			retval += setupArrowBuffer(bnode + retval,
									   &column->subfields[j],
									   &offset);
	}
	else
	{
		switch (column->arrow_type.node.tag)
		{
			/* inline type */
			case ArrowNodeTag__Int:
			case ArrowNodeTag__FloatingPoint:
			case ArrowNodeTag__Bool:
			case ArrowNodeTag__Decimal:
			case ArrowNodeTag__Date:
			case ArrowNodeTag__Time:
			case ArrowNodeTag__Timestamp:
			case ArrowNodeTag__Interval:
			case ArrowNodeTag__FixedSizeBinary:
				retval = 2;
				/* nullmap */
				if (column->nullcount == 0)
					offset += __setup_arrow_buffer(bnode, offset, 0);
				else
					offset += __setup_arrow_buffer(bnode, offset,
												   column->nullmap.usage);
				/* inline values */
				offset += __setup_arrow_buffer(bnode+1, offset,
											   column->values.usage);
				break;

			/* variable length type */
			case ArrowNodeTag__Utf8:
			case ArrowNodeTag__Binary:
			case ArrowNodeTag__LargeUtf8:
			case ArrowNodeTag__LargeBinary:
				retval = 3;
				/* nullmap */
				if (column->nullcount == 0)
					offset += __setup_arrow_buffer(bnode, offset, 0);
				else
					offset += __setup_arrow_buffer(bnode, offset,
												   column->nullmap.usage);
				/* index values */
				offset += __setup_arrow_buffer(bnode+1, offset,
											   column->values.usage);
				/* extra values */
				offset += __setup_arrow_buffer(bnode+2, offset,
											   column->extra.usage);
				break;

			default:
				Elog("Bug? Arrow Type %s is not supported right now",
					 column->arrow_typename);
				break;
		}
	}
	*p_offset = offset;
	return retval;
}

size_t
estimateArrowBufferLength(SQLfield *column, size_t nitems)
{
	size_t		len = 0;
	int			j;

	if (column->nitems != nitems)
		Elog("Bug? number of items mismatch");
	
	if (column->enumdict)
	{
		/* Enum data types */
		assert(column->arrow_type.node.tag == ArrowNodeTag__Utf8);
		if (column->nullcount > 0)
			len += ARROWALIGN(column->nullmap.usage);
		len += ARROWALIGN(column->values.usage);
		assert(column->extra.usage == 0);
	}
	else if (column->element)
	{
		/* Array data types */
		assert(column->arrow_type.node.tag == ArrowNodeTag__List ||
			   column->arrow_type.node.tag == ArrowNodeTag__LargeList);
		if (column->nullcount > 0)
			len += ARROWALIGN(column->nullmap.usage);
		len += ARROWALIGN(column->values.usage);
		assert(column->extra.usage == 0);
		len += estimateArrowBufferLength(column->element, nitems);
	}
	else if (column->subfields)
	{
		/* Composite data types */
		assert(column->arrow_type.node.tag == ArrowNodeTag__Struct);
		if (column->nullcount > 0)
			len += ARROWALIGN(column->nullmap.usage);
		assert(column->values.usage == 0 ||
			   column->extra.usage == 0);
		for (j=0; j < column->nfields; j++)
			len += estimateArrowBufferLength(&column->subfields[j], nitems);
	}
	else
	{
		switch (column->arrow_type.node.tag)
		{
			/* inline type */
			case ArrowNodeTag__Int:
			case ArrowNodeTag__FloatingPoint:
			case ArrowNodeTag__Bool:
			case ArrowNodeTag__Decimal:
			case ArrowNodeTag__Date:
			case ArrowNodeTag__Time:
			case ArrowNodeTag__Timestamp:
			case ArrowNodeTag__Interval:
			case ArrowNodeTag__FixedSizeBinary:
				if (column->nullcount > 0)
					len += ARROWALIGN(column->nullmap.usage);
				len += ARROWALIGN(column->values.usage);
				assert(column->extra.usage == 0);
				break;

			/* variable length type */
			case ArrowNodeTag__Utf8:
			case ArrowNodeTag__Binary:
			case ArrowNodeTag__LargeUtf8:
			case ArrowNodeTag__LargeBinary:
				if (column->nullcount > 0)
					len += ARROWALIGN(column->nullmap.usage);
				len += ARROWALIGN(column->values.usage);
				len += ARROWALIGN(column->extra.usage);
				break;

			default:
				Elog("Bug? Arrow Type %s is not supported right now",
					 column->arrow_typename);
				break;
		}
	}
	return len;
}

static void
writeArrowBuffer(int fdesc, SQLfield *column)
{
	if (column->enumdict)
	{
		/* Enum data types */
		assert(column->arrow_type.node.tag == ArrowNodeTag__Utf8);
		if (column->nullcount > 0)
			sql_buffer_write(fdesc, &column->nullmap);
		sql_buffer_write(fdesc, &column->values);
	}
	else if (column->element)
	{
		/* Array data types */
		assert(column->arrow_type.node.tag == ArrowNodeTag__List ||
			   column->arrow_type.node.tag == ArrowNodeTag__LargeList);
		if (column->nullcount > 0)
			sql_buffer_write(fdesc, &column->nullmap);
		sql_buffer_write(fdesc, &column->values);
		writeArrowBuffer(fdesc, column->element);
	}
	else if (column->subfields)
	{
		int		j;

		/* Composite data types */
		assert(column->arrow_type.node.tag == ArrowNodeTag__Struct);
		if (column->nullcount > 0)
			sql_buffer_write(fdesc, &column->nullmap);
		for (j=0; j < column->nfields; j++)
			writeArrowBuffer(fdesc, &column->subfields[j]);
	}
	else
	{
		switch (column->arrow_type.node.tag)
		{
			/* inline type */
			case ArrowNodeTag__Int:
			case ArrowNodeTag__FloatingPoint:
			case ArrowNodeTag__Bool:
			case ArrowNodeTag__Decimal:
			case ArrowNodeTag__Date:
			case ArrowNodeTag__Time:
			case ArrowNodeTag__Timestamp:
			case ArrowNodeTag__Interval:
			case ArrowNodeTag__FixedSizeBinary:
				if (column->nullcount > 0)
					sql_buffer_write(fdesc, &column->nullmap);
				sql_buffer_write(fdesc, &column->values);
				break;

			/* variable length type */
			case ArrowNodeTag__Utf8:
			case ArrowNodeTag__Binary:
			case ArrowNodeTag__LargeUtf8:
			case ArrowNodeTag__LargeBinary:
				if (column->nullcount > 0)
					sql_buffer_write(fdesc, &column->nullmap);
				sql_buffer_write(fdesc, &column->values);
				sql_buffer_write(fdesc, &column->extra);
				break;

			default:
				Elog("Bug? Arrow Type %s is not supported right now",
					 column->arrow_typename);
				break;
		}
	}
}

static void
sql_field_clear(SQLfield *column)
{
	int		j;
	
	column->nitems = 0;
	column->nullcount = 0;
	sql_buffer_clear(&column->nullmap);
	sql_buffer_clear(&column->values);
	sql_buffer_clear(&column->extra);
	column->__curr_usage__ = 0;

	if (column->element)
		sql_field_clear(column->element);
	if (column->nfields > 0)
	{
		for (j=0; j < column->nfields; j++)
			sql_field_clear(&column->subfields[j]);
	}
}

int
writeArrowRecordBatch(SQLtable *table)
{
	ArrowMessage	message;
	ArrowRecordBatch *rbatch;
	ArrowFieldNode *nodes;
	ArrowBuffer	   *buffers;
	ArrowBlock	   *block;
	int32			i, j;
	int				index;
	off_t			currPos;
	size_t			metaLength;
	size_t			bodyLength = 0;

	assert(table->nitems > 0);
	/* adjust current file position */
	currPos = lseek(table->fdesc, 0, SEEK_CUR);
	if (currPos < 0)
		Elog("unable to get current position of the file");
	if (currPos != LONGALIGN(currPos))
	{
		uint64  zero = 0;
		size_t  gap = LONGALIGN(currPos) - currPos;

		if (write(table->fdesc, &zero, gap) != gap)
			Elog("unable to fill up alignment gap: %m");
	}

	/* fill up [nodes] vector */
	nodes = alloca(sizeof(ArrowFieldNode) * table->numFieldNodes);
	for (i=0, j=0; i < table->nfields; i++)
	{
		assert(table->nitems == table->columns[i].nitems);
		j += setupArrowFieldNode(&nodes[j], &table->columns[i]);
	}
	assert(j == table->numFieldNodes);

	/* fill up [buffers] vector */
	buffers = alloca(sizeof(ArrowBuffer) * table->numBuffers);
	for (i=0, j=0; i < table->nfields; i++)
	{
		j += setupArrowBuffer(&buffers[j], &table->columns[i],
							  &bodyLength);
	}
	assert(j == table->numBuffers);

	/* setup Message of Schema */
	initArrowNode(&message, Message);
	message.version = ArrowMetadataVersion__V4;
	message.bodyLength = bodyLength;

	rbatch = &message.body.recordBatch;
	initArrowNode(rbatch, RecordBatch);
	rbatch->length = table->nitems;
	rbatch->nodes = nodes;
	rbatch->_num_nodes = table->numFieldNodes;
	rbatch->buffers = buffers;
	rbatch->_num_buffers = table->numBuffers;
	/* serialization */
	metaLength = writeFlatBufferMessage(table->fdesc, &message);
	for (j=0; j < table->nfields; j++)
		writeArrowBuffer(table->fdesc, &table->columns[j]);

	/* save the offset/length at ArrowBlock */
	index = table->numRecordBatches++;
	if (index == 0)
		table->recordBatches = palloc(sizeof(ArrowBlock) * 32);
	else
		table->recordBatches = repalloc(table->recordBatches,
										sizeof(ArrowBlock) * (index+1));
	block = &table->recordBatches[index];
	initArrowNode(block, Block);
	block->offset = currPos;
	block->metaDataLength = metaLength;
	block->bodyLength = bodyLength;

	/* make the local buffer empty again */
	for (j=0; j < table->nfields; j++)
		sql_field_clear(&table->columns[j]);
	table->nitems = 0;

	return index;
}

/*
 * writeArrowFooter
 */
ssize_t
writeArrowFooter(SQLtable *table)
{
	ArrowFooter		footer;
	ArrowSchema	   *schema;
	int				i;

	/* setup Footer */
	initArrowNode(&footer, Footer);
	footer.version = ArrowMetadataVersion__V4;

	/* setup Schema of Footer */
	schema = &footer.schema;
	initArrowNode(schema, Schema);
	schema->endianness = ArrowEndianness__Little;
	schema->fields = alloca(sizeof(ArrowField) * table->nfields);
	schema->_num_fields = table->nfields;
	for (i=0; i < table->nfields; i++)
		setupArrowField(&schema->fields[i], &table->columns[i]);
	schema->custom_metadata = table->customMetadata;
	schema->_num_custom_metadata = table->numCustomMetadata;

	/* [dictionaries] */
	footer.dictionaries = table->dictionaries;
	footer._num_dictionaries = table->numDictionaries;

	/* [recordBatches] */
	footer.recordBatches = table->recordBatches;
	footer._num_recordBatches = table->numRecordBatches;

	/* serialization */
	return writeFlatBufferFooter(table->fdesc, &footer);
}
