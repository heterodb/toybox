#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>

typedef char                cl_bool;
typedef char                cl_char;
typedef unsigned char       cl_uchar;
typedef short               cl_short;
typedef unsigned short      cl_ushort;
typedef int                 cl_int;
typedef unsigned int        cl_uint;
typedef long                cl_long;
typedef unsigned long       cl_ulong;

#define true		((cl_bool) 1)
#define false		((cl_bool) 0)
#define Assert(x)	assert(x)

static cl_uint
__gstoreFdwAllocateRowId(cl_ulong *base, cl_uint nrooms, cl_uint min_id,
						 int depth, cl_uint offset,
						 cl_bool *p_has_unused_rowids)
{
	cl_ulong   *next = NULL;
	cl_ulong	map, mask = (1UL << ((min_id >> 24) & 0x3fU)) - 1;
	cl_uint		rowid = UINT_MAX;
	int			k, start = (min_id >> 30);

//	printf("depth=%d start=%d mask=%016lx\n", depth, start, mask);
	
	if ((offset << 8) >= nrooms)
		return UINT_MAX;	/* obviously, out of range */
	
	switch (depth)
	{
		case 0:
			if (nrooms > 256)
				next = base + 4;
			break;
		case 1:
			if (nrooms > 65536)
				next = base + 4 + (4 << 8);
			break;
		case 2:
			if (nrooms > 16777216)
				next = base + 4 + (4 << 8) + (4 << 16);
			break;
		case 3:
			next = NULL;
			break;
		default:
			*p_has_unused_rowids = false;
			return UINT_MAX;	/* Bug? */
	}
	base += (4 * offset);
	for (k=start; k < 4; k++, mask=0)
	{
		cl_ulong	map = base[k] | mask;
		cl_ulong	bit;
	retry:
		if (map != ~0UL)
		{
			/* lookup the first zero position */
			rowid = (__builtin_ffsl(~map) - 1);
			bit = (1UL << rowid);
			rowid |= (offset << 8) | (k << 6);  /* add offset */

			if (!next)
			{
				if (rowid < nrooms)
					base[k] |= bit;
				else
				{
					rowid = UINT_MAX;	/* not a valid RowID */
				}
			}
			else
			{
				cl_bool		has_unused_rowids;

				rowid = __gstoreFdwAllocateRowId(next, nrooms,
												 min_id << 8,
												 depth+1, rowid,
												 &has_unused_rowids);
				if (!has_unused_rowids)
					base[k] |= bit;
				if (rowid == UINT_MAX)
				{
					//printf("map=%016lx bit=%016lx\n", map, bit);
					map |= bit;
					min_id = 0;
					goto retry;
				}
			}
			break;
		}
	}

	if (p_has_unused_rowids)
	{
		if ((base[0] & base[1] & base[2] & base[3]) == ~0UL)
			*p_has_unused_rowids = false;
		else
			*p_has_unused_rowids = true;
	}
	return rowid;
}

static cl_uint
gstoreFdwAllocateRowId(cl_ulong *base, cl_uint nrooms, cl_uint min_rowid)
{
	if (min_rowid < nrooms)
	{
		if (nrooms <= (1U << 8))
			min_rowid = (min_rowid << 24);
		else if (nrooms <= (1U << 16))
			min_rowid = (min_rowid << 16);
		else if (nrooms <= (1U << 24))
			min_rowid = (min_rowid << 8);
		return __gstoreFdwAllocateRowId(base, nrooms, min_rowid, 0, 0, NULL);
	}
	return UINT_MAX;
}

static cl_bool
__gstoreFdwReleaseRowId(cl_ulong *base, cl_uint nrooms, cl_uint rowid)
{
	cl_ulong   *bottom;
	
	if (rowid < nrooms)
	{
		if (nrooms <= (1U << 8))
		{
			/* single level */
			Assert((rowid & 0xffffff00) == 0);
			bottom = base;
			if ((bottom[rowid >> 6] & (1UL << (rowid & 0x3f)))== 0)
				return false;	/* RowID is not allocated yet */
			base[rowid >> 6] &= ~(1UL << (rowid & 0x3f));
		}
		else if (nrooms <= (1U << 16))
		{
			/* double level */
			Assert((rowid & 0xffff0000) == 0);
			bottom = base + 4;
			if ((bottom[rowid >> 6] & (1UL << (rowid & 0x3f))) == 0)
				return false;	/* RowID is not allocated yet */
			base[(rowid >> 14)] &= ~(1UL << ((rowid >> 8) & 0x3f));
			base += 4;
			base[(rowid >>  6)] &= ~(1UL << (rowid & 0x3f));
		}
		else if (nrooms <= (1U << 24))
		{
			/* triple level */
			Assert((rowid & 0xff000000) == 0);
			bottom = base + 4 + 1024;
			if ((bottom[rowid >> 6] & (1UL << (rowid & 0x3f))) == 0)
				return false;	/* RowID is not allocated yet */
			base[(rowid >> 22)] &= ~(1UL << ((rowid >> 16) & 0x3f));
			base += 4;
			base[(rowid >> 14)] &= ~(1UL << ((rowid >>  8) & 0x3f));
			base += 1024;
			base[(rowid >>  6)] &= ~(1UL << (rowid & 0x3f));
	   }
	   else
	   {
		   /* full level */
		   bottom = base + 4 + 1024 + 262144;
		   if ((bottom[rowid >> 6] & (1UL << (rowid & 0x3f))) == 0)
			   return false;	 /* RowID is not allocated yet */
		   base[(rowid >> 30)] &= ~(1UL << ((rowid >> 24) & 0x3f));
		   base += 4;
		   base[(rowid >> 22)] &= ~(1UL << ((rowid >> 16) & 0x3f));
		   base += 1024;
		   base[(rowid >> 14)] &= ~(1UL << ((rowid >>  8) & 0x3f));
		   base += 262144;
		   base[(rowid >>  6)] &= ~(1UL << (rowid & 0x3f));
	   }
	   return true;
   }
   return false;
}

int main(int argc, char *argv[])
{
	cl_ulong	base[4 + 4 * 256];
	cl_uint		rowid, nrooms = 500;
	cl_bool		rv;
	int			i, j, val;

	memset(base, 0, sizeof(base));
	for (i=1; i < argc; i++)
	{
		val = atoi(argv[i]);

		if (val < 0)
		{
			while (val++ < 0)
			{
				rowid = gstoreFdwAllocateRowId(base, nrooms, 250);
				printf("new RowId = %u\n", rowid);
			}
		}
		else
		{
			rv = __gstoreFdwReleaseRowId(base, nrooms, val);
			printf("release RowId = %u %s\n",
				   val, rv ? "success" : "failed");
		}
		printf("%016lx,%016lx\n"
			   "%016lx,%016lx,%016lx,%016lx,%016lx,%016lx\n",
			   base[1], base[0],
			   base[9], base[8], base[7], base[6], base[5], base[4]);
	}
	return 0;
}
