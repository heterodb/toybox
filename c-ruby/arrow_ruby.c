#include <ruby.h>

static VALUE
my_initialize(VALUE self, VALUE arg)
{
	rb_ivar_set(self, rb_intern("monu"), arg);

	printf("my_initialize is called\n");

	return Qnil;
}

static VALUE
my_action(VALUE self)
{
	VALUE	datum;

	datum = rb_ivar_get(self, rb_intern("monu"));

	switch (TYPE(datum))
	{
		case T_STRING:
			printf("my_action is called [%s]\n", StringValueCStr(datum));
			break;
		default:
			printf("my_action is called\n");
			break;
	}
	return Qnil;
}

static int
__showall_callback(VALUE key, VALUE val, VALUE priv)
{
	const char *__key;
	const char *__val;

	__key = StringValueCStr(key);
	__val = StringValueCStr(val);
	
	printf("key=[%s] val=[%s]\n", __key, __val);

	return ST_CONTINUE;
}

static VALUE
my_showall(VALUE self, VALUE arg)
{
	if (TYPE(arg) != T_HASH)
		rb_raise(rb_cHash, "datum is not hash table");

	st_foreach(RHASH_TBL(arg), __showall_callback, Qnil);

	return Qnil;
}

static VALUE
my_show(VALUE self, VALUE hash)
{
	const char *key = "aaa";
	VALUE	datum;

	datum = rb_hash_fetch(hash, rb_str_new_cstr(key));
	if (datum == Qnil)
		printf("key=(%s) datum=NULL\n", key);
	else
		printf("key=(%s) datum=(%s)\n", key, StringValueCStr(datum));

	return Qnil;
}

static VALUE
my_cleanup(VALUE self)
{
	printf("my_cleanup is called\n");

	return Qnil;
}

void
Init_ArrowFile(void)
{
	VALUE	klass;

	klass = rb_define_class("ArrowFile", rb_cObject);
	rb_define_method(klass, "initialize", my_initialize, 1);
	rb_define_method(klass, "action", my_action, 0);
	rb_define_method(klass, "showall", my_showall, 1);
	rb_define_method(klass, "show", my_show, 1);
	rb_define_method(klass, "cleanup", my_cleanup, 0);
}
