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
	rb_define_method(klass, "cleanup", my_cleanup, 0);
}
