#!/usr/bin/ruby -I . --debug

require 'ArrowFile'

class MyTest
  def hoge
    proc {
      @af.cleanup
    }
  end
  def initialize(f_name = "/tmp/hoge.arrow")
    @af = ArrowFile.new(f_name)
    ObjectSpace.define_finalizer(self, hoge)
  end
  def action
    @af.action
  end
end

v = MyTest.new("hogehoge")
v.action
