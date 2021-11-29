#!/usr/bin/ruby -I .

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
  def showall(hash)
    @af.showall(hash)
  end
  def show(hash)
    @af.show(hash)
  end
end

hash = { "aaa" => "panda",
         "bbb" => "lion",
         "ccc" => "capybara",
         "ddd" => "pug" }

v = MyTest.new("hogehoge unkounko")
v.action
v.showall(hash)
v.show(hash)
