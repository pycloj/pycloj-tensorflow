(ns tensorflow.contrib.learn.python.learn.learn-io.generator-io.FunctionType
  "Create a function object.

  code
    a code object
  globals
    the globals dictionary
  name
    a string that overrides the name from the code object
  argdefs
    a tuple that specifies the default argument values
  closure
    a tuple that supplies the bindings for free variables"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce generator-io (import-module "tensorflow.contrib.learn.python.learn.learn_io.generator_io"))

(defn FunctionType 
  "Create a function object.

  code
    a code object
  globals
    the globals dictionary
  name
    a string that overrides the name from the code object
  argdefs
    a tuple that specifies the default argument values
  closure
    a tuple that supplies the bindings for free variables"
  [ code globals name argdefs closure ]
  (py/call-attr generator-io "FunctionType"  code globals name argdefs closure ))
