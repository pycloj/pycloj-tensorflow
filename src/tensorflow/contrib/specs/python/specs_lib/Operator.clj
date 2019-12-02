(ns tensorflow.contrib.specs.python.specs-lib.Operator
  "A wrapper for an operator.

  This takes an operator and an argument list and returns
  the result of applying the operator to the results of applying
  the functions in the argument list.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce specs-lib (import-module "tensorflow.contrib.specs.python.specs_lib"))

(defn Operator 
  "A wrapper for an operator.

  This takes an operator and an argument list and returns
  the result of applying the operator to the results of applying
  the functions in the argument list.
  "
  [ op ]
  (py/call-attr specs-lib "Operator"  op ))

(defn funcall 
  ""
  [ self x ]
  (py/call-attr self "funcall"  self x ))
