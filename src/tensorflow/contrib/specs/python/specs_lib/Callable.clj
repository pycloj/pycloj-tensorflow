(ns tensorflow.contrib.specs.python.specs-lib.Callable
  "A composable function that simply defers to a callable function.
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

(defn Callable 
  "A composable function that simply defers to a callable function.
  "
  [ f ]
  (py/call-attr specs-lib "Callable"  f ))

(defn funcall 
  ""
  [ self x ]
  (py/call-attr self "funcall"  self x ))
