(ns tensorflow.contrib.data.Reducer
  "A reducer is used for reducing a set of elements.

  A reducer is represented as a tuple of the three functions:
    1) initialization function: key => initial state
    2) reduce function: (old state, input) => new state
    3) finalization function: state => result
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce data (import-module "tensorflow.contrib.data"))

(defn Reducer 
  "A reducer is used for reducing a set of elements.

  A reducer is represented as a tuple of the three functions:
    1) initialization function: key => initial state
    2) reduce function: (old state, input) => new state
    3) finalization function: state => result
  "
  [ init_func reduce_func finalize_func ]
  (py/call-attr data "Reducer"  init_func reduce_func finalize_func ))

(defn finalize-func 
  ""
  [ self ]
    (py/call-attr self "finalize_func"))

(defn init-func 
  ""
  [ self ]
    (py/call-attr self "init_func"))

(defn reduce-func 
  ""
  [ self ]
    (py/call-attr self "reduce_func"))
