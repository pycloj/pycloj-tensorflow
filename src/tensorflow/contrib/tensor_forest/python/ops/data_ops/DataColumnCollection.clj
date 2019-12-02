(ns tensorflow.contrib.tensor-forest.python.ops.data-ops.DataColumnCollection
  "Collection of DataColumns, meant to mimic a proto repeated field."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce data-ops (import-module "tensorflow.contrib.tensor_forest.python.ops.data_ops"))

(defn DataColumnCollection 
  "Collection of DataColumns, meant to mimic a proto repeated field."
  [  ]
  (py/call-attr data-ops "DataColumnCollection"  ))

(defn SerializeToString 
  ""
  [ self  ]
  (py/call-attr self "SerializeToString"  self  ))

(defn add 
  ""
  [ self  ]
  (py/call-attr self "add"  self  ))

(defn size 
  ""
  [ self  ]
  (py/call-attr self "size"  self  ))
