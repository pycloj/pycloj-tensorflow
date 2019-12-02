(ns tensorflow.contrib.tensor-forest.python.ops.data-ops.TensorForestDataSpec
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce data-ops (import-module "tensorflow.contrib.tensor_forest.python.ops.data_ops"))

(defn TensorForestDataSpec 
  ""
  [  ]
  (py/call-attr data-ops "TensorForestDataSpec"  ))

(defn SerializeToString 
  ""
  [ self  ]
  (py/call-attr self "SerializeToString"  self  ))
