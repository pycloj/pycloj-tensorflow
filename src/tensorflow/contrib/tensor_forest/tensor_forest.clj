(ns tensorflow.contrib.tensor-forest.python.tensor-forest
  "Extremely random forest graph builder. go/brain-tree."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tensor-forest (import-module "tensorflow.contrib.tensor_forest.python.tensor_forest"))

(defn build-params-proto 
  "Build a TensorForestParams proto out of the V4ForestHParams object."
  [ params ]
  (py/call-attr tensor-forest "build_params_proto"  params ))

(defn get-epoch-variable 
  "Returns the epoch variable, or [0] if not defined."
  [  ]
  (py/call-attr tensor-forest "get_epoch_variable"  ))

(defn parse-number-or-string-to-proto 
  ""
  [ proto param ]
  (py/call-attr tensor-forest "parse_number_or_string_to_proto"  proto param ))
