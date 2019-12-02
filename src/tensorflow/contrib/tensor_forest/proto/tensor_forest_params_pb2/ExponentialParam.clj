(ns tensorflow.contrib.tensor-forest.proto.tensor-forest-params-pb2.ExponentialParam
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tensor-forest-params-pb2 (import-module "tensorflow.contrib.tensor_forest.proto.tensor_forest_params_pb2"))
