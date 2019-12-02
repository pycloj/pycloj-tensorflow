(ns tensorflow.contrib.tensor-forest.client.random-forest.ModelBuilderOutputType
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce random-forest (import-module "tensorflow.contrib.tensor_forest.client.random_forest"))

(defn ModelBuilderOutputType 
  ""
  [  ]
  (py/call-attr random-forest "ModelBuilderOutputType"  ))
