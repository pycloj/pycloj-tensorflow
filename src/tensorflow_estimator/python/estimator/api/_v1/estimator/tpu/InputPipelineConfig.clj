(ns tensorflow-estimator.python.estimator.api.-v1.estimator.tpu.InputPipelineConfig
  "Please see the definition of these values in TPUConfig."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tpu (import-module "tensorflow_estimator.python.estimator.api._v1.estimator.tpu"))

(defn InputPipelineConfig 
  "Please see the definition of these values in TPUConfig."
  [  ]
  (py/call-attr tpu "InputPipelineConfig"  ))
